"""Rating engine. Deterministic, testable, and the ONLY place a premium
is produced. The model never computes one - it asks for one.

InsureMO-shaped: same inputs, same breakup, same idempotency semantics, so
swapping to the real interface is a config change and not a rewrite.
"""
from __future__ import annotations

from app.coremock.catalog import catalog, product

CITY_ZONE = {
    "mumbai": "A", "navi mumbai": "A", "thane": "A", "delhi": "A",
    "new delhi": "A", "gurgaon": "A", "gurugram": "A", "noida": "A",
    "bengaluru": "B", "bangalore": "B", "pune": "B", "hyderabad": "B",
    "chennai": "B", "kolkata": "B", "ahmedabad": "B", "jaipur": "B",
    "lucknow": "B", "chandigarh": "B", "kochi": "B", "indore": "B",
}


def _zone(city: str) -> str:
    return CITY_ZONE.get((city or "").strip().lower(), "C")


def _age_band(age: int, bands: dict[str, float]) -> float:
    """Look a value up in a banded table.

    Bands are written the way an actuary writes them - "18-35", "56-65",
    "10+" - so the open-ended one has to be read as open-ended rather than
    parsed as an integer. `int("10+")` is the bug this function exists to
    not have.
    """
    for band, factor in bands.items():
        raw = str(band).strip()
        if raw.endswith("+"):
            if age >= int(raw[:-1]):
                return factor
            continue
        lo, _, hi = raw.partition("-")
        if not hi:                                  # a single value, "0"
            if age == int(lo):
                return factor
            continue
        if int(lo) <= age <= int(hi):
            return factor
    return max(bands.values())                      # off the top of the table


def rate_health(product_id: str, sum_insured: int, member_ages: list[int],
                city: str, addons: list[str],
                as_of: str | None = None) -> dict:
    p = product(product_id, as_of=as_of)
    if p["lob"] != "health":
        raise ValueError(f"{product_id} is not a health product")
    if sum_insured not in p["sum_insured"]:
        raise ValueError(
            f"sum insured {sum_insured} not offered on {product_id}; "
            f"available: {p['sum_insured']}")
    if not member_ages:
        raise ValueError("at least one member age is required")
    for age in member_ages:
        if not (p["min_age"] <= age <= p["max_age"]):
            raise ValueError(
                f"age {age} outside the entry band {p['min_age']}-{p['max_age']} "
                f"for {p['name']}")

    lakhs = sum_insured / 100_000
    zone_f = p["zone_factor"][_zone(city)]
    base = p["base_rate_per_lakh"] * lakhs * zone_f

    lines: list[dict] = []
    subtotal = 0.0
    for i, age in enumerate(sorted(member_ages, reverse=True)):
        loading = _age_band(age, p["age_loading"])
        amount = base * loading
        if i > 0:                                   # floater: eldest at full rate
            amount *= (1 - p["floater_discount"])
        amount = round(amount, 2)
        lines.append({"member_age": age, "loading": loading,
                      "amount": amount,
                      "note": "floater discount applied" if i else "primary"})
        subtotal += amount

    addon_lines = []
    by_code = {a["code"]: a for a in p.get("addons", [])}
    for code in addons:
        a = by_code.get(code)
        if not a:
            raise ValueError(
                f"add-on {code} not available on {product_id}; "
                f"available: {sorted(by_code)}")
        amt = round(subtotal * a["rate_pct"], 2) if "rate_pct" in a else a["flat"]
        addon_lines.append({"code": code, "name": a["name"], "amount": amt})
        subtotal += amt

    gst_pct = catalog()["tax"]["gst_pct"]
    gst = round(subtotal * gst_pct / 100, 2)
    return {
        "lob": "health",
        "product_id": product_id,
        "product_name": p["name"],
        "uin": p["uin"],
        "wording_version": p.get("wording_version"),
        "sum_insured": sum_insured,
        "zone": _zone(city),
        "city": city,
        "member_lines": lines,
        "addon_lines": addon_lines,
        "net_premium": round(subtotal, 2),
        "gst_pct": gst_pct,
        "gst": gst,
        "gross_premium": round(subtotal + gst, 2),
        "waiting_periods": p["waiting_periods"],
        "rating_basis": {
            "base_rate_per_lakh": p["base_rate_per_lakh"],
            "zone_factor": p["zone_factor"][_zone(city)],
            "floater_discount": p["floater_discount"],
        },
    }


def rate_motor(product_id: str, idv: int, ncb_pct: int,
               vehicle_age_years: float, addons: list[str],
               as_of: str | None = None) -> dict:
    p = product(product_id, as_of=as_of)
    if p["lob"] != "motor":
        raise ValueError(f"{product_id} is not a motor product")
    if ncb_pct not in p["ncb_slabs"].values():
        raise ValueError(
            f"NCB {ncb_pct}% is not a valid slab; "
            f"valid: {sorted(set(p['ncb_slabs'].values()))}")
    if idv <= 0:
        raise ValueError("IDV must be positive")

    loading = _age_band(int(vehicle_age_years), p["od_age_loading"])
    od_gross = idv * p["od_base_rate_pct"] / 100 * loading
    ncb_amount = round(od_gross * ncb_pct / 100, 2)
    od_net = round(od_gross - ncb_amount, 2)

    addon_lines = []
    by_code = {a["code"]: a for a in p.get("addons", [])}
    addon_total = 0.0
    for code in addons:
        a = by_code.get(code)
        if not a:
            raise ValueError(
                f"add-on {code} not available on {product_id}; "
                f"available: {sorted(by_code)}")
        amt = round(idv * a["rate_pct"] / 100, 2) if "rate_pct" in a else a["flat"]
        addon_lines.append({"code": code, "name": a["name"], "amount": amt})
        addon_total += amt

    tp, cpa = p["tp_premium"], p["cpa_premium"]
    net = round(od_net + addon_total + tp + cpa, 2)
    gst_pct = catalog()["tax"]["gst_pct"]
    gst = round(net * gst_pct / 100, 2)
    return {
        "lob": "motor",
        "product_id": product_id,
        "product_name": p["name"],
        "uin": p["uin"],
        "wording_version": p.get("wording_version"),
        "vehicle_class": p.get("vehicle_class"),
        "idv": idv,
        "vehicle_age_years": vehicle_age_years,
        "od_age_loading": loading,
        "od_gross": round(od_gross, 2),
        "ncb_pct": ncb_pct,
        "ncb_amount": ncb_amount,            # NCB never applies to TP
        "od_net": od_net,
        "tp_premium": tp,
        "cpa_premium": cpa,
        "addon_lines": addon_lines,
        "net_premium": net,
        "gst_pct": gst_pct,
        "gst": gst,
        "gross_premium": round(net + gst, 2),
    }
