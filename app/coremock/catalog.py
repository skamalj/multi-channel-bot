"""Loads products.yaml - the single source of truth for every figure.

Two things this file is careful about:

* **Effective dating (KB-6, NF-4).** A product has a current shape and a list
  of dated wording versions. `product(pid)` gives you today's; `product(pid,
  as_of=date)` gives you the one that was in force then. A question about a
  policy incepted in 2025 is answered from the 2025 wording, because that is
  the contract the customer actually bought.
* **Nothing is computed here.** The catalogue returns figures; `rating.py`
  does arithmetic on them and the model does neither.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parents[2] / "data"
LOBS = ("health", "motor")


@lru_cache
def catalog() -> dict:
    return yaml.safe_load((DATA / "products.yaml").read_text(encoding="utf-8"))


def config_version() -> str:
    return str(catalog().get("meta", {}).get("config_version", "unversioned"))


def _as_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def versions(product_id: str) -> list[dict]:
    """Dated wording versions, oldest first. Empty when a product has one."""
    return list(_raw(product_id).get("versions", []))


def _raw(product_id: str) -> dict:
    for lob in LOBS:
        for p in catalog().get(lob, []):
            if p["product_id"] == product_id:
                return p
    raise KeyError(f"unknown product {product_id}")


def version_in_force(product_id: str, as_of: str | date) -> dict | None:
    """The wording version in force on a given date, or None."""
    when = _as_date(as_of)
    for v in versions(product_id):
        frm = _as_date(v.get("effective_from"))
        to = _as_date(v.get("effective_to"))
        if frm and when < frm:
            continue
        if to and when > to:
            continue
        return v
    return None


def product(product_id: str, as_of: str | date | None = None) -> dict:
    """Product as at a date. Without `as_of` you get the current wording.

    The version overlay is a shallow merge of the dated fields onto the
    product, so a caller that does not care about dating sees no difference.
    """
    p = dict(_raw(product_id))
    p["lob"] = lob_of(product_id)
    if as_of is None:
        p["wording_version"] = (versions(product_id)[-1]["version"]
                                if versions(product_id) else "V1")
        return p
    v = version_in_force(product_id, as_of)
    if v is None:
        p["wording_version"] = "unknown"
        return p
    overlay = {k: val for k, val in v.items()
               if k not in ("version", "effective_from", "effective_to", "note")}
    p |= overlay
    p |= {"wording_version": v["version"],
          "wording_effective_from": v.get("effective_from"),
          "wording_effective_to": v.get("effective_to"),
          "wording_note": v.get("note")}
    return p


def lob_of(product_id: str) -> str:
    for lob in LOBS:
        for p in catalog().get(lob, []):
            if p["product_id"] == product_id:
                return lob
    raise KeyError(f"unknown product {product_id}")


def products_for(lob: str, **filters) -> list[dict]:
    """Catalogue listing, filtered on the things a caller can actually vary.

    Filters are applied here rather than left to the model: an age outside a
    product's band is an eligibility fact, not a preference.
    """
    min_age = filters.get("min_age")
    max_age = filters.get("max_age")
    family_size = filters.get("family_size")
    vehicle_class = filters.get("vehicle_class")

    out = []
    for p in catalog().get(lob, []):
        if min_age is not None and min_age > p.get("max_age", 999):
            continue
        if max_age is not None and max_age < p.get("min_age", 0):
            continue
        if vehicle_class and p.get("vehicle_class") != vehicle_class:
            continue
        row = {
            "product_id": p["product_id"],
            "name": p["name"],
            "uin": p["uin"],
            "type": p["type"],
            "lob": lob,
        }
        if lob == "health":
            row |= {
                "sum_insured": p.get("sum_insured"),
                "deductible": p.get("deductible"),
                "entry_age": f"{p['min_age']}-{p['max_age']}",
                "waiting_periods": p.get("waiting_periods"),
                "room_rent": p.get("room_rent"),
                "copay": p.get("copay"),
            }
            if family_size and family_size > 1 and p.get("floater_discount"):
                row["floater_discount_pct"] = round(p["floater_discount"] * 100)
        else:
            row |= {
                "vehicle_class": p.get("vehicle_class"),
                "tp_premium": p.get("tp_premium"),
                "cpa_premium": p.get("cpa_premium"),
                "ncb_slabs": p.get("ncb_slabs"),
            }
        row["addons"] = [a["code"] for a in p.get("addons", [])]
        out.append(row)
    return out


def gates_for(lob: str) -> list[dict]:
    """CO-5: the gate chain that must clear before a policy can issue."""
    return list(catalog().get("gates", {}).get(lob, []))


def underwriting_rules(lob: str) -> dict:
    return dict(catalog().get("underwriting", {}).get(lob, {}))
