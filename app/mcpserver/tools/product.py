"""Product catalogue and quotation.

The premium is computed by the core, never by the model. That is what the
`authority="core"` tag means: the responder must quote these numbers verbatim
and the glass box shows them as decisions.

Note the grain (AG-9). There is no `execute_rest(payload)` here. Each
operation is narrow, typed and business-meaningful, with its own schema, its
own authorization check and its own idempotency key - because a generic core
call hands the model the ability to compose any transaction the core can
perform, which is exactly the authority this design withholds from it.
"""
from __future__ import annotations

from app.coremock import rating, store
from app.coremock.catalog import catalog, products_for
from app.coremock.store import CoreError
from app.mcpserver.registry import tool


def _ids(lob: str):
    return lambda: [p["product_id"] for p in catalog()[lob]]


def _addons(lob: str):
    return lambda: sorted({a["code"] for p in catalog()[lob]
                           for a in p.get("addons", [])})


@tool(tags={"lob": "health", "persona": "customer|agent"}, authority="core")
def product_list_health(min_age: int | None = None,
                        max_age: int | None = None,
                        family_size: int | None = None) -> list[dict]:
    """List eligible HEALTH products with sum-insured options and key terms.

    Filter by the ages actually being proposed - a product outside its entry
    band is not a preference, it is an eligibility fact."""
    return products_for("health", min_age=min_age, max_age=max_age,
                        family_size=family_size)


@tool(tags={"lob": "motor", "persona": "customer|agent"}, authority="core")
def product_list_motor(vehicle_class: str | None = None) -> list[dict]:
    """List eligible MOTOR products. vehicle_class is private_car or
    two_wheeler."""
    return products_for("motor", vehicle_class=vehicle_class)


@tool(tags={"lob": "health", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="identified",
      consent_purpose="quotation",
      choices={"product_id": _ids("health"), "addons": _addons("health")})
def quote_create_health(product_id: str, sum_insured: int,
                        member_ages: list[int], city: str,
                        addons: list[str] | None = None,
                        _idempotency_key: str | None = None) -> dict:
    """Create a rated HEALTH quote. The premium comes from the rating engine,
    never from the model. Returns quote_id, the full premium breakup and the
    validity date."""
    try:
        q = rating.rate_health(product_id, sum_insured, member_ages, city,
                               addons or [])
        return store.save_quote(q, idempotency_key=_idempotency_key)
    except CoreError as exc:
        return exc.as_result()
    except ValueError as exc:
        return {"error": "not_ratable", "detail": str(exc)}


@tool(tags={"lob": "motor", "persona": "customer|agent"},
      effect="write", authority="core", pii=True, auth="identified",
      consent_purpose="quotation",
      choices={"product_id": _ids("motor"), "addons": _addons("motor")})
def quote_create_motor(product_id: str, idv: int, ncb_pct: int,
                       vehicle_age_years: float,
                       addons: list[str] | None = None,
                       _idempotency_key: str | None = None) -> dict:
    """Create a rated MOTOR quote: own damage plus third party plus
    compulsory personal accident, with no claim bonus applied to own damage
    only."""
    try:
        q = rating.rate_motor(product_id, idv, ncb_pct, vehicle_age_years,
                              addons or [])
        return store.save_quote(q, idempotency_key=_idempotency_key)
    except CoreError as exc:
        return exc.as_result()
    except ValueError as exc:
        return {"error": "not_ratable", "detail": str(exc)}


@tool(tags={"lob": "health|motor", "persona": "customer|agent"},
      authority="core", pii=True, auth="identified")
def quote_get(quote_id: str) -> dict:
    """Retrieve a quote, including whether it has expired. An expired quote is
    re-priced by the core, never resurrected."""
    try:
        return store.get_quote(quote_id)
    except CoreError as exc:
        return exc.as_result()
