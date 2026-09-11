"""Product catalogue and quotation.

The premium is computed by the core, never by the model - that is what
`authority="core"` in `extras` means: the responder quotes these numbers
verbatim. Each operation is narrow and typed; there is no generic
`execute(payload)`, because a generic core call would hand the model the
authority to compose any transaction the core can perform, which is exactly
what this design withholds from it.

What each tool IS and what its parameters mean lives here, on the tool. What
to do BEFORE calling it - look products up first, ask before quoting - lives
in the prompt, not in these descriptions.
"""
from __future__ import annotations

from typing import Annotated, Optional

from langchain.tools import tool

from app.agents import confirm
from app.coremock import rating, store
from app.coremock.catalog import products_for
from app.coremock.store import CoreError


@tool(extras={"tags": {"lob": "health", "persona": "customer|agent"},
              "authority": "core"})
def product_list_health(min_age: Optional[int] = None,
                        max_age: Optional[int] = None,
                        family_size: Optional[int] = None) -> list[dict]:
    """List eligible health products with sum-insured options and key terms.

    Filter by the ages actually being proposed - a product outside its entry
    band is an eligibility fact, not a preference."""
    return products_for("health", min_age=min_age, max_age=max_age,
                        family_size=family_size)


@tool(extras={"tags": {"lob": "motor", "persona": "customer|agent"},
              "authority": "core"})
def product_list_motor(
    vehicle_class: Annotated[Optional[str], "private_car or two_wheeler."] = None,
) -> list[dict]:
    """List eligible motor products."""
    return products_for("motor", vehicle_class=vehicle_class)


@tool(extras={"tags": {"lob": "health", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "identified", "consent_purpose": "quotation"})
def quote_create_health(
    product_id: str,
    sum_insured: Annotated[int, "Cover in rupees (10 lakh = 1000000)."],
    member_ages: Annotated[list[int], "Age of each person covered, incl. proposer."],
    city: str,
    addons: Annotated[Optional[list[str]], "Add-on codes, e.g. MAT. Omit for base."] = None,
) -> dict:
    """Create a rated health quote (premium from the core). Returns quote_id,
    the premium breakup and the validity date."""
    key = confirm.token_for("quote_create_health", {
        "product_id": product_id, "sum_insured": sum_insured,
        "member_ages": member_ages, "city": city, "addons": addons})
    try:
        q = rating.rate_health(product_id, sum_insured, member_ages, city,
                               addons or [])
        return store.save_quote(q, idempotency_key=key)
    except CoreError as exc:
        return exc.as_result()
    except ValueError as exc:
        return {"error": "not_ratable", "detail": str(exc)}


@tool(extras={"tags": {"lob": "motor", "persona": "customer|agent"},
              "effect": "write", "authority": "core", "pii": True,
              "auth": "identified", "consent_purpose": "quotation"})
def quote_create_motor(
    product_id: str,
    idv: Annotated[int, "Insured Declared Value in rupees - the agreed value a total loss pays out."],
    ncb_pct: Annotated[int, "No Claim Bonus already earned, as a percentage (0/20/25/35/45/50)."],
    vehicle_age_years: Annotated[float, "Whole years since first registration."],
    addons: Annotated[Optional[list[str]], "Add-on codes, e.g. ZD. Omit for base."] = None,
) -> dict:
    """Create a rated motor quote: own damage plus third party plus compulsory
    personal accident, with no claim bonus applied to own damage only."""
    key = confirm.token_for("quote_create_motor", {
        "product_id": product_id, "idv": idv, "ncb_pct": ncb_pct,
        "vehicle_age_years": vehicle_age_years, "addons": addons})
    try:
        q = rating.rate_motor(product_id, idv, ncb_pct, vehicle_age_years,
                              addons or [])
        return store.save_quote(q, idempotency_key=key)
    except CoreError as exc:
        return exc.as_result()
    except ValueError as exc:
        return {"error": "not_ratable", "detail": str(exc)}


@tool(extras={"tags": {"lob": "health|motor", "persona": "customer|agent"},
              "authority": "core", "pii": True, "auth": "identified"})
def quote_get(quote_id: str) -> dict:
    """Retrieve a quote, including whether it has expired. An expired quote is
    re-priced by the core, never resurrected."""
    try:
        return store.get_quote(quote_id)
    except CoreError as exc:
        return exc.as_result()
