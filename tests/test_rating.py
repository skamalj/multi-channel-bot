"""The premium is arithmetic, so it is testable. This is the point of keeping
rating out of the model.

Every assertion here is something a customer might one day ask you to
explain, or a regulator might ask you to prove.
"""
import pytest

from app.coremock import rating
from app.coremock.rating import _age_band


def test_health_floater_discount_applies_to_second_member_only():
    one = rating.rate_health("PHS", 1000000, [34], "Pune", [])
    two = rating.rate_health("PHS", 1000000, [34, 37], "Pune", [])
    assert two["member_lines"][0]["note"] == "primary"
    assert two["member_lines"][1]["note"] == "floater discount applied"
    # second member costs less than a standalone primary of the same band
    assert two["member_lines"][1]["amount"] < one["net_premium"]


def test_floater_discount_never_touches_the_eldest():
    """The member who sets the price is always rated at the full rate."""
    q = rating.rate_health("PHS", 500000, [30, 58, 44], "Pune", [])
    ages = [line["member_age"] for line in q["member_lines"]]
    assert ages == [58, 44, 30]                     # eldest first
    assert q["member_lines"][0]["note"] == "primary"


def test_health_zone_changes_premium():
    a = rating.rate_health("PHS", 500000, [40], "Mumbai", [])
    c = rating.rate_health("PHS", 500000, [40], "Nashik", [])
    assert a["zone"] == "A" and c["zone"] == "C"
    assert a["net_premium"] > c["net_premium"]


def test_health_rejects_unoffered_sum_insured():
    with pytest.raises(ValueError, match="not offered"):
        rating.rate_health("PHS", 750000, [30], "Pune", [])


def test_health_rejects_age_outside_the_entry_band():
    with pytest.raises(ValueError, match="outside the entry band"):
        rating.rate_health("PHS", 500000, [72], "Pune", [])


def test_gst_is_applied_once_on_the_total():
    q = rating.rate_health("PHS", 500000, [30], "Pune", ["OPD"])
    assert q["gross_premium"] == pytest.approx(
        round(q["net_premium"] * 1.18, 2), abs=0.02)


def test_motor_ncb_never_touches_third_party():
    no_ncb = rating.rate_motor("PMS", 480000, 0, 3.0, [])
    with_ncb = rating.rate_motor("PMS", 480000, 35, 3.0, [])
    assert with_ncb["tp_premium"] == no_ncb["tp_premium"]
    assert with_ncb["cpa_premium"] == no_ncb["cpa_premium"]
    assert with_ncb["od_net"] < no_ncb["od_net"]
    assert with_ncb["ncb_amount"] > 0


def test_motor_rejects_invalid_ncb_slab():
    with pytest.raises(ValueError, match="not a valid slab"):
        rating.rate_motor("PMS", 480000, 33, 2.0, [])


def test_open_ended_age_band_does_not_crash():
    """"10+" is how an actuary writes a band and how int() raises ValueError.

    A vehicle over ten years old is an ordinary renewal, so this path is on
    the happy road, not an edge case.
    """
    assert _age_band(12, {"0-3": 1.0, "3-5": 1.12, "10+": 1.45}) == 1.45
    old = rating.rate_motor("PMS", 200000, 0, 12.0, [])
    young = rating.rate_motor("PMS", 200000, 0, 1.0, [])
    assert old["od_gross"] > young["od_gross"]


def test_effective_dated_wording_changes_the_waiting_period():
    """KB-6: the same product, rated on two dates, quotes two wordings."""
    old = rating.rate_health("PHS", 500000, [30], "Pune", [], as_of="2025-06-01")
    new = rating.rate_health("PHS", 500000, [30], "Pune", [], as_of="2026-06-01")
    assert old["waiting_periods"]["ped_months"] == 48
    assert new["waiting_periods"]["ped_months"] == 36
    assert old["wording_version"] == "V1" and new["wording_version"] == "V2"
    # The PRICE is unchanged: a wording amendment is not a rate revision.
    assert old["net_premium"] == new["net_premium"]


def test_addon_not_on_the_product_is_refused_by_name():
    with pytest.raises(ValueError, match="not available"):
        rating.rate_health("PHST", 500000, [30], "Pune", ["MAT"])
