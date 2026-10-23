"""The features must only depend on data up to the cutoff week.

The test scrambles everything after the cutoff (spend, trips, new transactions, extra
campaigns) and checks that the features don't move. It also checks the label DOES move,
so we know the scrambling actually reached the data.
"""

import pandas as pd
import pytest

from couponroi.features import FEATURES, features_at, labels_at

CUTOFF = 30


def _scramble_future(con, cutoff: int) -> None:
    con.execute(
        f"UPDATE hh_week SET spend = spend * 7 + 50, spend_xf = spend_xf * 7 + 50, trips = trips + 3, "
        f"is_active = true, redemptions = redemptions + 2 WHERE week > {cutoff}"
    )
    con.execute(
        f"UPDATE fct_transactions SET sales_value = sales_value * 9, department = 'DELI' WHERE week > {cutoff}"
    )
    con.execute(f"""INSERT INTO fct_campaign_exposure
                    SELECT household_id, 999, 'Type A', {cutoff} + 1, {cutoff} + 6 FROM dim_household""")


def test_features_ignore_the_future(writable_copy):
    before = features_at(writable_copy, CUTOFF).set_index("household_id")[FEATURES]
    _scramble_future(writable_copy, CUTOFF)
    after = features_at(writable_copy, CUTOFF).set_index("household_id")[FEATURES]
    pd.testing.assert_frame_equal(before.sort_index(), after.sort_index())


def test_labels_do_see_the_future(writable_copy):
    before = labels_at(writable_copy, CUTOFF).set_index("household_id")
    _scramble_future(writable_copy, CUTOFF)
    after = labels_at(writable_copy, CUTOFF).set_index("household_id")
    assert not before["future_weekly"].equals(after["future_weekly"])


@pytest.mark.parametrize("cutoff", [17, 29, 45])
def test_features_have_no_unexpected_nulls(con, cutoff):
    f = features_at(con, cutoff)
    allowed_null = {"momentum_4v16", "momentum_8v16", "basket_value_16w"}  # ratios with a zero denominator
    nulls = f[FEATURES].isna().sum()
    assert not nulls[[c for c in FEATURES if c not in allowed_null]].any(), nulls[nulls > 0]
    assert f["recency_weeks"].between(0, 7).all()  # population = shopped in the last 8 weeks
