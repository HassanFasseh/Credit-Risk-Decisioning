"""
Leakage tests paired with the bureau.csv point-in-time filter
(src/bureau_features.py). Two levels:
  1. against the real data -- no row surviving the filter violates the cutoff.
  2. against a synthetic case -- the aggregation actually excludes a leaky
     row's values from the result, not just that the filter step looks right
     in isolation.
"""

import pandas as pd

from bureau_features import aggregate_bureau, filter_point_in_time, load_bureau_raw

BUREAU_COLUMNS = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY",
    "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
    "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE",
    "CREDIT_TYPE", "DAYS_CREDIT_UPDATE", "AMT_ANNUITY",
]


def make_row(**overrides) -> dict:
    row = dict(
        SK_ID_CURR=1, SK_ID_BUREAU=1, CREDIT_ACTIVE="Active", CREDIT_CURRENCY="currency 1",
        DAYS_CREDIT=-100, CREDIT_DAY_OVERDUE=0, DAYS_CREDIT_ENDDATE=200, DAYS_ENDDATE_FACT=None,
        AMT_CREDIT_MAX_OVERDUE=None, CNT_CREDIT_PROLONG=0, AMT_CREDIT_SUM=1000.0,
        AMT_CREDIT_SUM_DEBT=500.0, AMT_CREDIT_SUM_LIMIT=0.0, AMT_CREDIT_SUM_OVERDUE=0.0,
        CREDIT_TYPE="Consumer credit", DAYS_CREDIT_UPDATE=-10, AMT_ANNUITY=None,
    )
    row.update(overrides)
    return row


def test_no_post_application_rows_survive_filter():
    raw = load_bureau_raw()
    filtered = filter_point_in_time(raw)

    assert (filtered["DAYS_CREDIT"] >= 0).sum() == 0
    assert (filtered["DAYS_CREDIT_UPDATE"] >= 0).sum() == 0
    assert len(filtered) < len(raw)  # the filter actually removes the known bad rows


def test_aggregation_excludes_leaky_status_update():
    """A credit that predates the application (DAYS_CREDIT < 0) but whose
    bureau status was refreshed on/after the application (DAYS_CREDIT_UPDATE
    >= 0) must not contribute its status-dependent values to the aggregate."""
    df = pd.DataFrame([
        make_row(SK_ID_BUREAU=1, DAYS_CREDIT=-100, DAYS_CREDIT_UPDATE=-10, AMT_CREDIT_SUM=1000.0, CREDIT_ACTIVE="Active"),
        make_row(SK_ID_BUREAU=2, DAYS_CREDIT=-50, DAYS_CREDIT_UPDATE=5, AMT_CREDIT_SUM=99_999.0, CREDIT_ACTIVE="Closed"),
    ])[BUREAU_COLUMNS]

    filtered = filter_point_in_time(df)
    agg = aggregate_bureau(filtered)

    row = agg[agg["SK_ID_CURR"] == 1].iloc[0]
    assert row["BUREAU_COUNT"] == 1
    assert row["BUREAU_AMT_CREDIT_SUM_SUM"] == 1000.0
    assert row["BUREAU_COUNT_ACTIVE"] == 1
    assert row["BUREAU_COUNT_CLOSED"] == 0


def test_aggregation_excludes_leaky_origination_date():
    """A credit applied for on/after the application (DAYS_CREDIT >= 0) must
    be excluded even if its last update timestamp looks fine."""
    df = pd.DataFrame([
        make_row(SK_ID_BUREAU=1, DAYS_CREDIT=-100, DAYS_CREDIT_UPDATE=-10, AMT_CREDIT_SUM=1000.0),
        make_row(SK_ID_BUREAU=2, DAYS_CREDIT=0, DAYS_CREDIT_UPDATE=-1, AMT_CREDIT_SUM=99_999.0),
    ])[BUREAU_COLUMNS]

    filtered = filter_point_in_time(df)
    agg = aggregate_bureau(filtered)

    row = agg[agg["SK_ID_CURR"] == 1].iloc[0]
    assert row["BUREAU_COUNT"] == 1
    assert row["BUREAU_AMT_CREDIT_SUM_SUM"] == 1000.0


def test_applicant_with_only_leaky_rows_is_absent_not_zeroed():
    """If every bureau row for an applicant fails the point-in-time filter,
    that applicant must not appear in the aggregate at all -- the downstream
    left join is what turns "absent" into NaN, never 0."""
    df = pd.DataFrame([
        make_row(SK_ID_CURR=1, SK_ID_BUREAU=1, DAYS_CREDIT=-100, DAYS_CREDIT_UPDATE=-10),
        make_row(SK_ID_CURR=2, SK_ID_BUREAU=2, DAYS_CREDIT=0, DAYS_CREDIT_UPDATE=-1),
    ])[BUREAU_COLUMNS]

    filtered = filter_point_in_time(df)
    agg = aggregate_bureau(filtered)

    assert set(agg["SK_ID_CURR"]) == {1}


def test_all_nan_field_sums_to_nan_not_zero():
    """A groupby sum() over an all-NaN column defaults to 0 in pandas; the
    aggregation must override that (min_count=1) so a genuinely all-missing
    field stays NaN instead of manufacturing a fake zero."""
    df = pd.DataFrame([
        make_row(SK_ID_CURR=1, SK_ID_BUREAU=1, AMT_ANNUITY=None),
        make_row(SK_ID_CURR=1, SK_ID_BUREAU=2, AMT_ANNUITY=None),
    ])[BUREAU_COLUMNS]

    filtered = filter_point_in_time(df)
    agg = aggregate_bureau(filtered)

    row = agg[agg["SK_ID_CURR"] == 1].iloc[0]
    assert pd.isna(row["BUREAU_AMT_ANNUITY_SUM"])
