"""
Component 4: bureau.csv aggregation, point-in-time safe.

Point-in-time cutoff (see DECISIONS.md for the full reasoning):
    DAYS_CREDIT < 0            AND
    DAYS_CREDIT_UPDATE < 0

DAYS_CREDIT is when the bureau credit was applied for, relative to the
current application; DAYS_CREDIT_UPDATE is when the bureau last refreshed
that record's status fields (CREDIT_ACTIVE, AMT_CREDIT_SUM_DEBT, etc.),
also relative to the current application. A row can satisfy the first and
fail the second: an old credit whose bureau-reported status was refreshed
on or after the current application would leak post-decision information
through CREDIT_ACTIVE / AMT_CREDIT_SUM_DEBT / CREDIT_DAY_OVERDUE / etc. even
though the credit itself predates the application. Both cutoffs are
therefore required, applied together, before any aggregation.

Confirmed empirically (not assumed) that this is sufficient: among rows
passing both cutoffs, DAYS_ENDDATE_FACT (actual closure date) never occurs
on or after the application (max -1 day), so no row's closure status is
leaking forward either. DAYS_CREDIT_ENDDATE (the planned/contractual
maturity date) is legitimately positive for ~37% of kept rows -- that's a
known-at-origination future date, not leakage, and is left untouched.

Applicants with no bureau history, or whose entire bureau history fails the
point-in-time filter (19 applicants), get NaN across every aggregate column
via the left join in build_bureau_features -- never 0. Every .sum() below
uses min_count=1 for the same reason: pandas' default sum() of an all-NaN
group is 0, which would silently manufacture a fake "zero" for applicants
whose bureau rows exist but happen to be null on that particular field.
"""

import pandas as pd

from splits import DATA_DIR


def load_bureau_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "bureau.csv")


def filter_point_in_time(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["DAYS_CREDIT"] < 0) & (df["DAYS_CREDIT_UPDATE"] < 0)
    return df[mask].copy()


def aggregate_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """df must already be point-in-time filtered."""
    g = df.groupby("SK_ID_CURR")

    active = df["CREDIT_ACTIVE"] == "Active"
    closed = df["CREDIT_ACTIVE"] == "Closed"

    agg = pd.DataFrame(index=g.size().index)
    agg["BUREAU_COUNT"] = g.size()
    agg["BUREAU_COUNT_ACTIVE"] = df[active].groupby(df["SK_ID_CURR"]).size().reindex(agg.index, fill_value=0)
    agg["BUREAU_COUNT_CLOSED"] = df[closed].groupby(df["SK_ID_CURR"]).size().reindex(agg.index, fill_value=0)
    agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_COUNT_ACTIVE"] / agg["BUREAU_COUNT"]
    agg["BUREAU_CREDIT_TYPE_NUNIQUE"] = g["CREDIT_TYPE"].nunique()

    agg["BUREAU_DAYS_CREDIT_MIN"] = g["DAYS_CREDIT"].min()
    agg["BUREAU_DAYS_CREDIT_MAX"] = g["DAYS_CREDIT"].max()
    agg["BUREAU_DAYS_CREDIT_MEAN"] = g["DAYS_CREDIT"].mean()

    agg["BUREAU_CREDIT_DAY_OVERDUE_MAX"] = g["CREDIT_DAY_OVERDUE"].max()
    agg["BUREAU_CREDIT_DAY_OVERDUE_MEAN"] = g["CREDIT_DAY_OVERDUE"].mean()

    agg["BUREAU_DAYS_CREDIT_ENDDATE_MIN"] = g["DAYS_CREDIT_ENDDATE"].min()
    agg["BUREAU_DAYS_CREDIT_ENDDATE_MAX"] = g["DAYS_CREDIT_ENDDATE"].max()

    agg["BUREAU_CNT_CREDIT_PROLONG_SUM"] = g["CNT_CREDIT_PROLONG"].sum(min_count=1)

    for col in ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE"]:
        agg[f"BUREAU_{col}_SUM"] = g[col].sum(min_count=1)
        agg[f"BUREAU_{col}_MEAN"] = g[col].mean()
        agg[f"BUREAU_{col}_MAX"] = g[col].max()

    # debt utilization: total owed vs total credit ever extended. NaN (not 0)
    # when the sum of AMT_CREDIT_SUM is 0 or missing -- a ratio isn't defined
    # for "no recorded credit amount", so div-by-zero producing inf is wrong too.
    total_credit = agg["BUREAU_AMT_CREDIT_SUM_SUM"]
    total_debt = agg["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"]
    agg["BUREAU_DEBT_CREDIT_RATIO"] = (total_debt / total_credit).where(total_credit > 0)

    agg["BUREAU_AMT_ANNUITY_SUM"] = g["AMT_ANNUITY"].sum(min_count=1)
    agg["BUREAU_AMT_ANNUITY_MEAN"] = g["AMT_ANNUITY"].mean()

    return agg.reset_index()


def build_bureau_features() -> pd.DataFrame:
    raw = load_bureau_raw()
    filtered = filter_point_in_time(raw)
    return aggregate_bureau(filtered)


def main() -> None:
    raw = load_bureau_raw()
    filtered = filter_point_in_time(raw)
    print(f"bureau.csv: {len(raw):,} rows -> {len(filtered):,} after point-in-time filter "
          f"({len(raw) - len(filtered):,} dropped)")
    features = aggregate_bureau(filtered)
    print(f"aggregated to {len(features):,} applicants, {features.shape[1] - 1} features")
    print(features.describe().T[["count", "mean", "min", "max"]])


if __name__ == "__main__":
    main()
