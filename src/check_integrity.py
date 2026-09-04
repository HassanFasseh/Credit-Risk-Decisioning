"""
Component 1: data loading + schema/integrity sanity checks.

Covers, for every table in data/: row counts, application_train's target rate,
primary-key uniqueness where a table defines one, and join-key coverage between
parent and child tables (SK_ID_CURR / SK_ID_PREV / SK_ID_BUREAU).

No feature engineering, no modeling. Only the ID/key columns are loaded for the
large behavioral tables (bureau_balance, POS_CASH_balance, installments_payments,
credit_card_balance) since row counts and key coverage don't need the rest of
the columns.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_ids(filename: str, cols: list[str]) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, usecols=cols)


def report_uniqueness(name: str, df: pd.DataFrame, key: str) -> set:
    n_rows = len(df)
    n_unique = df[key].nunique()
    n_null = df[key].isna().sum()
    status = "OK" if n_unique == n_rows and n_null == 0 else "MISMATCH"
    print(f"  [{status}] {name}: {n_rows:,} rows, {n_unique:,} unique {key}, {n_null:,} null {key}")
    return set(df[key].dropna().unique())


def report_coverage(child_name: str, child_ids: set, parent_name: str, parent_ids: set, key: str) -> None:
    orphans = child_ids - parent_ids
    pct_covered = 100 * (len(child_ids) - len(orphans)) / len(child_ids) if child_ids else float("nan")
    status = "OK" if not orphans else "ORPHANS FOUND"
    print(
        f"  [{status}] {child_name}.{key} -> {parent_name}: "
        f"{len(child_ids):,} distinct values, {pct_covered:.4f}% found in {parent_name}, "
        f"{len(orphans):,} orphaned"
    )


def report_parent_hit_rate(parent_name: str, parent_ids: set, child_name: str, child_ids: set) -> None:
    hits = parent_ids & child_ids
    pct = 100 * len(hits) / len(parent_ids) if parent_ids else float("nan")
    print(
        f"  [INFO] {pct:.2f}% of {parent_name} rows ({len(hits):,}/{len(parent_ids):,}) "
        f"have at least one matching row in {child_name}"
    )


def main() -> None:
    print("=" * 80)
    print("APPLICATION TABLES")
    print("=" * 80)

    train = load_ids("application_train.csv", ["SK_ID_CURR", "TARGET"])
    test = load_ids("application_test.csv", ["SK_ID_CURR"])

    train_ids = report_uniqueness("application_train", train, "SK_ID_CURR")
    test_ids = report_uniqueness("application_test", test, "SK_ID_CURR")

    overlap = train_ids & test_ids
    status = "OK" if not overlap else "OVERLAP FOUND"
    print(f"  [{status}] train/test SK_ID_CURR overlap: {len(overlap):,} ids")

    n_pos = int(train["TARGET"].sum())
    n_total = len(train)
    print(f"  [INFO] TARGET distribution: {n_pos:,} positive / {n_total:,} total = {100 * n_pos / n_total:.3f}%")
    n_target_null = train["TARGET"].isna().sum()
    print(f"  [INFO] TARGET nulls: {n_target_null:,}")

    all_app_ids = train_ids | test_ids

    print()
    print("=" * 80)
    print("BUREAU TABLES")
    print("=" * 80)

    bureau = load_ids("bureau.csv", ["SK_ID_CURR", "SK_ID_BUREAU"])
    bureau_bureau_ids = report_uniqueness("bureau", bureau, "SK_ID_BUREAU")
    bureau_curr_ids = set(bureau["SK_ID_CURR"].dropna().unique())
    report_coverage("bureau", bureau_curr_ids, "application_{train,test}", all_app_ids, "SK_ID_CURR")
    report_parent_hit_rate("application_{train,test}", all_app_ids, "bureau", bureau_curr_ids)

    bureau_balance = load_ids("bureau_balance.csv", ["SK_ID_BUREAU"])
    print(f"  [INFO] bureau_balance: {len(bureau_balance):,} rows")
    bb_bureau_ids = set(bureau_balance["SK_ID_BUREAU"].dropna().unique())
    report_coverage("bureau_balance", bb_bureau_ids, "bureau", bureau_bureau_ids, "SK_ID_BUREAU")
    report_parent_hit_rate("bureau", bureau_bureau_ids, "bureau_balance", bb_bureau_ids)

    print()
    print("=" * 80)
    print("PREVIOUS APPLICATION TABLES")
    print("=" * 80)

    prev = load_ids("previous_application.csv", ["SK_ID_CURR", "SK_ID_PREV"])
    prev_prev_ids = report_uniqueness("previous_application", prev, "SK_ID_PREV")
    prev_curr_ids = set(prev["SK_ID_CURR"].dropna().unique())
    report_coverage("previous_application", prev_curr_ids, "application_{train,test}", all_app_ids, "SK_ID_CURR")
    report_parent_hit_rate("application_{train,test}", all_app_ids, "previous_application", prev_curr_ids)

    for filename, table_name in [
        ("POS_CASH_balance.csv", "POS_CASH_balance"),
        ("installments_payments.csv", "installments_payments"),
        ("credit_card_balance.csv", "credit_card_balance"),
    ]:
        df = load_ids(filename, ["SK_ID_CURR", "SK_ID_PREV"])
        print(f"  [INFO] {table_name}: {len(df):,} rows")
        child_prev_ids = set(df["SK_ID_PREV"].dropna().unique())
        report_coverage(table_name, child_prev_ids, "previous_application", prev_prev_ids, "SK_ID_PREV")

        # cross-check: for rows shared on SK_ID_PREV, does SK_ID_CURR agree with
        # the owning row in previous_application? a mismatch would mean the two
        # tables disagree about which applicant a previous loan belongs to.
        merged = df.drop_duplicates("SK_ID_PREV").merge(
            prev.drop_duplicates("SK_ID_PREV"), on="SK_ID_PREV", how="inner", suffixes=("_child", "_prev")
        )
        mismatches = (merged["SK_ID_CURR_child"] != merged["SK_ID_CURR_prev"]).sum()
        status = "OK" if mismatches == 0 else "MISMATCH"
        print(f"  [{status}] {table_name} vs previous_application: SK_ID_CURR agreement on shared SK_ID_PREV, {mismatches:,} mismatches")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
