"""
Component 4: application + bureau.csv aggregates.

Same application-side cleaning and categorical handling as baseline.py, same
walk-forward blocks, same LightGBM config -- the only change is joining the
point-in-time bureau aggregates (bureau_features.py) onto the application
table by SK_ID_CURR before training. Bureau aggregates are all numeric
(counts, sums, means, ratios), so no new categorical columns are introduced.

Applicants with no bureau history, or whose bureau history was entirely
filtered out by the point-in-time cutoff, get NaN across every BUREAU_*
column via this left join -- exactly the missing-not-zero behavior required.
"""

import pandas as pd

from baseline import load_features
from bureau_features import build_bureau_features
from splits import build_split_labels, load_sorted_ids
from train_eval import run_walk_forward


def build_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Returns (features, target, split_label, categorical_columns), aligned by row."""
    app = load_features()
    bureau = build_bureau_features()

    print()
    print("=" * 80)
    print("JOIN")
    print("=" * 80)
    n_before = len(app)
    df = app.merge(bureau, on="SK_ID_CURR", how="left")
    assert len(df) == n_before, "left join must not change row count"
    n_with_bureau = df["BUREAU_COUNT"].notna().sum()
    print(f"  {len(bureau):,} applicants have bureau features; "
          f"{n_with_bureau:,}/{n_before:,} ({100 * n_with_bureau / n_before:.2f}%) of application rows matched")
    print(f"  added {bureau.shape[1] - 1} bureau features")

    split_labels = build_split_labels(load_sorted_ids())
    df = df.merge(split_labels, on="SK_ID_CURR", how="left")

    target = df["TARGET"]
    features = df.drop(columns=["SK_ID_CURR", "TARGET", "split"])

    cat_cols = features.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        features[col] = features[col].astype("category")

    return features, target, df["split"], cat_cols


def main() -> None:
    features, target, split, cat_cols = build_dataset()
    print()
    print("=" * 80)
    print("WALK-FORWARD TRAINING (holdout not touched)")
    print("=" * 80)
    run_walk_forward(features, target, split, cat_cols)


if __name__ == "__main__":
    main()
