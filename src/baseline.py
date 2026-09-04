"""
Component 3: baseline model, application_train only.

No joins, no engineered features: every raw column in application_train.csv
(minus SK_ID_CURR and TARGET) goes in as-is. This is the reference point every
later table addition (bureau, previous_application, ...) has to beat. Trained
and evaluated on the component-2 walk-forward blocks only; the holdout is
never touched here.

Cleaning is deliberately minimal:
  - DAYS_EMPLOYED has a known sentinel value (365243, ~365 years) used to mark
    "not currently employed" (pensioners, unemployed). Left as a raw number it
    reads as a bizarre outlier to the model; replaced with NaN so LightGBM's
    native missing-value handling takes over.
  - Other anomalies (CODE_GENDER 'XNA', ORGANIZATION_TYPE 'XNA', an extreme
    AMT_INCOME_TOTAL outlier) are flagged with counts below but not touched --
    trees split on thresholds/categories directly, so a handful of rare
    category levels or one extreme numeric outlier doesn't distort the fit the
    way it would for a linear model. Fixing them now would be over-engineering
    a baseline that hasn't earned the extra complexity yet.

Categoricals: every object-dtype column is cast to pandas 'category' dtype and
passed to LightGBM's categorical_feature explicitly (not 'auto'-detected) so
it's clear which columns get this treatment. Under the hood, LightGBM does not
one-hot encode: it takes the integer category codes pandas assigns and, at
each split, searches for the best way to partition the category set into two
groups by sorting categories by their gradient statistics -- an optimal
grouped split, not an arbitrary ordinal cut. Missing values (both native NaN
and the DAYS_EMPLOYED replacement above) are handled the same way: LightGBM
learns, per split node, which direction minimizes loss and sends missing rows
there. No manual imputation.

No class-imbalance reweighting (no scale_pos_weight / is_unbalance) in this
baseline -- flagging that explicitly since it's a real modeling choice, just
deferred rather than defaulted-and-hidden. Revisit if/when tuning becomes the
focus.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from splits import DATA_DIR, build_split_labels, load_sorted_ids, walk_forward_folds

DAYS_EMPLOYED_SENTINEL = 365243

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)


def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "application_train.csv")

    print("=" * 80)
    print("ANOMALY CHECK (flagged, not all fixed)")
    print("=" * 80)

    n_sentinel = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum()
    print(f"  DAYS_EMPLOYED == {DAYS_EMPLOYED_SENTINEL}: {n_sentinel:,} rows -> replaced with NaN")
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan

    n_xna_gender = (df["CODE_GENDER"] == "XNA").sum()
    print(f"  CODE_GENDER == 'XNA': {n_xna_gender:,} rows -> left as-is (native categorical handling)")

    n_xna_org = (df["ORGANIZATION_TYPE"] == "XNA").sum()
    print(f"  ORGANIZATION_TYPE == 'XNA': {n_xna_org:,} rows -> left as-is (native categorical handling)")

    income_max = df["AMT_INCOME_TOTAL"].max()
    income_p99 = df["AMT_INCOME_TOTAL"].quantile(0.99)
    print(f"  AMT_INCOME_TOTAL: max={income_max:,.0f} vs 99th pct={income_p99:,.0f} -> left as-is (tree splits, not distance-based)")

    return df


def main() -> None:
    df = load_features()
    split_labels = build_split_labels(load_sorted_ids())
    df = df.merge(split_labels, on="SK_ID_CURR", how="left")

    target = df["TARGET"]
    features = df.drop(columns=["SK_ID_CURR", "TARGET", "split"])

    cat_cols = features.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        features[col] = features[col].astype("category")

    print()
    print("=" * 80)
    print(f"CATEGORICAL COLUMNS ({len(cat_cols)}) -- native LightGBM handling")
    print("=" * 80)
    for col in cat_cols:
        print(f"  {col} ({features[col].cat.categories.size} levels)")

    print()
    print("=" * 80)
    print("WALK-FORWARD TRAINING (holdout not touched)")
    print("=" * 80)

    results = []
    for fold_idx, (train_blocks, val_block) in enumerate(walk_forward_folds()):
        train_labels = [f"block_{b}" for b in train_blocks]
        val_label = f"block_{val_block}"

        train_mask = df["split"].isin(train_labels)
        val_mask = df["split"] == val_label

        X_train, y_train = features[train_mask], target[train_mask]
        X_val, y_val = features[val_mask], target[val_mask]

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_X=X_val, eval_y=y_val,
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        scores = model.predict_proba(X_val)[:, 1]

        pr_auc = average_precision_score(y_val, scores)
        roc_auc = roc_auc_score(y_val, scores)
        lift_10 = compute_lift(y_val.values, scores, 0.10)
        lift_20 = compute_lift(y_val.values, scores, 0.20)

        results.append(dict(
            fold=fold_idx, train_blocks=train_blocks, val_block=val_block,
            n_train=len(X_train), n_val=len(X_val),
            best_iteration=model.best_iteration_,
            pr_auc=pr_auc, roc_auc=roc_auc, lift_10=lift_10, lift_20=lift_20,
        ))

        print(
            f"  fold {fold_idx}: train=blocks{train_blocks} ({len(X_train):,}) "
            f"val=block_{val_block} ({len(X_val):,}) best_iter={model.best_iteration_} "
            f"PR-AUC={pr_auc:.4f} ROC-AUC={roc_auc:.4f} lift@10={lift_10:.2f} lift@20={lift_20:.2f}"
        )

    print()
    print("=" * 80)
    print("SUMMARY (mean +/- std across folds)")
    print("=" * 80)
    res_df = pd.DataFrame(results)
    for metric in ["pr_auc", "roc_auc", "lift_10", "lift_20"]:
        print(f"  {metric:>10s}: mean={res_df[metric].mean():.4f}  std={res_df[metric].std():.4f}  per-fold={res_df[metric].round(4).tolist()}")


def compute_lift(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> float:
    """
    Lift at top top_frac of scored rows: precision among the top-scored slice
    divided by the overall positive rate. 1.0 = no better than random ranking;
    e.g. 3.0 means the top decile is 3x as dense in actual positives as the
    base rate.
    """
    n_top = max(1, int(round(len(y_score) * top_frac)))
    top_idx = np.argsort(-y_score)[:n_top]
    top_rate = y_true[top_idx].mean()
    base_rate = y_true.mean()
    return top_rate / base_rate


if __name__ == "__main__":
    main()
