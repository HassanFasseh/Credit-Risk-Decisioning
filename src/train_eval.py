"""
Shared walk-forward training/evaluation loop, extracted out of baseline.py so
every table addition (bureau, previous_application, ...) is scored through
the exact same mechanics as the baseline. Nothing in here is model- or
feature-set-specific.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from splits import walk_forward_folds

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
)


def compute_lift(y_true: np.ndarray, y_score: np.ndarray, top_frac: float) -> float:
    """
    Lift at top top_frac of scored rows: precision among the top-scored slice
    divided by the overall positive rate. 1.0 = no better than random ranking.
    """
    n_top = max(1, int(round(len(y_score) * top_frac)))
    top_idx = np.argsort(-y_score)[:n_top]
    top_rate = y_true[top_idx].mean()
    base_rate = y_true.mean()
    return top_rate / base_rate


def run_walk_forward(features: pd.DataFrame, target: pd.Series, split: pd.Series, cat_cols: list[str]) -> pd.DataFrame:
    """
    Trains/evaluates on the component-2 expanding-window walk-forward blocks.
    features/target/split must be aligned (same index / row order). Returns a
    DataFrame of per-fold metrics; prints per-fold and summary lines.
    """
    results = []
    for fold_idx, (train_blocks, val_block) in enumerate(walk_forward_folds()):
        train_labels = [f"block_{b}" for b in train_blocks]
        val_label = f"block_{val_block}"

        train_mask = split.isin(train_labels)
        val_mask = split == val_label

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

    res_df = pd.DataFrame(results)
    print()
    print("  SUMMARY (mean +/- std across folds)")
    for metric in ["pr_auc", "roc_auc", "lift_10", "lift_20"]:
        print(f"    {metric:>10s}: mean={res_df[metric].mean():.4f}  std={res_df[metric].std():.4f}")
    return res_df
