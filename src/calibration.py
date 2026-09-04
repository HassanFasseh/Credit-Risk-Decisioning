"""
Component 5: probability calibration.

Slice: model trains on blocks 0-3 (209,108 rows) -- the exact same model that
was fold 3 in components 3/4, early-stopped by monitoring block_4. Calibrator
fits on block_4 (52,276 rows), the last walk-forward validation block: forward
of every row the model trained on, never used for training, and NOT the final
holdout (which stays untouched per the component-2 rule). See DECISIONS.md for
why this still isn't a fully clean out-of-sample calibration check -- block_4
is doing early-stopping, calibrator-fitting, and calibration-evaluation duty
all at once, for lack of a dedicated calibration split.

Isotonic vs Platt: isotonic regression fits a non-decreasing step function via
pool-adjacent-violators (PAV) -- it repeatedly merges adjacent groups of
(sorted-by-raw-score) points whenever their average outcome would otherwise
violate monotonicity, so the result is the least-squares-optimal monotonic
fit to (raw_score, label). It's more flexible than Platt's single sigmoid,
which is a real advantage if miscalibration isn't sigmoid-shaped, but it can
overfit with too few points -- the common rule of thumb is it needs on the
order of 1,000+ calibration examples to be trustworthy. Decision made below
after checking the actual slice size and positive count, not assumed.

Ranking preservation: isotonic is monotonic non-decreasing, so it cannot
change the relative order of two scores that were already different, and
PR-AUC/ROC-AUC depend only on ranking. The one place ranking COULD move is
where isotonic pools multiple distinct raw scores into one tied calibrated
value, which can only ever appear inside a block that PAV decided was already
misordered relative to outcomes -- so this can only matter between fold
outcomes, and it is checked below rather than just asserted.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import lightgbm as lgb
from train_eval import LGBM_PARAMS, compute_lift
from with_bureau import build_dataset

ISOTONIC_MIN_EXAMPLES = 1000


def fit_calibration(verbose: bool = True) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """
    Trains the fold-3 model (blocks 0-3) and calibrates it on block_4.
    Returns (X_calib, y_calib, raw_scores, calibrated_scores) so downstream
    components (e.g. threshold selection) reuse the exact same fit rather
    than retraining and possibly drifting from the calibration numbers
    reported here.
    """
    features, target, split, cat_cols = build_dataset()

    train_mask = split.isin(["block_0", "block_1", "block_2", "block_3"])
    calib_mask = split == "block_4"

    X_train, y_train = features[train_mask], target[train_mask]
    X_calib, y_calib = features[calib_mask], target[calib_mask]

    if verbose:
        print()
        print("=" * 80)
        print("CALIBRATION SLICE")
        print("=" * 80)
        print(f"  train: blocks 0-3, {len(X_train):,} rows, {int(y_train.sum()):,} positive")
        print(f"  calibration slice: block_4, {len(X_calib):,} rows, {int(y_calib.sum()):,} positive")
        print("  holdout: untouched")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_X=X_calib, eval_y=y_calib,
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    raw_scores = model.predict_proba(X_calib)[:, 1]

    n_calib, n_pos = len(X_calib), int(y_calib.sum())
    use_isotonic = n_calib >= ISOTONIC_MIN_EXAMPLES and n_pos >= ISOTONIC_MIN_EXAMPLES
    if verbose:
        print()
        print("=" * 80)
        print("CALIBRATOR CHOICE")
        print("=" * 80)
        print(f"  calibration slice: {n_calib:,} rows, {n_pos:,} positive "
              f"(rule of thumb: need >= {ISOTONIC_MIN_EXAMPLES:,} of each to trust isotonic)")
        print(f"  -> using {'isotonic' if use_isotonic else 'Platt/sigmoid'} regression")

    if use_isotonic:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_scores, y_calib)
        calibrated_scores = calibrator.predict(raw_scores)
    else:
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression()
        calibrator.fit(raw_scores.reshape(-1, 1), y_calib)
        calibrated_scores = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    return X_calib, y_calib, raw_scores, calibrated_scores


def main() -> None:
    X_calib, y_calib, raw_scores, calibrated_scores = fit_calibration()

    print()
    print("=" * 80)
    print("BRIER SCORE (lower is better; 0 = perfect, 0.25 = uninformative at 50% base rate)")
    print("=" * 80)
    brier_before = brier_score_loss(y_calib, raw_scores)
    brier_after = brier_score_loss(y_calib, calibrated_scores)
    print(f"  before calibration: {brier_before:.5f}")
    print(f"  after calibration:  {brier_after:.5f}")
    print(f"  delta: {brier_after - brier_before:+.5f}")

    print()
    print("=" * 80)
    print("RELIABILITY CURVE (10 quantile bins, mean predicted vs observed default rate)")
    print("=" * 80)
    print_reliability("before", y_calib, raw_scores)
    print_reliability("after", y_calib, calibrated_scores)

    print()
    print("=" * 80)
    print("RANKING PRESERVATION (raw vs calibrated, on the calibration slice)")
    print("=" * 80)
    pr_before, pr_after = average_precision_score(y_calib, raw_scores), average_precision_score(y_calib, calibrated_scores)
    roc_before, roc_after = roc_auc_score(y_calib, raw_scores), roc_auc_score(y_calib, calibrated_scores)
    n_distinct_before, n_distinct_after = len(np.unique(raw_scores)), len(np.unique(calibrated_scores))
    lift10_before, lift10_after = compute_lift(y_calib.values, raw_scores, 0.10), compute_lift(y_calib.values, calibrated_scores, 0.10)
    lift20_before, lift20_after = compute_lift(y_calib.values, raw_scores, 0.20), compute_lift(y_calib.values, calibrated_scores, 0.20)
    print(f"  PR-AUC:  before={pr_before:.6f}  after={pr_after:.6f}  delta={pr_after - pr_before:+.6f}")
    print(f"  ROC-AUC: before={roc_before:.6f}  after={roc_after:.6f}  delta={roc_after - roc_before:+.6f}")
    print(f"  lift@10: before={lift10_before:.4f}  after={lift10_after:.4f}  delta={lift10_after - lift10_before:+.4f}")
    print(f"  lift@20: before={lift20_before:.4f}  after={lift20_after:.4f}  delta={lift20_after - lift20_before:+.4f}")
    print(f"  distinct score values: before={n_distinct_before:,}  after={n_distinct_after:,} "
          f"({n_distinct_before - n_distinct_after:,} raw scores pooled into ties by isotonic)")

    # np.argsort is not stable across ties by default in the way compute_lift
    # uses it (quicksort) -- when isotonic produces thousands of exact ties at
    # the decile boundary, which of the tied rows lands "in" the top 10% is
    # arbitrary. Report how many calibration-slice rows sit inside a tied
    # group that straddles the top-10% cutoff, since that's the group whose
    # membership is not well-defined post-calibration.
    order = np.argsort(-calibrated_scores)
    cutoff_score = calibrated_scores[order[int(round(len(order) * 0.10)) - 1]]
    n_at_cutoff_value = int((calibrated_scores == cutoff_score).sum())
    print(f"  rows tied at the exact top-10% cutoff score after calibration: {n_at_cutoff_value:,} "
          f"(their relative order within the cutoff is arbitrary, not a calibration bug)")


def print_reliability(label: str, y_true: pd.Series, scores: np.ndarray) -> None:
    observed, predicted = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    print(f"  {label} calibration:")
    print(f"    {'bin':>4s} {'mean predicted':>15s} {'observed rate':>15s} {'gap':>10s}")
    for i, (p, o) in enumerate(zip(predicted, observed)):
        print(f"    {i:>4d} {p:>15.4f} {o:>15.4f} {o - p:>+10.4f}")
    mean_abs_gap = np.mean(np.abs(observed - predicted))
    print(f"    mean |observed - predicted| across bins: {mean_abs_gap:.4f}")


if __name__ == "__main__":
    main()
