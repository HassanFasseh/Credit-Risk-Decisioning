"""
Component 6: cost-based threshold selection.

Cost matrix (agreed with the user, not picked unilaterally -- see DECISIONS.md):
    C_FN(i) = LGD_FRACTION    * AMT_CREDIT_i   -- approve a defaulter: lose LGD% of principal
    C_FP(i) = MARGIN_FRACTION * AMT_CREDIT_i   -- reject a good applicant: lose the margin
    LGD_FRACTION = 0.75, MARGIN_FRACTION = 0.10  ->  ratio 7.5:1

Both costs scale linearly with AMT_CREDIT, so the amount cancels out of the
optimal decision rule (derived below) -- a single flat probability threshold
applies to every applicant regardless of loan size. AMT_CREDIT-scaling is used
only to report costs in dollars; verified empirically below that the flat
(count-weighted) and dollar (amount-weighted) sweeps land on the same
threshold, rather than just asserting the math holds.

Decision rule, derived from minimizing expected cost per applicant:
    approve if p*C_FN(i) < (1-p)*C_FP(i)  =>  approve if p < C_FP(i)/(C_FN(i)+C_FP(i))
Since C_FN(i)/C_FP(i) = LGD_FRACTION/MARGIN_FRACTION regardless of i, this is
    approve if p < MARGIN_FRACTION/(LGD_FRACTION+MARGIN_FRACTION) = 1/(ratio+1)
a single closed-form threshold, reported below alongside the empirical sweep
as a cross-check on whether the calibrated scores behave the way that formula
assumes.

Tie-break rule: decline if calibrated_score >= threshold (not strictly >).
Isotonic calibration (component 5) pooled 52,276 scores into 56 distinct
plateaus, so thousands of applicants can share the exact cutoff value -- this
makes >= vs > a real, visible policy choice, not a rounding detail. Ties go to
decline: when the model is exactly indifferent, tilt toward the cheaper error
given FN costs 7.5x more than FP.
"""

import numpy as np
import pandas as pd

from calibration import fit_calibration

MARGIN_FRACTION = 0.10
RATIO_SCENARIOS = [("5:1 (lower LGD)", 0.50), ("7.5:1 (primary)", 0.75), ("10:1 (higher LGD)", 1.00)]


def evaluate_policy(scores: np.ndarray, y: np.ndarray, cost_fn: np.ndarray, cost_fp: np.ndarray, threshold: float) -> dict:
    """Realized cost and confusion matrix for the rule: decline if score >= threshold."""
    declined = scores >= threshold
    approved = ~declined

    tp = int(np.sum(declined & (y == 1)))   # declined a true defaulter
    fp = int(np.sum(declined & (y == 0)))   # declined a good applicant
    fn = int(np.sum(approved & (y == 1)))   # approved a true defaulter
    tn = int(np.sum(approved & (y == 0)))   # approved a good applicant

    total_cost = cost_fn[approved & (y == 1)].sum() + cost_fp[declined & (y == 0)].sum()
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn, total_cost=total_cost, mean_cost=total_cost / len(y))


def sweep(scores: np.ndarray, y: np.ndarray, cost_fn: np.ndarray, cost_fp: np.ndarray) -> pd.DataFrame:
    """Empirical cost at every candidate threshold (the 56 distinct calibrated
    score plateaus). Cost is a step function between candidates, so these are
    the only thresholds that can possibly change the decision for anyone."""
    candidates = np.unique(scores)
    rows = [evaluate_policy(scores, y, cost_fn, cost_fp, t) for t in candidates]
    return pd.DataFrame(rows)


def main() -> None:
    _, _, X_calib, y_calib, _, calibrated_scores = fit_calibration(verbose=False)
    y = y_calib.values
    amt_credit = X_calib["AMT_CREDIT"].values

    print("=" * 80)
    print("PRIMARY COST MATRIX: LGD=0.75, margin=0.10, ratio=7.5:1")
    print("=" * 80)
    closed_form = MARGIN_FRACTION / (0.75 + MARGIN_FRACTION)
    print(f"  closed-form threshold = margin/(LGD+margin) = {MARGIN_FRACTION}/{0.75 + MARGIN_FRACTION:.2f} = {closed_form:.4f}")

    cost_fn_dollar = 0.75 * amt_credit
    cost_fp_dollar = MARGIN_FRACTION * amt_credit
    cost_fn_flat = np.full_like(amt_credit, 0.75, dtype=float)
    cost_fp_flat = np.full_like(amt_credit, MARGIN_FRACTION, dtype=float)

    sweep_dollar = sweep(calibrated_scores, y, cost_fn_dollar, cost_fp_dollar)
    sweep_flat = sweep(calibrated_scores, y, cost_fn_flat, cost_fp_flat)

    best_dollar = sweep_dollar.loc[sweep_dollar["total_cost"].idxmin()]
    best_flat = sweep_flat.loc[sweep_flat["total_cost"].idxmin()]

    print()
    print(f"  empirical sweep over {len(sweep_dollar)} candidate thresholds (the calibrated score plateaus)")
    print(f"  flat/count-weighted optimum: threshold={best_flat['threshold']:.4f}  total_cost={best_flat['total_cost']:.2f} (unitless) <- DRIVES THE DECISION, per agreed design")
    print(f"  dollar-weighted optimum:  threshold={best_dollar['threshold']:.4f}  total_cost=${best_dollar['total_cost']:,.0f} (reference only)")
    same = np.isclose(best_dollar["threshold"], best_flat["threshold"])
    print(f"  dollar- and flat-weighted sweeps agree on the threshold: {same}")
    if not same:
        print("  they disagree on this finite/discrete grid -- the 'amount cancels out' argument is exact only for")
        print("  a continuously adjustable threshold; on 56 discrete score plateaus, large and small loans are not")
        print("  evenly spread across plateaus, so which plateau minimizes dollar cost vs. count cost can differ.")
        print("  the agreed design uses the flat optimum for the actual decision; dollar figures below are reported")
        print("  AT that same flat-chosen threshold, not re-optimized in dollar terms.")
    print(f"  flat empirical optimum vs closed-form estimate: {best_flat['threshold']:.4f} vs {closed_form:.4f}")

    chosen_threshold = float(best_flat["threshold"])

    print()
    print("=" * 80)
    print(f"CHOSEN THRESHOLD = {chosen_threshold:.4f}  (decline if calibrated_score >= threshold)")
    print("=" * 80)

    chosen = evaluate_policy(calibrated_scores, y, cost_fn_dollar, cost_fp_dollar, chosen_threshold)
    chosen_flat = evaluate_policy(calibrated_scores, y, cost_fn_flat, cost_fp_flat, chosen_threshold)
    print("  confusion matrix (positive = actual default) -- depends only on the threshold, same for both weightings:")
    print(f"    {'':>18s} {'pred: decline':>15s} {'pred: approve':>15s}")
    print(f"    {'actual: default':>18s} {chosen['tp']:>15,d} {chosen['fn']:>15,d}")
    print(f"    {'actual: good':>18s} {chosen['fp']:>15,d} {chosen['tn']:>15,d}")
    n = len(y)
    print(f"  decline rate: {(chosen['tp'] + chosen['fp']) / n:.2%}  ({chosen['tp'] + chosen['fp']:,}/{n:,})")

    print()
    print("=" * 80)
    print("COMPARISON VS BASELINES, evaluated at each policy's own threshold rule")
    print("=" * 80)
    thresholds_to_compare = {
        "chosen (flat 7.5:1 optimum)": chosen_threshold,
        "threshold = 0.5": 0.50,
        "approve-all": np.inf,
        "reject-all": -np.inf,
    }
    print(f"  {'policy':<28s} {'decline rate':>13s} {'unitless cost':>14s} {'total cost ($)':>16s} {'mean cost/applicant ($)':>24s}")
    dollar_results = {}
    for name, t in thresholds_to_compare.items():
        r_dollar = evaluate_policy(calibrated_scores, y, cost_fn_dollar, cost_fp_dollar, t)
        r_flat = evaluate_policy(calibrated_scores, y, cost_fn_flat, cost_fp_flat, t)
        dollar_results[name] = r_dollar
        decline_rate = (r_dollar["tp"] + r_dollar["fp"]) / n
        print(f"  {name:<28s} {decline_rate:>13.2%} {r_flat['total_cost']:>14.2f} {r_dollar['total_cost']:>16,.0f} {r_dollar['mean_cost']:>24.2f}")

    savings_vs_050 = dollar_results["threshold = 0.5"]["total_cost"] - chosen["total_cost"]
    savings_vs_approve_all = dollar_results["approve-all"]["total_cost"] - chosen["total_cost"]
    print()
    print(f"  savings vs 0.5 threshold: ${savings_vs_050:,.0f} "
          f"({100 * savings_vs_050 / dollar_results['threshold = 0.5']['total_cost']:.1f}% reduction)")
    print(f"  savings vs approve-all:   ${savings_vs_approve_all:,.0f} "
          f"({100 * savings_vs_approve_all / dollar_results['approve-all']['total_cost']:.1f}% reduction)")

    print()
    print("=" * 80)
    print("SENSITIVITY: vary LGD_FRACTION (margin held at 0.10), full empirical sweep each time")
    print("=" * 80)
    print(f"  {'scenario':<20s} {'ratio':>7s} {'closed-form t':>14s} {'empirical t (flat)':>19s} {'total cost ($) at that t':>24s}")
    for label, lgd in RATIO_SCENARIOS:
        ratio = lgd / MARGIN_FRACTION
        cf_t = MARGIN_FRACTION / (lgd + MARGIN_FRACTION)
        s_flat = sweep(calibrated_scores, y, np.full_like(amt_credit, lgd, dtype=float), np.full_like(amt_credit, MARGIN_FRACTION, dtype=float))
        best_t = float(s_flat.loc[s_flat["total_cost"].idxmin(), "threshold"])
        dollar_cost_at_best = evaluate_policy(calibrated_scores, y, lgd * amt_credit, MARGIN_FRACTION * amt_credit, best_t)["total_cost"]
        print(f"  {label:<20s} {ratio:>6.1f}:1 {cf_t:>14.4f} {best_t:>19.4f} {dollar_cost_at_best:>24,.0f}")


if __name__ == "__main__":
    main()
