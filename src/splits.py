"""
Component 2: validation split.

No true calendar date exists in this dataset (see DECISIONS.md). SK_ID_CURR
ascending order is the only available ordering proxy across applicants, used
as a stand-in for time. The split is a pure function of that order: sort by
SK_ID_CURR, carve the last HOLDOUT_FRAC as a final holdout touched once at the
end, and cut the remaining rows into N_BLOCKS contiguous chunks used for
walk-forward (expanding-window) model selection. No shuffling, no random
seed anywhere in this file: the split is fully deterministic from the sorted
ID order, so every component that imports this module gets identical splits
without needing to persist and reload a fold-assignment file.

PSI (population stability index) is computed between the earliest and latest
blocks as a diagnostic on how much the ID-order proxy actually moves the
feature distributions. It is informational only: the walk-forward scheme is
the validation scheme regardless of what PSI shows (see DECISIONS.md) because
PSI on observed features cannot detect label drift or a shifted feature-target
relationship, only shifted marginal feature distributions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

HOLDOUT_FRAC = 0.15
N_BLOCKS = 5  # over the remaining ~85%, walk-forward validates on blocks 1..4


def load_sorted_ids() -> pd.DataFrame:
    """application_train sorted ascending by SK_ID_CURR."""
    df = pd.read_csv(DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
    return df.sort_values("SK_ID_CURR", kind="stable").reset_index(drop=True)


def build_split_labels(df: pd.DataFrame, holdout_frac: float = HOLDOUT_FRAC, n_blocks: int = N_BLOCKS) -> pd.DataFrame:
    """
    Assign each row (already sorted by SK_ID_CURR ascending) a split label:
    'block_0' .. 'block_{n_blocks-1}' for the walk-forward portion, or
    'holdout' for the final untouched slice. Purely positional, deterministic.
    """
    n = len(df)
    n_holdout = round(n * holdout_frac)
    n_train_pool = n - n_holdout

    labels = np.empty(n, dtype=object)
    labels[n_train_pool:] = "holdout"

    # np.array_split handles the remainder fairly across blocks (sizes differ
    # by at most 1 row) while keeping blocks contiguous in ID order.
    block_positions = np.array_split(np.arange(n_train_pool), n_blocks)
    for block_idx, positions in enumerate(block_positions):
        labels[positions] = f"block_{block_idx}"

    out = df[["SK_ID_CURR"]].copy()
    out["split"] = labels
    return out


def walk_forward_folds(n_blocks: int = N_BLOCKS) -> list[tuple[list[int], int]]:
    """
    Expanding-window walk-forward folds over block indices 0..n_blocks-1.
    Fold k trains on blocks [0..k] and validates on block k+1. Block 0 is
    never a validation block since there's nothing earlier to train on.
    Returns a list of (train_block_indices, val_block_index).
    """
    return [(list(range(k + 1)), k + 1) for k in range(n_blocks - 1)]


def compute_psi(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index between two samples of the same feature.
    Mechanics: cut the reference sample into `bins` quantile buckets (equal
    population in each, by construction), including a dedicated bucket for
    missing values. Assign the comparison sample into the same bucket edges.
    PSI = sum over buckets of (comp_share - ref_share) * ln(comp_share / ref_share).
    Rule of thumb: <0.10 no meaningful shift, 0.10-0.25 moderate, >0.25 large.
    A small epsilon guards against zero-share buckets in the log/division.
    """
    eps = 1e-6

    ref_null = reference.isna()
    comp_null = comparison.isna()
    ref_valid = reference[~ref_null]
    comp_valid = comparison[~comp_null]

    quantiles = np.unique(np.quantile(ref_valid, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        # not enough distinct values to bin meaningfully (e.g. a near-constant feature)
        return float("nan")
    quantiles[0], quantiles[-1] = -np.inf, np.inf

    ref_bucket = pd.cut(ref_valid, quantiles, duplicates="drop")
    comp_bucket = pd.cut(comp_valid, quantiles, duplicates="drop")

    ref_share = ref_bucket.value_counts(normalize=True, sort=False)
    comp_share = comp_bucket.value_counts(normalize=True, sort=False).reindex(ref_share.index, fill_value=0.0)

    # fold the null rate in as one more bucket on both sides
    ref_null_share = ref_null.mean()
    comp_null_share = comp_null.mean()
    ref_shares = np.append(ref_share.values, ref_null_share)
    comp_shares = np.append(comp_share.values, comp_null_share)

    ref_shares = np.clip(ref_shares, eps, None)
    comp_shares = np.clip(comp_shares, eps, None)

    return float(np.sum((comp_shares - ref_shares) * np.log(comp_shares / ref_shares)))


def main() -> None:
    df = load_sorted_ids()
    split_labels = build_split_labels(df)

    print("=" * 80)
    print("SPLIT SIZES (sorted by SK_ID_CURR ascending)")
    print("=" * 80)
    counts = split_labels["split"].value_counts().reindex(
        [f"block_{i}" for i in range(N_BLOCKS)] + ["holdout"]
    )
    total = len(split_labels)
    for label, n in counts.items():
        print(f"  {label:>10s}: {n:>7,} rows ({100 * n / total:5.2f}%)")
    print(f"  {'total':>10s}: {total:>7,} rows")

    print()
    print("=" * 80)
    print("WALK-FORWARD FOLDS (expanding window, no shuffling)")
    print("=" * 80)
    for i, (train_blocks, val_block) in enumerate(walk_forward_folds()):
        train_n = counts[[f"block_{b}" for b in train_blocks]].sum()
        val_n = counts[f"block_{val_block}"]
        print(f"  fold {i}: train = blocks {train_blocks} ({train_n:,} rows), val = block {val_block} ({val_n:,} rows)")

    # persist the mapping for convenience/debugging; not the source of truth
    # (the source of truth is this file's deterministic functions)
    out_dir = DATA_DIR / "splits"
    out_dir.mkdir(exist_ok=True)
    split_labels.to_csv(out_dir / "application_train_splits.csv", index=False)
    print(f"\n  fold assignment written to {out_dir / 'application_train_splits.csv'} (derived artifact, regenerable)")

    print()
    print("=" * 80)
    print("PSI: block_0 (earliest) vs block_4 (latest pre-holdout) vs holdout")
    print("=" * 80)

    feature_cols = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "DAYS_EMPLOYED",
    ]
    full = pd.read_csv(DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"] + feature_cols)
    full = full.merge(split_labels, on="SK_ID_CURR", how="left").sort_values("SK_ID_CURR", kind="stable")

    ref = full[full["split"] == "block_0"]
    late_block = full[full["split"] == f"block_{N_BLOCKS - 1}"]
    holdout = full[full["split"] == "holdout"]

    print(f"  {'feature':<18s} {'PSI block_0 vs block_4':>24s} {'PSI block_4 vs holdout':>24s}")
    for col in feature_cols:
        psi_within_train = compute_psi(ref[col], late_block[col])
        psi_train_vs_holdout = compute_psi(late_block[col], holdout[col])
        print(f"  {col:<18s} {psi_within_train:>24.4f} {psi_train_vs_holdout:>24.4f}")

    print()
    print("  [INFO] TARGET rate by block (diagnostic only, not PSI: PSI on features cannot see label drift):")
    train_target = load_sorted_ids().merge(split_labels, on="SK_ID_CURR", how="left")
    rate_by_block = train_target.groupby("split")["TARGET"].mean().reindex(
        [f"block_{i}" for i in range(N_BLOCKS)] + ["holdout"]
    )
    for label, rate in rate_by_block.items():
        print(f"    {label:>10s}: {100 * rate:.3f}%")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
