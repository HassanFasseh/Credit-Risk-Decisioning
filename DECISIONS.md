# Decisions

## Validation split (component 2)

**Decision:** no true calendar date exists in this dataset, only per-applicant
relative DAYS_* offsets. Use SK_ID_CURR ascending order as a proxy for time.
Sort application_train by SK_ID_CURR, carve the last 15% as a final holdout
(untouched until the end), and split the remaining 85% into 5 contiguous
blocks used for expanding-window walk-forward validation (train on blocks
0..k, validate on block k+1; no shuffling at any stage).

**Rejected alternative:** plain random k-fold (even stratified by TARGET).
Rejected regardless of measured drift, because it assumes away exactly the
risk we can't verify — that applicants who applied later differ systematically
from earlier ones (macro conditions, underwriting policy, population mix). A
random split would let "future-like" rows leak into training and could
overstate offline performance relative to production, where the model only
ever scores applicants who haven't applied yet.

**Rejected alternative:** single ID-ordered holdout with no walk-forward
blocks. Simpler, but gives no tuning signal without either leaking the holdout
or falling back to random CV for hyperparameter search, which would contradict
the point of the exercise.

**PSI diagnostic:** computed PSI on 7 representative features (EXT_SOURCE_1/2/3,
AMT_INCOME_TOTAL, AMT_CREDIT, DAYS_BIRTH, DAYS_EMPLOYED) between block_0 vs
block_4 and block_4 vs holdout. All values were 0.0002-0.0007, far under the
0.10 "no shift" threshold; TARGET rate was flat across blocks (7.9-8.2%).

**This does not change the decision.** PSI on observed features cannot detect
label drift or a shifted feature-target relationship, only shifted marginal
feature distributions. A flat PSI reading here means the split cost nothing
relative to random on these particular features, not that order is safe to
ignore. The walk-forward scheme is the validation scheme regardless of what
PSI shows, at every stage (tuning and final evaluation), not just at the final
holdout gate.

**Open item:** this split reserves no dedicated block for probability
calibration. When the calibration component is built, the calibration slice
must be a forward block relative to whatever data trained the final model
(not a random carve-out of training), sitting between the walk-forward
training data and the final holdout in ID order. Sizing for that block still
needs to be carved out of the current scheme.

## Baseline model (component 3)

**Decision:** LightGBM (sklearn API) on application_train only, no joins, no
engineered features -- every raw column except SK_ID_CURR/TARGET goes in
as-is. This is the reference every later table addition must beat. Trained
and evaluated on the component-2 walk-forward blocks; holdout untouched.
n_estimators=1000 with early stopping (50 rounds, monitored on ROC-AUC --
the only metric LightGBM computes natively without a custom feval callback);
PR-AUC and lift are computed independently after fitting and are what we
actually judge the model on, not what stops training. learning_rate=0.05,
num_leaves=31, random_state=42 (all defaults/conventional choices, not tuned;
tuning is a separate later step once a table's inclusion is already justified).

**Categoricals:** 16 object-dtype columns cast to pandas 'category' and passed
explicitly to `categorical_feature` (not `'auto'`-detected), so it's on record
which columns get LightGBM's native grouped-split handling instead of a
one-hot or ordinal encoding.

**Cleaning, minimal by design:** DAYS_EMPLOYED's 365243 sentinel (55,374 rows,
"not currently employed") replaced with NaN. Noted but did not act on: this
count exactly matches ORGANIZATION_TYPE=='XNA' (55,374 rows) -- same
population flagged two different ways; left as two separate raw signals for
now rather than collapsing them, since that's a feature-engineering call, not
a baseline-cleaning one. CODE_GENDER=='XNA' (4 rows) and an AMT_INCOME_TOTAL
outlier (max 117M vs 99th pct 472.5K) flagged, left untouched -- trees split
on thresholds/category groups, not distances, so neither needs pre-baseline
handling.

**Rejected:** class-imbalance reweighting (scale_pos_weight / is_unbalance).
Not because it's wrong, but because this is the reference baseline -- adding
a reweighting knob here would make later comparisons ("did bureau.csv help?")
conflate two changes at once. Deferred to a tuning-focused component.

**Result (mean over 4 walk-forward folds, holdout untouched):**
PR-AUC = 0.2372 (std 0.0092), ROC-AUC = 0.7515 (reference only, std 0.0075),
lift@10 = 3.24 (std 0.14), lift@20 = 2.56 (std 0.07).

## Bureau features, point-in-time cutoff (component 4)

**Decision:** aggregate bureau.csv per SK_ID_CURR using only rows where
`DAYS_CREDIT < 0 AND DAYS_CREDIT_UPDATE < 0`, both strict. Computed the actual
distributions before choosing this rather than assuming: DAYS_CREDIT ranges
[-2922, 0] with only 25 rows at exactly 0 and none positive; DAYS_CREDIT_UPDATE
ranges [-41947, 372] with 605 rows at exactly 0 and 17 positive. Combined
filter drops 646 of 1,716,428 rows (0.038%) and costs 19 applicants their
entire bureau history (they correctly fall back to all-NaN, not zero).

**Why both fields, not just DAYS_CREDIT:** DAYS_CREDIT is the credit's fixed
origination date; DAYS_CREDIT_UPDATE is the "as-of" timestamp for every
mutable status field on that row (CREDIT_ACTIVE, AMT_CREDIT_SUM_DEBT,
CREDIT_DAY_OVERDUE, ...). A credit that predates the application can still
have its bureau-reported status refreshed on or after the application,
which would leak post-decision information through those status fields even
though DAYS_CREDIT itself looks safe. Verified empirically rather than
assumed: among rows passing both cutoffs, DAYS_ENDDATE_FACT (actual closure
date) never occurs on/after the application (max -1 day) -- confirming
DAYS_CREDIT_UPDATE really does bound every status field on the row, not just
the ones directly aggregated.

**Not leakage, left untouched:** DAYS_CREDIT_ENDDATE (planned/contractual
maturity date) is positive for ~37% of kept rows. That's a future date known
at origination (e.g. "this loan matures in 3 years"), not information that
arrived after the application decision.

**Missing-not-zero:** applicants with no bureau history, or whose entire
history fails the point-in-time filter, get NaN across every BUREAU_* column
via a left join, never 0 -- absence of a group in the post-filter groupby
naturally produces this. Separately, every `.sum()` aggregation uses
`min_count=1`: pandas' default `sum()` over an all-NaN group returns 0, which
would silently manufacture a fake zero for an applicant who has bureau rows
but happens to be null on that specific field (e.g. AMT_ANNUITY is often
missing). Both failure modes are covered by tests/test_bureau_leakage.py.

**Result (mean over 4 walk-forward folds, holdout untouched):**
PR-AUC = 0.2457 (std 0.0108) vs baseline 0.2372, +0.0085 (+3.6% relative).
ROC-AUC = 0.7575 (reference only) vs 0.7515.
lift@10 = 3.33 vs 3.24, lift@20 = 2.61 vs 2.56.
Improvement holds in all 4 folds individually, not just on average -- bureau
history earns its place in v1.1.

## Probability calibration (component 5)

**Decision:** calibration slice = block_4 (the last walk-forward validation
block), model trained on blocks 0-3 (the same model that was fold 3 in
components 3/4). Resolves the component-2 open item: forward of every row the
model trained on, never used for training, not the final holdout.

**Named cost of not having a dedicated calibration split:** block_4 now does
three jobs at once -- early-stopping monitor for the base model's tree count,
isotonic calibrator fit, and calibration-quality evaluation. Fitting and
evaluating the calibrator on the same data is optimistic in general. Isotonic
regression's effective flexibility is much lower than a general model (see
below: it collapsed to 56 plateaus over 52,276 points), which limits how much
this can be overfitting rather than genuine signal, but it isn't zero. The
holdout is the only place this gets a real out-of-sample check, and that
happens once, at the end.

**Isotonic vs Platt:** used isotonic. Calibration slice has 52,276 rows /
4,178 positives, both comfortably over the ~1,000-example rule of thumb for
trusting isotonic over Platt's single-sigmoid fit.

**Result -- calibration quality:** Brier score 0.06640 -> 0.06618 (small
improvement; this dataset's ~8% base rate caps how large a Brier improvement
can look regardless of calibration quality). Reliability curve mean
|observed - predicted| across 10 quantile bins: 0.0031 -> 0.0000 (isotonic is
fit by construction to match observed rate on its own training data --
expected, not a surprise, and part of why this reliability number specifically
needs the holdout to mean anything as a true generalization check).

**Result -- ranking preservation, checked not assumed:** isotonic pooled
52,220 of 52,276 raw scores into just 56 distinct calibrated values. This
lift-corrected ROC-AUC (+0.0014) and both business metrics (lift@10 +0.019,
lift@20 +0.004) essentially unchanged or marginally better, but PR-AUC
dropped 0.2608 -> 0.2557 (-0.0050, -1.9% relative) -- not fully "unchanged".
Reasoned explanation, not hand-waved: 56 plateaus over 52,276 points at an 8%
base rate is close to what's needed for each plateau to have enough positives
(~75 each) for a low-variance frequency estimate; PAV is doing what it's
supposed to given how much data actually supports distinguishing probability
levels this finely. PR-AUC is sensitive to fine-grained tie-breaking across
the full precision-recall curve in a way lift@k and ROC-AUC at the decile
level are not, which is why it moved and the others didn't. 1,869 calibration-
slice rows are tied at the exact score value straddling the top-10% cutoff --
their relative order within that tie is arbitrary post-calibration, which
matters for lift-based portfolio cuts and should stay visible, not buried.
