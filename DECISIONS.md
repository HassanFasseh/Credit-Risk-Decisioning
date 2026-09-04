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
