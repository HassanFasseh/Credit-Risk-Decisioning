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
