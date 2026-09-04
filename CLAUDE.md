# Project: credit-default risk decisioning system (PFE portfolio)

## What this is
A credit-default prediction system on the Home Credit dataset, for my end-of-studies
portfolio. Predicts whether a loan applicant will default, using the applicant's data
plus credit-bureau and prior-application history. Deployed as a FastAPI service, not a
notebook or Streamlit demo.

## Hard rule: I must understand and defend every line
For technical interviews across banks, consulting, and general DS roles.
- Explain the approach in plain English BEFORE writing code, wait for my OK.
- One component at a time. Small diffs. Stop after each.
- After each piece, give me the 2-3 interview questions it raises and check I can answer.
- On any modeling choice (validation, encoding, threshold, calibration), stop, explain
  the tradeoff and the rejected alternative, log it in DECISIONS.md.
- No magic. If you use a library feature, say what it does under the hood.

## Non-negotiable design decisions
- VALIDATION: temporal or grouped split, NEVER a plain random k-fold that ignores time.
  Explain the leakage risk.
- POINT-IN-TIME FEATURES: aggregating bureau / previous-application history uses ONLY
  records predating the application. No future leakage across the multi-table joins.
  This is the hard part and the main story.
- IMBALANCE: ~8% positive. Evaluate with PR-AUC and lift, not accuracy or plain ROC-AUC.
- CALIBRATION: calibrate probabilities (isotonic or Platt), validate with a reliability
  curve and Brier score.
- THRESHOLD: choose from an explicit business cost matrix, not 0.5.
- LEAKAGE TESTS: a test asserting no feature uses post-application data.
- SERVING: FastAPI /predict returning calibrated probability + decision + threshold.

## Stack (fixed)
Python, pandas/polars, scikit-learn, LightGBM or XGBoost, FastAPI, Docker, pytest.
MLflow only if it earns its place.

## Style
- Depth over breadth. One strong model, correct validation, honest evaluation.
- No em dashes, no AI-sounding prose, no filler comments.
- Every non-obvious decision gets a line in DECISIONS.md.
- No Streamlit. No 15-chart EDA dump.

## Things NOT to do
- Don't scaffold the whole project at once.
- Don't use a random split "just to start".
- Don't report accuracy as a headline metric.
- Don't build features that peek at post-application data.
- Don't add tools to look impressive.