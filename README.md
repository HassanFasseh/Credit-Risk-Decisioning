# Credit Risk Decisioning

A credit-default risk system on the Home Credit dataset that turns an applicant's
data into an approve or decline decision. It goes past the usual notebook:
leakage-safe features built with point-in-time correctness, walk-forward validation,
calibrated probabilities, a cost-based decision threshold, and the whole thing served
behind a FastAPI endpoint.

## Results

Walk-forward validation (application table, then with point-in-time bureau features):

| Model | PR-AUC | Lift@10 | Lift@20 | ROC-AUC |
|---|---|---|---|---|
| Baseline (application only) | 0.237 | 3.24 | 2.56 | 0.752 |
| + point-in-time bureau features | 0.246 | 3.33 | 2.61 | 0.758 |

The base default rate is 8.1 percent, so a top-decile lift of 3.3 means the riskiest
10 percent of applicants contain 3.3x more defaulters than a random 10 percent would.

Decision policy: a cost-based threshold of 0.1225 on the calibrated probability, built
from a 7.5:1 false-negative to false-positive cost ratio. It cuts expected cost by
21.8 percent versus the naive 0.5 threshold.

## Design highlights

**Point-in-time correctness.** Bureau history is aggregated using only records that
predate the application, filtered on both `DAYS_CREDIT < 0` and
`DAYS_CREDIT_UPDATE < 0` so no feature is computed from a bureau record refreshed after
the decision point. Every point-in-time filter is paired with a leakage test.

**Walk-forward validation.** The data has no real timestamps, only relative day
offsets, so a true temporal split is not available. IDs are used as an order proxy,
with a PSI drift check confirming the ordering carries no distributional shift, and an
order-respecting walk-forward scheme is kept as the conservative default regardless.
The final holdout is touched exactly once.

**Probability calibration.** Isotonic regression on a forward validation block,
validated with a reliability curve and Brier score. Calibration is treated honestly: it
costs about 1.9 percent PR-AUC in exchange for probabilities that mean what they say,
which is the trade a decisioning system needs in order to threshold on cost.

**Cost-based threshold.** The decision boundary comes from an explicit cost matrix
(loss given default versus forgone margin), not 0.5, with a sensitivity check across a
range of cost ratios.

**Train/serve consistency.** The serving layer calls the exact same point-in-time
aggregation functions used and leakage-tested in training, so there is no train/serve
feature skew.

## Stack

Python, LightGBM, scikit-learn, pandas, FastAPI, Docker, pytest.

## Data

The Home Credit Default Risk dataset from Kaggle:
https://www.kaggle.com/competitions/home-credit-default-risk/data

Accept the competition rules, then download the CSVs into `data/` (gitignored). The
main table has 307,511 labeled applications.

## Quickstart

```bash
# install
pip install -r requirements.txt

# train and save artifacts (model, calibrator, threshold)
python src/train_final_model.py

# run the service
uvicorn src.serve:app --reload
```

Or with Docker:

```bash
docker compose up --build
```

## API

```
GET /health
{"status": "ok", "n_features": 148, "threshold": 0.1225}

POST /predict
{"probability": 0.0706, "decision": "approve", "threshold": 0.1225}
```

The service accepts an applicant's fields plus raw bureau records, applies the same
point-in-time aggregation as training, and returns the calibrated default probability
and the approve or decline decision.

## Tests

```bash
pytest
```

11 tests covering the point-in-time leakage filters, the missing-not-zero aggregation
guards, and the API contract (health, predict shape, no-bureau-history path,
malformed-input rejection).

## Limitations and next steps

- The dataset has no real timestamps, so ID-order is a proxy for time. The PSI check
  shows it carries no drift, meaning the walk-forward split behaves like a random one
  on this particular data; the harness would surface real drift on data that has it.
- The behavioral panel tables (monthly balances, installments) are deferred; the
  current model uses the application and bureau tables.
- Single LightGBM model, no ensemble.
