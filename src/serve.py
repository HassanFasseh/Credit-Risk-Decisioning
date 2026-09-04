"""
Component 7: FastAPI serving layer.

Loads model.pkl / calibrator.pkl / manifest.pkl (written by
train_final_model.py) once at import time -- no retraining at startup.

Bureau features: the request sends raw bureau tradelines (bureau_records),
not pre-computed BUREAU_* aggregates. /predict reuses bureau_features.py's
filter_point_in_time() and aggregate_bureau() directly -- the same functions
components 4-6 were trained and leakage-tested against -- so there is exactly
one implementation of "what a bureau feature is," eliminating train/serve
skew by construction rather than trusting a second implementation to match.
See the chat response for the full tradeoff against accepting pre-computed
aggregates.

Missing features: any application field the caller omits (or sends as null)
stays None -> NaN through pandas, exactly like a raw application_train cell
that was never populated. No zero-filling anywhere on this path, consistent
with every prior component's missing-not-zero rule.

Categorical columns: reconstructed with pd.Categorical(value,
categories=manifest_categories) using the exact category list fitted at
training time, not inferred fresh from the single incoming request -- a
fresh inference would assign different integer codes than training did,
silently corrupting every split that column participates in. An unseen
category value (not in the fitted list) becomes NaN via this same
construction, which is the correct behavior: LightGBM already knows how to
route missing values, and inventing a new code for something never seen in
training would not be.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, create_model

from bureau_features import aggregate_bureau, filter_point_in_time

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

with open(ARTIFACTS_DIR / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open(ARTIFACTS_DIR / "calibrator.pkl", "rb") as f:
    CALIBRATOR = pickle.load(f)
with open(ARTIFACTS_DIR / "manifest.pkl", "rb") as f:
    MANIFEST = pickle.load(f)


class BureauRecord(BaseModel):
    """One tradeline from a raw bureau pull, bureau.csv schema. DAYS_CREDIT
    and DAYS_CREDIT_UPDATE are required -- the point-in-time filter can't run
    without them, so a record missing either is not a usable record."""
    DAYS_CREDIT: float
    DAYS_CREDIT_UPDATE: float
    CREDIT_ACTIVE: Optional[str] = None
    CREDIT_TYPE: Optional[str] = None
    DAYS_CREDIT_ENDDATE: Optional[float] = None
    DAYS_ENDDATE_FACT: Optional[float] = None
    CREDIT_DAY_OVERDUE: Optional[float] = None
    CNT_CREDIT_PROLONG: Optional[float] = None
    AMT_CREDIT_SUM: Optional[float] = None
    AMT_CREDIT_SUM_DEBT: Optional[float] = None
    AMT_CREDIT_SUM_LIMIT: Optional[float] = None
    AMT_CREDIT_SUM_OVERDUE: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None


def _build_application_model() -> type[BaseModel]:
    """
    Builds the request model for the ~120 raw application_train columns from
    the persisted manifest rather than hand-typing every field: at this
    column count, a hand-written list would drift from what the model
    actually expects the first time the training feature set changes, silently.
    pydantic.create_model does exactly what writing `field: Optional[T] = None`
    120 times by hand would do -- it's generated, not hidden behavior.
    Every field is optional and defaults to None, which becomes NaN on the
    way into the model, matching an unpopulated cell in the raw data.
    """
    fields = {}
    for col in MANIFEST["application_feature_columns"]:
        field_type = str if col in MANIFEST["categorical_columns"] else float
        fields[col] = (Optional[field_type], None)
    return create_model("ApplicationFeatures", **fields)


ApplicationFeatures = _build_application_model()


class PredictRequest(BaseModel):
    application: ApplicationFeatures
    bureau_records: list[BureauRecord] = []


class PredictResponse(BaseModel):
    probability: float
    decision: str
    threshold: float


app = FastAPI(title="Credit Default Risk API")


def _build_application_row(application: BaseModel) -> pd.DataFrame:
    row = pd.DataFrame([application.model_dump()])
    for col, categories in MANIFEST["categorical_columns"].items():
        if col in row.columns:
            row[col] = pd.Categorical(row[col], categories=categories)
    for col in MANIFEST["numeric_columns"]:
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    return row


BUREAU_NUMERIC_FIELDS = [
    "DAYS_CREDIT", "DAYS_CREDIT_UPDATE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT",
    "CREDIT_DAY_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY",
]


def _build_bureau_row(bureau_records: list[BureauRecord]) -> pd.DataFrame:
    bureau_cols = MANIFEST["bureau_feature_columns"]
    if not bureau_records:
        return pd.DataFrame([{c: np.nan for c in bureau_cols}])

    raw = pd.DataFrame([r.model_dump() for r in bureau_records])
    raw["SK_ID_CURR"] = 0  # single applicant per request; group key is a placeholder
    # a field the caller omits arrives as Python None; a column built entirely
    # (or partly) from None lands as object dtype, not float NaN, and survives
    # that way into the aggregate -- LightGBM's predict() rejects object dtype
    # outright. pd.read_csv never has this problem (missing cells are already
    # float NaN), so this only shows up here, at the request boundary.
    for col in BUREAU_NUMERIC_FIELDS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    filtered = filter_point_in_time(raw)
    if len(filtered) == 0:
        return pd.DataFrame([{c: np.nan for c in bureau_cols}])

    agg = aggregate_bureau(filtered)
    return agg.drop(columns=["SK_ID_CURR"]).iloc[[0]].reset_index(drop=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_features": len(MANIFEST["feature_order"]), "threshold": MANIFEST["threshold"]}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    app_row = _build_application_row(payload.application)
    bureau_row = _build_bureau_row(payload.bureau_records)

    full_row = pd.concat([app_row.reset_index(drop=True), bureau_row], axis=1)
    full_row = full_row[MANIFEST["feature_order"]]  # KeyError if anything is missing/misaligned -- fail loudly, not silently

    raw_score = MODEL.predict_proba(full_row)[:, 1][0]
    calibrated_prob = float(CALIBRATOR.predict([raw_score])[0])
    threshold = MANIFEST["threshold"]
    decision = "decline" if calibrated_prob >= threshold else "approve"  # component 6 tie-break: ties go to decline

    return PredictResponse(probability=calibrated_prob, decision=decision, threshold=threshold)
