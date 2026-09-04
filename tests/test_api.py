"""
API contract tests for the FastAPI serving layer (component 7).

Requires artifacts/ to exist (run src/train_final_model.py first) -- serve.py
loads model.pkl/calibrator.pkl/manifest.pkl at import time, same as it does
at real startup, so importing the app here exercises exactly that path.

Covers: /health shape, /predict shape and decision-rule consistency, the
no-bureau-history NaN path, and that the point-in-time filter (component 4)
actually runs inside the request path, not just in the offline training
pipeline -- a bureau record dated on/after "today" must not blow up or sneak
into the aggregate.
"""

from fastapi.testclient import TestClient

from serve import MANIFEST, app

client = TestClient(app)

MINIMAL_APPLICATION = {col: None for col in MANIFEST["application_feature_columns"]}

VALID_BUREAU_RECORD = dict(
    DAYS_CREDIT=-500, DAYS_CREDIT_UPDATE=-30, CREDIT_ACTIVE="Closed",
    CREDIT_TYPE="Consumer credit", AMT_CREDIT_SUM=100_000.0, AMT_CREDIT_SUM_DEBT=0.0,
)
LEAKY_BUREAU_RECORD = dict(
    DAYS_CREDIT=0, DAYS_CREDIT_UPDATE=5, CREDIT_ACTIVE="Active",
    CREDIT_TYPE="Consumer credit", AMT_CREDIT_SUM=999_999.0, AMT_CREDIT_SUM_DEBT=999_999.0,
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["n_features"], int) and body["n_features"] > 0
    assert body["threshold"] == MANIFEST["threshold"]


def test_predict_shape_no_bureau_history():
    r = client.post("/predict", json={"application": MINIMAL_APPLICATION, "bureau_records": []})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["decision"] in ("approve", "decline")
    assert body["threshold"] == MANIFEST["threshold"]


def test_predict_decision_matches_threshold_rule():
    r = client.post("/predict", json={"application": MINIMAL_APPLICATION, "bureau_records": []})
    body = r.json()
    expected = "decline" if body["probability"] >= body["threshold"] else "approve"
    assert body["decision"] == expected


def test_predict_with_bureau_history_runs_and_differs_from_no_history():
    no_history = client.post("/predict", json={"application": MINIMAL_APPLICATION, "bureau_records": []}).json()
    with_history = client.post(
        "/predict",
        json={"application": MINIMAL_APPLICATION, "bureau_records": [VALID_BUREAU_RECORD]},
    ).json()
    assert 0.0 <= with_history["probability"] <= 1.0
    assert with_history["probability"] != no_history["probability"]


def test_predict_filters_leaky_bureau_record_instead_of_erroring():
    """A bureau record dated on/after the application (DAYS_CREDIT=0,
    DAYS_CREDIT_UPDATE=5) must be silently excluded by the same point-in-time
    filter used in training, not crash the request and not corrupt the result
    for the valid record alongside it."""
    only_valid = client.post(
        "/predict",
        json={"application": MINIMAL_APPLICATION, "bureau_records": [VALID_BUREAU_RECORD]},
    ).json()
    valid_plus_leaky = client.post(
        "/predict",
        json={"application": MINIMAL_APPLICATION, "bureau_records": [VALID_BUREAU_RECORD, LEAKY_BUREAU_RECORD]},
    ).json()
    assert valid_plus_leaky["probability"] == only_valid["probability"]


def test_predict_rejects_bureau_record_missing_required_fields():
    bad_record = {"CREDIT_ACTIVE": "Active"}  # missing DAYS_CREDIT / DAYS_CREDIT_UPDATE
    r = client.post("/predict", json={"application": MINIMAL_APPLICATION, "bureau_records": [bad_record]})
    assert r.status_code == 422
