"""
Trains and persists the artifacts the serving layer (component 7) loads at
startup: the LightGBM model, the isotonic calibrator, and a feature manifest.
Not a new model -- this is the exact model from components 4-6 (trained on
blocks 0-3, calibrated on block_4), just saved to disk instead of only
printed.

The manifest exists so serve.py never has to duplicate or guess anything the
training pipeline already knows: the exact column order the model expects,
which columns are categorical and their exact fitted category levels (LightGBM's
categorical split logic depends on the integer codes pandas assigns those
categories -- constructing a fresh Categorical at serve time with different or
missing levels would silently shift what an integer code means), and which
columns are bureau-derived vs applicant-supplied.
"""

import pickle
from pathlib import Path

from calibration import fit_calibration

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
THRESHOLD = 0.1225  # component 6: flat 7.5:1 cost-ratio empirical optimum


def main() -> None:
    model, calibrator, X_calib, y_calib, _, _ = fit_calibration(verbose=False)

    cat_cols = [c for c in X_calib.columns if str(X_calib[c].dtype) == "category"]
    bureau_cols = [c for c in X_calib.columns if c.startswith("BUREAU_")]
    application_cols = [c for c in X_calib.columns if c not in bureau_cols]

    manifest = dict(
        feature_order=list(X_calib.columns),
        categorical_columns={col: X_calib[col].cat.categories.tolist() for col in cat_cols},
        numeric_columns=[c for c in X_calib.columns if c not in cat_cols],
        bureau_feature_columns=bureau_cols,
        application_feature_columns=application_cols,
        threshold=THRESHOLD,
        cost_assumptions=dict(lgd_fraction=0.75, margin_fraction=0.10, ratio=7.5),
    )

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACTS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
    with open(ARTIFACTS_DIR / "manifest.pkl", "wb") as f:
        pickle.dump(manifest, f)

    print(f"saved model.pkl, calibrator.pkl, manifest.pkl to {ARTIFACTS_DIR}")
    print(f"  {len(manifest['feature_order'])} total features: "
          f"{len(application_cols)} application + {len(bureau_cols)} bureau")
    print(f"  {len(cat_cols)} categorical columns (exact fitted levels persisted)")
    print(f"  threshold: {THRESHOLD}")


if __name__ == "__main__":
    main()
