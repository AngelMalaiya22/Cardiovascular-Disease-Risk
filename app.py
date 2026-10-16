"""
Cardiovascular Disease Risk Assessment & Prediction Model - Flask web app

Run locally :  python app.py                 -> http://127.0.0.1:5000
Production  :  gunicorn app:app              (see DEPLOYMENT_GUIDE.md)

Routes
------
GET  /                 the assessment page
POST /api/predict      JSON in (13 features) -> JSON out (probability, risk band, warnings)
GET  /api/model-info   model name, accuracies, feature ranges
GET  /health           liveness check for the hosting platform
"""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import EXAMPLES, FEATURE_GROUPS, FEATURES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise RuntimeError(
        "Model files not found in ./model. Train the model first:  python train_model.py"
    )

pipeline = joblib.load(MODEL_PATH)
with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

FEATURE_ORDER = meta["feature_order"]
RANGES = meta["feature_ranges"]          # training min / max for the numeric features
LOW_CUTOFF, HIGH_CUTOFF = 0.35, 0.65      # probability bands

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # requests are tiny; reject anything larger


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def parse_payload(data):
    """Return (values, errors). Every field is required and must be inside its allowed range."""
    values, errors = {}, {}
    for spec in FEATURES:
        name = spec["name"]
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            errors[name] = "This field is required."
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Enter a number."
            continue

        if spec["kind"] == "number":
            if not spec["min"] <= num <= spec["max"]:
                errors[name] = f"Enter a value between {spec['min']} and {spec['max']}."
                continue
            values[name] = num
        else:
            allowed = [code for code, _ in spec["options"]]
            if num not in allowed:
                errors[name] = "Choose one of the listed options."
                continue
            values[name] = int(num)
    return values, errors


def range_warnings(values):
    """Warn when a number is valid but outside the range seen during training."""
    warnings = []
    labels = {f["name"]: f["label"] for f in FEATURES}
    for name, (lo, hi) in RANGES.items():
        v = values[name]
        if v < lo or v > hi:
            warnings.append(
                f"{labels[name]} ({v:g}) is outside the range the model was trained on "
                f"({lo:g} to {hi:g}), so the estimate is less reliable."
            )
    return warnings


def risk_band(p):
    if p < LOW_CUTOFF:
        return "low", "Lower estimated risk"
    if p < HIGH_CUTOFF:
        return "moderate", "Moderate estimated risk"
    return "high", "Higher estimated risk"


def model_summary():
    m = meta["metrics"]
    return {
        "name": meta["model_name"],
        "cv_accuracy": round(m["cv_accuracy_after_tuning"] * 100, 1),
        "test_accuracy": round(m["test_accuracy_after_tuning"] * 100, 1),
        "n_test": meta["data"]["n_test"],
        "rows_used": meta["data"]["rows_used"],
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html", groups=FEATURE_GROUPS, examples=EXAMPLES,
                           model=model_summary())


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with the 13 feature values."}), 400

    values, errors = parse_payload(data)
    if errors:
        return jsonify({"errors": errors}), 422

    row = pd.DataFrame([[values[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    probability = float(pipeline.predict_proba(row)[0, 1])   # class 1 = disease present
    band, band_label = risk_band(probability)

    return jsonify({
        "probability": round(probability * 100, 1),
        "band": band,
        "band_label": band_label,
        "cutoffs": {"low": LOW_CUTOFF * 100, "high": HIGH_CUTOFF * 100},
        "warnings": range_warnings(values),
        "model": model_summary(),
    })


@app.get("/api/model-info")
def model_info():
    return jsonify({**model_summary(), "feature_order": FEATURE_ORDER,
                    "feature_ranges": RANGES, "trained_at_utc": meta["trained_at_utc"],
                    "versions": meta["versions"]})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)),
            debug=os.getenv("FLASK_DEBUG") == "1")
