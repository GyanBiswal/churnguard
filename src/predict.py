from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"


def load_model(path=MODEL_PATH):
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact["threshold"]


def predict_customer(customer, pipeline, threshold):
    X_new = pd.DataFrame([customer])
    proba = pipeline.predict_proba(X_new)[0, 1]
    return proba, int(proba >= threshold)