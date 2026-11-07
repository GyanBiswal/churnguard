from pathlib import Path

import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.data_prep import load_data, split_data
from src.models import get_models

THRESHOLD = 0.30
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = get_models()["Logistic Regression"]
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Recall:", round(recall_score(y_test, pred), 3))
    print("F1:", round(f1_score(y_test, pred), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

    joblib.dump({"pipeline": model, "threshold": THRESHOLD}, MODEL_PATH)
    print("Saved to", MODEL_PATH)


if __name__ == "__main__":
    main()