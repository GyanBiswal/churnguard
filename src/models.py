from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data_prep import build_preprocessor


def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", XGBClassifier(random_state=42, eval_metric="logloss")),
        ]),
    }