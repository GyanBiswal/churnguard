from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "telco_churn.csv"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )