import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.predict import load_model, predict_customer

from src.data_prep import CATEGORICAL_COLS, load_data, split_data

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


pipeline, threshold = get_model()

@st.cache_data
def get_data():
    return load_data()


df = get_data()

@st.cache_data
def get_test_results():
    _, X_test, _, y_test = split_data(df)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return y_test.to_numpy(), proba

st.title("ChurnGuard")
st.write("Customer churn prediction for a telecom company.")

tab_predict, tab_dashboard, tab_performance, tab_insights = st.tabs(
    ["Predict", "Dashboard", "Model performance", "Insights"]
)

YES_NO = ["No", "Yes"]
SERVICE = ["No", "Yes", "No internet service"]

with tab_predict:
    st.subheader("Customer details")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", YES_NO)
            partner = st.selectbox("Has a partner", YES_NO)
            dependents = st.selectbox("Has dependents", YES_NO)
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly charges", 15.0, 125.0, 70.0, step=0.5)

        with col2:
            phone = st.selectbox("Phone service", YES_NO, index=1)
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online security", SERVICE)
            backup = st.selectbox("Online backup", SERVICE)
            protection = st.selectbox("Device protection", SERVICE)

        with col3:
            support = st.selectbox("Tech support", SERVICE)
            tv = st.selectbox("Streaming TV", SERVICE)
            movies = st.selectbox("Streaming movies", SERVICE)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", YES_NO)
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        submitted = st.form_submit_button("Predict churn")

    if submitted:
        if phone == "No":
            multiple = "No phone service"
        if internet == "No":
            security = backup = protection = support = tv = movies = "No internet service"

        customer = {
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": tenure * monthly,
        }
        proba, flag = predict_customer(customer, pipeline, threshold)

        if proba >= 0.60:
            risk = "High"
        elif proba >= threshold:
            risk = "Medium"
        else:
            risk = "Low"

        st.divider()
        result1, result2 = st.columns(2)
        result1.metric("Churn probability", f"{proba:.1%}")
        result2.metric("Risk level", risk)

        if risk == "High":
            st.error("High risk: this customer is likely to leave. Consider a retention offer.")
        elif risk == "Medium":
            st.warning("Medium risk: flagged by the model. A light-touch follow-up may help.")
        else:
            st.success("Low risk: this customer is not flagged.")

        st.caption(
            f"Customers at or above {threshold:.0%} are flagged as likely to churn. "
            "Total charges are estimated as tenure × monthly charges."
        )

with tab_dashboard:
    st.subheader("Customer churn overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers", f"{len(df):,}")
    m2.metric("Churned", f"{int(df['Churn'].sum()):,}")
    m3.metric("Churn rate", f"{df['Churn'].mean():.1%}")

    st.subheader("Churn rate by category")
    feature = st.selectbox("Choose a feature", CATEGORICAL_COLS)
    rate = df.groupby(feature)["Churn"].mean() * 100
    st.bar_chart(rate)
    st.caption("Churn rate (%) for each category of the chosen feature.")

with tab_performance:
    st.subheader("Final model: Logistic Regression")

    y_true, proba = get_test_results()
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Precision", f"{precision_score(y_true, pred):.3f}")
    p2.metric("Recall", f"{recall_score(y_true, pred):.3f}")
    p3.metric("F1", f"{f1_score(y_true, pred):.3f}")
    p4.metric("ROC-AUC", f"{auc:.3f}")
    st.caption(
        f"Measured on {len(y_true):,} test customers the model never saw, "
        f"using a decision threshold of {threshold:.2f}."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, pred)
        st.table(pd.DataFrame(
            cm,
            index=["Actually stayed", "Actually churned"],
            columns=["Predicted stay", "Predicted churn"],
        ))

    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y_true, proba)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate (recall)")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Comparison of all four models")
    st.table(pd.DataFrame(
        {
            "Accuracy": [0.806, 0.721, 0.786, 0.776],
            "Precision": [0.657, 0.475, 0.624, 0.590],
            "Recall": [0.559, 0.487, 0.484, 0.519],
            "F1": [0.604, 0.481, 0.545, 0.552],
            "ROC-AUC": [0.842, 0.646, 0.817, 0.820],
        },
        index=["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"],
    ))
    st.caption(
        "Test-set results from model evaluation, all at default settings and a 0.50 threshold. "
        "The final model above uses a 0.30 threshold, chosen on training data to catch more churners."
    )


with tab_insights:
    st.subheader("What drives churn in this model")

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    weights = pd.Series(coefs, index=[n.split("__", 1)[1] for n in names]).sort_values()

    top = pd.concat([weights.head(8), weights.tail(8)])
    colors = ["tab:green" if w < 0 else "tab:red" for w in top]

    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Model weight (negative = stays, positive = churns)")
    st.pyplot(fig)

    st.caption(
        "Red bars push the prediction toward churn; green bars push toward staying. "
        "These are patterns the model learned, not proven causes."
    )