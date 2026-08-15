"""
Streamlit app — Breast Cancer Classification Demo
Assignment 2 | Machine Learning | BITS WILP M.Tech (AIML/DSE)

Features:
  a. CSV upload of test data
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix / classification report
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
}
NEEDS_SCALING = {"Logistic Regression", "kNN"}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    feature_cols = joblib.load("model/feature_columns.pkl")
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, feature_cols, models


scaler, feature_cols, models = load_artifacts()

st.title("🩺 Breast Cancer Wisconsin — Classification Demo")
st.caption(
    "Assignment 2 | Machine Learning | BITS WILP M.Tech (AIML/DSE) | Piyush Pandey | 2025AC05361 | "
    "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository"
)

st.markdown("### 1. Upload Test Data (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV containing the 30 numeric features plus a 'diagnosis' column "
    "(0 = malignant, 1 = benign). You can use the provided test_data.csv.",
    type=["csv"],
)

st.markdown("### 2. Select Model")
model_name = st.selectbox("Choose a classification model", list(models.keys()))

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    if "diagnosis" not in test_df.columns:
        st.error("Uploaded CSV must contain a 'diagnosis' column (ground truth labels).")
    else:
        X_test = test_df[feature_cols]
        y_test = test_df["diagnosis"]

        model = models[model_name]
        X_input = scaler.transform(X_test) if model_name in NEEDS_SCALING else X_test

        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]

        st.markdown("### 3. Evaluation Metrics")
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.4f}")

        st.markdown("### 4. Confusion Matrix & Classification Report")
        c1, c2 = st.columns(2)

        with c1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"], ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {model_name}")
            st.pyplot(fig)

        with c2:
            report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"])
            st.text("Classification Report")
            st.code(report)

        with st.expander("Show predictions table"):
            out_df = test_df.copy()
            out_df["predicted_diagnosis"] = y_pred
            out_df["predicted_probability_benign"] = np.round(y_proba, 4)
            st.dataframe(out_df)
else:
    st.info("👆 Upload a test CSV (e.g. test_data.csv from the repo) to see results.")

st.markdown("---")
st.caption("Built for BITS WILP ML Assignment 2 — Streamlit Community Cloud deployment.")
