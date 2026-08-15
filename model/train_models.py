"""
train_models.py
Trains 5 classification models on the Breast Cancer Wisconsin dataset,
evaluates each with 6 metrics, and saves the fitted models + scaler
to disk (as .pkl files) so the Streamlit app can load them directly
instead of retraining on every run.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42

# ---- Load data ----
df = pd.read_csv("../full_data.csv")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE
    ),
}

# Models that need scaled features to behave well
needs_scaling = {"Logistic Regression", "kNN"}

results = {}

for name, model in models.items():
    Xtr = X_train_scaled if name in needs_scaling else X_train
    Xte = X_test_scaled if name in needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results[name] = metrics

    # Save fitted model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, f"{fname}.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

with open("metrics_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print comparison table
results_df = pd.DataFrame(results).T
results_df.index.name = "ML Model Name"
print(results_df.to_string())
results_df.to_csv("comparison_table.csv")
