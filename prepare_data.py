"""
prepare_data.py
Loads the Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML Repository,
also bundled in scikit-learn), and creates train/test CSV splits.

Dataset source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
(569 instances, 30 numeric features, binary target: malignant / benign)
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)

print("Shape:", df.shape)
print("Features:", df.shape[1] - 1)
print("Instances:", df.shape[0])
print(df["diagnosis"].value_counts())

# Full dataset (for training)
df.to_csv("full_data.csv", index=False)

# Train / test split (test_data.csv is what gets uploaded to the Streamlit app)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["diagnosis"]
)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("\nSaved: full_data.csv, train_data.csv, test_data.csv")
