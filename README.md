# ML_classification_models
# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement
Breast cancer diagnosis is a critical binary classification problem in
healthcare: given a set of measurements extracted from a digitized image of
a breast mass (fine needle aspirate), predict whether the mass is
**malignant** or **benign**. Early and accurate classification directly
supports clinical decision-making. This project implements and compares
five classical machine learning classifiers on this task, and exposes them
through an interactive Streamlit web application for evaluation.

## b. Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository
  (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
  also available via `sklearn.datasets.load_breast_cancer`
- **Instances:** 569
- **Features:** 30 numeric features (mean, standard error, and "worst"
  values of 10 real-valued measurements computed for each cell nucleus,
  e.g. radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension)
- **Target:** `diagnosis` - 0 = malignant (212 cases), 1 = benign (357 cases)
- **Split:** 80% train (455 rows) / 20% test (114 rows), stratified on the
  target, `random_state=42`

## c. GitHub Repository Link
`https://github.com/psh2590/ML_classification_models`

## d. Models Used

### Comparison Table
<img width="771" height="175" alt="image" src="https://github.com/user-attachments/assets/cec9abb0-60ee-4df6-9d23-949a4ad09bfe" />
                                                                                                                            
																															 ### Observation Table
	
<img width="1453" height="378" alt="image" src="https://github.com/user-attachments/assets/cf74e490-bfb9-4eb5-ad28-a99b64d8743d" />

