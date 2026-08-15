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
---------------------------------------------------------------------------------------
| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
|-------------------------------------------------------------------------------------|
| Decision Tree            | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362 | 0.8341 |
|-------------------------------------------------------------------------------------|
| kNN                      | 0.9737   | 0.9884 | 0.9600    | 1.0000 | 0.9796 | 0.9442 |
|-------------------------------------------------------------------------------------|
| Naive Bayes              | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517 | 0.8676 |
|-------------------------------------------------------------------------------------|
| Random Forest (Ensemble) | 0.9561   | 0.9940 | 0.9589    | 0.9722 | 0.9655 | 0.9054 |
---------------------------------------------------------------------------------------

### Observations
 ML Model Name            	 Observation about model performance                                                                                          
 Logistic Regression      	 Best overall performer here. The classes are close to linearly separable after standard scaling, so a simple linear decision 
	 boundary generalizes very well - highest accuracy, F1, and MCC of all five models.                                           
 Decision Tree            	 Weakest of the five. A single tree (even depth-limited to 5) overfits the training split and produces a comparatively low AUC
	 , reflecting less stable probability estimates than the other models.                                                        
 kNN                      	 Strong performer, achieving perfect recall (no malignant case predicted as benign in this split) after feature scaling - but 
	 recall alone can be misleading; precision is slightly lower than Logistic Regression's.                                      
 Naive Bayes              	 Decent AUC despite its unrealistic feature-independence assumption (many of the 30 features are correlated, e.g. radius and  
	 area), which caps its accuracy/F1 below the top models.                                                                      
 Random Forest (Ensemble) 	 Very close to Logistic Regression on AUC and clearly better than a single Decision Tree on every metric - bagging trees      
	 the overfitting seen in the standalone Decision Tree.                                                                        
 **Overall Winner for your	 **Logistic Regression** - highest Accuracy, Precision, Recall, F1 and MCC among the five models on this test split.                                                                                                                                  
	
<img width="1453" height="378" alt="image" src="https://github.com/user-attachments/assets/cf74e490-bfb9-4eb5-ad28-a99b64d8743d" />

