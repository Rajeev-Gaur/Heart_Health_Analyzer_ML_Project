# Heart Health Analyzer

## Overview
The Heart Health Analyzer is a machine learning project designed to predict heart disease based on various health metrics. This project utilizes a Random Forest Classifier to analyze patient data and provide insights into heart health risks.

## Table of Contents
- [Installation]
- [Usage]
- [Features]
- [Data]
- [Results]
- [Visualization]
- [License]

## Installation
To run this project, you'll need to have Python and the following packages installed:

```bash
pip install pandas scikit-learn imbalanced-learn matplotlib

Clone the repository and navigate to the project directory:
git clone <repository-url>
cd C:\Users\DeLL\Desktop\Visual Studio Coding\Heart_Health_Analyzer_ML_Project

Usage
The project can be run by executing the following command in the terminal:
python src/main.py Data/heart_disease_uci.csv


Features
Data preprocessing including handling missing values and one-hot encoding of categorical variables.
Application of SMOTE for class balancing.
Model training using the Random Forest Classifier.
Model evaluation with accuracy, precision, recall, and F1-score metrics.
Visualization of results through various plots.


Data
This project uses the UCI Heart Disease dataset, which includes information such as:

Age
Sex
Chest pain type (cp)
Resting blood pressure (trestbps)
Serum cholesterol (chol)
Fasting blood sugar (fbs)
Resting electrocardiographic results (restecg)
Maximum heart rate achieved (thalach)
Exercise induced angina (exang)
Oldpeak
Slope of the peak exercise ST segment (slope)
Number of major vessels colored by fluoroscopy (ca)
Thalassemia (thal)
Diagnosis of heart disease (num)


Result:
Shapes of the datasets after SMOTE:
X_train shape: (1644, 18)
X_test shape: (411, 18)
y_train shape: (1644,)
y_test shape: (411,)
Accuracy: 0.82
Classification Report:
               precision    recall  f1-score   support

         0.0       0.78      0.86      0.82        85
         1.0       0.76      0.63      0.69        81
         2.0       0.77      0.86      0.81        72
         3.0       0.85      0.81      0.83        84
         4.0       0.94      0.94      0.94        89

    accuracy                           0.82       411
   macro avg       0.82      0.82      0.82       411
weighted avg       0.82      0.82      0.82       411



Visualization
The project includes visualizations to help understand the model performance and data distribution. These visualizations include:
Confusion Matrix
Feature Importance
ROC Curve


License
This project is licensed under the MIT License. See the LICENSE file for details.
