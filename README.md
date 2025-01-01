

# Vaccine Likelihood Prediction

## Overview

This project focuses on predicting the likelihood of individuals receiving XYZ and seasonal flu vaccines using machine learning techniques. The prediction is based on a dataset containing various features such as demographic information, health behaviors, and personal opinions. It addresses a multi-label classification problem, where the goal is to predict two binary outcomes simultaneously: XYZ vaccine uptake and seasonal flu vaccine uptake.

The project includes data preprocessing, model training using RandomForestClassifier within a MultiOutputClassifier, evaluation based on ROC AUC scores, and generating predictions for submission. Accurate predictions can assist healthcare providers and policymakers in understanding vaccination behaviors and developing targeted interventions to improve vaccination rates.

## Key Features

- **Dataset**: The dataset consists of 36 features, encompassing respondent demographics, health behaviors (e.g., handwashing frequency, mask usage), medical history (e.g., chronic conditions), and opinions on vaccine effectiveness and risks.
- **Target Variables**: 
  - `xyz_vaccine`: Whether the respondent received the XYZ flu vaccine (0 = No, 1 = Yes).
  - `seasonal_vaccine`: Whether the respondent received the seasonal flu vaccine (0 = No, 1 = Yes).
- **Modeling Approach**: 
  - RandomForestClassifier is utilized within a MultiOutputClassifier to handle the multi-label nature of the problem.
- **Evaluation Metric**: 
  - Performance is evaluated using the area under the Receiver Operating Characteristic curve (ROC AUC), separately for each target variable!

## Directory Structure

```plaintext
vaccine-prediction/
│
├── data/                    # Directory for dataset files
│   ├── training_set_features.csv     # Training set features
│   ├── test_set_features.csv         # Test set features
│   ├── training_set_labels.csv        # Labels for training set
│   └── submission_format.csv          # Submission format
│
├── scripts/                   # Directory for Python scripts
│   └── model_training.py           # Script for data preprocessing, model training, and prediction
│
├── notebooks/                  # Directory for Jupyter notebooks (optional)
│
├── README.md                    # This file
│
└── submission.csv                # Predictions for submission
```




---

This README.md file provides a clear and structured overview of your project, including how to set it up, run the model training, and contribute to it. Adjust the sections as per your specific project details and preferences.
