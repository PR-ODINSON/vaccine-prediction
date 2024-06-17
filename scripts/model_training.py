import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load the datasets
training_set_features = pd.read_csv('data/training_set_features.csv')
test_set_features = pd.read_csv('data/test_set_features.csv')
training_set_labels = pd.read_csv('data/training_set_labels.csv')
submission_format = pd.read_csv('data/submission_format.csv')

# Drop respondent_id as it is not a feature
train_features = training_set_features.drop(columns=['respondent_id'])
test_features = test_set_features.drop(columns=['respondent_id'])

# Separate features and target variables in training set
X_train = train_features
y_train = training_set_labels[['xyz_vaccine', 'seasonal_vaccine']]

# List of numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Apply preprocessing to training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(test_features)

# Define the base classifier
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a multi-output classifier
multi_target_classifier = MultiOutputClassifier(base_classifier, n_jobs=-1)

# Train the classifier
multi_target_classifier.fit(X_train_processed, y_train)

# Predict probabilities on the training dataset for evaluation
y_pred_prob_train = multi_target_classifier.predict_proba(X_train_processed)

# Extract probabilities for each target variable
xyz_vaccine_pred_prob_train = y_pred_prob_train[0][:, 1]
seasonal_vaccine_pred_prob_train = y_pred_prob_train[1][:, 1]

# Calculate ROC AUC scores
xyz_vaccine_auc = roc_auc_score(y_train['xyz_vaccine'], xyz_vaccine_pred_prob_train)
seasonal_vaccine_auc = roc_auc_score(y_train['seasonal_vaccine'], seasonal_vaccine_pred_prob_train)

# Print the ROC AUC scores
print(f'ROC AUC score for xyz_vaccine: {xyz_vaccine_auc}')
print(f'ROC AUC score for seasonal_vaccine: {seasonal_vaccine_auc}')
print(f'Mean ROC AUC score: {(xyz_vaccine_auc + seasonal_vaccine_auc) / 2}')

# Predict probabilities on the test dataset for submission
y_pred_prob_test = multi_target_classifier.predict_proba(X_test_processed)

# Extract probabilities for each target variable
xyz_vaccine_pred_prob_test = y_pred_prob_test[0][:, 1]
seasonal_vaccine_pred_prob_test = y_pred_prob_test[1][:, 1]

# Create the submission dataframe
submission = pd.DataFrame({
    'respondent_id': test_set_features['respondent_id'],
    'xyz_vaccine': xyz_vaccine_pred_prob_test,
    'seasonal_vaccine': seasonal_vaccine_pred_prob_test
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
