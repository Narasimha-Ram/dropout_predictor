import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv("data/student_data.csv")
model = joblib.load("model/dropout_model.pkl")

# Encode features exactly as during training
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

from sklearn.preprocessing import LabelEncoder
edu_encoder = LabelEncoder()
edu_encoder.classes_ = np.array(['Bachelor', 'High School', 'Master', 'PhD'])
df['parental_education'] = edu_encoder.transform(df['parental_education'])

df['financial_aid'] = df['financial_aid'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)

# Select exact feature columns
X = df[[
    "age",
    "gender",
    "attendance_rate",
    "avg_assignment_score",
    "study_hours_per_week",
    "parental_education",
    "financial_aid"
]]

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Debug shapes
print("X shape:", X.shape)
print("shap_values[1] shape:", shap_values[1].shape)

# Summary plot for class 1 (dropout)
shap.summary_plot(shap_values, X)
