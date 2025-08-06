import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

df = pd.read_csv("data/student_data.csv")

# Encode categorical features
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
edu_encoder = LabelEncoder()
df['parental_education'] = edu_encoder.fit_transform(df['parental_education'])

X = df.drop(['student_id', 'dropout'], axis=1)
y = df['dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/dropout_model.pkl")
print("✅ Model trained and saved to model/dropout_model.pkl")
