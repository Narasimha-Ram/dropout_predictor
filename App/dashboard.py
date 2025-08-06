import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/dropout_model.pkl")

# Define encoding for categorical features
edu_encoder_classes = ['Bachelor', 'High School', 'Master', 'PhD']
edu_map = {name: i for i, name in enumerate(edu_encoder_classes)}
gender_map = {"Male": 1, "Female": 0}

# Streamlit UI
st.title("📉 Student Dropout Predictor")

age = st.slider("Age", 18, 25, 20)
gender = st.selectbox("Gender", ["Male", "Female"])
attendance = st.slider("Attendance Rate (%)", 50, 100, 80)
score = st.slider("Average Assignment Score (%)", 40, 100, 70)
study_hours = st.slider("Study Hours per Week", 0, 40, 10)
parental_education = st.selectbox("Parental Education", edu_encoder_classes)
financial_aid = st.selectbox("Financial Aid Received", ["Yes", "No"])

# Process input
gender = gender_map[gender]
parental_education = edu_map[parental_education]
financial_aid = 1 if financial_aid.lower() == "yes" else 0

# Create input DataFrame
input_data = pd.DataFrame([[
    age,
    gender,
    attendance,
    score,
    study_hours,
    parental_education,
    financial_aid
]], columns=[
    "age", "gender", "attendance_rate", "avg_assignment_score",
    "study_hours_per_week", "parental_education", "financial_aid"
])

# Prediction button
if st.button("Predict Dropout"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely to DROP OUT (Risk Score: {probability:.2f})")
    else:
        st.success(f"✅ Likely to STAY (Risk Score: {probability:.2f})")

# Optional: Show raw input
with st.expander("🔍 See Model Input"):
    st.write(input_data)
