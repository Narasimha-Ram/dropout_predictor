from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = joblib.load("model/dropout_model.pkl")

# Prepare LabelEncoder for parental_education (must match training)
edu_encoder = LabelEncoder()
edu_encoder.classes_ = ['Bachelor', 'High School', 'Master', 'PhD']  # Use the exact order used during training

def preprocess_input(data):
    """
    Convert raw input JSON into model-ready DataFrame row,
    applying the same encoding as training.
    """

    # Map gender to numeric
    gender_map = {"Male": 1, "Female": 0}
    gender = gender_map.get(data.get("gender", "Female"), 0)

    # Encode parental education
    parental_education_raw = data.get("parental_education", "High School")
    try:
        parental_education = int(edu_encoder.transform([parental_education_raw])[0])
    except ValueError:
        # If unseen category, fallback to 0
        parental_education = 0

    # Financial aid: convert Yes/No or bool to int 0/1
    financial_aid_raw = data.get("financial_aid", 0)
    if isinstance(financial_aid_raw, str):
        financial_aid = 1 if financial_aid_raw.lower() in ['yes', '1', 'true'] else 0
    else:
        financial_aid = int(financial_aid_raw)

    # Construct DataFrame with columns in exact order expected by model
    input_df = pd.DataFrame([[
        int(data.get("age", 20)),
        gender,
        float(data.get("attendance_rate", 80.0)),
        float(data.get("avg_assignment_score", 70.0)),
        float(data.get("study_hours_per_week", 10.0)),
        parental_education,
        financial_aid
    ]], columns=[
        "age", "gender", "attendance_rate", "avg_assignment_score",
        "study_hours_per_week", "parental_education", "financial_aid"
    ])

    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    input_df = preprocess_input(data)

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(proba)
    })

if __name__ == '__main__':
    app.run(debug=True)
