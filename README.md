# 🎓 Student Dropout Predictor

A machine learning project that predicts whether a student is likely to drop out based on personal, academic, and socioeconomic features. The project includes:

- 🔍 A Streamlit Dashboard
- 🧠 A trained Random Forest model
- 📦 A Flask API for backend integration
- 📊 SHAP-based explainability

---

## 🚀 Features

- 🎯 Predict dropout risk based on 7 key student attributes
- 📈 Visualize model predictions via a Streamlit dashboard
- 🧪 Explain predictions with SHAP (feature importance)
- 🖥️ Backend REST API for integration
- 🛠️ Train and test on synthetic data

---

## 📁 Project Structure

```plaintext
dropout_predictor/
│
├── data/
│   └── student_data.csv         # Generated synthetic dataset
│
├── model/
│   └── dropout_model.pkl        # Trained Random Forest model
│
├── app/
│   ├── dashboard.py             # Streamlit dashboard
│   ├── api.py                   # Flask API for prediction
│   └── explainer.py             # SHAP summary plot generator
│
├── generate_data.py             # Generates synthetic student data
├── train_model.py               # Trains and saves the ML model
├── requirements.txt             # Project dependencies
# 🎓 Student Dropout Predictor

A machine learning project that predicts whether a student is likely to drop out based on personal, academic, and socioeconomic features. The project includes:

- 🔍 A Streamlit Dashboard
- 🧠 A trained Random Forest model
- 📦 A Flask API for backend integration
- 📊 SHAP-based explainability

---

## 🚀 Features

- 🎯 Predict dropout risk based on 7 key student attributes
- 📈 Visualize model predictions via a Streamlit dashboard
- 🧪 Explain predictions with SHAP (feature importance)
- 🖥️ Backend REST API for integration
- 🛠️ Train and test on synthetic data

---

## 📁 Project Structure

```plaintext
dropout_predictor/
│
├── data/
│   └── student_data.csv         # Generated synthetic dataset
│
├── model/
│   └── dropout_model.pkl        # Trained Random Forest model
│
├── app/
│   ├── dashboard.py             # Streamlit dashboard
│   ├── api.py                   # Flask API for prediction
│   └── explainer.py             # SHAP summary plot generator
│
├── generate_data.py             # Generates synthetic student data
├── train_model.py               # Trains and saves the ML model
├── requirements.txt             # Project dependencies
