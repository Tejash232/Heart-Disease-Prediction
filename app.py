
import streamlit as st
import pickle
import numpy as np

# Load trained model and feature names
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's medical details below.")

# Input fields
age = st.number_input("Age", 1, 120, 41)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 50, 250, 130)
chol = st.number_input("Cholesterol", 50, 600, 204)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate", 50, 250, 172)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.4)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prediction
if st.button("Predict Heart Disease"):
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction: Heart disease detected.")
    else:
        st.success("Prediction: No heart disease detected.")

    st.warning(
        "Educational demonstration only. "
        "This result is not a medical diagnosis."
    )
