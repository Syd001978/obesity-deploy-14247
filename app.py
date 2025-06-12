import streamlit as st
import joblib
import numpy as np

# Load model 
model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

label_mapping = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

# Title & Description
st.markdown("<h1 style='text-align: center;'>💡 Are We Obesity?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Input your data to predict.</p>", unsafe_allow_html=True)

# Form panel with border and background
st.markdown("""
<div style="border: 1px solid #4F4F4F; padding: 20px; border-radius: 10px; background-color: #1e1e1e;">
    <h5 style='color:white;'>🧍‍♂️ Personal Data & Physical Info</h5>
""", unsafe_allow_html=True)

# Two-column layout for form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

with col2:
    age = st.slider("Age", 14, 65, 25)
    height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)

fam_history = st.selectbox("Family history of obesity?", ["Yes", "No"])

# Close the panel
st.markdown("</div>", unsafe_allow_html=True)

# Encode input
gender_encoded = 1 if gender == "Male" else 0
fam_history_encoded = 1 if fam_history == "Yes" else 0
input_data = np.array([[age, gender_encoded, height, weight, fam_history_encoded]])
input_scaled = scaler.transform(input_data)

# Green button styling
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        border: none;
        padding: 0.6em 1em;
        font-weight: bold;
        border-radius: 6px;
    }
    div.stButton > button:hover {
        background-color: #218838;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Predict button and result
if st.button("🔍 Let's Find Out"):
    prediction = model.predict(input_scaled)
    result_label = label_mapping.get(int(prediction[0]), "Unknown")
    st.success(f"You are: **{result_label}**")
