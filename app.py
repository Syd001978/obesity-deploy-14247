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

st.title("💡 Are We Obesity?")
st.markdown("Input your data to predic")

# Panel Data Diri
with st.expander("🧍‍♂️ Data Diri & Fisik", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 14, 65, 25)

    with col2:
        height = st.number_input("Height(cm)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Weight(kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

    fam_history = st.selectbox("Riwayat keluarga dengan berat badan berlebih?", ["Yes", "No"])

# Encode input
gender_encoded = 1 if gender == "Male" else 0
fam_history_encoded = 1 if fam_history == "Yes" else 0

# Gabungkan sesuai urutan fitur
input_data = np.array([[age, gender_encoded, height, weight, fam_history_encoded]])
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("🔍 Let's Find out"):
    prediction = model.predict(input_scaled)
    result_label = label_mapping.get(int(prediction[0]), "Tidak diketahui")
    st.success(f"You are : **{result_label}**")
