import streamlit as st
import joblib
import numpy as np

# Load model dan preprocessing
model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# Mapping label hasil prediksi
label_mapping = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

st.title("💡 Prediksi Obesitas dengan XGBoost")
st.markdown("Masukkan data pribadi Anda untuk mengetahui tingkat risiko obesitas.")

# Border menggunakan markdown HTML
st.markdown("""
<div style="border: 1px solid #4F4F4F; padding: 20px; border-radius: 10px; background-color: #1e1e1e;">
    <h5 style='color:white;'>🧍‍♂️ Data Diri & Fisik</h5>
""", unsafe_allow_html=True)

# Dua kolom
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female"])
    age = st.slider("Umur (Age)", 14, 65, 25)

with col2:
    height = st.number_input("Tinggi (Height, dalam meter)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Berat Badan (Weight, dalam kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

fam_history = st.selectbox("Riwayat keluarga dengan berat badan berlebih?", ["Yes", "No"])

# Tutup div
st.markdown("</div>", unsafe_allow_html=True)

# Encoding
gender_encoded = 1 if gender == "Male" else 0
fam_history_encoded = 1 if fam_history == "Yes" else 0
input_data = np.array([[age, gender_encoded, height, weight, fam_history_encoded]])
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_scaled)
    result_label = label_mapping.get(int(prediction[0]), "Tidak diketahui")
    st.success(f"Hasil Prediksi: **{result_label}**")
