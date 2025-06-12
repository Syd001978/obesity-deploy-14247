import streamlit as st
import joblib
import numpy as np

# Load model dan preprocessing
model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")

# Mapping label (ubah sesuai urutan label aslinya)
label_mapping = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

# Title
st.title("Prediksi Obesitas Menggunakan XGBoost")
st.write("Masukkan data pribadi Anda untuk mengetahui klasifikasi obesitas.")

# Form input user
age = st.number_input("Usia (tahun)", min_value=0)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
height = st.number_input("Tinggi badan (cm)", min_value=0.0, format="%.2f")
weight = st.number_input("Berat badan (kg)", min_value=0.0, format="%.2f")
fam_history = st.selectbox("Apakah ada riwayat obesitas di keluarga?", ["Yes", "No"])

# Encode input kategorikal
gender_encoded = 1 if gender == "Male" else 0
fam_history_encoded = 1 if fam_history == "Yes" else 0

# Gabungkan input sesuai urutan fitur
input_data = np.array([[age, gender_encoded, height, weight, fam_history_encoded]])
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_scaled)
    result_label = label_mapping.get(prediction[0], "Tidak diketahui")
    st.success(f"Hasil Prediksi: {result_label}")
