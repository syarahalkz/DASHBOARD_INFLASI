import streamlit as st
from src.preprocessing import preprocess_and_update_histori
from src.inference import predict_inflasi
import pandas as pd
from datetime import datetime

# Load fitur training yang sudah lengkap dan urut
with open('data/features_training.txt') as f:
    features_training = [line.strip() for line in f.readlines()]

st.title("📈 Dashboard Prediksi Inflasi Indonesia")

st.markdown("""
📦 Masukkan data ekonomi **bulan sebelumnya** untuk memprediksi **inflasi bulan berikutnya**.
""")

# Input user
tahun = st.number_input("Tahun", value=datetime.now().year, min_value=2010, max_value=2030)
bulan = st.selectbox("Bulan", 
                     ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                      "Juli", "Agustus", "September", "Oktober", "November", "Desember"], index=6)
BI_Rate = st.number_input("BI Rate (%)", value=5.5, step=0.01)
BBM = st.number_input("Harga BBM (Rp/L)", value=10000, step=50)
Kurs_USD_IDR = st.number_input("Kurs USD/IDR", value=16418, step=10)
Harga_Beras = st.number_input("Harga Beras (Rp/kg)", value=13735, step=50)
Inflasi_Inti = st.number_input("Inflasi Inti (%)", value=0.08, step=0.01)
Inflasi_Total = st.number_input("Inflasi Total (%)", value=1.6, step=0.01)

if st.button("Prediksi Inflasi"):
    input_user = {
        'Tahun': tahun,
        'Bulan': bulan,
        'BI_Rate': BI_Rate,
        'BBM': BBM,
        'Kurs_USD_IDR': Kurs_USD_IDR,
        'Harga_Beras': Harga_Beras,
        'Inflasi_Inti': Inflasi_Inti,
        'Inflasi_Total': Inflasi_Total
    }
    csv_path = 'data/data_inflasi.csv'
    model_path = 'model/model_inflasi.model'

    df_infer, df_histori = preprocess_and_update_histori(csv_path, input_user, features_training)
    prediksi = predict_inflasi(model_path, df_infer, features_training)

    bulan_list = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                  "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    bulan_index = bulan_list.index(bulan)  # 0-based index
    bulan_pred = bulan_list[(bulan_index + 1) % 12]  # bulan berikutnya
    tahun_pred = tahun + 1 if bulan_index == 11 else tahun  # tambah tahun jika Desember

    st.success(f"📌 Prediksi Inflasi untuk **{bulan_pred} {tahun_pred}** adalah: **{prediksi:.2f}%**")

