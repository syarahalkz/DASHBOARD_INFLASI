# dashboard_inferensi.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from datetime import datetime

# ==============================
# 1. Load model dan scaler
# ==============================
MODEL_PATH = "model_inflasi_percobaan.pkl"
SCALER_PATH = "scaler.pkl"  # opsional, kalau data perlu scaling

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None

# ==============================
# 2. Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="Prediksi Inflasi Bulanan", page_icon="📈", layout="wide")
st.title("📈 Dashboard Prediksi Inflasi Bulanan - XGBoost")

st.markdown("Masukkan nilai variabel makroekonomi untuk memprediksi inflasi bulan berikutnya.")

# ==============================
# 3. Input Parameter
# ==============================
with st.sidebar:
    st.header("Input Parameter")
    tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=datetime.now().year)
    bulan = st.selectbox("Bulan", 
                         ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
                          "Juli", "Agustus", "September", "Oktober", "November", "Desember"], 
                         index=6)  # default Juli
    
    BI_Rate = st.number_input("BI Rate (%)", value=6.0, step=0.01)
    BBM = st.number_input("Harga BBM (Rp/L)", value=10000, step=50)
    Kurs_USD_IDR = st.number_input("Kurs USD/IDR", value=15000, step=10)
    Harga_Beras = st.number_input("Harga Beras (Rp/kg)", value=12000, step=50)
    Inflasi_Inti = st.number_input("Inflasi Inti (%)", value=2.5, step=0.01)

# ==============================
# 4. Preprocessing
# ==============================
# Encode bulan ke bentuk sin & cos (seasonality)
bulan_num = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
             "Juli", "Agustus", "September", "Oktober", "November", "Desember"].index(bulan) + 1
bulan_sin = np.sin(2 * np.pi * bulan_num / 12)
bulan_cos = np.cos(2 * np.pi * bulan_num / 12)

# Buat DataFrame input
input_data = pd.DataFrame([{
    "Tahun": tahun,
    "BI_Rate": BI_Rate,
    "BBM": BBM,
    "Kurs_USD_IDR": Kurs_USD_IDR,
    "Harga_Beras": Harga_Beras,
    "Inflasi_Inti": Inflasi_Inti,
    "bulan_sin": bulan_sin,
    "bulan_cos": bulan_cos
}])

# Scaling jika ada
if scaler:
    input_scaled = scaler.transform(input_data)
else:
    input_scaled = input_data

# ==============================
# 5. Prediksi
# ==============================
if st.sidebar.button("Prediksi Inflasi"):
    prediksi = model.predict(input_scaled)[0]
    
    st.subheader("📊 Hasil Prediksi")
    st.metric(label="Prediksi Inflasi Bulanan (%)", value=f"{prediksi:.2f}")

    # Opsional: tampilkan tabel input
    with st.expander("Lihat Data Input"):
        st.dataframe(input_data)

    # Opsional: keterangan prediksi
    if prediksi > 5:
        st.warning("⚠️ Inflasi cukup tinggi, waspada kenaikan harga barang.")
    elif prediksi > 3:
        st.info("ℹ️ Inflasi dalam batas wajar namun perlu diwaspadai.")
    else:
        st.success("✅ Inflasi rendah dan terkendali.")

