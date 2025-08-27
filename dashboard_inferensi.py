import streamlit as st
from src.preprocessing import preprocess_and_update_histori
from src.inference import predict_inflasi
import pandas as pd
from datetime import datetime
import xgboost as xgb  # <- FIX 1: import xgboost

# ------------------ Konfigurasi & data global ------------------
with open('data/features_training.txt') as f:
    features_training = [line.strip() for line in f.readlines()]

# FIX 2: taruh path di level global agar tersedia untuk semua tombol
csv_path = 'data/data_inflasi.csv'
model_path = 'model/model_inflasi.model'

# FIX 3: siapkan session_state untuk menyimpan hasil prediksi
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None
if 'last_pred_label' not in st.session_state:
    st.session_state.last_pred_label = ""

# ------------------ UI ------------------
st.title("📈 Dashboard Prediksi Inflasi Indonesia")
st.markdown("📦 Masukkan data ekonomi **bulan sebelumnya** untuk memprediksi **inflasi bulan berikutnya**.")

tahun = st.number_input("Tahun", value=datetime.now().year, min_value=2010, max_value=2030)
bulan = st.selectbox("Bulan", 
    ["Januari","Februari","Maret","April","Mei","Juni",
     "Juli","Agustus","September","Oktober","November","Desember"], index=6)

BI_Rate = st.number_input("BI Rate (%)", value=5.5, step=0.01)
BBM = st.number_input("Harga BBM (Rp/L)", value=10000, step=50)
Kurs_USD_IDR = st.number_input("Kurs USD/IDR", value=16418, step=10)
Harga_Beras = st.number_input("Harga Beras (Rp/kg)", value=13735, step=50)
Inflasi_Inti = st.number_input("Inflasi Inti (%)", value=0.08, step=0.01)
Inflasi_Total = st.number_input("Inflasi Total (%)", value=1.6, step=0.01)

# ------------------ Tombol Prediksi ------------------
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

    df_infer, df_histori = preprocess_and_update_histori(csv_path, input_user, features_training)
    prediksi = float(predict_inflasi(model_path, df_infer, features_training))

    bulan_list = ["Januari","Februari","Maret","April","Mei","Juni",
                  "Juli","Agustus","September","Oktober","November","Desember"]
    bulan_index = bulan_list.index(bulan)
    bulan_pred = bulan_list[(bulan_index + 1) % 12]
    tahun_pred = tahun + 1 if bulan_index == 11 else tahun

    # Simpan ke session_state agar tidak hilang saat app rerun
    st.session_state.last_pred = prediksi
    st.session_state.last_pred_label = f"{bulan_pred} {tahun_pred}"

# ------------------ Tombol Feature Importance ------------------
if st.button("Tampilkan Feature Importance"):
    # Load model XGBoost
    model = xgb.Booster()
    model.load_model(model_path)

    # Ambil importance; bisa 'weight'/'gain'/'cover'. Umumnya 'gain' lebih informatif.
    raw_imp = model.get_score(importance_type='gain')

    # Map 'f0','f1',... ke nama fitur asli
    def key_to_name(k: str) -> str:
        if k.startswith('f'):
            try:
                idx = int(k[1:])
                if 0 <= idx < len(features_training):
                    return features_training[idx]
            except:
                pass
        return k

    imp_named = {key_to_name(k): v for k, v in raw_imp.items()}
    imp_df = (pd.DataFrame(list(imp_named.items()), columns=['Fitur', 'Skor'])
                .sort_values('Skor', ascending=False))

    st.subheader("📊 Feature Importance (XGBoost, tipe: gain)")
    if imp_df.empty:
        st.info("Model tidak memiliki importance yang terbaca.")
    else:
        st.dataframe(imp_df, use_container_width=True)
        st.bar_chart(imp_df.set_index("Fitur"))  # tanpa matplotlib

# ------------------ Selalu tampilkan hasil prediksi terakhir ------------------
if st.session_state.last_pred is not None:
    st.success(f"📌 Prediksi Inflasi untuk **{st.session_state.last_pred_label}** adalah: **{st.session_state.last_pred:.2f}%**")
