from xgboost import Booster, DMatrix

def predict_inflasi(model_path, df_features, feature_list):
    model = Booster()
    model.load_model(model_path)
    
    # Buang kolom target kalau ada di df_features sebelum prediksi
    X = df_features.copy()
    if 'Inflasi_Total' in X.columns:
        X = X.drop(columns=['Inflasi_Total'])
    
    # Ambil fitur yang sesuai urutan training (yang sudah tanpa target)
    X = X[feature_list]

    dmatrix = DMatrix(X)
    preds = model.predict(dmatrix)
    return preds[0]
