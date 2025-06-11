import streamlit as st
import joblib
import numpy as np

def show_single_regressor():
    st.header("🪙 Single Prediction Regressor")

    st.subheader("🔢 Masukkan Nilai Fitur:")

    # Input fitur regresi yang sesuai dengan model training
    features = {
        'V1': 0.0,
        'V2': 0.0,
        'V4': 0.0,
        'V5': 0.0,
        'V7': 0.0,
        'V20': 0.0,
        'V22': 0.0,
        'V23': 0.0,
        'V26': 0.0,
        'V28': 0.0
    }

    # Ambil nilai dari pengguna
    for key in features:
        features[key] = st.number_input(f"{key}", value=0.0, step=0.01)

    input_data = np.array(list(features.values())).reshape(1, -1)

    # Model yang dapat dipilih
    st.subheader("🧠 Pilih Model:")
    use_knn = st.checkbox("K-Nearest Neighbors")
    use_svm = st.checkbox("Support Vector Machine")
    use_nn  = st.checkbox("Neural Network")
    use_dt  = st.checkbox("Decision Tree")

    # Tombol prediksi
    if st.button("🔍 Prediksi"):
        if not any([use_knn, use_svm, use_nn, use_dt]):
            st.warning("⚠️ Silakan pilih setidaknya satu model untuk diprediksi.")
            return

        # Fungsi menampilkan prediksi
        def predict_and_show(model_path, model_name):
            try:
                model = joblib.load(model_path)
                pred = model.predict(input_data)
                st.success(f"💡 Prediksi oleh {model_name}: {pred[0]:,.2f}")
            except Exception as e:
                st.error(f"❌ Gagal memuat model {model_name}: {str(e)}")

        # Eksekusi prediksi berdasarkan pilihan
        if use_knn:
            predict_and_show("modelJb_Regresi_KNN.joblib", "K-Nearest Neighbors")
        if use_svm:
            predict_and_show("modelJb_Regresi_SVM.joblib", "Support Vector Machine")
        if use_nn:
            predict_and_show("Regresi_modelJb_NN.joblib", "Neural Network")
        if use_dt:
            predict_and_show("modelJb_Regresi_DecisionTree.joblib", "Decision Tree")
