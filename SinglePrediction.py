import streamlit as st
import joblib
import numpy as np

def show_single():
    st.header("🪙 Single Prediction Classifier")

    st.subheader("🔢 Input Features:")
    features = {
        'V3': 0.0,
        'V4': 0.0,
        'V7': 0.0,
        'V10': 0.0,
        'V11': 0.0,
        'V12': 0.0,
        'V14': 0.0,
        'V16': 0.0,
        'V17': 0.0,
        'V18': 0.0
    }

    # Ambil nilai input dari user
    for key in features:
        features[key] = st.number_input(f"{key}", value=0.0, step=0.01)

    input_data = np.array(list(features.values())).reshape(1, -1)

    # Pilih model
    st.subheader("🧠 Choose Models:")
    use_knn = st.checkbox("K-Nearest Neighbors")
    use_svm = st.checkbox("Support Vector Machine")
    use_nn  = st.checkbox("Neural Network")
    use_dt  = st.checkbox("Decision Tree")

    if st.button("🔍 Prediksi"):
        if not any([use_knn, use_svm, use_nn, use_dt]):
            st.warning("⚠️ Silakan pilih setidaknya satu model untuk diprediksi.")
            return

        # Fungsi untuk menampilkan hasil prediksi
        def predict_and_show(model_path, model_name):
            try:
                model = joblib.load(model_path)
                pred = model.predict(input_data)
                label = "Transaksi Normal" if pred[0] == 0 else "Transaksi Penipuan"
                st.success(f"🔎 Prediksi oleh {model_name}: {pred[0]} → {label}")
            except Exception as e:
                st.error(f"❌ Gagal memuat model {model_name}: {str(e)}")

        # Jalankan prediksi
        if use_knn:
            predict_and_show("modelJb_Klasifikasi_KNN.joblib", "K-Nearest Neighbors")
        if use_svm:
            predict_and_show("modelJb_Klasifikasi_SVM.joblib", "Support Vector Machine")
        if use_nn:
            predict_and_show("Klasifikasi_modelJb_NN.joblib", "Neural Network")
        if use_dt:
            predict_and_show("modelJb_Klasifikasi_DecisionTree.joblib", "Decision Tree")
