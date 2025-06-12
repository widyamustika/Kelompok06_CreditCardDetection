import streamlit as st
import pandas as pd
import numpy as np
import joblib

def get_label(pred):
    return "Transaksi Normal" if pred == 0 else "Transaksi Penipuan" if pred == 1 else "Unknown"

def show_batch():
    st.header("💴 Batch Prediction Classifier")

    uploaded_file = st.file_uploader("📤 Upload CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### 📄 Uploaded Data", df.head())

            # Pilih model yang akan digunakan
            st.subheader("🧠 Choose Models")
            use_knn = st.checkbox("K-Nearest Neighbors")
            use_svm = st.checkbox("Support Vector Machine")
            use_nn  = st.checkbox("Neural Network")
            use_dt  = st.checkbox("Decision Tree")

            if st.button("🔍 Prediksi"):
                if not any([use_knn, use_svm, use_nn, use_dt]):
                    st.warning("⚠️ Silakan pilih setidaknya satu model untuk diprediksi.")
                    return

                # Ambil fitur dan pastikan kolom sesuai
                try:
                    X = df[['V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']]
                except KeyError:
                    st.error("❌ Kolom input tidak sesuai. Pastikan file memiliki kolom: V3, V4, V7, V10, V11, V12, V14, V16, V17, V18")
                    return

                # Fungsi prediksi per model
                def predict_batch(model_path, model_name, col_prefix):
                    try:
                        model = joblib.load(model_path)
                        preds = model.predict(X)
                        result_df = df.copy()
                        result_df[f"{col_prefix} Class"] = preds
                        result_df[f"{col_prefix} Label"] = [get_label(p) for p in preds]
                        st.markdown(f"### 🔍 Hasil Prediksi: {model_name}")
                        st.dataframe(result_df[[f"{col_prefix} Class", f"{col_prefix} Label"]])
                    except Exception as e:
                        st.error(f"❌ Gagal memuat model {model_name}: {str(e)}")

                # Jalankan prediksi berdasarkan model yang dipilih
                if use_knn:
                    predict_batch("modelJb_Klasifikasi_KNN.joblib", "K-Nearest Neighbors", "KNN")
                if use_svm:
                    predict_batch("modelJb_Klasifikasi_SVM.joblib", "Support Vector Machine", "SVM")
                if use_nn:
                    predict_batch("Klasifikasi_modelJb_NN.joblib", "Neural Network", "NN")
                if use_dt:
                    predict_batch("modelJb_Klasifikasi_DecisionTree.joblib", "Decision Tree", "DT")

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {str(e)}")
