import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fungsi untuk membuat plot prediksi vs aktual
def plot_prediction_error(y_true, y_pred, model_name):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Garis ideal
    ax.set_xlabel("Actual Amount")
    ax.set_ylabel("Predicted Amount")
    ax.set_title(f"{model_name} - Actual vs Predicted")
    return fig

# Fungsi utama batch regressor
def show_batch_regressor():
    st.header("💴 Batch Prediction Regressor")

    uploaded_file = st.file_uploader("📤 Upload CSV file for prediction", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### 📄 Uploaded Data", df.head())

        # Pilih model yang akan digunakan
        st.subheader("🧠 Choose Models")
        use_knn = st.checkbox("Use K-Nearest Neighbors")
        use_svm = st.checkbox("Use Support Vector Machine")
        use_nn  = st.checkbox("Use Neural Network")
        use_dt  = st.checkbox("Use Decision Tree")

        if st.button("🔍 Predict"):
            if not any([use_knn, use_svm, use_nn, use_dt]):
                st.warning("Please select at least one model.")
                return

            # Ambil fitur - pastikan sesuai dengan model saat training
            X = df.iloc[:, :10]
            X.columns = ['V1', 'V2', 'V4', 'V5', 'V7', 'V20', 'V22', 'V23', 'V26', 'V28']

            # Cek apakah kolom Amount tersedia untuk evaluasi
            actual_amount = df["Amount"] if "Amount" in df.columns else None

            # Fungsi prediksi dan evaluasi
            def predict_and_display(model_path, model_name, col_name):
                model = joblib.load(model_path)
                preds = model.predict(X)
                result_df = df.copy()
                result_df[col_name] = preds

                st.markdown(f"### 📊 {model_name} Predictions")

                if actual_amount is not None:
                    result_df["Actual Amount"] = actual_amount
                    mae = mean_absolute_error(actual_amount, preds)
                    rmse = mean_squared_error(actual_amount, preds, squared=False)

                    st.metric("📏 Mean Absolute Error (MAE)", f"{mae:.2f}")
                    st.metric("📐 Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

                    fig = plot_prediction_error(actual_amount, preds, model_name)
                    st.pyplot(fig)

                st.dataframe(result_df[[col_name] + (["Actual Amount"] if actual_amount is not None else [])])

            # Jalankan masing-masing model sesuai pilihan
            if use_knn:
                predict_and_display("modelJb_Regresi_KNN.joblib", "K-Nearest Neighbors", "KNN Predicted Amount")

            if use_svm:
                predict_and_display("modelJb_Regresi_SVM.joblib", "Support Vector Machine", "SVM Predicted Amount")

            if use_nn:
                predict_and_display("Regresi_modelJb_NN.joblib", "Neural Network", "NN Predicted Amount")

            if use_dt:
                predict_and_display("modelJb_Regresi_DecisionTree.joblib", "Decision Tree", "DT Predicted Amount")
