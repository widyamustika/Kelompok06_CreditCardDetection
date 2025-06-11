import streamlit as st
import joblib
import numpy as np

#model = joblib.load("modelJb_klasifikasiIris.joblib")

def show_single():
    st.header("🪙Single Prediction Classifier")
    st.subheader("Input Features:")
    v3 = float(st.number_input("V3", value=0.0))
    v4 = float(st.number_input("V4", value=0.0))
    v7 = float(st.number_input("V7", value=0.0))
    v10 = float(st.number_input("V10", value=0.0))
    v11 = float(st.number_input("V11", value=0.0))
    v12 = float(st.number_input("V12", value=0.0))
    v14 = float(st.number_input("V14", value=0.0))
    v16 = float(st.number_input("V16", value=0.0))
    v17 = float(st.number_input("V17", value=0.0))
    v18 = float(st.number_input("V18", value=0.0))

    # Checkbox untuk memilih model
    use_knn = st.checkbox("Use KNN")
    use_svm = st.checkbox("Use SVM")
    use_nn = st.checkbox("Use Neural Network")
    use_dt = st.checkbox("Use Decision Tree")

    btn = st.button("Predict")

    if btn:
        input_data = np.array([v3, v4, v7, v10, v11, v12, v14, v16, v17, v18]).reshape(1, -1)

        def show_prediction(model_name, model_file):
            model = joblib.load(model_file)
            pred = model.predict(input_data)
            label = ""
            if pred[0] == 0:
                label = "Transaksi Normal"
            elif pred[0] == 1:
                label = "Transaksi Penipuan"
            else:
                label = "Unknown"
            st.subheader(f"{model_name} Prediction: {pred[0]} → {label}")

        if use_knn:
            show_prediction("K-Nearest Neighbors", "modelJb_Klasifikasi_KNN.joblib")
        if use_svm:
            show_prediction("Support Vector Machine", "modelJb_Klasifikasi_SVM.joblib")
        if use_nn:
            show_prediction("Neural Network", "Klasifikasi_modelJb_NN.joblib")
        if use_dt:
            show_prediction("Decision Tree", "modelJb_Klasifikasi_DecisionTree.joblib")