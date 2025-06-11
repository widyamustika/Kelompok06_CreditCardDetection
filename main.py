import streamlit as st
from SinglePrediction import show_single
from SinglePredictionRegresi import show_single_regressor
#from BatchUploadPrediction import show_batch
from BatchUpload import show_batch
from BatchUploadRegresi import show_batch_regressor

st.title("💳 Credit Card Fraud Detection")

# Menu utama
main_menu = st.sidebar.radio("Choose a mode:", ["Single Prediction", "Batch Upload"])

if main_menu == "Single Prediction":
    # Sub-menu untuk Single Prediction
    sub_menu = st.sidebar.radio("Choose model type:", ["Classifier", "Regressor"])
    
    if sub_menu == "Classifier":
        show_single()
    elif sub_menu == "Regressor":
        show_single_regressor()

elif main_menu == "Batch Upload":
    # Sub-menu untuk Batch Upload
    sub_menu = st.sidebar.radio("Choose model type:", ["Classifier", "Regressor"])
    
    if sub_menu == "Classifier":
        show_batch()
    elif sub_menu == "Regressor":
        show_batch_regressor()