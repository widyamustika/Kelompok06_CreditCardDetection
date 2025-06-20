import streamlit as st
import joblib
import numpy as np

# Custom CSS 
def load_custom_css():
    st.markdown("""
    <style>
    /* Background gradient pink */
    .stApp {
        background: linear-gradient(135deg, #ffeef8 0%, #f8e8f5 50%, #f0d9e7 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #e91e63, #f06292);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(233, 30, 99, 0.3);
    }
    
    /* Feature input container */
    .feature-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(233, 30, 99, 0.1);
        border: 2px solid rgba(233, 30, 99, 0.2);
    }
    
    /* Model selection styling */
    .model-container {
        background: linear-gradient(135deg, #fce4ec, #f8bbd9);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(233, 30, 99, 0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e91e63, #f06292);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 30, 99, 0.4);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #ad1457;
        font-weight: 600;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid rgba(233, 30, 99, 0.3);
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(90deg, #4caf50, #81c784);
        border-radius: 10px;
    }
    
    /* Warning message styling */
    .stWarning {
        background: linear-gradient(90deg, #ff9800, #ffb74d);
        border-radius: 10px;
    }
    
    /* Feature group styling */
    .feature-group {
        background: rgba(255, 182, 193, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #e91e63;
    }
    
    /* Amount display styling */
    .amount-display {
        background: linear-gradient(135deg, #4caf50, #81c784);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2em;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

def show_single_regressor():
    load_custom_css()
    
    # Header dengan styling custom
    st.markdown("""
    <div class="main-header">
        <h1>💎 Single Prediction Regressor 💎</h1>
        <p>Prediksi jumlah transaksi dengan akurasi tinggi</p>
    </div>
    """, unsafe_allow_html=True)

    # Container untuk input fitur
    st.markdown('<div class="feature-container">', unsafe_allow_html=True)
    st.markdown("### 🔢 **Input Features**")

    features = {
        'V1': 0.0, 'V2': 0.0, 'V4': 0.0, 'V5': 0.0, 'V7': 0.0,
        'V20': 0.0, 'V22': 0.0, 'V23': 0.0, 'V26': 0.0, 'V28': 0.0
    }

    # Satu grup fitur dengan dua kolom
    st.markdown('<div class="feature-group">', unsafe_allow_html=True)
    st.markdown("#### 💎 **Fitur Transaksi**")
    
    col1, col2 = st.columns(2)
    feature_keys = list(features.keys())
    
    with col1:
        for key in feature_keys[:5]:
            features[key] = st.number_input(
                f"📊 {key}", 
                value=0.0, 
                step=0.01, 
                key=f"regressor_{key}",
                help=f"Masukkan nilai untuk fitur {key}"
            )
    
    with col2:
        for key in feature_keys[5:]:
            features[key] = st.number_input(
                f"📊 {key}", 
                value=0.0, 
                step=0.01, 
                key=f"regressor_{key}",
                help=f"Masukkan nilai untuk fitur {key}"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    input_data = np.array(list(features.values())).reshape(1, -1)

    # Container untuk pemilihan model
    st.markdown('<div class="model-container">', unsafe_allow_html=True)
    st.markdown("### 🧠 **Pilih Model Machine Learning**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_knn = st.checkbox("🎯 K-Nearest Neighbors", key="regressor_knn")
        use_svm = st.checkbox("⚡ Support Vector Machine", key="regressor_svm")
    
    with col2:
        use_nn = st.checkbox("🧠 Neural Network", key="regressor_nn")
        use_dt = st.checkbox("🌳 Decision Tree", key="regressor_dt")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Tombol prediksi
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("💰 **PREDIKSI JUMLAH**", key="regressor_predict")

    if predict_button:
        if not any([use_knn, use_svm, use_nn, use_dt]):
            st.warning("⚠️ Silakan pilih setidaknya satu model untuk diprediksi.")
            return

        st.markdown("---")
        st.markdown("### 💎 **Hasil Prediksi**")

        def predict_and_show(model_path, model_name, emoji):
            try:
                model = joblib.load(model_path)
                pred = model.predict(input_data)
                
                # Format amount tanpa dolar
                formatted_amount = f"{pred[0]:,.2f}"
                
                st.markdown(f"""
                <div class="amount-display">
                    {emoji} <strong>{model_name}</strong><br>
                    💰 Prediksi Jumlah: {formatted_amount}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Gagal memuat model {model_name}: {str(e)}")

        # Jalankan prediksi dengan emoji yang menarik
        if use_knn:
            predict_and_show("modelJb_Regresi_KNN.joblib", "K-Nearest Neighbors", "🎯")
        if use_svm:
            predict_and_show("modelJb_Regresi_SVM.joblib", "Support Vector Machine", "⚡")
        if use_nn:
            predict_and_show("Regresi_modelJb_NN.joblib", "Neural Network", "🧠")
        if use_dt:
            predict_and_show("modelJb_Regresi_DecisionTree.joblib", "Decision Tree", "🌳")
