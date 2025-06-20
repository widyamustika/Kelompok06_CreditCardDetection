import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    
    /* Upload container */
    .upload-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(233, 30, 99, 0.1);
        border: 2px dashed rgba(233, 30, 99, 0.5);
        text-align: center;
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
    
    /* Dataframe styling */
    .dataframe {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Result container */
    .result-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(233, 30, 99, 0.1);
        border-left: 5px solid #e91e63;
    }
    
    /* File upload styling */
    .stFileUploader > div > div {
        background: rgba(255, 182, 193, 0.1);
        border: 2px dashed #e91e63;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def get_label(pred):
    return "Transaksi Normal" if pred == 0 else "Transaksi Penipuan" if pred == 1 else "Unknown"

def show_batch():
    load_custom_css()
    
    # Header dengan styling custom
    st.markdown("""
    <div class="main-header">
        <h1>💼 Batch Prediction Classifier 💼</h1>
        <p>Prediksi multiple transaksi sekaligus dari file CSV</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload container
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown("### 📤 **Upload File CSV**")
    st.markdown("Pilih file CSV yang berisi data transaksi untuk diprediksi")
    
    uploaded_file = st.file_uploader(
        "📁 Drag & Drop atau Browse File", 
        type=["csv"],
        help="File harus berformat CSV dengan kolom: V3, V4, V7, V10, V11, V12, V14, V16, V17, V18"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data dengan styling (simplified)
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("### 📊 **Data yang Diupload**")
            st.markdown(f"**Jumlah Baris:** {len(df)} | **Jumlah Kolom:** {len(df.columns)}")
            
            # Show first few rows only
            st.dataframe(df.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Container untuk pemilihan model
            st.markdown('<div class="model-container">', unsafe_allow_html=True)
            st.markdown("### 🧠 **Pilih Model untuk Prediksi**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_knn = st.checkbox("🎯 K-Nearest Neighbors", key="batch_knn")
                use_svm = st.checkbox("⚡ Support Vector Machine", key="batch_svm")
            
            with col2:
                use_nn = st.checkbox("🧠 Neural Network", key="batch_nn")
                use_dt = st.checkbox("🌳 Decision Tree", key="batch_dt")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Tombol prediksi
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                predict_button = st.button("🔮 **MULAI PREDIKSI BATCH**", key="batch_predict")

            if predict_button:
                if not any([use_knn, use_svm, use_nn, use_dt]):
                    st.warning("⚠️ Silakan pilih setidaknya satu model untuk diprediksi.")
                    return

                # Validasi kolom
                required_columns = ['V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Kolom berikut tidak ditemukan dalam file: {missing_columns}")
                    st.info("💡 Pastikan file CSV memiliki kolom: V3, V4, V7, V10, V11, V12, V14, V16, V17, V18")
                    return

                # Ambil fitur
                try:
                    X = df[required_columns]
                    
                    st.markdown("---")
                    st.markdown("### 🎯 **Hasil Prediksi Batch**")

                    # Fungsi prediksi per model
                    def predict_batch(model_path, model_name, col_prefix, emoji):
                        try:
                            with st.spinner(f"Memproses prediksi dengan {model_name}..."):
                                model = joblib.load(model_path)
                                preds = model.predict(X)
                                
                                # Buat dataframe hasil
                                result_df = df.copy()
                                result_df[f"{col_prefix}_Prediction"] = preds
                                result_df[f"{col_prefix}_Label"] = [get_label(p) for p in preds]
                                
                                # Hitung statistik
                                normal_count = sum(1 for p in preds if p == 0)
                                fraud_count = sum(1 for p in preds if p == 1)
                                
                                st.markdown(f"""
                                <div class="result-container">
                                    <h4>{emoji} <strong>{model_name}</strong></h4>
                                    <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                                        <div style="text-align: center;">
                                            <h3 style="color: #4caf50; margin: 0;">{normal_count}</h3>
                                            <p style="margin: 0;">✅ Transaksi Normal</p>
                                        </div>
                                        <div style="text-align: center;">
                                            <h3 style="color: #f44336; margin: 0;">{fraud_count}</h3>
                                            <p style="margin: 0;">⚠️ Transaksi Penipuan</p>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Tampilkan hasil dalam expander
                                with st.expander(f"📋 Lihat Detail Hasil {model_name}"):
                                    st.dataframe(
                                        result_df[[f"{col_prefix}_Prediction", f"{col_prefix}_Label"]],
                                        use_container_width=True
                                    )
                                
                        except Exception as e:
                            st.error(f"❌ Gagal memuat model {model_name}: {str(e)}")

                    # Jalankan prediksi berdasarkan model yang dipilih
                    if use_knn:
                        predict_batch("modelJb_Klasifikasi_KNN.joblib", "K-Nearest Neighbors", "KNN", "🎯")
                    if use_svm:
                        predict_batch("modelJb_Klasifikasi_SVM.joblib", "Support Vector Machine", "SVM", "⚡")
                    if use_nn:
                        predict_batch("Klasifikasi_modelJb_NN.joblib", "Neural Network", "NN", "🧠")
                    if use_dt:
                        predict_batch("modelJb_Klasifikasi_DecisionTree.joblib", "Decision Tree", "DT", "🌳")

                except KeyError as e:
                    st.error(f"❌ Kolom tidak ditemukan: {str(e)}")
                    st.info("💡 Pastikan file CSV memiliki semua kolom yang diperlukan")

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {str(e)}")
            st.info("💡 Pastikan file yang diupload adalah format CSV yang valid")