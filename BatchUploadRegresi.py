import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Custom CSS untuk nuansa pink dan styling yang lebih menarik
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
    
    /* Result container */
    .result-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(233, 30, 99, 0.1);
        border-left: 5px solid #e91e63;
    }
    
    /* Statistics container */
    .stats-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(233, 30, 99, 0.2);
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

# Fungsi untuk membuat plot prediksi vs aktual
def plot_prediction_error(y_true, y_pred, model_name):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set background color
    fig.patch.set_facecolor('#ffeef8')
    ax.set_facecolor('#ffeef8')
    
    # Scatter plot
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, c='#e91e63', s=50, edgecolors='white', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel("Actual Amount", fontsize=12, color='#ad1457')
    ax.set_ylabel("Predicted Amount", fontsize=12, color='#ad1457')
    ax.set_title(f"{model_name} - Actual vs Predicted", fontsize=14, color='#ad1457', fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, color='#e91e63')
    ax.legend()
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#e91e63')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    return fig

# Fungsi utama batch regressor
def show_batch_regressor():
    load_custom_css()
    
    # Header dengan styling custom
    st.markdown("""
    <div class="main-header">
        <h1>💰 Batch Prediction Regressor 💰</h1>
        <p>Prediksi jumlah transaksi multiple sekaligus dengan analisis mendalam</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload container
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown("### 📤 **Upload File CSV**")
    st.markdown("Pilih file CSV yang berisi data untuk prediksi regresi")
    
    uploaded_file = st.file_uploader(
        "📁 Drag & Drop atau Browse File", 
        type=["csv"],
        help="File harus berformat CSV dengan kolom: V1, V2, V4, V5, V7, V20, V22, V23, V26, V28"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Display uploaded data dengan styling
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("### 📊 **Data yang Diupload**")
            st.markdown(f"**Jumlah Baris:** {len(df)} | **Jumlah Kolom:** {len(df.columns)}")
            
            # Show first few rows
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Container untuk pemilihan model
            st.markdown('<div class="model-container">', unsafe_allow_html=True)
            st.markdown("### 🧠 **Pilih Model untuk Prediksi**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                use_knn = st.checkbox("🎯 K-Nearest Neighbors", key="batch_reg_knn")
                use_svm = st.checkbox("⚡ Support Vector Machine", key="batch_reg_svm")
            
            with col2:
                use_nn = st.checkbox("🧠 Neural Network", key="batch_reg_nn")
                use_dt = st.checkbox("🌳 Decision Tree", key="batch_reg_dt")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Tombol prediksi
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                predict_button = st.button("💎 **MULAI PREDIKSI REGRESI**", key="batch_reg_predict")

            if predict_button:
                if not any([use_knn, use_svm, use_nn, use_dt]):
                    st.warning("⚠️ Silakan pilih setidaknya satu model untuk diprediksi.")
                    return

                # Validasi dan persiapan data
                try:
                    # Ambil fitur - sesuaikan dengan model training
                    required_columns = ['V1', 'V2', 'V4', 'V5', 'V7', 'V20', 'V22', 'V23', 'V26', 'V28']
                    
                    # Check if we have the required columns or use first 10 columns
                    if all(col in df.columns for col in required_columns):
                        X = df[required_columns]
                    else:
                        if len(df.columns) >= 10:
                            X = df.iloc[:, :10]
                            X.columns = required_columns
                            st.info("💡 Menggunakan 10 kolom pertama sebagai fitur input")
                        else:
                            st.error("❌ File harus memiliki minimal 10 kolom untuk prediksi")
                            return

                    # Cek apakah kolom Amount tersedia untuk evaluasi
                    actual_amount = df["Amount"] if "Amount" in df.columns else None

                    st.markdown("---")
                    st.markdown("### 💎 **Hasil Prediksi Regresi**")

                    # Fungsi prediksi dan evaluasi
                    def predict_and_display(model_path, model_name, col_name, emoji):
                        try:
                            with st.spinner(f"Memproses prediksi dengan {model_name}..."):
                                model = joblib.load(model_path)
                                preds = model.predict(X)
                                
                                # Buat hasil dataframe
                                result_df = df.copy()
                                result_df[col_name] = preds

                                st.markdown(f"""
                                <div class="result-container">
                                    <h4>{emoji} <strong>{model_name}</strong></h4>
                                </div>
                                """, unsafe_allow_html=True)

                                # Evaluasi jika actual amount tersedia
                                if actual_amount is not None:
                                    mae = mean_absolute_error(actual_amount, preds)
                                    rmse = mean_squared_error(actual_amount, preds, squared=False)
                                    
                                    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                                    st.markdown("#### 📊 **Evaluasi Model**")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("📏 Mean Absolute Error (MAE)", f"${mae:,.2f}")
                                    with col2:
                                        st.metric("📐 Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)

                                    # Plot prediksi vs aktual
                                    fig = plot_prediction_error(actual_amount, preds, model_name)
                                    st.pyplot(fig)
                                    plt.close(fig)

                                # Tampilkan hasil dalam expander
                                with st.expander(f"📋 Lihat Detail Hasil {model_name}"):
                                    display_columns = [col_name]
                                    if actual_amount is not None:
                                        result_df["Actual_Amount"] = actual_amount
                                        display_columns.append("Actual_Amount")
                                    
                                    st.dataframe(result_df[display_columns], use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Gagal memuat model {model_name}: {str(e)}")

                    # Jalankan masing-masing model sesuai pilihan
                    if use_knn:
                        predict_and_display("modelJb_Regresi_KNN.joblib", "K-Nearest Neighbors", "KNN_Predicted_Amount", "🎯")

                    if use_svm:
                        predict_and_display("modelJb_Regresi_SVM.joblib", "Support Vector Machine", "SVM_Predicted_Amount", "⚡")

                    if use_nn:
                        predict_and_display("Regresi_modelJb_NN.joblib", "Neural Network", "NN_Predicted_Amount", "🧠")

                    if use_dt:
                        predict_and_display("modelJb_Regresi_DecisionTree.joblib", "Decision Tree", "DT_Predicted_Amount", "🌳")

                except Exception as e:
                    st.error(f"❌ Error dalam pemrosesan data: {str(e)}")
                    st.info("💡 Pastikan file CSV memiliki format yang sesuai")

        except Exception as e:
            st.error(f"❌ Gagal membaca file: {str(e)}")
            st.info("💡 Pastikan file yang diupload adalah format CSV yang valid")