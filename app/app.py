import sys
import os

# --- 1. SETUP PATH OTOMATIS (Logic Tambahan) ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(current_dir)                 
sys.path.append(root_dir)                               

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io

# Coba import konfigurasi dari src (jika ada), jika tidak pakai default
try:
    from src.config import MODEL_PATH, CLASS_NAMES
    from src.preprocess import prepare_image
except ImportError:
    # Fallback jika src belum siap (agar app tidak crash saat dev UI)
    MODEL_PATH = os.path.join(current_dir, 'best_mobnet.h5')
    CLASS_NAMES = ['Anthracnose', 'Golmichi', 'Healthy', 'Powdery Mildew', 'Turning Brown']
    def prepare_image(file): return file # Dummy

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="MangiferaNet - Deteksi Penyakit Daun Mangga",
    page_icon="🥭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CSS CUSTOM (TETAP SAMA SEPERTI KODE ANDA) ---
st.markdown("""
    <style>
    /* Hide default sidebar */
    [data-testid="stSidebar"] { display: none; }
    
    /* Navigation Bar */
    .navbar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        align-items:center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .nav-menu {
        display: flex;
        gap: 2rem;
        list-style: none;
        margin: 0;
        padding: 0;
    }
    
    .nav-item {
        color: white;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 500;
    }
    
    .nav-item:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .nav-item.active {
        background-color: rgba(255, 255, 255, 0.3);
        border-bottom: 2px solid #FF9F43;
    }
    
    .main-header {
        font-size: 3rem;
        color: #FF9F43;
        text-align: center;
        font-weight: bold;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    .disease-card {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF9F43;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: transparent;
        color: #667eea;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Better content margins */
    .block-container {
        padding-left: 5rem !important;
        padding-right: 5rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    /* Navigation buttons container */
    .nav-buttons {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem 0;
    }
    
    /* Info box for disease classes */
    .info-box {
        background-color: #1a1a1a;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA PENYAKIT (UI RICH DATA) ---
# Tetap di app.py karena mengandung data UI (warna, html list) yang tidak ada di config.py biasa
DISEASE_INFO = {
    'Anthracnose': {
        'name_id': 'Anthracnose',
        'description': 'Penyakit jamur yang menyerang buah, daun, dan ranting mangga. Disebabkan oleh Colletotrichum gloeosporioides.',
        'symptoms': [
            'Bercak coklat kehitaman pada daun',
            'Buah busuk dengan bintik-bintik gelap',
            'Daun mengering dan gugur'
        ],
        'treatment': [
            'Semprot dengan fungisida berbasis tembaga',
            'Buang bagian tanaman yang terinfeksi',
            'Jaga kebersihan kebun',
            'Lakukan pemangkasan untuk sirkulasi udara'
        ],
        'color': '#E74C3C'
    },
    'Golmichi': {
        'name_id': 'Golmichi',
        'description': 'Penyakit yang menyebabkan deformasi dan pertumbuhan abnormal pada daun mangga.',
        'symptoms': [
            'Daun menggulung dan berkerut',
            'Pertumbuhan daun tidak normal',
            'Warna daun pucat atau kekuningan'
        ],
        'treatment': [
            'Gunakan insektisida untuk mengendalikan vektor',
            'Potong dan musnahkan bagian yang terinfeksi',
            'Tingkatkan nutrisi tanaman',
            'Monitor pertumbuhan secara rutin'
        ],
        'color': '#9B59B6'
    },
    'Healthy': {
        'name_id': 'Healthy',
        'description': 'Daun mangga dalam kondisi sehat tanpa tanda-tanda penyakit.',
        'symptoms': [
            'Warna hijau segar dan merata',
            'Permukaan daun halus dan bersih',
            'Tidak ada bercak atau deformasi'
        ],
        'treatment': [
            'Pertahankan perawatan rutin',
            'Berikan pupuk secara teratur',
            'Pastikan penyiraman yang cukup',
            'Lakukan pemantauan berkala'
        ],
        'color': '#2ECC71'
    },
    'Powdery Mildew': {
        'name_id': 'Powdery Mildew',
        'description': 'Infeksi jamur yang menyebabkan lapisan putih seperti tepung pada permukaan daun.',
        'symptoms': [
            'Lapisan putih seperti bedak pada daun',
            'Daun mengkerut dan melengkung',
            'Pertumbuhan terhambat',
            'Daun menjadi kuning dan gugur'
        ],
        'treatment': [
            'Aplikasi fungisida sulfur atau potassium bikarbonat',
            'Tingkatkan sirkulasi udara',
            'Kurangi kelembaban berlebih',
            'Buang daun yang terinfeksi parah'
        ],
        'color': '#3498DB'
    },
    'Turning Brown': {
        'name_id': 'Turning Brown',
        'description': 'Kondisi daun yang mulai mencoklat, bisa disebabkan oleh berbagai faktor stress atau penyakit.',
        'symptoms': [
            'Daun berubah warna coklat',
            'Tepi daun mengering',
            'Tekstur daun menjadi rapuh',
            'Kemungkinan gugur prematur'
        ],
        'treatment': [
            'Identifikasi penyebab (penyakit, kekurangan nutrisi, atau stress air)',
            'Perbaiki pola penyiraman',
            'Berikan pupuk seimbang',
            'Lindungi dari stress lingkungan'
        ],
        'color': '#FF6B6B'
    }
}

# --- FUNGSI LOAD MODEL (DISESUAIKAN DENGAN SRC) ---
@st.cache_resource
def load_model_app():
    """Load model menggunakan path dari config src"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model tidak ditemukan di path: {MODEL_PATH}")
        st.info("Pastikan file .h5 sudah dicopy ke folder app/.")
        return None
    try:
        # Load model TensorFlow
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- FUNGSI PREDIKSI (MENGGUNAKAN LOGIKA SRC) ---
def predict_disease(model, image_file):
    """Melakukan prediksi penyakit menggunakan model dan preprocessing dari src"""
    
    # 1. Preprocessing (Menggunakan fungsi prepare_image dari src/preprocess.py)
    # Fungsi ini sudah handle resize 224x224 dan rescale 1/255
    try:
        processed_img = prepare_image(image_file)
    except Exception as e:
        # Biarkan caller menampilkan pesan kesalahan yang sesuai
        raise ValueError(f"Gagal memproses gambar untuk prediksi: {e}")

    # 2. Prediksi
    predictions = model.predict(processed_img)
    
    # 3. Interpretasi Hasil
    # CLASS_NAMES diambil dari src/config.py agar konsisten dengan training
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    # Mengambil probabilitas semua kelas
    probabilities = {CLASS_NAMES[i]: predictions[0][i] * 100 for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, probabilities

# --- MAIN APP LOGIC ---

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Header Navigation
st.markdown("""
    <p class="main-header">🥭 MangiferaNet</p>
    <p class="sub-header">Sistem Deteksi Penyakit Daun Mangga Berbasis Deep Learning</p>
""", unsafe_allow_html=True)

# Navigation buttons
st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = 'Home'
with col2:
    if st.button("🔍 Detect", use_container_width=True):
        st.session_state.page = 'Detect'
with col3:
    if st.button("📊 Info", use_container_width=True):
        st.session_state.page = 'Info'
with col4:
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = 'About'
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# --- HALAMAN: BERANDA ---
if st.session_state.page == 'Home':
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>5</h2>
            <p>Kelas Penyakit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖</h2>
            <p>Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>📸</h2>
            <p>Upload & Prediksi</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 🌟 Fitur Utama")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Deteksi Otomatis
        Sistem dapat mendeteksi 5 jenis kondisi daun mangga:
        - **Anthracnose** - Penyakit jamur antraknosa
        - **Golmichi** - Deformasi daun
        - **Healthy** - Daun sehat
        - **Powdery Mildew** - Embun tepung
        - **Turning Brown** - Daun mencoklat
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Akurat & Cepat
        - Prediksi real-time dalam hitungan detik
        - Menggunakan teknologi Deep Learning
        - Interface yang user-friendly
        - Informasi lengkap tentang penyakit
        - Rekomendasi penanganan
        """)
    
    st.markdown("---")
    
    st.markdown("## 📖 Cara Menggunakan")
    st.markdown("""
    1. **Upload Gambar** - Pilih foto daun mangga dari device Anda
    2. **Prediksi** - Klik tombol "Deteksi Penyakit"
    3. **Hasil** - Lihat hasil prediksi dan tingkat kepercayaan
    4. **Informasi** - Baca informasi detail dan cara penanganan
    """)
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Kelas Penyakit yang Dapat Dideteksi:</h4>
        <ul>
            <li><strong>Anthracnose</strong> - Penyakit jamur antraknosa</li>
            <li><strong>Golmichi</strong> - Deformasi daun</li>
            <li><strong>Healthy</strong> - Daun sehat</li>
            <li><strong>Powdery Mildew</strong> - Embun tepung</li>
            <li><strong>Turning Brown</strong> - Daun mencoklat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- HALAMAN: DETEKSI ---
elif st.session_state.page == 'Detect':
    st.markdown("## 📸 Upload Gambar Daun Mangga")
    
    # Load model menggunakan fungsi baru
    # Tampilkan spinner saat memuat model dan simpan status di session_state
    with st.spinner('Memuat model...'):
        model = load_model_app()
    st.session_state.model_loaded = True if model is not None else False
    # Tampilkan status singkat model
    if st.session_state.model_loaded:
        st.success("Model terload dan siap melakukan prediksi.")
    else:
        st.info("Model belum terload. Pastikan file `.h5` ada di folder `app/`.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Pilih gambar daun mangga (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png']
        )

        # Hanya terima gambar dari upload; tidak menggunakan dummy/sample image
        image_to_use = None
        if uploaded_file is not None:
            # Baca bytes untuk metadata lalu rewind agar stream tetap bisa dipakai
            try:
                uploaded_bytes = uploaded_file.read()
                img = Image.open(io.BytesIO(uploaded_bytes))
                width, height = img.size
                size_kb = len(uploaded_bytes) / 1024
                uploaded_file.seek(0)
            except Exception:
                width, height, size_kb = None, None, None

            st.image(uploaded_file, caption='Gambar yang diupload', use_column_width=True)
            st.markdown("**Metadata File**")
            md_cols = st.columns(3)
            md_cols[0].markdown(f"- **Nama:** {getattr(uploaded_file, 'name', 'uploaded_image')}")
            md_cols[1].markdown(f"- **Ukuran:** {size_kb:.1f} KB" if size_kb is not None else "- **Ukuran:** -")
            md_cols[2].markdown(f"- **Resolusi:** {width} x {height}" if width is not None else "- **Resolusi:** -")

        # Jika model belum siap, beritahu user — tombol Deteksi baru muncul jika model siap
        if not st.session_state.get('model_loaded', False):
            st.warning("Model belum tersedia. Mohon tunggu atau pastikan file `.h5` ada di folder `app/`.")
        else:
            if st.button("🔍 Deteksi Penyakit", key="predict_btn"):
                if model is not None:
                    # Gunakan file yang diupload sebagai input prediksi
                    input_for_predict = uploaded_file
                    if input_for_predict is None:
                        st.error('Tidak ada gambar untuk dideteksi. Silakan upload atau gunakan contoh.')
                    else:
                        with st.spinner('Menganalisis gambar...'):
                            try:
                                predicted_class, confidence, probabilities = predict_disease(model, input_for_predict)
                                st.session_state.predicted_class = predicted_class
                                st.session_state.confidence = confidence
                                st.session_state.probabilities = probabilities
                            except ValueError as e:
                                st.error(str(e))
                            except Exception as e:
                                st.error(f"Prediksi gagal: {e}")
                else:
                    st.error("Model belum tersedia. Pastikan file .h5 ada di folder app/.")
    
    with col2:
        if 'predicted_class' in st.session_state:
            st.markdown("### 🎯 Hasil Prediksi")
            
            pred_key = st.session_state.predicted_class
            # Fallback jika prediksi tidak ada di dictionary info
            disease_info = DISEASE_INFO.get(pred_key, {'color': '#333', 'name_id': pred_key, 'description': '', 'symptoms': [], 'treatment': []})
            
            st.markdown(f"""
            <div style="background-color: {disease_info['color']}22; padding: 2rem; border-radius: 10px; border-left: 5px solid {disease_info['color']};">
                <h2 style="color: {disease_info['color']}; margin: 0;">{disease_info['name_id']}</h2>
                <h3 style="color: #666; margin-top: 0.5rem;">Confidence: {st.session_state.confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### 📊 Distribusi Probabilitas")
            
            # Chart Plotly
            fig = go.Figure(data=[
                go.Bar(
                    x=list(st.session_state.probabilities.values()),
                    y=list(st.session_state.probabilities.keys()),
                    orientation='h',
                    marker=dict(
                        color=[DISEASE_INFO.get(k, {'color': '#ccc'})['color'] for k in st.session_state.probabilities.keys()],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{v:.2f}%' for v in st.session_state.probabilities.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Probabilitas Setiap Kelas',
                xaxis_title='Probabilitas (%)',
                yaxis_title='Kelas Penyakit',
                height=400,
                showlegend=False,
                xaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Tampilkan Top-3 Prediksi dan threshold
            st.markdown("### 🏆 Top-3 Prediksi")
            # Slider untuk threshold (minimal confidence untuk ditampilkan)
            threshold = st.slider("Tampilkan hanya kelas dengan confidence minimal (%)", min_value=0, max_value=100, value=0)

            # Ambil probabilities dari session state (nilai persen)
            probs = st.session_state.probabilities if 'probabilities' in st.session_state else {}
            # Urutkan berdasar confidence
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            # Filter berdasarkan threshold
            filtered = [p for p in sorted_probs if p[1] >= threshold]
            top_k = filtered[:3]

            if not top_k:
                st.info("Tidak ada kelas yang memenuhi threshold.")
            else:
                for idx, (name, val) in enumerate(top_k, start=1):
                    cols = st.columns([1, 4, 1])
                    cols[0].markdown(f"**{idx}.**")
                    cols[1].markdown(f"**{name}**")
                    # Small visual bar using progress (value 0-1)
                    cols[2].progress(int(val)/100.0)
                    st.markdown(f"Confidence: **{val:.2f}%**")
            
            st.markdown("### 📝 Informasi Detail")
            st.markdown(f"**Deskripsi:** {disease_info['description']}")
            
            st.markdown("**Gejala:**")
            for symptom in disease_info['symptoms']:
                st.markdown(f"- {symptom}")
            
            st.markdown("**Penanganan:**")
            for treatment in disease_info['treatment']:
                st.markdown(f"- {treatment}")

# --- HALAMAN: INFO ---
elif st.session_state.page == 'Info':
    st.markdown("## 📚 Informasi Lengkap Penyakit Daun Mangga")
    
    tabs = st.tabs(["Anthracnose", "Golmichi", "Healthy", "Powdery Mildew", "Turning Brown"])
    
    for idx, (disease_key, tab) in enumerate(zip(DISEASE_INFO.keys(), tabs)):
        with tab:
            disease_info = DISEASE_INFO[disease_key]
            
            st.markdown(f"### {disease_info['name_id']}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="disease-card" style="background-color: #1a1a1a; color: #e0e0e0;">
                    <h4>📖 Deskripsi</h4>
                    <p>{disease_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="disease-card" style="background-color: #1a1a1a; color: #e0e0e0;">
                    <h4>🔍 Gejala</h4>
                    <ul>
                        {''.join([f'<li>{symptom}</li>' for symptom in disease_info['symptoms']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: #2a2a2a; padding: 1.5rem; border-radius: 10px; border: 2px solid {disease_info['color']};">
                    <h4 style="color: {disease_info['color']};">💡 Penanganan</h4>
                    <ul style="color: #e0e0e0;">
                        {''.join([f'<li>{treatment}</li>' for treatment in disease_info['treatment']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# --- HALAMAN: TENTANG ---
elif st.session_state.page == 'About':
    st.markdown("## ℹ️ Tentang MangiferaNet")
    
    st.markdown("""
    ### 🎯 Tujuan
    Sistem ini bertujuan untuk membantu petani dan peneliti dalam:
    - Mendeteksi penyakit daun mangga secara dini
    - Memberikan informasi akurat tentang jenis penyakit
    - Menyediakan rekomendasi penanganan yang tepat
    - Meningkatkan produktivitas perkebunan mangga
    
    ### 🤖 Teknologi
    - **Deep Learning**: MobileNetV2
    - **Framework**: TensorFlow/Keras
    - **Interface**: Streamlit
    - **Visualization**: Plotly
    
    ### 📊 Dataset
    Dataset yang digunakan terdiri dari 5 kelas:
    1. **Anthracnose** - Penyakit jamur antraknosa
    2. **Golmichi** - Deformasi daun
    3. **Healthy** - Daun sehat
    4. **Powdery Mildew** - Embun tepung
    5. **Turning Brown** - Daun mencoklat
    """)
    
    st.markdown("### 📈 Statistik Sistem")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Kelas", "5", "")
    with col2:
        st.metric("Akurasi Model", "~91%", "")
    with col3:
        st.metric("Input Size", "224x224", "")
    with col4:
        st.metric("Framework", "TensorFlow", "")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🥭 MangiferaNet - Sistem Deteksi Penyakit Daun Mangga</p>
</div>
""", unsafe_allow_html=True)