# 🥭 MangiferaNet: AI-Powered Mango Disease Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

**MangiferaNet** adalah sistem deteksi dini penyakit pada tanaman mangga (*Mangifera indica*) yang dikembangkan menggunakan teknologi *Deep Learning* dan *Computer Vision*. Proyek ini dirancang untuk membantu petani dan agronomis dalam mengidentifikasi patogen tanaman secara cepat, akurat, dan *real-time* melalui citra digital.

Sistem ini dikembangkan sebagai bagian dari proyek akhir **Deep Learning Course**, dengan fokus pada implementasi arsitektur *Transfer Learning* yang efisien untuk *deployment* pada perangkat *edge* atau web.

---

## 📋 Table of Contents
- [Latar Belakang](#-latar-belakang)
- [Dataset & Model](#-dataset--model)
- [Arsitektur Proyek](#-arsitektur-proyek)
- [Instalasi & Setup](#-instalasi--setup)
- [Panduan Penggunaan](#-panduan-penggunaan)
- [Tim Pengembang](#-tim-pengembang)
- [Lisensi](#-lisensi--citation)

---

## 🎯 Latar Belakang

Penyakit pada daun mangga seringkali terlambat dideteksi secara visual, menyebabkan penurunan hasil panen yang signifikan. MangiferaNet hadir sebagai solusi cerdas yang mampu mengklasifikasikan 5 kondisi kesehatan daun dengan presisi tinggi:

1.  **Anthracnose** (Antraknosa - *Colletotrichum gloeosporioides*)
2.  **Golmichi** (Gall Midge)
3.  **Powdery Mildew** (Embun Tepung - *Oidium mangiferae*)
4.  **Turning Brown** (Nekrosis/Stress Nutrisi)
5.  **Healthy** (Tanaman Sehat)

---

## 🧠 Dataset & Model

### Dataset

Kami menggunakan **SAR-MLD1-2025 (High Quality Mango Leaf Dataset)** yang telah melalui proses *preprocessing* ketat:
- Resize: 224×224 piksel
- Normalisasi: Min-Max scaling (0-1)
- Augmentasi Data: Rotasi, Flip, Brightness adjustment

### Performance Benchmark

Dalam pengembangan ini, kami melakukan komparasi eksperimental antara dua arsitektur *State-of-the-Art*:

| Model Architecture | Test Accuracy | Model Size | Inference Speed | Deployment Status |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet50V2** | ~91.9% | ~101 MB | Medium | Research Benchmark |
| **MobileNetV2** | ~90.3% | ~11 MB | **Very Fast** | **Production (Deployed)** ✅ |

**Keputusan:** **MobileNetV2** dipilih untuk versi produksi karena menawarkan *trade-off* terbaik antara akurasi dan latensi, memungkinkan aplikasi berjalan mulus di lingkungan web (Streamlit).

---

## 🛠️ Arsitektur Proyek

Struktur repositori ini disusun secara modular mengikuti standar *Data Science Cookiecutter* untuk memudahkan *maintenance* dan kolaborasi:

```text
MangiferaNet/
│
├── app/                        # Production Code (Streamlit)
│   ├── app.py                  # Main Application Entry Point
│   └── best_mobnet.h5          # Serialized Model (Ready for Inference)
│   └── requirements.txt        # Project Dependencies
│
├── data/                       # Data Management (Local only)
│   ├── raw/                    # Immutable Original Data
│   └── README.md               # Project Documentation
│
├── notebooks/                  # Research & Experiments
│   ├── mangiferanet-eda.ipynb       # Data Distribution & Visual Analysis
│   └── mangiferanet-training.ipynb  # Model Training, Tuning & Evaluation
│
├── outputs/                     # Outputs
│   ├── models                   # .h5 Model
│   └── plots                    # Plots Comparative
|
├── report/                       # Report
│   ├── mangiferanet-proposal.pdf # Project Proposal
│   ├── mangiferanet-report.pdf   # Project Final Report
│   └── README.md               # Project Documentation
|
├── src/                        # Modular Source Code
│   ├── __init__.py
│   ├── config.py               # Global Configuration & Constants
│   └── preprocess.py           # Unified Preprocessing Pipeline
│
├── requirements.txt            # Project Dependencies
└── README.md                   # Project Documentation
```

---

## ⚙️ Instalasi & Setup

Ikuti langkah-langkah berikut untuk menjalankan sistem di lingkungan lokal Anda:

### 1. Clone Repository

```bash
git clone https://github.com/SergioWinn/MangiferaNet.git
cd MangiferaNet
```

### 2. Setup Virtual Environment (Recommended)

Isolasi dependensi proyek menggunakan venv:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Pastikan Anda berada di root directory (`MangiferaNet/`), lalu jalankan:

```bash
pip install -r requirements.txt
```

### 4. Setup Model File

Pastikan file model pre-trained (`best_mobnet.h5`) telah ditempatkan di dalam direktori `app/`.

> **Catatan:** Jika Anda mengunduh dari rilis GitHub/Kaggle, pindahkan file tersebut secara manual ke folder `app/`.

---

## 📖 Panduan Penggunaan

Setelah instalasi selesai, Anda dapat menjalankan aplikasi web interaktif MangiferaNet.

### 1. Menjalankan Server

Jalankan perintah berikut dari root directory proyek:

```bash
streamlit run app/app.py
```

Aplikasi akan otomatis terbuka di browser default Anda (biasanya `http://localhost:8501`).

### 2. Menggunakan Aplikasi

#### Navigasi
Gunakan Navigation Bar untuk berpindah antara menu:
- **🏠 Home** - Informasi umum dan fitur sistem
- **🔍 Detect** - Upload gambar dan deteksi penyakit
- **📊 Info** - Informasi lengkap setiap penyakit
- **ℹ️ About** - Tentang sistem dan statistik model

#### Upload Gambar
1. Masuk ke menu **Detect**
2. Klik tombol "Browse files"
3. Pilih citra daun mangga (format JPG/JPEG/PNG)
4. Metadata file akan ditampilkan

#### Analisis & Hasil
1. Klik tombol **🔍 Deteksi Penyakit**
2. Sistem akan memproses citra dalam hitungan detik
3. Hasil yang ditampilkan:
   - Prediksi utama dengan confidence score
   - Top-3 prediksi dengan threshold slider
   - Grafik distribusi probabilitas
   - Informasi lengkap dan rekomendasi penanganan

---

## 👥 Tim Pengembang

Proyek ini dikembangkan dengan dedikasi oleh tim yang berpengalaman di bidang Data Science dan AI:

| Nama | NIM | Role | Fokus Area |
| :--- | :---: | :--- | :--- |
| **Sergio Winnero** | 2702240166 | Lead Developer & AI Engineer | Data Preprocessing, Model Training, Arsitektur Model, Deployment Streamlit, Integrasi Sistem, Model Deployment, CI/CD Pipeline |
| **Samuel Setiawan** | 2702258024 | Data Scientist & Research Lead | Data Preprocessing, EDA, Augmentation Strategy, Production Monitoring |
| **Karina Vanya Wardoyo** | 2702350024 | Full Stack Engineer & UI/UX Designer | Frontend Development, Web Interface, User Experience Design |

---

## 📜 Lisensi & Citation

**License:** MIT License

Jika Anda menggunakan repositori ini untuk penelitian atau referensi, harap cantumkan kredit ke repositori ini.

---

**Developed with ❤️ for the advancement of Smart Agriculture.**