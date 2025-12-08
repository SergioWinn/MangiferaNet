# 📓 Research & Experiment Notebooks

Folder ini berisi *Jupyter Notebooks* yang digunakan untuk eksplorasi data, eksperimen model, dan pelatihan (*training*).

Untuk performa terbaik (terutama saat Training), disarankan menjalankan notebook ini menggunakan **Kaggle** atau **Google Colab** untuk memanfaatkan akses **GPU Gratis**.

---

## 🚀 Daftar Notebook

| Notebook | Deskripsi | Link Eksekusi |
| :--- | :--- | :---: |
| **1. Exploratory Data Analysis** | Analisis distribusi data, visualisasi sampel citra, dan pengecekan keseimbangan kelas dataset. | [![Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/sergiowinn/mangiferanet-eda) |
| **2. Training & Evaluation** | Proses *preprocessing*, *data augmentation*, pelatihan model (MobileNetV2 vs ResNet50), dan evaluasi performa. | [![Kaggle](https://img.shields.io/badge/Kaggle-Open%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/sergiowinn/mangiferanet-training) |

---

## 🛠️ Cara Menjalankan di Kaggle

1.  Klik tombol **"Open Notebook"** pada tabel di atas.
2.  Jika dataset belum terhubung, klik **"Add Input"** di sebelah kanan dan cari dataset **"A High Quality Mango Leaf Dataset"** (atau upload ulang dataset raw).
3.  Untuk notebook **Training**, pastikan mengaktifkan **GPU Accelerator** (GPU T4 x2) pada menu *Session Options*.
4.  Klik **Run All**.