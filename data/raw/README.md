# 🍃 Dataset Penyakit Daun Mangga (Raw Data)

**⚠️ PENTING:**
Karena ukuran dataset yang besar, file gambar **TIDAK DISIMPAN** di dalam repository GitHub ini untuk menjaga repositori tetap ringan.

Dataset yang digunakan dalam proyek ini bersumber dari repositori publik Mendeley Data.

## 📥 Sumber & Unduhan Dataset

Silakan unduh dataset asli melalui tautan resmi berikut:

1.  **SAR-MLD1-2025: A High Quality Mango Leaf Dataset for Disease Classification (part1):**
    👉 [Mendeley Data - Part 1](https://data.mendeley.com/datasets/sd8hzpg69b/4)
    
2.  **SAR-MLD1-2025: A High Quality Mango Leaf Dataset for Disease Classification (part2):**
    👉 [Mendeley Data - Part 2](https://data.mendeley.com/datasets/j3bn63t4sp/4)

---

## ⚙️ Cara Setup Folder

1.  **Download:** Unduh file `.zip` dari kedua link di atas.
2.  **Ekstrak & Gabungkan:**
    * Ekstrak file tersebut.
    * Pilih gambar-gambar yang relevan sesuai dengan 5 kelas yang kita gunakan:
        * Anthracnose
        * Golmichi
        * Healthy
        * Powdery Mildew
        * Turning Brown
3.  **Susun Folder:**
    Letakkan hasil kurasi/gabungan gambar tersebut di folder ini (`data/raw/`) dengan struktur berikut:

    ```text
    MangiferaNet/
    ├── data/
    │   └── raw/
    │       ├── Anthracnose/    # Masukkan semua gambar Antraknosa di sini
    │       ├── Golmichi/       # Masukkan semua gambar Golmichi di sini
    │       ├── Healthy/        # ... dst
    │       ├── Powdery Mildew/
    │       └── Turning Brown/
    ```

---

## 📊 Informasi Dataset

* **Total Gambar:** ± 5.000 Citra (setelah penggabungan)
* **Kelas:** 5 Kategori
* **Format:** JPG
* **Resolusi:** High Quality

### Deskripsi Kelas
1.  **Anthracnose:** Bercak coklat/hitam yang melekuk pada daun.
2.  **Golmichi:** Bintil-bintil (gall) menonjol pada permukaan daun.
3.  **Healthy:** Daun sehat, hijau segar tanpa bercak.
4.  **Powdery Mildew:** Lapisan serbuk putih seperti tepung pada daun.
5.  **Turning Brown:** Daun mencoklat/kering (nekrosis).

---

## ⚖️ Lisensi & Kredit
Dataset ini bersumber dari Mendeley Data dan digunakan untuk tujuan akademik (Final Project Deep Learning). Hak cipta tetap pada kontributor asli dataset tersebut.