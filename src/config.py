import os

# Konfigurasi Model & Data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'app', 'best_mobnet.h5') # Sesuaikan nama file model

# Parameter Gambar
IMAGE_SIZE = (224, 224) 

# Daftar Kelas
CLASS_NAMES = [
    'Anthracnose', 
    'Golmichi', 
    'Healthy', 
    'Powdery Mildew', 
    'Turning Brown'
]

# Info Penyakit (Dictionary)
DISEASE_INFO = {
    'Anthracnose': 'Penyakit jamur yang menyebabkan bintik-bintik gelap/hitam melekuk pada daun dan buah.',
    'Golmichi': 'Disebabkan oleh hama Gall Midge, menimbulkan bintil-bintil menonjol pada permukaan daun.',
    'Healthy': 'Tanaman dalam kondisi prima. Pertahankan pemupukan dan penyiraman yang baik.',
    'Powdery Mildew': 'Jamur yang tampak seperti serbuk putih tepung melapisi permukaan daun.',
    'Turning Brown': 'Daun mengalami pencoklatan/nekrosis, bisa karena faktor usia, terbakar matahari, atau defisiensi nutrisi.'
}