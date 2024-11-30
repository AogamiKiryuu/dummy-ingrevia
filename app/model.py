import tensorflow as tf
import os
from google.cloud import storage
from tensorflow.keras.models import load_model
import numpy as np

# Setup Google Cloud Storage Client
client = storage.Client()
bucket_name = "dummy-capstone"  # Ganti dengan nama bucket Google Cloud Storage Anda
model_file_name = "siamese_model.h5"  # Ganti dengan nama model Anda

# Fungsi untuk mendownload model dari Google Cloud Storage
def download_model():
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(model_file_name)
    blob.download_to_filename(model_file_name)
    print(f"Model {model_file_name} downloaded from GCS.")

# Fungsi untuk memuat model
def load_ml_model():
    if not os.path.exists(model_file_name):
        download_model()
    model = load_model(model_file_name)
    print("Model loaded successfully.")
    return model

# Fungsi untuk melakukan prediksi dengan model
def get_recommendations(user_input):
    # Load model terlebih dahulu
    model = load_ml_model()

    # Proses input dan lakukan prediksi berdasarkan model
    # Misalnya, kita anggap input user berupa array fitur untuk rekomendasi
    # Perlu disesuaikan dengan input yang dibutuhkan model Anda
    input_features = np.array([user_input])

    # Melakukan prediksi
    recommendations = model.predict(input_features)

    # Misalnya output berupa rekomendasi berupa ID resep atau nilai lainnya
    return recommendations.tolist()  # Mengembalikan hasil sebagai list
