from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import requests
import os

app = FastAPI()

# Model Path
MODEL_PATH = "gs://dummy-capstone/siamese_model.h5"  # Path to model in Cloud Storage

# Load model from Google Cloud Storage
def load_model_from_gcs():
    # Download model from Cloud Storage
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Pydantic model to define input structure
class UserData(BaseModel):
    weight: int
    height: int
    preferences: str

@app.post("/recommend/")
async def recommend(user_data: UserData):
    # Load model from GCS (this could be optimized to load only once)
    model = load_model_from_gcs()

    # Preprocess the input and get recommendation (dummy process)
    user_features = [user_data.weight, user_data.height]  # Just a dummy feature processing

    # Get recommendations (dummy prediction)
    recommendation = model.predict([user_features])  # This is where your actual model will predict

    # Return the recommendations (for simplicity, using a dummy return)
    return {"recommended_recipe": "Spaghetti Carbonara", "confidence": 0.92}

