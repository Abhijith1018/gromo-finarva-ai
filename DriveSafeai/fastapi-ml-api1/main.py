from fastapi import FastAPI, HTTPException
from typing import List
from models import TripData
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Load model with error handling
try:
    with open("ml_models/drive_score_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    # Log the error and raise a more informative exception
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load ML model. Please ensure the model file is properly saved.")

@app.post("/predict-drive-score")
def predict_drive_score(data: TripData):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)
    return {"drive_score": round(float(prediction[0]), 2)}

@app.post("/predict-drive-score-batch")
def predict_drive_score_batch(data: List[TripData]):
    df = pd.DataFrame([d.model_dump() for d in data])
    prediction = model.predict(df)
    avg_score = float(np.mean(prediction))
    return {"drive_score": round(avg_score, 2)}