from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load models
drive_model = joblib.load("drive_score_model.pkl")
drisc_model = joblib.load("drisc_score_model.pkl")  # optional, if using a different model for DRISC score

# Define input schema
class TripData(BaseModel):
    speed: float
    rpm: float
    acceleration: float
    throttle_position: float
    engine_temperature: float
    system_voltage: float
    engine_load_value: float
    distance_travelled: float
    brake: float

#DRIVE SCORE Endpoint
@app.post("/predict/drive-score")
def predict_drive_score(trip: TripData):
    features = np.array([[trip.speed, trip.rpm, trip.acceleration,
                          trip.throttle_position, trip.engine_temperature,
                          trip.system_voltage, trip.engine_load_value,
                          trip.distance_travelled, trip.brake]])
    score = drive_model.predict(features)[0]
    return {"score": round(float(score), 2)}

#DRISC SCORE Endpoint
@app.post("/predict/drisc-score")
def predict_drisc_score(trips: list[TripData]):
    df = pd.DataFrame([t.dict() for t in trips])
    anomaly = drisc_model.decision_function(df)
    risk_score = (1 - anomaly.mean()) * 100
    return {"score": round(float(risk_score), 2)}
