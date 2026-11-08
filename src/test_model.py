import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/accident_risk_model.pkl")

# Create a sample accident scenario
sample = pd.DataFrame([{
    "city": "Chennai",
    "state": "Tamil Nadu",
    "latitude": 13.08,
    "longitude": 80.27,
    "hour": 18,
    "day_of_week": "Friday",
    "is_weekend": 0,
    "road_type": "urban",
    "lanes": 4,
    "traffic_signal": 1,
    "weather": "rain",
    "visibility": "low",
    "temperature": 28,
    "traffic_density": "high",
    "cause": "overspeeding",
    "is_peak_hour": 1,
    "year": 2025,
    "month": 8,
    "day": 15
}])

# Predict risk
prediction = model.predict(sample)

print("Predicted Accident Risk:", prediction[0])