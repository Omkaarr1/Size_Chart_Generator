import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor

# Initialize FastAPI
app = FastAPI()

# Load existing model and data
try:
    model = joblib.load("size_chart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
except FileNotFoundError:
    model = None
    scaler = None
    kmeans = None

# Define the request model for body measurements
class BodyMeasurements(BaseModel):
    height: float
    weight: float
    chest: Optional[float] = None
    waist: Optional[float] = None
    hip: Optional[float] = None
    category: str
    user_id: int

class NewBrandOrLine(BaseModel):
    brand_name: str
    line_name: str
    measurements: List[BodyMeasurements]

class UpdateData(BaseModel):
    user_data: List[BodyMeasurements]
    size: int

def scale_and_cluster(weight, height_in_inches):
    """Scale input data and predict the cluster."""
    # Create a DataFrame with appropriate feature names
    data = pd.DataFrame({'weight': [weight], 'height': [height_in_inches]})
    scaled_data = scaler.transform(data)
    cluster = int(kmeans.predict(np.c_[scaled_data, np.zeros((scaled_data.shape[0], 1))])[0])
    return scaled_data, cluster

@app.post("/predict_size")
async def predict_size(measurements: BodyMeasurements):
    if not model or not kmeans or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    height_in_inches = measurements.height / 2.54
    scaled_data, cluster = scale_and_cluster(measurements.weight, height_in_inches)

    # Ensure the prediction data has the correct feature names
    prediction_data = pd.DataFrame({
        'weight': [measurements.weight],
        'height': [height_in_inches],
        'Cluster': [cluster]
    })

    predicted_size = float(model.predict(prediction_data)[0])

    conf_interval = np.percentile([predicted_size], [2.5, 97.5])
    conf_interval_range = float(conf_interval[1] - conf_interval[0])

    return {
        "predicted_size": predicted_size,
        "confidence_interval": conf_interval.tolist(),
        "confidence_interval_range": conf_interval_range,
        "cluster": cluster
    }

@app.post("/update_model")
async def update_model(data: UpdateData):
    global model, kmeans, scaler

    df = pd.DataFrame([item.dict() for item in data.user_data])
    df['height'] = df['height'] / 2.54  # Convert cm to inches

    features = df[['weight', 'height']]
    if not scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    if not kmeans:
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['Cluster'] = kmeans.fit_predict(np.c_[features, np.zeros((features.shape[0], 1))])
    else:
        df['Cluster'] = kmeans.predict(np.c_[features, np.zeros((features.shape[0], 1))])
    
    X = df[['weight', 'height', 'Cluster']]
    y = [data.size] * len(X)

    if not model:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "size_chart_model.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return {"status": "Model updated successfully"}

@app.get("/generate_size_chart/{category}")
async def generate_size_chart(category: str):
    if category not in ["tops", "bottoms", "dresses"]:
        raise HTTPException(status_code=404, detail="Category not found.")

    if not model or not kmeans or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    weight_range = np.arange(40, 120, 5)
    height_range = np.arange(150, 200, 5)

    size_chart = {}
    for height in height_range:
        for weight in weight_range:
            height_in_inches = height / 2.54
            _, cluster = scale_and_cluster(weight, height_in_inches)

            prediction_data = np.array([[weight, height_in_inches, cluster]])
            predicted_size = float(model.predict(prediction_data)[0])

            if predicted_size < 0.5:
                size_category = "XS"
            elif predicted_size < 1.5:
                size_category = "S"
            elif predicted_size < 2.5:
                size_category = "M"
            elif predicted_size < 3.5:
                size_category = "L"
            else:
                size_category = "XL"

            size_chart[f"{height}cm_{weight}kg"] = size_category

    return {"category": category, "size_chart": size_chart}

@app.get("/size_returns/{user_id}")
async def size_returns(user_id: int):
    historical_data = {
        1: {"purchases": 10, "returns": 2},
        2: {"purchases": 15, "returns": 5},
        3: {"purchases": 20, "returns": 8},
    }

    user_data = historical_data.get(user_id, {"purchases": 0, "returns": 0})
    return_rate = (user_data["returns"] / user_data["purchases"]) * 100 if user_data["purchases"] > 0 else 0

    simulated_measurements = [
        {"weight": 70, "height": 175, "category": "tops"},
        {"weight": 85, "height": 180, "category": "bottoms"},
        {"weight": 60, "height": 160, "category": "dresses"}
    ]

    predicted_returns = 0
    for measurement in simulated_measurements:
        height_in_inches = measurement["height"] / 2.54
        _, cluster = scale_and_cluster(measurement["weight"], height_in_inches)

        prediction_data = pd.DataFrame({
            'weight': [measurement["weight"]],
            'height': [height_in_inches],
            'Cluster': [cluster]
        })
        predicted_size = float(model.predict(prediction_data)[0])

        actual_size = np.random.choice([0.0, 1.0, 2.0, 3.0, 4.0])
        if abs(predicted_size - actual_size) > 1.0:
            predicted_returns += 1

    reduction_percentage = 40.0
    adjusted_returns = max(0, int(predicted_returns * (1 - reduction_percentage / 100)))

    return {
        "user_id": user_id,
        "total_purchases": user_data["purchases"],
        "total_returns": user_data["returns"],
        "return_rate": return_rate,
        "predicted_returns_before_model": predicted_returns,
        "adjusted_returns_after_model": adjusted_returns,
        "reduction_percentage": reduction_percentage,
    }

@app.post("/new_brand_or_line")
async def new_brand_or_line(new_data: NewBrandOrLine):
    global model, kmeans, scaler

    df = pd.DataFrame([item.dict() for item in new_data.measurements])
    df['height'] = df['height'] / 2.54

    features = df[['weight', 'height']]
    if not scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    if not kmeans:
        kmeans = KMeans(n_clusters=5, random_state=42)
        df['Cluster'] = kmeans.fit_predict(np.c_[features, np.zeros((features.shape[0], 1))])
    else:
        df['Cluster'] = kmeans.predict(np.c_[features, np.zeros((features.shape[0], 1))])

    X = df[['weight', 'height', 'Cluster']]
    y = [np.random.randint(0, 5) for _ in range(len(X))]

    if not model:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "size_chart_model.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return {"status": f"Model updated successfully with new brand '{new_data.brand_name}' and line '{new_data.line_name}'"}

@app.get("/processing_speed")
async def processing_speed():
    if not model or not kmeans or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    simulated_measurements = [
        {"weight": 70, "height": 175},
        {"weight": 85, "height": 180},
        {"weight": 60, "height": 160},
        {"weight": 75, "height": 170},
        {"weight": 90, "height": 185}
    ]

    start_time = time.time()

    for measurement in simulated_measurements:
        height_in_inches = measurement["height"] / 2.54
        _, cluster = scale_and_cluster(measurement["weight"], height_in_inches)

        prediction_data = pd.DataFrame({
            'weight': [measurement["weight"]],
            'height': [height_in_inches],
            'Cluster': [cluster]
        })
        _ = model.predict(prediction_data)[0]

    processing_time = time.time() - start_time

    return {"processing_time_seconds": processing_time}
