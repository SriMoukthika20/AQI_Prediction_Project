import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

MODEL_PATH = "model/aqi_model.pkl"

# Load model if available, otherwise train automatically
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")

else:
    print("Model not found. Training model...")

    # Load dataset
    df = pd.read_csv("data/city_day.csv")

    # Keep required columns
    columns = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "AQI"
    ]

    df = df[columns]

    # Remove missing values
    df = df.dropna()

    # Features and target
    X = df.drop("AQI", axis=1)
    y = df["AQI"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved!")


@app.route("/")
def home():
    return "AQI Prediction API is Running!"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([[
        data["PM2.5"],
        data["PM10"],
        data["NO2"],
        data["SO2"],
        data["CO"],
        data["O3"]
    ]])

    prediction = model.predict(features)[0]

    return jsonify({
        "Predicted AQI": round(float(prediction), 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)