import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask, request, jsonify
import pytz

# Firebase URL
FIREBASE_URL = "https://iotsecurity-30d1a-default-rtdb.firebaseio.com"

# Initialize Flask app
app = Flask(__name__)


# Function to fetch data from Firebase
def fetch_firebase_data():
    lanes = ["lane_1", "lane_2"]
    all_data = []

    for lane in lanes:
        url = f"{FIREBASE_URL}/sensors/{lane}.json"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data:
                    # Convert nested Firebase data to list of records
                    for timestamp, values in data.items():
                        record = {
                            "lane": lane,
                            "vehicle_count": values["vehicle_count"],
                            "timestamp": values["timestamp"],
                        }
                        all_data.append(record)
        except Exception as e:
            print(f"Error fetching data for {lane}: {e}")

    return pd.DataFrame(all_data)


# Function to preprocess data and train model
def prepare_and_train_model():
    # Fetch data
    df = fetch_firebase_data()

    if df.empty:
        print("No data available to train model")
        return None

    # Convert timestamp to datetime features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Encode lane as numerical
    le = LabelEncoder()
    df["lane_encoded"] = le.fit_transform(df["lane"])

    # Features and target
    X = df[["hour", "day_of_week", "lane_encoded"]]
    y = df["vehicle_count"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model R² Score - Training: {train_score:.4f}, Testing: {test_score:.4f}")

    # Save model and label encoder
    with open("vehicle_predictor.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("lane_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return model, le


# Load model and encoder (for API use)
def load_model_and_encoder():
    try:
        with open("vehicle_predictor.pkl", "rb") as f:
            model = pickle.load(f)
        with open("lane_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return model, le
    except FileNotFoundError:
        return None, None


# Initial model training
model, lane_encoder = prepare_and_train_model()


# API endpoint to predict vehicle count
@app.route("/predict", methods=["POST"])
def predict_vehicle_count():
    global model, lane_encoder

    if model is None or lane_encoder is None:
        model, lane_encoder = load_model_and_encoder()
        if model is None:
            return jsonify({"error": "Model not trained yet"}), 500

    # Get request data
    data = request.get_json()

    try:
        # Extract and validate input
        lane = data["lane"]  # e.g., "lane_1" or "lane_2"
        timestamp = data.get(
            "timestamp",
            datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Convert timestamp to features
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        day_of_week = dt.dayofweek

        # Encode lane
        lane_encoded = lane_encoder.transform([lane])[0]

        # Prepare input for model
        input_data = np.array([[hour, day_of_week, lane_encoded]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify(
            {
                "lane": lane,
                "timestamp": timestamp,
                "predicted_vehicle_count": round(prediction),
            }
        ), 200

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API endpoint to retrain model
@app.route("/retrain", methods=["POST"])
def retrain_model():
    global model, lane_encoder
    model, lane_encoder = prepare_and_train_model()

    if model is None:
        return jsonify({"message": "Failed to retrain model - no data available"}), 500
    return jsonify({"message": "Model retrained successfully"}), 200


if __name__ == "__main__":
    # Run the Flask app (suitable for cloud deployment)
    app.run(host="0.0.0.0", port=5000, debug=False)
