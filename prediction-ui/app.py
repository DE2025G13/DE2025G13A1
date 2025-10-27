from flask import Flask, render_template, request
import requests
import os
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token
import traceback

app = Flask(__name__)
PREDICTOR_API_URL = os.environ.get("PREDICTOR_API_URL")
REQUIRED_FEATURES = [
    "type",
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]
MIN_QUALITY_SCORE = 3
MAX_QUALITY_SCORE = 9

def get_identity_token(audience):
    try:
        print(f"Requesting identity token for audience: {audience}")
        creds, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        token = id_token.fetch_id_token(auth_req, audience)
        print("Successfully acquired identity token.")
        return token
    except Exception as e:
        print(f"Fatal Error: Failed to acquire identity token for service-to-service authentication.")
        raise

def get_quality_rating(quality_score):
    quality_score = int(quality_score)
    if quality_score <= 4: stars, label = 0, "Poor Quality"
    elif quality_score <= 6: stars, label = 1, "Average Quality"
    elif quality_score == 7: stars, label = 2, "Good Quality"
    else: stars, label = 3, "Excellent Quality"
    percentage = ((quality_score - MIN_QUALITY_SCORE) / (MAX_QUALITY_SCORE - MIN_QUALITY_SCORE)) * 100
    return {"stars": stars, "label": label, "percentage": max(0, min(100, percentage))}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not PREDICTOR_API_URL:
        return render_template("result.html", error="Server configuration error: The prediction API URL is not set.")
    try:
        features = {}
        for feature_name in REQUIRED_FEATURES:
            value = request.form.get(feature_name)
            if value is None:
                return render_template("result.html", error=f"Missing required field: {feature_name}")
            if feature_name == "type":
                features[feature_name] = value
            else:
                try:
                    features[feature_name] = float(value)
                except ValueError:
                    return render_template("result.html", error=f"Invalid value for {feature_name}. Must be a number.")
        api_endpoint = f"{PREDICTOR_API_URL}/predict"
        auth_token = get_identity_token(audience=PREDICTOR_API_URL)
        headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
        print(f"Sending prediction request to the API at {api_endpoint}.")
        response = requests.post(api_endpoint, headers=headers, json=features, timeout=60)
        print(f"API responded with status code: {response.status_code}.")
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            return render_template("result.html", error=data["error"])
        quality = data.get("prediction")
        if quality is None:
            return render_template("result.html", error=f"API gave an unexpected response: {data}")
        rating_info = get_quality_rating(quality)
        # We need to pass the min/max quality to the results page as well.
        return render_template("result.html", quality=quality, features=features, rating=rating_info,
                             min_quality=MIN_QUALITY_SCORE, max_quality=MAX_QUALITY_SCORE)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)
        traceback.print_exc()
        return render_template("result.html", error=error_msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
