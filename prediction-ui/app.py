from flask import Flask, render_template, request
import requests
import os
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token

app = Flask(__name__)

# These define the expected range of wine quality scores
MIN_QUALITY_SCORE = 3
MAX_QUALITY_SCORE = 8

# Star rating thresholds
QUALITY_THRESHOLDS = {
    'poor': (3, 4),      # 0 stars
    'average': (5, 6),   # 1 star
    'good': (7, 7),      # 2 stars
    'excellent': (8, 10) # 3 stars
}

# Color gradient mapping for quality visualization
QUALITY_COLORS = {
    3: '#8B0000',  # Dark red
    4: '#A52A2A',  # Brown
    5: '#CD853F',  # Peru
    6: '#DAA520',  # Goldenrod
    7: '#9370DB',  # Medium purple
    8: '#8B4789',  # Deep purple
}

# Get the predictor API URL from environment
PREDICTOR_API_URL = os.environ.get("PREDICTOR_API_URL")

def get_identity_token(audience):
    """
    Fetches a Google-signed identity token for the given audience.
    This is used for authenticating service-to-service calls.
    """
    try:
        creds, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        
        print(f"Fetching identity token for audience: {audience}")
        token = id_token.fetch_id_token(auth_req, audience)
        print("Successfully fetched identity token.")
        return token
    except Exception as e:
        print(f"CRITICAL: Failed to fetch identity token. Error: {e}")
        raise

def get_quality_rating(quality_score):
    """
    Determines the star rating and descriptive label for a quality score.
    
    Args:
        quality_score: Integer quality score (typically 3-8)
    
    Returns:
        dict: Contains 'stars', 'label', 'color', and 'percentage'
    """
    quality_score = int(quality_score)
    
    if QUALITY_THRESHOLDS['poor'][0] <= quality_score <= QUALITY_THRESHOLDS['poor'][1]:
        stars = 0
        label = 'Poor Quality'
        base_color = '#8B0000'
    elif QUALITY_THRESHOLDS['average'][0] <= quality_score <= QUALITY_THRESHOLDS['average'][1]:
        stars = 1
        label = 'Average Quality'
        base_color = '#CD853F'
    elif QUALITY_THRESHOLDS['good'][0] <= quality_score <= QUALITY_THRESHOLDS['good'][1]:
        stars = 2
        label = 'Good Quality'
        base_color = '#9370DB'
    else:  # excellent
        stars = 3
        label = 'Excellent Quality'
        base_color = '#8B4789'
    
    # Calculate percentage within the range for gradient visualization
    percentage = ((quality_score - MIN_QUALITY_SCORE) / (MAX_QUALITY_SCORE - MIN_QUALITY_SCORE)) * 100
    percentage = max(0, min(100, percentage))  # Clamp between 0-100
    
    # Get the specific color for this quality score
    color = QUALITY_COLORS.get(quality_score, base_color)
    
    return {
        'stars': stars,
        'label': label,
        'color': color,
        'percentage': round(percentage, 1)
    }

@app.route('/')
def index():
    """Render the main input form with quality range info."""
    return render_template('index.html', 
                         min_quality=MIN_QUALITY_SCORE,
                         max_quality=MAX_QUALITY_SCORE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, calls the prediction API, and renders the result."""
    if not PREDICTOR_API_URL:
        return render_template('result.html', error="Predictor API URL not configured on the server.")

    try:
        # Extract features from form
        form_data = request.form.to_dict()
        features = {key: float(value) for key, value in form_data.items()}
        
        api_endpoint = f"{PREDICTOR_API_URL}/predict"
        
        # Get an identity token for the private API service
        auth_token = get_identity_token(audience=PREDICTOR_API_URL)
        
        # Add the token to the authorization header
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

        print(f"Sending authenticated request to {api_endpoint} with payload: {features}")
        response = requests.post(api_endpoint, headers=headers, json=features, timeout=10)
        
        # This will raise an exception for 4xx or 5xx status codes
        response.raise_for_status() 
        
        data = response.json()
        
        # Check if there's an error in the response
        if 'error' in data:
            return render_template('result.html', error=data['error'])
        
        # Get the prediction (handle both 'prediction' and 'quality' keys for compatibility)
        quality = data.get('prediction') or data.get('quality')
        
        if quality is None:
            return render_template('result.html', error="Invalid response from prediction API.")
        
        # Get rating information
        rating_info = get_quality_rating(quality)
        
        return render_template('result.html', 
                             quality=quality,
                             features=features,
                             rating=rating_info,
                             min_quality=MIN_QUALITY_SCORE,
                             max_quality=MAX_QUALITY_SCORE)

    except requests.exceptions.RequestException as e:
        # The response.raise_for_status() will likely trigger this for 403 errors
        return render_template('result.html', error=f"Could not connect to the prediction API: {e}")
    except ValueError as e:
        return render_template('result.html', error=f"Invalid input. Please ensure all fields are numbers: {e}")
    except Exception as e:
        return render_template('result.html', error=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))