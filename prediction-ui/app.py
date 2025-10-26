from flask import Flask, request, render_template
import requests
import os
import google.auth
import google.auth.transport.requests

app = Flask(__name__)

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
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, audience)
        print("Successfully fetched identity token.")
        return id_token
    except Exception as e:
        print(f"CRITICAL: Failed to fetch identity token. Error: {e}")
        raise

@app.route('/')
def index():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, calls the prediction API, and renders the result."""
    if not PREDICTOR_API_URL:
        return render_template('result.html', error="Predictor API URL not configured on the server.")

    try:
        form_data = request.form.to_dict()
        json_payload = {key: float(value) for key, value in form_data.items()}

        api_endpoint = f"{PREDICTOR_API_URL}/predict"
        
        # Get an identity token for the private API service.
        id_token = get_identity_token(audience=PREDICTOR_API_URL)
        
        # Add the token to the authorization header.
        headers = {
            "Authorization": f"Bearer {id_token}",
            "Content-Type": "application/json"
        }

        print(f"Sending authenticated request to {api_endpoint} with payload: {json_payload}")
        response = requests.post(api_endpoint, headers=headers, json=json_payload, timeout=10)
        
        # This will raise an exception for 4xx or 5xx status codes.
        response.raise_for_status() 
        
        data = response.json()
        if 'error' in data:
             return render_template('result.html', error=data['error'])
        
        return render_template('result.html', prediction=data.get('prediction'))

    except requests.exceptions.RequestException as e:
        # The response.raise_for_status() will likely trigger this for 403 errors.
        return render_template('result.html', error=f"Could not connect to the prediction API: {e}")
    except ValueError:
        return render_template('result.html', error="Invalid input. Please ensure all fields are numbers.")
    except Exception as e:
        return render_template('result.html', error=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))