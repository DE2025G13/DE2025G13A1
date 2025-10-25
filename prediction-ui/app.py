from flask import Flask, request, render_template
import requests
import os

app = Flask(__name__)

PREDICTOR_API_URL = os.environ.get("PREDICTOR_API_URL")

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
        
        print(f"Sending request to {api_endpoint} with payload: {json_payload}")
        response = requests.post(api_endpoint, json=json_payload, timeout=10)
        
        response.raise_for_status()
        
        data = response.json()
        if 'error' in data:
             return render_template('result.html', error=data['error'])
        
        return render_template('result.html', prediction=data.get('prediction'))

    except requests.exceptions.RequestException as e:
        return render_template('result.html', error=f"Could not connect to the prediction API: {e}")
    except ValueError:
        return render_template('result.html', error="Invalid input. Please ensure all fields are numbers.")
    except Exception as e:
        return render_template('result.html', error=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))