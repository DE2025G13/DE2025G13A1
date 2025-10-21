from flask import Flask, request, render_template, redirect, url_for
import requests
import os

app = Flask(__name__)
PREDICTOR_API_URL = os.environ.get("PREDICTOR_API_URL")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file and PREDICTOR_API_URL:
        files = {'file': (file.filename, file.stream, file.mimetype)}
        try:
            response = requests.post(f"{PREDICTOR_API_URL}/predict", files=files, timeout=20)
            response.raise_for_status() 
            
            data = response.json()
            if 'error' in data:
                 return render_template('result.html', error=data['error'])
            
            return render_template('result.html', 
                                   prediction=data.get('prediction', 'N/A'), 
                                   confidence=f"{data.get('confidence', 0):.2f}",
                                   image_with_box=data.get('image_with_box'))
        except requests.exceptions.RequestException as e:
            return f"Could not connect to or get a valid response from the prediction API: {e}", 503
        except Exception as e:
            return f"An unexpected error occurred: {e}", 500
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))