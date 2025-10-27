from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# ===== GLOBAL CONSTANTS =====
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
    """Handle prediction request and display results."""
    try:
        # Extract features from form
        features = {
            'fixed_acidity': float(request.form['fixed_acidity']),
            'volatile_acidity': float(request.form['volatile_acidity']),
            'citric_acid': float(request.form['citric_acid']),
            'residual_sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free_sulfur_dioxide': float(request.form['free_sulfur_dioxide']),
            'total_sulfur_dioxide': float(request.form['total_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['pH']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol']),
        }
        
        # Get prediction from API
        api_url = os.getenv('PREDICTION_API_URL', 'http://prediction-api:8080/predict')
        response = requests.post(api_url, json=features, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        quality = result.get('quality', 0)
        
        # Get rating information
        rating_info = get_quality_rating(quality)
        
        return render_template('result.html', 
                             quality=quality,
                             features=features,
                             rating=rating_info,
                             min_quality=MIN_QUALITY_SCORE,
                             max_quality=MAX_QUALITY_SCORE)
    
    except requests.exceptions.RequestException as e:
        error_message = f"Error connecting to prediction API: {str(e)}"
        return render_template('result.html', error=error_message)
    
    except ValueError as e:
        error_message = f"Invalid input values: {str(e)}"
        return render_template('result.html', error=error_message)
    
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        return render_template('result.html', error=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)