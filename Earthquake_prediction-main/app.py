from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from geopy import distance
from geopy.geocoders import Nominatim
import time

app = Flask(__name__)

# Load the data
data = pd.read_excel('earthquake_predictions.xlsx', engine='openpyxl')

# Initialize geocoder
geolocator = Nominatim(user_agent="earthquake_predictor")

def find_nearest_location(lat, lon):
    locations = data[['Latitude', 'Longitude']].values
    query_location = np.array([[lat, lon]])
    distances = cdist(query_location, locations)
    nearest_index = distances.argmin()
    return data.iloc[nearest_index]

def get_city_name(lat, lon):
    max_retries = 3
    for _ in range(max_retries):
        try:
            location = geolocator.reverse(f"{lat}, {lon}", language='en')
            if location:
                address = location.raw['address']
                city = address.get('city', address.get('town', address.get('village', address.get('county', 'Unknown'))))
                return city
            else:
                return "Unknown"
        except Exception as e:
            print(f"Error in geocoding: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
    return "Unknown"

def get_prediction(lat, lon):
    nearest = find_nearest_location(lat, lon)
    
    # Calculate distance
    user_location = (lat, lon)
    epicenter = (nearest['Latitude'], nearest['Longitude'])
    distance_km = distance.distance(user_location, epicenter).km
    
    # Get city names
    user_city = get_city_name(lat, lon)
    epicenter_city = get_city_name(nearest['Latitude'], nearest['Longitude'])
    
    result = {
        'predicted_magnitude': round(nearest['Predicted_Magnitude'], 2),
        'epicenter': {
            'latitude': round(nearest['Latitude'], 4),
            'longitude': round(nearest['Longitude'], 4),
            'city': epicenter_city
        },
        'user_location': {
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'city': user_city
        },
        'distance': round(distance_km, 2),
        'depth': round(nearest['Depth'], 2)
    }
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    lat = float(data['latitude'])
    lon = float(data['longitude'])
    result = get_prediction(lat, lon)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)