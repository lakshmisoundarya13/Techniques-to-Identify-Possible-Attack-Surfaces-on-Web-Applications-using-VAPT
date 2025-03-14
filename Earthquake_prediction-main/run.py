import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load the data
data = pd.read_csv('earthquake_1995-2023.csv')

# Preprocess the data
data['date_time'] = pd.to_datetime(data['date_time'])
data['year'] = data['date_time'].dt.year
data['month'] = data['date_time'].dt.month
data['day'] = data['date_time'].dt.day

# Select features for the model
features = ['year', 'month', 'day', 'latitude', 'longitude', 'depth']
target = 'magnitude'

X = data[features]
y = data[target]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a Random Forest Regressor on the entire dataset
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Make predictions for all entries
y_pred = rf_model.predict(X_scaled)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Create a DataFrame with all the information
results = pd.DataFrame({
    'Date': data['date_time'],
    'Year': data['year'],
    'Month': data['month'],
    'Day': data['day'],
    'Latitude': data['latitude'],
    'Longitude': data['longitude'],
    'Depth': data['depth'],
    'Actual_Magnitude': y,
    'Predicted_Magnitude': y_pred
})

# Save predictions to Excel file
results.to_excel('earthquake_predictions.xlsx', index=False)
print("Predictions saved to earthquake_predictions.xlsx")

# Save model and scaler as pickle files
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved as pickle files")
print(f"Total number of entries processed: {len(data)}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)