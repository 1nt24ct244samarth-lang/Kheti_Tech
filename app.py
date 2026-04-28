from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoders
with open('kheti_tech_models.pkl', 'rb') as f:
    model_data = pickle.load(f) # Correct

crop_model = model_data['crop_model']
irr_model = model_data['irr_model']
encoders = model_data['encoders']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Prepare data for model
    input_df = pd.DataFrame([{
        'Soil_Moisture': float(data['moisture']),
        'Rainfall_mm': float(data['rainfall']),
        'Temperature_C': float(data['temp']),
        'Wind_Speed_kmh': float(data['wind']),
        'Previous_Irrigation_mm': float(data['prevIrrigation']),
        'Humidity': float(data['humidity']),
        'Soil_Type': data['soilType'],
        'Sunlight_Hours': float(data['sunlight']),
        'Season': data['season'],
        'Region': data['region'],
        'Soil_pH': float(data['ph'])
    }])

    # Encode categorical fields
    for col in ['Soil_Type', 'Season', 'Region']:
        input_df[col] = encoders[col].transform(input_df[col])

    # Predictions
    predicted_crop = crop_model.predict(input_df)[0]
    predicted_irrigation = irr_model.predict(input_df)[0]

    return jsonify({
        'crop': predicted_crop,
        'irrigation': predicted_irrigation
    })

if __name__ == '__main__':
    app.run(debug=True)