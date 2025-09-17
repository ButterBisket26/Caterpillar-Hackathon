from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Component to parameter mapping
component_param_map = {
    'Drive': [
        'Brake Control', 'Pedal Sensor', 'Transmission Pressure', 'Hydraulic Pump Rate'
    ],
    'Engine': [
        'Engine Oil Pressure', 'Engine Speed', 'Engine Temperature', 'Exhaust Gas Temperature'
    ],
    'Fuel': [
        'Fuel Level', 'Fuel Pressure', 'Fuel Temperature', 'Water in Fuel'
    ],
    'Misc': [
        'System Voltage', 'Air Filter Pressure'
    ]
}

machine_options = [
    'Excavator_1', 'Articulated_Truck_1', 'Backhoe_Loader_1', 'Dozer_1', 'Asphalt_Paver_1'
]
component_options = list(component_param_map.keys())

# For label encoding, flatten all possible parameters
all_parameters = sum(component_param_map.values(), [])

def get_label_encoder(options):
    le = LabelEncoder()
    le.fit(options)
    return le
le_machine = get_label_encoder(machine_options)
le_component = get_label_encoder(component_options)
le_parameter = get_label_encoder(all_parameters)

# Load model and scaler
MODEL_FILE = 'model.joblib'
def load_model():
    if os.path.exists(MODEL_FILE):
        data = joblib.load(MODEL_FILE)
        return data['model'], data['scaler'], data['features'], data.get('metrics', None)
    return None, None, None, None

@app.route('/get_parameters', methods=['POST'])
def get_parameters():
    component = request.json['component']
    params = component_param_map.get(component, [])
    return jsonify(params)

@app.route('/static/<path:filename>')
def staticfiles(filename):
    return send_from_directory('frontend_build/static', filename)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    machine = data.get('machine')
    component = data.get('component')
    parameter = data.get('parameter')
    value = float(data.get('value'))
    model, scaler, features, _ = load_model()
    if not all([model, scaler, features]):
        return jsonify({'error': 'Model not trained'}), 400
    X_input = pd.DataFrame([{
        'Machine': le_machine.transform([machine])[0],
        'Component': le_component.transform([component])[0],
        'Parameter': le_parameter.transform([parameter])[0],
        'Value': value
    }])[features]
    X_input_scaled = scaler.transform(X_input)
    prediction = model.predict(X_input_scaled)[0]
    # Save feature importance plot
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=importances, y=features, ax=ax, palette='viridis')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    feature_importance_path = 'static/feature_importance.png'
    plt.savefig(feature_importance_path)
    plt.close(fig)
    return jsonify({'prediction': float(prediction)})

@app.route('/', methods=['GET'])
def root():
    return send_from_directory('frontend_build', 'index.html')

@app.route('/<path:path>', methods=['GET'])
def catch_all(path):
    # For any other route, serve the React app (for client-side routing)
    return send_from_directory('frontend_build', 'index.html')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True) 