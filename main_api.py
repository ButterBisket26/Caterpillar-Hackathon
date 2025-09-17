import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# --- FastAPI app ---
app = FastAPI()

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and encoders ---
MODEL_FILE = 'model.joblib'
component_param_map = {
    'Drive': [
        'Brake Control', 'Pedal Sensor', 'Transmission Pressure', 'Hydraulic Pump Rate'
    ],
    'Engine': [
        'Engine Oil Pressure', 'Engine Speed', 'Engine Temperature',p 'Exhaust Gas Temperature'
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
all_parameters = sum(component_param_map.values(), [])
def get_label_encoder(options):
    le = LabelEncoder()
    le.fit(options)
    return le
le_machine = get_label_encoder(machine_options)
le_component = get_label_encoder(component_options)
le_parameter = get_label_encoder(all_parameters)

def load_model():
    if os.path.exists(MODEL_FILE):
        data = joblib.load(MODEL_FILE)
        return data['model'], data['scaler'], data['features'], data.get('metrics', None)
    return None, None, None, None

# --- API Models ---
class PredictRequest(BaseModel):
    machine: str
    component: str
    parameter: str
    value: float

# --- Prediction Endpoint ---
@app.post('/predict_api')
async def predict_api(req: PredictRequest):
    model, scaler, features, metrics = load_model()
    if not all([model, scaler, features]):
        return JSONResponse({"error": "Model not trained"}, status_code=400)
    X_input = pd.DataFrame([{
        'Machine': le_machine.transform([req.machine])[0],
        'Component': le_component.transform([req.component])[0],
        'Parameter': le_parameter.transform([req.parameter])[0],
        'Value': req.value
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
    response = {"prediction": float(prediction)}
    if metrics:
        response["metrics"] = metrics
    return response

# --- Serve React build and static files ---
if not os.path.exists('frontend_build/static'):
    os.makedirs('frontend_build/static')
app.mount("/static", StaticFiles(directory="frontend_build/static"), name="static")

@app.get("/", include_in_schema=False)
def serve_react_index():
    return FileResponse('frontend_build/index.html')

@app.get("/{full_path:path}", include_in_schema=False)
def serve_react_catchall(full_path: str):
    # Serve React index.html for any other route (client-side routing)
    index_path = os.path.join('frontend_build', 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return Response(content="Not found", status_code=404) 