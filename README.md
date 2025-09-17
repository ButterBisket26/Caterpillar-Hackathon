# Machine Part Time-to-Failure Prediction

This project predicts when machine parts will fail based on sensor data using machine learning. It provides a web interface for training the model and making predictions.

## Features

- **Machine Learning Model**: Random Forest Regressor for time-to-failure prediction
- **Web Interface**: Flask-based web application with Bootstrap UI
- **Feature Engineering**: Automatic feature importance visualization
- **Real-time Predictions**: Interactive form for making predictions

## Project Structure

```
caterpillar/
├── main.py              # Model training script
├── flask_app.py         # Flask web application
├── requirements.txt     # Python dependencies
├── model.joblib         # Trained model (generated)
├── original.xlsx        # Training data
├── templates/           # HTML templates
│   └── index.html      # Main web interface
└── static/             # Static files (generated)
    └── feature_importance.png
```

## Setup and Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the data**:
   - Ensure `original.xlsx` contains the training data
   - The Excel file should have sheets: 'Data Set' and 'Threshold'

## How to Run

### Option 1: Web Interface (Recommended)

1. **Start the Flask application**:
   ```bash
   python flask_app.py
   ```

2. **Open your browser** and go to `http://localhost:5000`

3. **Train the model**:
   - Click the "Train Model" button on the web interface
   - This will run the training script and save the model

4. **Make predictions**:
   - Select Machine, Component, Parameter, and enter a sensor Value
   - Click "Predict" to get the time-to-failure prediction
   - View feature importance visualization

### Option 2: Command Line Training

1. **Train the model directly**:
   ```bash
   python main.py
   ```

2. **Start the web interface**:
   ```bash
   python flask_app.py
   ```

## Usage

1. **Training**: The model uses Random Forest Regressor with cross-validation
2. **Prediction**: Input machine parameters to get time-to-failure estimates
3. **Visualization**: Feature importance is automatically generated for each prediction

## Data Format

The model expects:
- **Machine**: Equipment identifier
- **Component**: Part category (Drive, Engine, Fuel, Misc)
- **Parameter**: Sensor type (e.g., Engine Temperature, Fuel Pressure)
- **Value**: Sensor reading
- **Time**: Timestamp of the reading

## Model Performance

The model provides:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Cross-validation scores

## Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Machine learning
- joblib: Model serialization
- openpyxl: Excel file reading
- matplotlib/seaborn: Visualization
- flask: Web framework 