import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

# --- CONFIG ---
DATA_FILE = 'original.xlsx'
MODEL_FILE = 'model.joblib'
RANDOM_STATE = 42

# --- UTILS ---
def parse_threshold(thresh_str):
    """Parse threshold string into a dict of bounds."""
    thresh_str = str(thresh_str).replace(',', ' and ')
    bounds = {}
    for part in thresh_str.split('and'):
        part = part.strip()
        m = re.match(r'(Low|High)\s*([\d\.]+)', part, re.I)
        if m:
            if m.group(1).lower() == 'low':
                bounds['low'] = float(m.group(2))
            else:
                bounds['high'] = float(m.group(2))
    return bounds

def is_failure(value, bounds):
    """Return True if value is in failure region as per bounds."""
    if 'low' in bounds and value < bounds['low']:
        return True
    if 'high' in bounds and value > bounds['high']:
        return True
    return False

def compute_time_to_failure(df):
    """For each row, compute time (in days) to next failure for same Machine/Component/Parameter."""
    df = df.sort_values(['Machine', 'Component', 'Parameter', 'Time'])
    df['time_to_failure'] = np.nan
    for key, group in df.groupby(['Machine', 'Component', 'Parameter']):
        group = group.reset_index()
        failure_idxs = group.index[group['is_failure']].tolist()
        if not failure_idxs:
            continue
        for i, row in group.iterrows():
            # Find next failure after this row
            next_failures = [idx for idx in failure_idxs if idx > i]
            if next_failures:
                next_failure_idx = next_failures[0]
                t1 = row['Time']
                t2 = group.loc[next_failure_idx, 'Time']
                delta = (t2 - t1).total_seconds() / (3600*24)
                df.loc[row['index'], 'time_to_failure'] = delta
    return df

def main():
    # --- Load Data ---
    xls = pd.ExcelFile(DATA_FILE)
    df = pd.read_excel(xls, 'Data Set')
    df_thresh = pd.read_excel(xls, 'Threshold')

    # --- Preprocessing ---
    df['Time'] = pd.to_datetime(df['Time'])
    # Map parameter to threshold
    thresh_map = {row['Parameter'].strip().lower(): parse_threshold(row['Threshold']) for _, row in df_thresh.iterrows()}
    df['param_key'] = df['Parameter'].str.strip().str.lower()
    df['bounds'] = df['param_key'].map(thresh_map)
    df['is_failure'] = df.apply(lambda r: is_failure(r['Value'], r['bounds']) if isinstance(r['bounds'], dict) else False, axis=1)

    # --- Feature Engineering ---
    df = compute_time_to_failure(df)
    # Drop rows where time_to_failure is nan or negative
    df = df[df['time_to_failure'].notnull() & (df['time_to_failure'] > 0)]

    # Encode categoricals
    for col in ['Machine', 'Component', 'Parameter']:
        df[col] = LabelEncoder().fit_transform(df[col])
    # Feature selection
    features = ['Machine', 'Component', 'Parameter', 'Value']
    X = df[features]
    y = df['time_to_failure']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Modeling ---
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mae_scores = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error'))
    print(f'MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}')
    print(f'RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}')
    # Fit on all data
    model.fit(X_scaled, y)
    # Save model and scaler
    joblib.dump({'model': model, 'scaler': scaler, 'features': features}, MODEL_FILE)
    print(f'Model saved to {MODEL_FILE}')

if __name__ == '__main__':
    main() 