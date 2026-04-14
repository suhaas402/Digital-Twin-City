import os
import joblib
import numpy as np
import pandas as pd

DATA_PATH = "data/traffic_simulation.csv"
MODEL_DIR = "models"
TRAFFIC_MODEL_PATH = os.path.join(MODEL_DIR, "traffic_model.pkl")
AQI_MODEL_PATH = os.path.join(MODEL_DIR, "aqi_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.pkl")

# 1) Checks
for p in [DATA_PATH, TRAFFIC_MODEL_PATH, AQI_MODEL_PATH, FEATURE_COLS_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

# 2) Load artifacts
traffic_model = joblib.load(TRAFFIC_MODEL_PATH)
aqi_model = joblib.load(AQI_MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

if not feature_cols:
    raise ValueError("feature_cols.pkl is empty — retrain the model first.")

# 3) Load data and rebuild lags (must match training exactly)
df = pd.read_csv(DATA_PATH)

for c in ["hour", "day_of_week", "is_weekend"]:
    if c in df.columns:
        df[c] = df[c].astype(int)

for col in ["traffic", "aqi"]:
    for l in [1, 2, 3, 24]:
        df[f"{col}_lag_{l}"] = df[col].shift(l)

df = df.dropna().reset_index(drop=True)

if len(df) < 4:
    raise ValueError("Need at least 4 rows after lag creation for lag_1/2/3 features.")

missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    raise KeyError(f"Missing required feature columns in CSV: {missing_features}")

# 4) Construct a genuine T+1 feature row from the last known timestep
last = df.iloc[-1]
next_t = int(last["timestep"]) + 1
next_hour = next_t % 24
next_day = next_t // 24
next_dow = next_day % 7

next_row = {
    # Time features (T+1)
    "hour": next_hour,
    "day_of_week": next_dow,
    "is_weekend": int(next_dow >= 5),

    # Carry-forward congestion proxies (replace with live simulator values in production)
    "active_nodes": last["active_nodes"],
    "mean_node_load": last["mean_node_load"],
    "max_node_load": last["max_node_load"],
    "mean_edge_load": last.get("mean_edge_load", 0.0),
    "max_edge_load": last.get("max_edge_load", 0.0),

    # Weather proxy at T+1 (same diurnal pattern as simulation)
    "temperature": 22 + 6 * np.sin(2 * np.pi * next_hour / 24),
    "wind": max(0.0, 3 + 1.5 * np.cos(2 * np.pi * next_hour / 24)),

    # Lag features from known history
    "traffic_lag_1": last["traffic"],
    "traffic_lag_2": df["traffic"].iloc[-2],
    "traffic_lag_3": df["traffic"].iloc[-3],
    "traffic_lag_24": df["traffic"].iloc[-24] if len(df) >= 24 else last["traffic"],

    "aqi_lag_1": last["aqi"],
    "aqi_lag_2": df["aqi"].iloc[-2],
    "aqi_lag_3": df["aqi"].iloc[-3],
    "aqi_lag_24": df["aqi"].iloc[-24] if len(df) >= 24 else last["aqi"],
}

X_input = pd.DataFrame([next_row])[feature_cols]

# 5) Predict and clamp to non-negative
pred_traffic = max(0.0, float(traffic_model.predict(X_input)[0]))
pred_aqi = max(0.0, float(aqi_model.predict(X_input)[0]))

print(f"=== Next-step Prediction (T+1 = timestep {next_t}, hour {next_hour}) ===")
print(f"Predicted Traffic : {pred_traffic:.3f}")
print(f"Predicted AQI     : {pred_aqi:.3f}")
print()
print("Note: active_nodes / load features are carried forward from T.")
print("For live deployment, replace them with actual simulator/sensor values at T+1.")