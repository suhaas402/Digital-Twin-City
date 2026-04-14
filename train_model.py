"""
train.py  —  Digital Twin City  |  Phase 4  (disaster-aware edition)
=====================================================================
Models trained
--------------
  XGBoost (one-step prediction)
    1. traffic          — congestion index
    2. aqi              — air quality index
    3. evac_rate        — fraction of civilians evacuated
    4. avg_response_time — emergency vehicle response time

  PyTorch LSTM (multi-step forecasting)
    5. evac_rate  — 6-step ahead forecast (disaster management focus)

New vs previous train.py
-------------------------
  + disaster_type one-hot encoded  (flood / fire / earthquake / none)
  + disaster_zone, disaster_severity, zone_hazard, blocked_ratio
  + shelter_util, avg_response_time as features
  + evac_rate lag features  (1, 2, 3, 24)
  + avg_response_time target model
  + PyTorch LSTM multi-step evac forecaster
  + per-disaster-type metric breakdown in evaluation
  + residual plots saved alongside prediction plots
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/traffic_simulation.csv"
MODEL_DIR = "models"
PLOT_DIR  = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
LSTM_SEQ_LEN    = 24    # look-back window (hours)
LSTM_HORIZON    = 6     # steps ahead to forecast
LSTM_EPOCHS     = 60
LSTM_BATCH      = 32
LSTM_HIDDEN     = 64
LSTM_LAYERS     = 2
LSTM_LR         = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1) Load + validate data
# ═══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Simulation CSV not found at {DATA_PATH}. Run simulate.py first."
    )
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {DATA_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2) Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

# --- Cast integer columns ---
for c in ["hour", "day_of_week", "is_weekend"]:
    if c in df.columns:
        df[c] = df[c].astype(int)

# --- One-hot encode disaster_type ---
# Values: none / flood / fire / earthquake
if "disaster_type" in df.columns:
    # FIX: deterministic numeric dtype
    dummies = pd.get_dummies(df["disaster_type"], prefix="dis", dtype=int)
    # Ensure all four columns exist even if a type never appears in this run
    for col in ["dis_none", "dis_flood", "dis_fire", "dis_earthquake"]:
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df, dummies], axis=1)
else:
    for col in ["dis_none", "dis_flood", "dis_fire", "dis_earthquake"]:
        df[col] = 0

# --- Fill sentinel -1 in avg_response_time with rolling forward-fill ---
# -1 means no vehicle has arrived yet. We carry forward the last known value
# and initialise with the column median so early rows aren't NaN after dropna.
if "avg_response_time" in df.columns:
    df["avg_response_time"] = df["avg_response_time"].replace(-1, np.nan)
    df["avg_response_time"] = (
        df["avg_response_time"]
        .ffill()  # FIX: modern pandas style
        .fillna(df["avg_response_time"].median())
    )

# --- Lag features ---
LAG_TARGETS = ["traffic", "aqi", "evac_rate"]
LAGS        = [1, 2, 3, 24]
for col in LAG_TARGETS:
    if col in df.columns:
        for l in LAGS:
            df[f"{col}_lag_{l}"] = df[col].shift(l)

df = df.dropna().reset_index(drop=True)
print(f"After lag + dropna: {len(df)} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# 3) Feature columns
# ═══════════════════════════════════════════════════════════════════════════════
candidate_features = [
    # Temporal
    "hour", "day_of_week", "is_weekend",
    # Weather
    "temperature", "wind",
    # Network load
    "active_nodes", "mean_node_load", "max_node_load",
    "mean_edge_load", "max_edge_load",
    # Disaster state
    "dis_flood", "dis_fire", "dis_earthquake",   # dis_none excluded (collinear)
    "disaster_zone", "disaster_severity",
    "zone_hazard", "blocked_ratio",
    # Evacuation state
    "shelter_util", "avg_response_time",
    # Lags
    "traffic_lag_1",   "traffic_lag_2",   "traffic_lag_3",   "traffic_lag_24",
    "aqi_lag_1",       "aqi_lag_2",       "aqi_lag_3",       "aqi_lag_24",
    "evac_rate_lag_1", "evac_rate_lag_2", "evac_rate_lag_3", "evac_rate_lag_24",
]
feature_cols = [c for c in candidate_features if c in df.columns]
missing = set(candidate_features) - set(feature_cols)
if missing:
    print(f"Note: {len(missing)} candidate features not in CSV (skipped): {missing}")
print(f"Using {len(feature_cols)} features: {feature_cols}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4) Train / val / test split  (time-ordered, no shuffle)
# ════════════════════════════════════════════════════���══════════════════════════
split_idx  = int(len(df) * 0.80)
val_split  = int(split_idx * 0.90)

train_df = df.iloc[:val_split].copy()
val_df   = df.iloc[val_split:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()
print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

X_train = train_df[feature_cols]
X_val   = val_df[feature_cols]
X_test  = test_df[feature_cols]

TARGETS = ["traffic", "aqi", "evac_rate", "avg_response_time"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5) XGBoost — train one model per target
# ═══════════════════════════════════════════════════════════════════════════════
XGB_PARAMS = dict(
    n_estimators        = 1000,
    learning_rate       = 0.05,
    max_depth           = 6,
    subsample           = 0.8,
    colsample_bytree    = 0.8,
    reg_alpha           = 0.1,
    reg_lambda          = 1.0,
    random_state        = 42,
    n_jobs              = -1,
    eval_metric         = "rmse",
    early_stopping_rounds = 30,
)

xgb_models  = {}
xgb_metrics = {}

for target in TARGETS:
    if target not in df.columns:
        print(f"  Skipping {target} — not in CSV")
        continue
    print(f"\nTraining XGBoost → {target} ...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, train_df[target],
        eval_set=[(X_val, val_df[target])],
        verbose=False,
    )
    preds = model.predict(X_test)
    mae   = mean_absolute_error(test_df[target], preds)
    rmse  = np.sqrt(mean_squared_error(test_df[target], preds))
    print(f"  best_iteration={model.best_iteration}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    xgb_models[target]  = model
    xgb_metrics[target] = {"mae": round(mae, 4), "rmse": round(rmse, 4)}

# ═══════════════════════════════════════════════════════════════════════════════
# 6) Per-disaster-type metric breakdown (separate for all types)
# ══════════════════════════════��════════════════════════════════════════════════
print("\n=== Per-disaster-type metrics (test set) ===")

DISASTER_TYPES = ["none", "flood", "fire", "earthquake"]

def get_disaster_mask(frame: pd.DataFrame, dtype: str) -> pd.Series:
    """Return boolean mask for rows of a specific disaster type."""
    if dtype == "none":
        # none = no active disaster one-hot flags
        return ~(
            frame.get("dis_flood", pd.Series(0, index=frame.index)).astype(bool) |
            frame.get("dis_fire", pd.Series(0, index=frame.index)).astype(bool) |
            frame.get("dis_earthquake", pd.Series(0, index=frame.index)).astype(bool)
        )
    col = f"dis_{dtype}"
    if col in frame.columns:
        return frame[col].astype(bool)
    return pd.Series(False, index=frame.index)

breakdown = {}

for target, model in xgb_models.items():
    preds_all = model.predict(X_test)
    breakdown[target] = {}

    print(f"\nTarget: {target}")
    for dtype in DISASTER_TYPES:
        mask = get_disaster_mask(test_df, dtype)
        n = int(mask.sum())

        if n == 0:
            breakdown[target][dtype] = {"n": 0, "mae": None, "rmse": None}
            print(f"  [{dtype:12s}] n=   0  (no samples in test split)")
            continue

        y_true = test_df.loc[mask, target]
        y_pred = preds_all[mask.values]  # align numpy index with boolean mask
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        breakdown[target][dtype] = {
            "n": n,
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
        }
        print(f"  [{dtype:12s}] n={n:4d}  MAE={mae:.4f}  RMSE={rmse:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7) Save XGBoost models + metrics
# ═══════════════════════════════════════════════════════════════════════════════
for target, model in xgb_models.items():
    path = os.path.join(MODEL_DIR, f"{target}_model.pkl")
    joblib.dump(model, path)
    print(f"Saved: {path}")

joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

metrics_out = {
    "model":        "XGBRegressor",
    "xgb_version":  xgb.__version__,
    "split":        {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "features":     feature_cols,
    "overall":      xgb_metrics,
    "per_disaster": breakdown,
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f, indent=2)
print("Saved: metrics.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 8) Plots — prediction vs actual + residuals
# ═══════════════════════════════════════════════════════════════════════════════
def save_pred_plot(y_true, y_pred, title, ylabel, path):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(y_true.values, label="Actual",    lw=1.2, color="steelblue")
    axes[0].plot(y_pred,        label="Predicted", lw=1.2, color="tomato", alpha=0.85)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].legend()
    residuals = y_true.values - y_pred
    axes[1].axhline(0, color="gray", lw=0.8, linestyle="--")
    axes[1].fill_between(range(len(residuals)), residuals, alpha=0.4, color="darkorange")
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Test timestep")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def save_importance_plot(model, feature_cols, title, path):
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_cols) * 0.28)))
    imp.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

for target, model in xgb_models.items():
    preds = model.predict(X_test)
    save_pred_plot(
        test_df[target], preds,
        f"{target.replace('_',' ').title()}: Actual vs Predicted",
        target,
        os.path.join(PLOT_DIR, f"{target}_prediction_vs_actual.png"),
    )
    save_importance_plot(
        model, feature_cols,
        f"{target.replace('_',' ').title()} — Feature Importance",
        os.path.join(PLOT_DIR, f"{target}_feature_importance.png"),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 9) PyTorch LSTM — multi-step evac_rate forecaster
# ═══════════════════════════════════════════════════════════════════════════════
if "evac_rate" not in df.columns:
    print("\nSkipping LSTM — evac_rate not in CSV")
else:
    print(f"\n{'='*60}")
    print(f"PyTorch LSTM  |  seq_len={LSTM_SEQ_LEN}  horizon={LSTM_HORIZON}")
    print(f"{'='*60}")

    # --- Scale features + target jointly ---
    lstm_feature_cols = feature_cols
    target_col        = "evac_rate"

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    all_X = df[lstm_feature_cols].values.astype(np.float32)
    all_y = df[[target_col]].values.astype(np.float32)

    # Fit scalers only on training portion
    train_end = val_split
    scaler_X.fit(all_X[:train_end])
    scaler_y.fit(all_y[:train_end])

    X_scaled = scaler_X.transform(all_X)
    y_scaled = scaler_y.transform(all_y).flatten()

    # --- Build sliding-window sequences ---
    def make_sequences(X, y, seq_len, horizon):
        xs, ys = [], []
        for i in range(len(X) - seq_len - horizon + 1):
            xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len : i + seq_len + horizon])
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    X_seq, y_seq = make_sequences(X_scaled, y_scaled, LSTM_SEQ_LEN, LSTM_HORIZON)
    print(f"Sequences: {X_seq.shape}  Targets: {y_seq.shape}")

    # Time-ordered split (no shuffle)
    n_seq         = len(X_seq)
    train_end_seq = int(n_seq * 0.80)
    val_end_seq   = int(n_seq * 0.90)

    X_tr, y_tr = X_seq[:train_end_seq],             y_seq[:train_end_seq]
    X_vl, y_vl = X_seq[train_end_seq:val_end_seq],  y_seq[train_end_seq:val_end_seq]
    X_te, y_te = X_seq[val_end_seq:],               y_seq[val_end_seq:]

    def to_tensor(*arrays):
        return [torch.from_numpy(a).to(DEVICE) for a in arrays]

    Xtr, ytr = to_tensor(X_tr, y_tr)
    Xvl, yvl = to_tensor(X_vl, y_vl)
    Xte, yte = to_tensor(X_te, y_te)

    # FIX: no shuffling for time-series training windows
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=LSTM_BATCH, shuffle=False)
    val_loader   = DataLoader(TensorDataset(Xvl, yvl), batch_size=LSTM_BATCH, shuffle=False)

    class EvacLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, horizon):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, horizon),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    model_lstm = EvacLSTM(
        input_size  = len(lstm_feature_cols),
        hidden_size = LSTM_HIDDEN,
        num_layers  = LSTM_LAYERS,
        horizon     = LSTM_HORIZON,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=LSTM_LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss  = float("inf")
    best_state     = None
    train_losses, val_losses = [], []

    for epoch in range(1, LSTM_EPOCHS + 1):
        model_lstm.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model_lstm(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr)

        model_lstm.eval()
        with torch.no_grad():
            val_loss = criterion(model_lstm(Xvl), yvl).item()
        scheduler.step(val_loss)

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model_lstm.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{LSTM_EPOCHS}  "
                  f"train_loss={epoch_loss:.5f}  val_loss={val_loss:.5f}")

    model_lstm.load_state_dict(best_state)
    model_lstm.to(DEVICE)
    print(f"  Best val_loss: {best_val_loss:.5f}")

    model_lstm.eval()
    with torch.no_grad():
        pred_scaled = model_lstm(Xte).cpu().numpy()
        true_scaled = yte.cpu().numpy()

    pred_orig = scaler_y.inverse_transform(pred_scaled)
    true_orig = scaler_y.inverse_transform(true_scaled)

    mae_lstm  = mean_absolute_error(true_orig.flatten(), pred_orig.flatten())
    rmse_lstm = np.sqrt(mean_squared_error(true_orig.flatten(), pred_orig.flatten()))
    print(f"  LSTM test  MAE={mae_lstm:.4f}  RMSE={rmse_lstm:.4f}  "
          f"(averaged over {LSTM_HORIZON} steps)")

    print("  Per-step breakdown:")
    for h in range(LSTM_HORIZON):
        mae_h  = mean_absolute_error(true_orig[:, h], pred_orig[:, h])
        rmse_h = np.sqrt(mean_squared_error(true_orig[:, h], pred_orig[:, h]))
        print(f"    step+{h+1}  MAE={mae_h:.4f}  RMSE={rmse_h:.4f}")

    metrics_out["lstm_evac_rate"] = {
        "seq_len":    LSTM_SEQ_LEN,
        "horizon":    LSTM_HORIZON,
        "epochs":     LSTM_EPOCHS,
        "best_val_loss": round(best_val_loss, 6),
        "test_mae":   round(mae_lstm, 4),
        "test_rmse":  round(rmse_lstm, 4),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    torch.save(best_state, os.path.join(MODEL_DIR, "evac_lstm.pt"))
    joblib.dump(scaler_X,  os.path.join(MODEL_DIR, "lstm_scaler_X.pkl"))
    joblib.dump(scaler_y,  os.path.join(MODEL_DIR, "lstm_scaler_y.pkl"))
    print("Saved: evac_lstm.pt, lstm_scaler_X.pkl, lstm_scaler_y.pkl")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss", lw=1.2, color="steelblue")
    plt.plot(val_losses,   label="Val loss",   lw=1.2, color="tomato")
    plt.xlabel("Epoch"); plt.ylabel("MSE loss")
    plt.title("LSTM evac_rate — training curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lstm_training_curve.png"), dpi=150)
    plt.close()
    print("Saved: lstm_training_curve.png")

    n_show = min(96, len(pred_orig))
    plt.figure(figsize=(14, 5))
    plt.plot(true_orig[:n_show, 0],  label="Actual (step+1)",   lw=1.2, color="steelblue")
    plt.plot(pred_orig[:n_show, 0],  label="Pred   (step+1)",   lw=1.2, color="tomato",   alpha=0.85)
    plt.plot(true_orig[:n_show, -1], label=f"Actual (step+{LSTM_HORIZON})", lw=1.0, color="seagreen",  linestyle="--")
    plt.plot(pred_orig[:n_show, -1], label=f"Pred   (step+{LSTM_HORIZON})", lw=1.0, color="darkorange",linestyle="--", alpha=0.85)
    plt.xlabel("Test sequence index"); plt.ylabel("evac_rate")
    plt.title(f"LSTM evac_rate forecast  (step+1 and step+{LSTM_HORIZON})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "lstm_evac_forecast.png"), dpi=150)
    plt.close()
    print("Saved: lstm_evac_forecast.png")

print("\n" + "="*60)
print("All models trained and saved.")
print(f"  XGBoost models : {list(xgb_models.keys())}")
print(f"  LSTM model     : evac_rate  ({LSTM_HORIZON}-step horizon)")
print(f"  Metrics        : {MODEL_DIR}/metrics.json")
print(f"  Plots          : {PLOT_DIR}/")
print("="*60)