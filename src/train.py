# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.preprocess import load_csv, basic_clean, feature_engineer
from src.models import get_models

import joblib
import numpy as np

# =======================
# Config
# =======================
RAW_DATA_PATH = "data/raw/matches.csv"               # adjust path if needed
PROCESSED_DATA_PATH = "data/processed/processed_matches.csv"
OUTPUT_DIR = "outputs"
TARGET_COLUMN = "final_score"                        # change to your target column

# =======================
# Step 1: Preprocess Data
# =======================
# create processed folder if it doesn't exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

print("📥 Loading raw data...")
df = load_csv(RAW_DATA_PATH)
df = basic_clean(df)
df = feature_engineer(df)
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"✅ Data preprocessing complete. Saved to {PROCESSED_DATA_PATH}")

# =======================
# Step 2: Split Features/Target
# =======================
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in processed data")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# =======================
# Step 3: Train Models
# =======================
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = get_models()
results = {}

for name, model in models.items():
    print(f"\n▶ Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{name.replace(' ','_')}.joblib")
    joblib.dump(model, model_path)
    print(f"✅ {name} trained and saved to {model_path}")
    print(f"   Metrics -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

# =======================
# Step 4: Print Summary
# =======================
print("\n📊 Training Summary:")
for name, metrics in results.items():
    print(f"{name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.2f}")
