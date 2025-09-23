# src/predict.py
# src/predict.py

import pandas as pd
import joblib
from src.preprocess import feature_engineer

# -------------------------
# Config
# -------------------------
MODEL_PATH = "outputs/Random_Forest.joblib"   # path to your trained model

# -------------------------
# Load trained model
# -------------------------
model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model: {MODEL_PATH}")

# -------------------------
# Example new match input
# -------------------------
# Make sure the keys match your original columns before one-hot encoding
new_match = pd.DataFrame([{
    'overs_completed': 5,
    'runs_so_far': 55,
    'wickets_so_far': 1,
    'batting_team': 'Chennai Super Kings',
    'bowling_team': 'Mumbai Indians',
    'venue': 'Chennai'
}])

# -------------------------
# Preprocess new input
# -------------------------
new_match_processed = feature_engineer(new_match)

# Align columns with the model's training features
trained_columns = model.feature_names_in_  # scikit-learn stores training columns
new_match_processed = new_match_processed.reindex(columns=trained_columns, fill_value=0)

# -------------------------
# Predict first-innings score
# -------------------------
predicted_score = model.predict(new_match_processed)
print(f"Predicted first-innings score: {predicted_score[0]:.0f}")
