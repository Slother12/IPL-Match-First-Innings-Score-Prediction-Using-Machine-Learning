import streamlit as st
import pandas as pd
import joblib
from src.preprocess import feature_engineer

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = "outputs/Random_Forest.joblib"
model = joblib.load(MODEL_PATH)

st.title("IPL First-Innings Score Predictor 🏏")
st.write("Enter match details below to predict the first-innings score:")

# -------------------------
# Input fields
# -------------------------
overs_completed = st.number_input("Overs Completed", min_value=0, max_value=20, value=5)
runs_so_far = st.number_input("Runs Scored So Far", min_value=0, value=50)
wickets_so_far = st.number_input("Wickets Lost", min_value=0, max_value=10, value=1)

batting_team = st.selectbox("Batting Team", [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Delhi Capitals",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings"
])

bowling_team = st.selectbox("Bowling Team", [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Delhi Capitals",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings"
])

venue = st.selectbox("Venue", [
    "Chennai", "Mumbai", "Delhi", "Bangalore", "Kolkata", "Rajasthan", "Hyderabad", "Punjab"
])

# -------------------------
# Predict button
# -------------------------
if st.button("Predict Score"):
    # Create DataFrame
    new_match = pd.DataFrame([{
        'overs_completed': overs_completed,
        'runs_so_far': runs_so_far,
        'wickets_so_far': wickets_so_far,
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'venue': venue
    }])
    
    # Preprocess
    new_match_processed = feature_engineer(new_match)
    
    # Align columns
    trained_columns = model.feature_names_in_
    new_match_processed = new_match_processed.reindex(columns=trained_columns, fill_value=0)
    
    # Predict
    predicted_score = model.predict(new_match_processed)
    
    st.success(f"Predicted First-Innings Score: {int(predicted_score[0])} 🏏")
