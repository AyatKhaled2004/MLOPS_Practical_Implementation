import streamlit as st
import requests
import json
from src.config import COLUMNS_PATH

st.set_page_config(page_title="Real Estate Predictor", page_icon="🏠", layout="centered")

st.title("🏠 Bangalore Real Estate Price Predictor")
st.write("Enter the property details below to get an estimated price from the ML model.")

# Load locations from the JSON file
try:
    with open(COLUMNS_PATH, "r") as f:
        columns = json.load(f)["data_columns"]
        # Assuming the first 3 columns are sqft, bath, bhk, and the rest are one-hot encoded locations
        locations = columns[3:] 
except Exception:
    locations = ["other"]

# User input form
with st.form("prediction_form"):
    location = st.selectbox("Select Location:", locations)
    sqft = st.number_input("Total Area (sqft):", min_value=150.0, max_value=20000.0, value=1000.0, step=50.0)
    bhk = st.slider("Bedrooms (BHK):", min_value=1, max_value=8, value=2)
    bath = st.slider("Bathrooms:", min_value=1, max_value=8, value=2)
    
    submit_button = st.form_submit_button("💡 Predict Price")

# Trigger API call on form submission
if submit_button:
    payload = {
        "location": location,
        "sqft": sqft,
        "bath": bath,
        "bhk": bhk
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            price = result["predicted_price"]
            st.success(f"💰 Estimated Property Price: **{price} Lakhs**")
        else:
            st.error(f"Server Error: {response.json().get('detail')}")
            
    except Exception:
        st.error("Failed to connect to the FastAPI backend. Make sure the server is running!")