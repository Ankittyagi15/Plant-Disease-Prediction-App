import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Load the model
model_path = "plant_disease_detection_model-Copy1.pk1"
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    model = None
    st.warning("Model file not found. Running in demo mode.")

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Predictor", layout="centered")
st.title("🌱 Plant Disease Prediction App")

st.markdown("Enter environmental conditions to predict if the plant is **diseased** or **healthy**.")

# Input form
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0)

if st.button("Predict"):
    if model:
        input_data = np.array([[temperature, humidity, soil_moisture]])
        prediction = model.predict(input_data)[0]
        st.success(f"Prediction: {'Diseased' if prediction else 'Healthy'}")
    else:
        st.info("Demo prediction (model not loaded): This is a simulated result.")





