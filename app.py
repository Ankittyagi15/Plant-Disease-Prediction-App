#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Streamlit UI
st.set_page_config(page_title="🌾 Plant Disease Predictor", layout="centered")

st.title("🌱 Plant Disease Prediction App")
st.markdown("Enter environmental conditions to predict if the plant is **diseased** or **healthy**.")

# Check if model file exists and load it
model_path = 'plant_disease_detection_model.pkl'
try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        model_loaded = True
    else:
        st.warning("⚠️ Model file not found. Running in demo mode.")
        model_loaded = False
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Input fields
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=20.0, step=0.1)
soil_pH = st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[temperature, humidity, rainfall, soil_pH]])
    
    if model_loaded:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("🚨 The plant is likely to be **Diseased**.")
        else:
            st.success("✅ The plant is likely to be **Healthy**.")
    else:
        # Demo mode - just a placeholder prediction
        st.info("📝 Demo prediction (model not loaded): This is a simulated result.")
        if temperature > 30 and humidity > 70:
            st.error("🚨 The plant is likely to be **Diseased**.")
        else:
            st.success("✅ The plant is likely to be **Healthy**.")


# In[ ]:


get_ipython().system('pip freeze > requirements.txt')


# In[ ]:


get_ipython().system('pip install -r requirements.txt')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:




