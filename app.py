import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.title("Smart Manufacturing Predictor")

# Inputs
temperature = st.number_input("Temperature")
vibration = st.number_input("Vibration")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")
energy = st.number_input("Energy Consumption")

# Input dataframe
input_data = pd.DataFrame({
    "temperature": [temperature],
    "vibration": [vibration],
    "humidity": [humidity],
    "pressure": [pressure],
    "energy_consumption": [energy]
})

# Encode (basic)
input_data = pd.get_dummies(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
