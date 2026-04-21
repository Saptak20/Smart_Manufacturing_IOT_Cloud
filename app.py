import streamlit as st
import joblib
import pandas as pd

# Load model + columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Smart Manufacturing Predictor")

# Inputs
temperature = st.number_input("Temperature")
vibration = st.number_input("Vibration")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")
energy = st.number_input("Energy Consumption")

machine_status = st.selectbox("Machine Status", ["Running", "Stopped", "Maintenance"])

# Create input dataframe
input_data = pd.DataFrame({
    "temperature": [temperature],
    "vibration": [vibration],
    "humidity": [humidity],
    "pressure": [pressure],
    "energy_consumption": [energy],
    "machine_status": [machine_status]
})

# Convert categorical → numeric
input_data = pd.get_dummies(input_data)

# 🔧 THIS IS THE IMPORTANT PART (alignment fix)
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
