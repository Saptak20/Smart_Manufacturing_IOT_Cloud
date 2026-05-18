import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Smart Manufacturing Predictor")

temperature = st.number_input("Temperature")
vibration = st.number_input("Vibration")
humidity = st.number_input("Humidity")
pressure = st.number_input("Pressure")
energy = st.number_input("Energy Consumption")

machine_status = st.selectbox(
    "Machine Status",
    ["Running", "Stopped", "Maintenance"]
)

input_data = pd.DataFrame({
    "temperature": [temperature],
    "vibration": [vibration],
    "humidity": [humidity],
    "pressure": [pressure],
    "energy_consumption": [energy],
    "machine_status": [machine_status]
})

input_data = pd.get_dummies(input_data)

input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict"):

    # 🔥 NEW: use probability instead of direct prediction
    prob = model.predict_proba(input_data)[0][1]

    # 🔥 Threshold tuning (fix bias)
    if prob > 0.3:
        st.error(f"⚠️ Maintenance Needed (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Machine Healthy (Confidence: {1-prob:.2f})")
