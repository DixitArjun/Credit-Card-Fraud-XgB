import streamlit as st
import numpy as np
import joblib

# Load trained XGBoost model
model = joblib.load("xgb_fraud_model_weighted.joblib")

# UI layout
st.title("💳 Credit Card Fraud Detector")

st.markdown("Adjust the input features below to simulate a transaction:")

# Inputs
time = st.slider("Transaction Time", 0, 86400, 40000)
amount = st.number_input("Transaction Amount ($)", 0.0, 5000.0, 200.0)
distance = st.number_input("Distance from Last Transaction", 0.0, 1000.0, 100.0)
avg_value = st.number_input("Avg. Transaction Value", 0.0, 5000.0, 150.0)
past_24h = st.slider("Num Transactions in Past 24h", 0, 50, 5)
device_type = st.selectbox("Device Type", ["Desktop", "Mobile", "Tablet"])
prev_fraud = st.slider("Previous Fraudulent Transactions", 0, 10, 0)
merchant_risk = st.slider("Merchant Risk Score", 0.0, 10.0, 5.0)
is_international = st.selectbox("Is International?", ["No", "Yes"])

# Encoding and weights (same as training)
device_map = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
device_encoded = device_map[device_type]
is_international_val = 1 if is_international == "Yes" else 0

# Define weights
feature_weights = {
    'Time': 1.0,
    'Amount': 4.0,
    'Distance_From_Last_Transaction': 4.0,
    'Avg_Transaction_Value': 2.5,
    'Num_Transactions_Past_24H': 1.5,
    'Device_Type': 0.8,
    'Is_International': 1.5,
    'Merchant_Risk_Score': 1.0,  # No scaling for this one
    'Previous_Fraudulent_Transactions': 1.0
}

# Apply weights
inputs = np.array([
    time * feature_weights['Time'],
    amount * feature_weights['Amount'],
    distance * feature_weights['Distance_From_Last_Transaction'],
    avg_value * feature_weights['Avg_Transaction_Value'],
    past_24h * feature_weights['Num_Transactions_Past_24H'],
    device_encoded * feature_weights['Device_Type'],
    is_international_val * feature_weights['Is_International'],
    merchant_risk,  # raw value
    prev_fraud * feature_weights['Previous_Fraudulent_Transactions']
]).reshape(1, -1)

# Threshold slider
threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.2, step=0.01)

# Predict
if st.button("Predict Fraud Probability"):
    prob = model.predict_proba(inputs)[0][1]
    st.write(f"🔍 Fraud Probability: **{prob:.4f}**")

    if prob > threshold:
        st.error("🚨 Likely Fraudulent Transaction!")
    else:
        st.success("✅ Likely Legitimate Transaction.")
