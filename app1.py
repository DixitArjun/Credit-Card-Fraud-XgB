import streamlit as st
import numpy as np
import joblib

# Load the trained XGBoost model
model = joblib.load("xgb_fraud_model_weighted.joblib")

st.title("🔍 Credit Card Fraud Detection")

st.markdown("Enter the transaction details below to check for potential fraud.")

# Input fields
time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)
distance = st.number_input("Distance From Last Transaction", min_value=0.0)
avg_transaction = st.number_input("Average Transaction Value", min_value=0.0)
past_24h = st.number_input("Number of Transactions in Past 24 Hours", min_value=0)
device_type = st.selectbox("Device Type", options=["Mobile", "Desktop"])
is_international = st.selectbox("Is International Transaction?", options=["No", "Yes"])
merchant_risk = st.number_input("Merchant Risk Score", min_value=0.0)
prev_fraud = st.number_input("Previous Fraudulent Transactions", min_value=0)

# Encode categorical variables
device_type_encoded = 1 if device_type == "Desktop" else 0
is_international_encoded = 1 if is_international == "Yes" else 0

# Apply feature weights (same as training)
input_data = np.array([
    time * 3.0,
    amount * 3.0,
    distance * 3.0,
    avg_transaction * 3.0,
    past_24h * 1.5,
    device_type_encoded * 0.8,
    is_international_encoded * 0.5,
    merchant_risk,  # raw value
    prev_fraud * 1.0
]).reshape(1, -1)

# Predict probability
if st.button("Check for Fraud"):
    prob = model.predict_proba(input_data)[0][1]
    is_fraud = prob > 0.32  # lower threshold for fewer FNs

    st.markdown(f"**Fraud Probability:** {prob:.2f}")
    if is_fraud:
        st.error("🚨 This transaction is likely fraudulent!")
    else:
        st.success("✅ This transaction appears legitimate.")
