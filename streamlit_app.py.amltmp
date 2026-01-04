# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------
# 1. Load model, scaler, feature columns
# --------------------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_cols.pkl", "rb"))

# --------------------------
# 2. Streamlit UI
# --------------------------
st.title("💳 Fraud Detection in Financial Transactions")

st.write("Enter transaction details below to check if it's likely fraudulent:")

# User input
step = st.number_input("Step (time unit)", min_value=0, value=1)
amount = st.number_input("Amount", min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input("Old balance of origin account", min_value=0.0, value=0.0)
newbalanceOrig = st.number_input("New balance of origin account", min_value=0.0, value=0.0)
oldbalanceDest = st.number_input("Old balance of destination account", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New balance of destination account", min_value=0.0, value=0.0)
type_option = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])

# --------------------------
# 3. Create input DataFrame
# --------------------------
input_df = pd.DataFrame([{
    'step': step,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'type': type_option
}])

# --------------------------
# 4. Encode categorical columns
# --------------------------
input_df = pd.get_dummies(input_df, columns=['type'], drop_first=True)

# --------------------------
# 5. Align columns with training
# --------------------------
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0  # add missing dummy columns

# Remove extra columns
input_df = input_df[feature_cols]

# --------------------------
# 6. Scale features
# --------------------------
input_scaled = scaler.transform(input_df)

# --------------------------
# 7. Make prediction
# --------------------------
pred_prob = rf_model.predict_proba(input_scaled)[0, 1]
pred_class = rf_model.predict(input_scaled)[0]

st.write("---")
st.subheader("Prediction Results")
st.write(f"**Fraud Probability:** {pred_prob*100:.2f}%")
st.write(f"**Predicted Class:** {'Fraud' if pred_class == 1 else 'Legitimate'}")
