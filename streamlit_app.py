import streamlit as st
import pandas as pd
import pickle

# Load trained artifacts
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_cols.pkl", "rb"))

# Streamlit input form
st.title("Fraud Detection Demo")

step = st.number_input("Step", value=1)
amount = st.number_input("Amount", value=5000.0)
oldbalanceOrg = st.number_input("Old Balance Origin", value=200.0)
newbalanceOrig = st.number_input("New Balance Origin", value=0.0)
oldbalanceDest = st.number_input("Old Balance Destination", value=0.0)
newbalanceDest = st.number_input("New Balance Destination", value=5000.0)
txn_type = st.selectbox("Transaction Type", ['PAYMENT','TRANSFER','CASH_OUT','DEBIT','CASH_IN'])

# Create DataFrame
input_df = pd.DataFrame([{
    'step': step,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'type': txn_type
}])

# Dummy encode
input_df = pd.get_dummies(input_df, columns=['type'], drop_first=True)

# Add missing columns from training
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Remove extra columns
input_df = input_df[feature_cols]

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction
pred_prob = rf_model.predict_proba(input_scaled)[0, 1]
threshold = 0.01  # demo-friendly
pred_class = 1 if pred_prob >= threshold else 0

st.write(f"Fraud Probability: {pred_prob*100:.2f}%")
st.write(f"Predicted Class: {'Fraud' if pred_class==1 else 'Legitimate'}")
