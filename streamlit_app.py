# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Load trained model and scaler
# ------------------------------
# Make sure you save your model and scaler after training
# Example: 
#   pickle.dump(rf_model, open("rf_model.pkl", "wb"))
#   pickle.dump(scaler, open("scaler.pkl", "wb"))

rf_model = pickle.load(open("fraud_detection_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------------------
# Streamlit Interface
# ------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Fraud Detection in Transactions")

st.markdown("""
Enter transaction details below to get a fraud prediction.
""")

# ------------------------------
# User Inputs
# ------------------------------
# Example features from your dataset: amount, type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=10.0)
type_options = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
tx_type = st.selectbox("Transaction Type", type_options)
oldbalanceOrg = st.number_input("Origin Account Old Balance", min_value=0.0, value=5000.0, step=10.0)
newbalanceOrig = st.number_input("Origin Account New Balance", min_value=0.0, value=4000.0, step=10.0)
oldbalanceDest = st.number_input("Destination Account Old Balance", min_value=0.0, value=1000.0, step=10.0)
newbalanceDest = st.number_input("Destination Account New Balance", min_value=0.0, value=2000.0, step=10.0)

# ------------------------------
# Prepare Data
# ------------------------------
# Create dataframe
# Example input
input_df = pd.DataFrame([{
    'step': 1,
    'type': 'PAYMENT',   # must match your training data categories
    'amount': 9839.64,
    'oldbalanceOrg': 170136.0,
    'newbalanceOrig': 160296.36,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 0.0,
    # 'isFraud' and 'isFlaggedFraud' are target columns -> not needed for prediction
}])


# Encode 'type' column (dummy variables)
input_df = pd.get_dummies(input_df, columns=["type"], drop_first=True)

# Make sure all expected columns exist (if some types are missing)
# Define the features manually
# Encode 'type' column (dummy variables)

# Manually define columns used in training
expected_cols = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest', 
    'type_CASH_OUT', 'type_CASH_IN', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# Add missing columns
for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[expected_cols]

# Scale features
input_scaled = scaler.transform(input_df)


# ------------------------------
# Predict Fraud
# ------------------------------
if st.button("Predict Fraud"):
    prob = rf_model.predict_proba(input_scaled)[0][1]
    prediction = rf_model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Probability: {prob:.2f}")
    else:
        st.success(f"✅ Transaction Legitimate. Fraud Probability: {prob:.2f}")

# ------------------------------
# Optional: Show feature importance
# ------------------------------
if st.checkbox("Show Feature Importance"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fi = pd.DataFrame({
        "Feature": rf_model.feature_names_in_,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=fi, palette="viridis", ax=ax)
    ax.set_title("Top 15 Feature Importances")
    st.pyplot(fig)
