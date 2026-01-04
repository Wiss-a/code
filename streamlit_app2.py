import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load trained artifacts
# ----------------------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_cols = pickle.load(open("feature_cols.pkl", "rb"))

# ----------------------------
# Streamlit app UI
# ----------------------------
st.title("💳 Fraud Detection Demo")
st.write("This app predicts whether a financial transaction is likely Fraud or Legitimate.")

# ----------------------------
# Buttons for demo scenarios
# ----------------------------
demo_choice = st.radio("Choose a demo transaction:", ["Fraud", "Legitimate"])

# ----------------------------
# Define demo transactions
# ----------------------------
if demo_choice == "Fraud":
    input_df = pd.DataFrame([{
        'step': 1,
        'amount': 5000.0,           # large amount
        'oldbalanceOrg': 200.0,     # low origin balance
        'newbalanceOrig': 0.0,      # drained balance
        'oldbalanceDest': 0.0,
        'newbalanceDest': 5000.0,   # suddenly increased
        'type': 'TRANSFER'
    }])
else:  # Legitimate transaction
    input_df = pd.DataFrame([{
        'step': 1,
        'amount': 100.0,            # small normal amount
        'oldbalanceOrg': 500.0,
        'newbalanceOrig': 400.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 100.0,
        'type': 'PAYMENT'
    }])

st.write("### Input Transaction")
st.dataframe(input_df)

# ----------------------------
# Preprocess input
# ----------------------------
# Dummy encode
input_df = pd.get_dummies(input_df, columns=['type'], drop_first=True)

# Add missing columns from training
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Keep only the training columns
input_df = input_df[feature_cols]

# Scale input
input_scaled = scaler.transform(input_df)

# ----------------------------
# Prediction with threshold
# ----------------------------
pred_prob = rf_model.predict_proba(input_scaled)[0, 1]


# Very low threshold for demo
threshold = 0.002  # 0.2%
pred_class = 1 if pred_prob >= threshold else 0

st.write(f"Fraud Probability: {pred_prob*100:.2f}%")
st.write(f"Predicted Class: {'Fraud 🚨' if pred_class==1 else 'Legitimate ✅'}")

