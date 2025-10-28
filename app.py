import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# -------------------------#
# Page setup
# -------------------------#
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")

MODEL_PATH = "churn_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}\nPlease make sure churn_pipeline.pkl is in the same folder as app.py")
    st.stop()

model = joblib.load(MODEL_PATH)
st.success("✅ Model loaded successfully!")

st.markdown("### Enter Customer Details Below")

# -------------------------#
# Input Section
# -------------------------#
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], help="1 = Yes, 0 = No")
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=300, value=12)
    PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
    MultipleLines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
    PaymentMethod = st.selectbox("PaymentMethod", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=10000.0, value=70.0, format="%.2f")
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=700.0, format="%.2f")

    submitted = st.form_submit_button("🔍 Predict")

# -------------------------#
# Prediction Section
# -------------------------#
if submitted:
    # Build DataFrame
    input_dict = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure SeniorCitizen is numeric
    if 'SeniorCitizen' in input_df.columns and input_df['SeniorCitizen'].dtype == 'object':
        input_df['SeniorCitizen'] = input_df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    st.write("### Input Preview")
    st.dataframe(input_df)

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error(f"🚨 The customer is **likely to CHURN**.\nProbability: {prob:.2%}")
    else:
        st.success(f"💚 The customer is **likely to STAY**.\nProbability: {prob:.2%}")

