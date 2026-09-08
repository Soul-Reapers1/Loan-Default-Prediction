import streamlit as st
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("loan_pipeline.pkl")

# Page config
st.set_page_config(page_title="Loan Predictor", page_icon="💳")

st.title("💳 Loan Default Prediction")

# Load sample only once
if "sample" not in st.session_state:
    st.session_state.sample = pd.read_csv("loan.csv").iloc[0].to_dict()

# 🔥 Clean text inputs (no +/- buttons)
loan_amnt = st.text_input("💰 Loan Amount", placeholder="Enter loan amount")
annual_inc = st.text_input("📊 Annual Income", placeholder="Enter annual income")
int_rate = st.text_input("📈 Interest Rate (%)", placeholder="Enter interest rate")

# Predict button
if st.button("Predict"):

    try:
        # Convert inputs to float
        loan_amnt = float(loan_amnt)
        annual_inc = float(annual_inc)
        int_rate = float(int_rate)

        sample = st.session_state.sample.copy()

        # Update values
        sample['loan_amnt'] = loan_amnt
        sample['annual_inc'] = annual_inc
        sample['int_rate'] = int_rate

        # Predict
        df = pd.DataFrame([sample])
        prob = pipeline.predict_proba(df)[:, 1][0]

        # Result
        if prob >= 0.15:
            st.error(f"❌ Loan Rejected\n\nDefault Probability: {prob:.2f}")
        else:
            st.success(f"✅ Loan Approved\n\nDefault Probability: {prob:.2f}")

    except ValueError:
        st.warning("⚠️ Please enter valid numeric values")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning")