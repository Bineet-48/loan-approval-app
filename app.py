import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("loan_model_compressed.pkl")
expected_columns = list(model.feature_names_in_)

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("💰 Loan Approval Predictor")

with st.expander("🔍 About This App", expanded=False):
    st.write("""
    This app predicts whether a loan application will be **approved or denied** based on applicant financial and personal details.
    """)

# Sidebar Inputs
st.sidebar.header("📋 Applicant Information")

person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, value=50000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
person_emp_length = st.sidebar.slider("Employment Length (Years)", 0, 30, 5)

loan_intent = st.sidebar.selectbox("Loan Purpose", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.sidebar.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=1000, value=10000)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
loan_percent_income = st.sidebar.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2)

cb_person_default_on_file = st.sidebar.selectbox("Previous Default?", ['Y', 'N'])
cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (Years)", 1, 30, 5)

if st.button("🚀 Predict Loan Approval"):
    input_dict = {
        'person_age': [person_age],
        'person_income': [person_income],
        'person_emp_length': [person_emp_length],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],

        'person_home_ownership_MORTGAGE': [1 if person_home_ownership == 'MORTGAGE' else 0],
        'person_home_ownership_OWN': [1 if person_home_ownership == 'OWN' else 0],
        'person_home_ownership_OTHER': [1 if person_home_ownership == 'OTHER' else 0],

        'loan_intent_EDUCATION': [1 if loan_intent == 'EDUCATION' else 0],
        'loan_intent_HOMEIMPROVEMENT': [1 if loan_intent == 'HOMEIMPROVEMENT' else 0],
        'loan_intent_MEDICAL': [1 if loan_intent == 'MEDICAL' else 0],
        'loan_intent_VENTURE': [1 if loan_intent == 'VENTURE' else 0],
        'loan_intent_DEBTCONSOLIDATION': [1 if loan_intent == 'DEBTCONSOLIDATION' else 0],

        'loan_grade_B': [1 if loan_grade == 'B' else 0],
        'loan_grade_C': [1 if loan_grade == 'C' else 0],
        'loan_grade_D': [1 if loan_grade == 'D' else 0],
        'loan_grade_E': [1 if loan_grade == 'E' else 0],
        'loan_grade_F': [1 if loan_grade == 'F' else 0],
        'loan_grade_G': [1 if loan_grade == 'G' else 0],

        'cb_person_default_on_file_Y': [1 if cb_person_default_on_file == 'Y' else 0],
    }

    input_df = pd.DataFrame(input_dict)

    # Add any missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    with st.spinner("Making prediction..."):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

    result = "✅ **Loan Approved**" if prediction == 1 else "❌ **Loan Denied**"
    st.success(result)
    st.progress(int(proba * 100))
    st.write(f"**Confidence Score:** {proba:.2%}")
