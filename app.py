import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import base64

# -----------------------
# Load the trained model
# -----------------------
model = joblib.load("loan_model_compressed.pkl")
expected_columns = list(model.feature_names_in_)

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰")

# -----------------------
# Theme Toggle
# -----------------------
theme = st.sidebar.radio("🌗 Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #0E1117; color: white; }
            .stApp { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# -----------------------
# Title and About
# -----------------------
st.title("💰 Loan Approval Predictor")

with st.expander("🔍 About This App"):
    st.write("""
    This Streamlit app uses a machine learning model to predict whether a loan application is likely to be approved or denied based on key applicant details.
    
    It is built using:
    - 🧠 Scikit-learn (RandomForest)
    - 📦 Joblib for model loading
    - 🎈 Streamlit for UI
    """)

# -----------------------
# Sidebar Form
# -----------------------
with st.sidebar.form("input_form"):
    st.markdown("### 📋 Enter Applicant Information")

    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Annual Income ($)", min_value=1000, value=50000)
    person_home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    person_emp_length = st.slider("Employment Length (Years)", 0, 30, 5)
    loan_intent = st.selectbox("Loan Purpose", ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, value=10000)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
    loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2)
    cb_person_default_on_file = st.selectbox("Previous Default?", ['Y', 'N'])
    cb_person_cred_hist_length = st.slider("Credit History Length (Years)", 1, 30, 5)

    submitted = st.form_submit_button("🔮 Predict Loan Approval")

# -----------------------
# Prediction Logic
# -----------------------
if submitted:
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

    # Add any missing columns (in case of model training mismatches)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    # Result
    result_label = "✅ **Loan Approved**" if prediction == 1 else "❌ **Loan Denied**"
    st.subheader(result_label)
    st.write(f"**Confidence Score:** {proba:.2%}")

    # -----------------------
    # PDF Report Generation
    # -----------------------
    def create_pdf(result: str, score: float) -> str:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Loan Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Result: {result}", ln=True)
        pdf.cell(200, 10, txt=f"Confidence Score: {score:.2%}", ln=True)
        filename = "loan_report.pdf"
        pdf.output(filename)
        return filename

    def download_pdf_button(file_path: str):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="Loan_Report.pdf">📄 Download Report as PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

    pdf_file = create_pdf("Loan Approved" if prediction == 1 else "Loan Denied", proba)
    download_pdf_button(pdf_file)
