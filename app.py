import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------------
# Title
# -------------------------------
st.title("🏦 Loan Approval Prediction System")
st.write(
    "Enter the applicant details below to predict whether the loan "
    "application is likely to be approved."
)

st.divider()

# -------------------------------
# Input Section
# -------------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=500000,
        step=10000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000,
        value=300000,
        step=1000
    )

with col2:

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        700
    )

    num_dependents = st.selectbox(
        "Number of Dependents",
        [0,1,2,3,4,5]
    )

    existing_loans_count = st.selectbox(
        "Existing Loans Count",
        [0,1,2,3,4,5]
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["Unemployed","Self Employed","Salaried"]
    )

st.divider()

# -------------------------------
# Encoding
# -------------------------------

gender = 1 if gender == "Male" else 0

marital_status = 1 if marital_status == "Married" else 0

employment_dict = {
    "Unemployed":0,
    "Self Employed":1,
    "Salaried":2
}

employment_status = employment_dict[employment_status]

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Loan Status", use_container_width=True):

    input_data = np.array([[
        age,
        gender,
        marital_status,
        annual_income,
        loan_amount,
        credit_score,
        num_dependents,
        existing_loans_count,
        employment_status
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("✅ Loan Approved")

        st.metric(
            "Approval Probability",
            f"{probability[1]*100:.2f}%"
        )

    else:

        st.error("❌ Loan Rejected")

        st.metric(
            "Rejection Probability",
            f"{probability[0]*100:.2f}%"
        )

st.divider()

st.caption(
    "Machine Learning Model: AdaBoost Classifier | "
    "Built with Streamlit"
)