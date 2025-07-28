import streamlit as st
import pandas as pd
import pickle

# Load the pipeline using pickle (no joblib)
with open("churn_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# List of features used during model training
feature_order = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'Contract', 'PaperlessBilling', 'MonthlyCharges',
    'TotalCharges', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Helper function to convert "Yes"/"No" to 1/0
def yes_no_to_binary(choice):
    return 1 if choice == "Yes" else 0

st.title("📊 Customer Churn Predictor")

with st.form("churn_form"):
    st.markdown("### 🧾 Enter Customer Details")

    gender = st.radio("Gender", ["Female", "Male"])
    gender = 0 if gender == "Female" else 1

    SeniorCitizen = yes_no_to_binary(st.radio("Senior Citizen?", ["No", "Yes"]))
    Partner = yes_no_to_binary(st.radio("Has Partner?", ["No", "Yes"]))
    Dependents = yes_no_to_binary(st.radio("Has Dependents?", ["No", "Yes"]))
    tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100)
    PhoneService = yes_no_to_binary(st.radio("Phone Service?", ["No", "Yes"]))
    
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    Contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[Contract]
    
    PaperlessBilling = yes_no_to_binary(st.radio("Paperless Billing?", ["No", "Yes"]))
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0)

    st.markdown("### ⚙️ Service Options")
    ohe_inputs = {}
    for col in feature_order[10:]:
        label = col.replace("_", " ").replace("No internet service", "No Internet").replace("No phone service", "No Phone")
        ohe_inputs[col] = yes_no_to_binary(st.radio(f"{label}?", ["No", "Yes"], key=col))

    submit = st.form_submit_button("🔍 Predict")

if submit:
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        **ohe_inputs
    }

    input_df = pd.DataFrame([[input_data[col] for col in feature_order]], columns=feature_order)

    try:
        prediction = pipeline.predict(input_df)[0]

        st.subheader("📢 Prediction Result")
        if prediction == 1:
            st.error("⚠️ The customer is **likely to CHURN**.")
        else:
            st.success("✅ The customer is **NOT likely to churn**.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

