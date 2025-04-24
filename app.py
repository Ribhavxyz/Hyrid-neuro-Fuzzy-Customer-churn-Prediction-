import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the model
model = load_model("model/churn_model.h5")

# Define all expected feature columns in the correct order
expected_columns = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain',
    'Gender_Female', 'Gender_Male'
]

# Streamlit UI
st.title("Customer Churn Prediction - Hybrid Neuro-Fuzzy Model")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", value=20000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):
    # Initial 13 base features
    base_input = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if geography == "France" else 0,
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Gender_Male': 1 if gender == "Male" else 0
    }

    # Create DataFrame with all expected features in order
    input_df = pd.DataFrame([base_input], columns=expected_columns)

    # Convert to numpy array
    input_data = input_df.to_numpy()

    # Predict
    prediction = model.predict(input_data)

    # Display result
    result = "Customer will churn." if prediction[0][0] > 0.5 else "Customer will stay."
    st.success(f"Prediction: {result}")
