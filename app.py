import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Завантажуємо збережену модель та скейлер
model = load_model("churn_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction")
st.write("Введіть дані клієнта для прогнозу відтоку:")

# Вводимо ключові числові ознаки
tenure = st.number_input("Tenure (місяці)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# Вводимо категоріальні ознаки
contract_options = ["Month-to-month", "One year", "Two year"]
contract = st.selectbox("Contract Type", contract_options)

internet_options = ["DSL", "Fiber optic", "No"]
internet_service = st.selectbox("Internet Service", internet_options)

payment_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
payment_method = st.selectbox("Payment Method", payment_options)

# ❗ Додаємо ВСІ ознаки, які були у початковому тренувальному датасеті
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],

    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],

    "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
    "InternetService_No": [1 if internet_service == "No" else 0],

    "PaymentMethod_Credit card (automatic)": [1 if payment_method == "Credit card (automatic)" else 0],
    "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
    "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0],

    # Додаємо ВСІ категоріальні змінні, які були у початковому датасеті
    "Dependents": [0],  # Якщо не хочеш вводити - ставимо дефолтне значення
    "DeviceProtection_No internet service": [0],
    "DeviceProtection_Yes": [0],
    "MultipleLines_No phone service": [0],
    "MultipleLines_Yes": [0]
})

# 🔹 Переконуємось, що input_data має ВСІ необхідні колонки
expected_features = scaler.feature_names_in_
missing_features = set(expected_features) - set(input_data.columns)
for feature in missing_features:
    input_data[feature] = 0  # Додаємо відсутні колонки з нульовими значеннями

# 🔹 Перевіряємо, що всі ознаки є у правильному порядку
input_data = input_data[expected_features]

if st.button("Predict Churn"):
    # Масштабуємо вхідні дані
    input_data_scaled = scaler.transform(input_data)

    # Отримуємо прогноз
    prediction_prob = model.predict(input_data_scaled)[0][0]
    prediction = "Churn" if prediction_prob > 0.5 else "No Churn"

    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Probability:** {prediction_prob:.2f}")
