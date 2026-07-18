import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("LR_ford_car.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

st.title("Ford Car Price Predictor")
st.write("Enter the car details below to predict its selling price.")

year = st.number_input(
    "Manufacturing Year", 
    min_value=2000, 
    max_value=2026, 
    value=2018
    )

Mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=50000,
    value=5000
)

tax = st.number_input(
    "Road Tax (tax)",
    min_value=0,
    max_value=200,
    value=100
)

mpg = st.number_input(
    "MPG",
    min_value=10,
    max_value=100,
    value=55
)

engine = st.number_input(
    "Engine Size",
    min_value=1,
    max_value=7,
    value=2
)

transmission = st.selectbox(
    "Transmission",
    [
        "Automatic",
        "Manual",
        "Semi-Auto"
    ]
)

fuel_type = st.selectbox(
    "Fuel Type",
    [
        "Petrol",
        "Diesel",
        "Hybrid",
        "Other"
    ]
)

car_model = st.text_input("Enter Car Model Name...")

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [Mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    st.write("Encoded Data:")
    st.write(input_data)

    numerical_columns = ["year", "mileage", "tax", "mpg", "engineSize"]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: £{prediction[0]:,.2f}")