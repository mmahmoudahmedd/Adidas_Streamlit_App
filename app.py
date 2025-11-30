import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# 1️⃣ Load model, scaler, and encoders
# =========================
rf = joblib.load("random_forest_sales_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load LabelEncoders
le_cols = ['Region', 'State', 'City', 'Product', 'Sales Method', 'Retailer']
le_dict = {}
for col in le_cols:
    le_dict[col] = joblib.load(f"{col}_encoder.pkl")

# =========================
# 2️⃣ Load dataset for dropdown options
# =========================
df = pd.read_excel("Adidas US Sales Datasets.csv.xlsx", header=4)

# =========================
# 3️⃣ Streamlit App Layout
# =========================
st.title("Adidas US Sales Category Prediction")
st.write("Predict Sales Category (Low / Medium / High) based on inputs")

# Numeric inputs
price = st.number_input("Price per Unit", min_value=0.0, value=50.0)
units = st.number_input("Units Sold", min_value=0, value=100)
operating_profit = st.number_input("Operating Profit", value=1000.0)
operating_margin = st.number_input("Operating Margin", min_value=0.0, max_value=1.0, value=0.2)
year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
month = st.number_input("Month", min_value=1, max_value=12, value=11)
quarter = ((month-1)//3)+1
profit_margin_percent = st.number_input("Profit Margin %", min_value=0.0, max_value=1.0, value=0.2)

# Dropdowns for categorical columns
region = st.selectbox("Region", df['Region'].unique())
state = st.selectbox("State", df['State'].unique())
city = st.selectbox("City", df['City'].unique())
product = st.selectbox("Product", df['Product'].unique())
sales_method = st.selectbox("Sales Method", df['Sales Method'].unique())
retailer = st.selectbox("Retailer", df['Retailer'].unique())

# =========================
# 4️⃣ Predict Button
# =========================
if st.button("Predict Sales Category"):
    # Encode categorical variables
    new_data = pd.DataFrame({
        'Price per Unit': [price],
        'Units Sold': [units],
        'Operating Profit': [operating_profit],
        'Operating Margin': [operating_margin],
        'Year': [year],
        'Month': [month],
        'Quarter': [quarter],
        'Profit Margin %': [profit_margin_percent],
        'Region_encoded': [le_dict['Region'].transform([region])[0]],
        'State_encoded': [le_dict['State'].transform([state])[0]],
        'City_encoded': [le_dict['City'].transform([city])[0]],
        'Product_encoded': [le_dict['Product'].transform([product])[0]],
        'Sales Method_encoded': [le_dict['Sales Method'].transform([sales_method])[0]],
        'Retailer_encoded': [le_dict['Retailer'].transform([retailer])[0]]
    })

    # Scale features
    X_scaled = scaler.transform(new_data)

    # Predict
    prediction = rf.predict(X_scaled)[0]
    st.success(f"Predicted Sales Category: **{prediction}**")
