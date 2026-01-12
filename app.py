import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('house_model.pkl')

st.title("House Price Predictor")
st.write("Enter the details below to estimate the house price.")

# Create input fields for user data
sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=1500)
bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)

# Prediction logic
if st.button("Predict Price"):
    features = np.array([[sqft, bedrooms, bathrooms]])
    prediction = model.predict(features)
    st.success(f"The estimated price is ${prediction[0]:,.2f}")
