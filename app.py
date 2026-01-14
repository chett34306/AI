import streamlit as st
import tensorflow as tf
import numpy as np

# Load the model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('house_model.h5')

model = load_my_model()

st.title("🏠 House Price Predictor")
st.write("Enter the number of bedrooms to estimate the price.")

# User input
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=3)

if st.button("Predict Price"):
    # Make prediction
    prediction = model.predict(np.array([[float(bedrooms)]]))[0][0]
    # Convert back from 100ks to dollars
    final_price = prediction * 100000

    st.success(f"Estimated Price: ${final_price:,.2f}")