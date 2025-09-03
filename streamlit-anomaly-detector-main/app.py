import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the trained model
try:
    # Use st.cache_resource to cache the model loading
    @st.cache_resource
    def load_model():
        # Load the model and scaler
        model = joblib.load('best_random_forest_model.pkl')
        # We need the scaler used during training
        # If you saved the scaler, load it here. Otherwise, you may need to recreate it.
        # For simplicity, let's assume you've saved it as a separate file
        scaler = joblib.load('scaler.pkl')
        return model, scaler

    model, scaler = load_model()

except FileNotFoundError:
    st.error("Error: The model file 'best_random_forest_model.pkl' or scaler file 'scaler.pkl' was not found.")
    st.stop()

# Load the feature names from your training data
# It's important to have the features in the correct order for the model
# You can save the list of feature names during training and load it here.
# For example, save the columns of your training set X_train
# X_train.columns.tolist() -> save to a file
# For this example, we'll hardcode them based on your provided code
feature_names = [
    'temperature', 'pressure', 'vibration', 'humidity',
    'equipment_B', 'equipment_C', 'location_North', 'location_South'
]

# Set up the Streamlit app layout
st.set_page_config(page_title="Equipment Anomaly Detector", layout="wide")

st.title("⚙️ Equipment Anomaly Detector")
st.markdown("""
This application uses a trained machine learning model to predict whether a piece of equipment is faulty based on its sensor readings.
""")

# Create input widgets for the user
st.sidebar.header("Input Sensor Readings")

# Collect user input for numerical features
temperature = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
pressure = st.sidebar.slider("Pressure (psi)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
vibration = st.sidebar.slider("Vibration (mm/s)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.1)

# Collect user input for categorical features
equipment = st.sidebar.selectbox("Equipment Type", options=['A', 'B', 'C'])
location = st.sidebar.selectbox("Location", options=['North', 'South', 'Central'])

# Create a dictionary to hold user input
user_input_dict = {
    'temperature': temperature,
    'pressure': pressure,
    'vibration': vibration,
    'humidity': humidity
}

# Add one-hot encoded categorical features
user_input_dict['equipment_B'] = 1 if equipment == 'B' else 0
user_input_dict['equipment_C'] = 1 if equipment == 'C' else 0
user_input_dict['location_North'] = 1 if location == 'North' else 0
user_input_dict['location_South'] = 1 if location == 'South' else 0

# Ensure the features are in the same order as during training
user_input = pd.DataFrame([user_input_dict], columns=feature_names)

st.subheader("Sensor Readings Provided:")
st.dataframe(user_input, use_container_width=True)

# Scale the numerical features
numerical_features = ['temperature', 'pressure', 'vibration', 'humidity']
scaled_user_input = user_input.copy()
scaled_user_input[numerical_features] = scaler.transform(scaled_user_input[numerical_features])

# Make a prediction when the user clicks a button
if st.button("Predict Anomaly"):
    # Make prediction using the loaded model
    prediction = model.predict(scaled_user_input)[0]
    prediction_proba = model.predict_proba(scaled_user_input)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("🚨 **Prediction: Equipment is likely FAULTY**")
        st.write(f"Confidence (Probability of Fault): **{prediction_proba:.2f}**")
        st.info("Based on the readings, the model suggests that this equipment is experiencing an anomaly. Immediate inspection is recommended.")
    else:
        st.success("✅ **Prediction: Equipment is NORMAL**")
        st.write(f"Confidence (Probability of Fault): **{prediction_proba:.2f}**")
        st.info("The sensor readings appear to be within the normal range. Continuous monitoring is advised.")