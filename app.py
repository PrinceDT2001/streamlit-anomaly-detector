import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- MODEL TRAINING AND SAVING (Run this part first) ---

def train_and_save_model():
    """Trains the model and saves the model artifacts."""
    st.write("Training model and saving artifacts...")
    
    # Create dummy data for demonstration (you would load your real data here)
    data = {'temperature': np.random.rand(100) * 50,
            'pressure': np.random.rand(100) * 100,
            'vibration': np.random.rand(100) * 10,
            'humidity': np.random.rand(100) * 50,
            'equipment': np.random.choice(['A', 'B', 'C'], 100),
            'location': np.random.choice(['North', 'South', 'Central'], 100),
            'faulty': np.random.randint(0, 2, 100)}
    df = pd.DataFrame(data)

    # Prepare data for training
    X = df.drop('faulty', axis=1)
    y = df['faulty']
    
    categorical_features = ['equipment', 'location']
    numerical_features = ['temperature', 'pressure', 'vibration', 'humidity']
    
    # Perform one-hot encoding
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # 💡 IMPORTANT: Save the exact feature names after one-hot encoding
    feature_names = X.columns.tolist()
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    # Split data and scale numerical features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model and scaler
    joblib.dump(model, 'best_random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    st.write("Model training complete. Files saved: best_random_forest_model.pkl, scaler.pkl, feature_names.json")

# 💡 Make sure this training function is called before your app runs
# You would typically run this once in a separate script.
# For this "start-afresh" example, we will include a button to train the model.
# In a real app, you would train your model offline and upload the saved files.
# if not all([os.path.exists(f) for f in ['best_random_forest_model.pkl', 'scaler.pkl', 'feature_names.json']]):
#     train_and_save_model()


# --- STREAMLIT APP LOGIC (Run this part on every app load) ---

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('best_random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading resources: {e}. Please ensure 'best_random_forest_model.pkl', 'scaler.pkl', and 'feature_names.json' are in the project directory.")
        st.stop()

# Load the resources at the start of the app
model, scaler, feature_names = load_resources()

# Set up the Streamlit app layout
st.set_page_config(page_title="Equipment Anomaly Detector", layout="wide")
st.title("⚙️ Equipment Anomaly Detector")
st.markdown("""
This application uses a trained machine learning model to predict whether a piece of equipment is faulty based on its sensor readings.
""")

# Create input widgets for the user
st.sidebar.header("Input Sensor Readings")
temperature = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
pressure = st.sidebar.slider("Pressure (psi)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
vibration = st.sidebar.slider("Vibration (mm/s)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
equipment = st.sidebar.selectbox("Equipment Type", options=['A', 'B', 'C'])
location = st.sidebar.selectbox("Location", options=['North', 'South', 'Central'])

# Make a prediction when the user clicks a button
if st.button("Predict Anomaly"):
    # Create a DataFrame from the user input
    user_input_dict = {
        'temperature': [temperature], 'pressure': [pressure], 'vibration': [vibration],
        'humidity': [humidity],
        'equipment_B': [1 if equipment == 'B' else 0],
        'equipment_C': [1 if equipment == 'C' else 0],
        'location_North': [1 if location == 'North' else 0],
        'location_South': [1 if location == 'South' else 0],
    }
    user_input_df = pd.DataFrame(user_input_dict)

    # 💡 RE-INDEX: Reorder the columns to match the training data
    user_input_df = user_input_df.reindex(columns=feature_names, fill_value=0)

    st.subheader("Sensor Readings Provided:")
    st.dataframe(user_input_df, use_container_width=True)
    
    # Scale the numerical features
    scaled_user_input = user_input_df.copy()
    numerical_features = ['temperature', 'pressure', 'vibration', 'humidity']
    scaled_user_input[numerical_features] = scaler.transform(scaled_user_input[numerical_features])

    # Now, make a prediction using the re-indexed data
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