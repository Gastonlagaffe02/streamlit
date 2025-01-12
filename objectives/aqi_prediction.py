import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
import numpy as np

def run():
    # Load the saved models, PolynomialFeatures, and scalers
    poly_model = joblib.load(os.path.join("models", "polynomial_model_degree_2.pkl"))
    poly_features = joblib.load(os.path.join("models", "poly_features_degree_2.pkl"))
    scaler_poly = joblib.load(os.path.join("models", "scaler.pkl"))  # Load the saved scaler for polynomial regression

    # Load the decision tree model and scaler
    decision_tree_model = joblib.load(os.path.join("models", "best_decision_tree_model.pkl"))
    scaler_decision_tree = joblib.load(os.path.join("models", "scaler_decision_tree.pkl"))  # The scaler used for decision tree
    label_encoder = joblib.load(os.path.join("models", "label_encoder.pkl"))  # The label encoder for AQI Category

    # Load the neural network model and scaler
    model_mohamed = load_model(os.path.join("models", "nn_model_mohamed.h5"))
    scaler_mohamed = joblib.load(os.path.join("models", "scaler_nn_mohamed.pkl"))

    # Title and description
    st.title("Air Quality Prediction")
    st.write("""
    This app predicts air quality levels (AQI value or AQI Category) based on air pollutants using either a Polynomial Regression, Decision Tree, or Neural Network model.
    Upload your data or input feature values manually for predictions.
    """)

    # Sidebar for selecting model and input method
    st.sidebar.header("Input Options")
    model_type = st.sidebar.selectbox("Choose the prediction model:", ["Polynomial Regression", "Decision Tree", "Neural Network"])
    input_type = st.sidebar.selectbox("How would you like to provide input data?", ["Manual Input", "Upload File"])

    # Display input fields based on selected model type
    if model_type == "Neural Network":
        # Neural Network Model inputs with 6 decimal places
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        no = st.number_input("NO (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        nox = st.number_input("NOx (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        co = st.number_input("CO (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        benzene = st.number_input("Benzene (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        toluene = st.number_input("Toluene (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        xylene = st.number_input("Xylene (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        city = st.number_input("City", max_value=2100, step=1)
        year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
        day = st.number_input("Day", min_value=1, max_value=31, step=1)
        month = st.number_input("Month", min_value=1, max_value=12, step=1)

        # Combine features into a dataframe
        input_data = pd.DataFrame([[pm25, pm10, no, no2, nox, nh3, co, so2, ozone, benzene, toluene, xylene, city, year, month, day]],
                                  columns=["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "City", "Year", "Month", "Day"])

    else:
        # Previous input form for Polynomial Regression or Decision Tree models with more decimal precision
        pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)
        ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, format="%.6f", step=0.000001)

        # Combine features into a dataframe
        input_data = pd.DataFrame([[pm25, pm10, ozone]], columns=["PM2.5", "PM10", "Ozone"])

    # Scale the input data according to the selected model type
    if input_type == "Manual Input":
        if model_type == "Neural Network":
            input_data_scaled = scaler_mohamed.transform(input_data)  # Use scaler for the NN model
        else:
            input_data_scaled = scaler_poly.transform(input_data)  # Use scaler for Polynomial Regression

        st.write("Your input data:")
        st.write(input_data)

    # Upload a CSV file
    else:
        uploaded_file = st.file_uploader("Upload a CSV file with the required features")
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)

            # Drop the 'AQI value' column if it exists (because it's the target for predictions)
            if 'AQI value' in input_data.columns:
                input_data = input_data.drop(columns=['AQI value'])
            
            st.write("Uploaded data:")
            st.write(input_data)

            # Scale the input data according to the selected model type
            if model_type == "Neural Network":
                input_data_scaled = scaler_mohamed.transform(input_data)  # Use scaler for the NN model
            else:
                input_data_scaled = scaler_poly.transform(input_data)  # Use scaler for Polynomial Regression

        else:
            input_data = None

    # Perform predictions
    if input_data is not None:
        try:
            if model_type == "Polynomial Regression":
                # Predict using the polynomial regression model
                input_data_poly = poly_features.transform(input_data_scaled)
                predictions = poly_model.predict(input_data_poly)

                # Display predictions
                st.write("Predicted AQI Values (Polynomial Regression):")
                st.write(predictions[:5])  # Show only first 5 predictions to avoid large output

            elif model_type == "Decision Tree":
                # Scale input data using the Decision Tree scaler
                input_data_scaled = scaler_decision_tree.transform(input_data)

                # Predict using the decision tree model
                predictions = decision_tree_model.predict(input_data_scaled)

                # Decode the predictions (AQI Category) if necessary
                decoded_predictions = label_encoder.inverse_transform(predictions)

                # Display predictions
                st.write("Predicted AQI Categories (Decision Tree):")
                st.write(decoded_predictions[:5])  # Show only first 5 predictions to avoid large output

            elif model_type == "Neural Network":
                # Predict using the neural network model
                predictions = model_mohamed.predict(input_data_scaled)

                # Display predictions
                st.write("Predicted AQI Values (Neural Network):")
                st.write(predictions[:5])  # Show only first 5 predictions to avoid large output

        except Exception as e:
            st.error(f"Error in prediction: {e}")
