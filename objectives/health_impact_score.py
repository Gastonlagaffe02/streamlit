import streamlit as st
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model

def run():
    # Load the saved models and scalers for health impact score prediction
    poly_model_health_impact = joblib.load(os.path.join("models", "poly_model_health_impact.pkl"))
    scaler_health_impact = joblib.load(os.path.join("models", "scaler_health_impact.pkl"))

    # Load the neural network model and scaler
    nn_model = load_model(os.path.join("models", "nn_model.h5"))
    scaler_nn = joblib.load(os.path.join("models", "scaler_nn.pkl"))

    # Load the Random Forest Classifier model
    classification_model = joblib.load(os.path.join("models", "classification_model1.pkl"))  # New classifier model

    # Title and description
    st.title("Health Impact Score and Risk Classification Prediction")
    st.write("""
    This app predicts:
    - The Health Impact Score using Polynomial Regression or Neural Network models.
    - The Health Risk Classification using a Random Forest Classifier model.
    You can either upload your data or input feature values manually for predictions.
    """)

    # Sidebar for selecting model and input method
    st.sidebar.header("Input Options")
    model_type = st.sidebar.selectbox("Choose the prediction model:", 
                                      ["Polynomial Regression", "Neural Network", "Random Forest Classifier"])
    input_type = st.sidebar.selectbox("How would you like to provide input data?", ["Manual Input", "Upload File"])

    # Input data manually
    if input_type == "Manual Input":
        if model_type in ["Polynomial Regression", "Neural Network"]:
            # Input for health impact score prediction
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, step=0.1)
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, step=0.1)
            ozone = st.number_input("Ozone (µg/m³)", min_value=0.0, step=0.1)
            no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, step=0.1)
            so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, step=0.1)
            co = st.number_input("CO (µg/m³)", min_value=0.0, step=0.1)
            temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
            population_density = st.number_input("Population Density (people/km²)", min_value=0.0, step=1.0)

            input_data = pd.DataFrame([[pm25, pm10, ozone, no2, so2, co, temperature, humidity, population_density]],
                                      columns=["PM2.5", "PM10", "Ozone", "NO2", "SO2", "CO", "Temperature", "Humidity", "Population Density"])

        elif model_type == "Random Forest Classifier":
            # Input for classification model
            AQI = st.number_input("AQI", min_value=0.0, step=0.1)
            PM10 = st.number_input("PM10 (µg/m³)", min_value=0.0, step=0.1)
            PM2_5 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, step=0.1)
            NO2 = st.number_input("NO2 (µg/m³)", min_value=0.0, step=0.1)
            SO2 = st.number_input("SO2 (µg/m³)", min_value=0.0, step=0.1)
            O3 = st.number_input("O3 (µg/m³)", min_value=0.0, step=0.1)
            Temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
            Humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
            WindSpeed = st.number_input("Wind Speed (km/h)", min_value=0.0, step=0.1)
            RespiratoryCases = st.number_input("Respiratory Cases", min_value=0, step=1)
            CardiovascularCases = st.number_input("Cardiovascular Cases", min_value=0, step=1)
            HospitalAdmissions = st.number_input("Hospital Admissions", min_value=0, step=1)

            # Combine input into a DataFrame
            input_data = pd.DataFrame([[AQI, PM10, PM2_5, NO2, SO2, O3, Temperature, Humidity, WindSpeed, RespiratoryCases, CardiovascularCases, HospitalAdmissions]],
                                    columns=['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity', 'WindSpeed', 'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions'])


        st.write("Your input data:")
        st.write(input_data)

    # Upload a CSV file
    else:
        uploaded_file = st.file_uploader("Upload a CSV file with the required features for the selected model.")
        if uploaded_file:
            input_data = pd.read_csv(uploaded_file)
            st.write("Uploaded data:")
            st.write(input_data)
        else:
            input_data = None

    # Perform predictions
    if input_data is not None:
        try:
            if model_type in ["Polynomial Regression", "Neural Network"]:
                input_data_scaled = scaler_health_impact.transform(input_data) if model_type == "Polynomial Regression" else scaler_nn.transform(input_data)

                if model_type == "Polynomial Regression":
                    predictions = poly_model_health_impact.predict(input_data_scaled)
                    st.write("Predicted Health Impact Scores (Polynomial Regression):")
                    st.write(predictions[:5])

                elif model_type == "Neural Network":
                    predictions = nn_model.predict(input_data_scaled)
                    st.write("Predicted Health Impact Scores (Neural Network):")
                    st.write(predictions[:5])

            elif model_type == "Random Forest Classifier":
                predictions = classification_model.predict(input_data)

                # Display predictions
                st.write("Predicted Health Impact Class (Random Forest Classifier):")
                st.write(predictions)



        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    run()
