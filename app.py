import streamlit as st
import objectives.aqi_prediction as aqi
import objectives.health_impact_score as health
import objectives.death as death
# Display home page and objective selection
st.title("Air Pollution Prediction App")
st.write("""
Welcome to the Air Pollution Prediction App! Select an objective below to start:
""")

# Sidebar for selecting objective
objective = st.sidebar.selectbox("Choose an Objective", ["AQI Prediction", "Health Impact Score Prediction" ,"Deaths Prediction"])

# Navigate to the corresponding objective's app
if objective == "AQI Prediction":
    aqi.run()  # This calls the run() function from aqi_prediction.py
elif objective == "Health Impact Score Prediction":
    health.run()  # This calls the run() function from aqi_prediction.py
elif objective == "Deaths Prediction":
    death.run()  # This calls the run() function from aqi_prediction.py
    pass
