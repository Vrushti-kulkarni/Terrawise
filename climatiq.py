import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIMATIQ_API_KEY = os.getenv("CLIMATIQ_API_KEY")
CLIMATIQ_BASE_URL = "https://api.climatiq.io/estimate"


HEADERS = {
    "Authorization": f"Bearer {CLIMATIQ_API_KEY}",
    "Content-Type": "application/json"
}

def calculate_emissions(activity_id, parameters):
    url = f"{CLIMATIQ_BASE_URL}/estimate"
    data = {
        "emission_factor": {"activity_id": activity_id},
        "parameters": parameters
    }

    response = requests.post(url, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}


import streamlit as st

st.set_page_config(page_title="Terrawise Emissions Estimator")

st.title("🌱 Terrawise Carbon Emissions Estimator")

# Input section
st.header("Estimate Electricity Emissions")
country = st.selectbox("Select Country", ["IN", "US", "UK", "DE", "FR"])
electricity_unit = st.selectbox("Electricity Unit", ["kWh"])
amount = st.number_input("Electricity Used (kWh)", min_value=0.0, value=5.0)

if st.button("Calculate Emissions"):
    activity_id = f"electricity-energy_source_grid_mix_country_{country.lower()}_renewable_mix"
    params = {"energy": amount, "energy_unit": electricity_unit}

    result = calculate_emissions(activity_id, params)

    if "co2e" in result:
        st.success(f"Estimated Emissions: {result['co2e']} kg CO₂e")
        st.caption(f"Confidence: {result['co2e_confidence']}")
    else:
        st.error("Error calculating emissions")
        st.text(result.get("error", "Unknown error occurred."))
