import streamlit as st
import joblib
import numpy as np
import datetime
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to model relative to script location
model_path = os.path.join(script_dir, "..", "models", "xgboost_model.pkl")
# Normalize the path to handle .. correctly
model_path = os.path.normpath(model_path)

# Load trained model
model = joblib.load(model_path)

# Title & Header
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 House Price Prediction")
st.markdown("""
This application predicts the **median house value** based on housing data.

:information_source: **Note:**
- `Median Income` is scaled: **1 unit = $1,000**. So enter `4.5` for $4,500.
- Inputs should be within realistic housing ranges.
""")

# Collect user input
st.subheader("Enter the features below:")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-118.0, step=0.01)
    latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.2, step=0.01)
    housing_median_age = st.number_input("Median Age of House", min_value=1, max_value=100, value=30)
    total_rooms = st.number_input("Total Rooms", min_value=1, max_value=10000, value=5000)
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, max_value=5000, value=1000)

with col2:
    population = st.number_input("Population in Area", min_value=1, max_value=50000, value=1500)
    households = st.number_input("Number of Households", min_value=1, max_value=10000, value=500)
    median_income = st.number_input("Median Income (1 = $1,000)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
    ocean_proximity = st.selectbox("Ocean Proximity", 
                                   ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND", "<1H OCEAN"],
                                   index=0)

# Calculate all engineered features (same as preprocessing pipeline)
# Avoid division by zero
safe_divide = lambda a, b: a / b if b != 0 else 0

# Basic engineered features
rooms_per_household = safe_divide(total_rooms, households)
bedrooms_per_room = safe_divide(total_bedrooms, total_rooms)
population_per_household = safe_divide(population, households)

# Advanced engineered features
distance_to_center = np.sqrt((longitude - (-118.0))**2 + (latitude - 36.0)**2)
income_per_room = safe_divide(median_income, total_rooms)
income_per_person = safe_divide(median_income, population)
household_density = safe_divide(households, population)
age_log = np.log1p(housing_median_age)
bedroom_ratio = safe_divide(total_bedrooms, households)
income_times_rooms = median_income * rooms_per_household
income_times_age = median_income * housing_median_age

# One-hot encode ocean_proximity (drop_first means <1H OCEAN is reference = all zeros)
ocean_proximity_INLAND = 1 if ocean_proximity == "INLAND" else 0
ocean_proximity_ISLAND = 1 if ocean_proximity == "ISLAND" else 0
ocean_proximity_NEAR_BAY = 1 if ocean_proximity == "NEAR BAY" else 0
ocean_proximity_NEAR_OCEAN = 1 if ocean_proximity == "NEAR OCEAN" else 0

# Prepare input for model - MUST match exact order of 22 features after feature selection
# Features removed: households, age_squared, lat_times_lon (correlation-based removal)
user_data = np.array([[
    longitude,                          # 1
    latitude,                           # 2
    housing_median_age,                 # 3
    total_rooms,                        # 4
    total_bedrooms,                     # 5
    population,                         # 6
    median_income,                      # 7
    rooms_per_household,                # 8
    bedrooms_per_room,                  # 9
    population_per_household,           # 10
    distance_to_center,                 # 11
    income_per_room,                    # 12
    income_per_person,                  # 13
    household_density,                  # 14
    age_log,                            # 15
    bedroom_ratio,                      # 16
    income_times_rooms,                 # 17
    income_times_age,                   # 18
    ocean_proximity_INLAND,             # 19
    ocean_proximity_ISLAND,             # 20
    ocean_proximity_NEAR_BAY,           # 21
    ocean_proximity_NEAR_OCEAN          # 22
]])

# Store prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Predict
if st.button("Predict Price"):
    prediction = model.predict(user_data)
    result = round(prediction[0], 2)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success(f"Estimated House Price: ${result:,.2f}")

    # Add to history
    st.session_state.history.append({
        "time": timestamp,
        "price": result,
        "input": user_data.tolist()[0]
    })

# Show prediction history
if st.session_state.history:
    st.subheader("📈 Prediction History")
    for item in reversed(st.session_state.history[-5:]):
        st.write(f"{item['time']} → ${item['price']:,.2f}")

