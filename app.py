import streamlit as st
import pandas as pd 
import numpy as np
import pickle

st.set_page_config(
    page_title="Crop recommendation system",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    with open('model.pkl','rb') as f_model :
        model = pickle.load(f_model)
    with open('label_encoder.pkl','rb') as f_le:
        le = pickle.load(f_le)
    return model,le
model,le=load_artifacts()

st.title("🌾 Smart Crop Recommendation System")
st.write("Enter soil and weather conditions to get crop recommendation")
st.markdown("---")

col1,col2 =st.columns(2)
with col1:
    st.subheader("🧪 Soil Nutrients")
    N = st.number_input("Nitrogen (N)",min_value=0,max_value=140,value=50)
    P = st.number_input("Phosphorus (P)",min_value=5,max_value=145,value=50)
    K = st.number_input("Potassium (K)",min_value=5,max_value=205,value=50)
    temperature = st.number_input("Temperature (°C)",min_value=0.0,max_value=50.0,value=25.0)
with col2:
    st.subheader("🌡️ Climate Conditions")
    humidity = st.number_input("Humidity (%)",min_value=0.0, max_value=100.0, value=70.0)
    ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

if st.button("Recommend Crop",type="primary"):
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)
    prediction = model.predict(input_data)
    crop = le.inverse_transform(prediction)[0]
    st.success(f"### Recommended Crop: **{crop.upper()}**")
    st.balloons()
    summary_df = pd.DataFrame({
        "Parameter": ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH Level", "Rainfall"],
        "Value": [f"{N}", f"{P}", f"{K}", f"{temperature} °C", f"{humidity} %", f"{ph}", f"{rainfall} mm"]
    })
    st.table(summary_df)
