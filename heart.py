import streamlit as st
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# import matplotlib.pyplot as pl
import numpy as np

st.set_page_config(page_title="HeartGaurd AI", layout="wide")

model = joblib.load("ann_model.pkl")
# scaler = joblib.load("scaler.pkl")

st.sidebar.image("image.png",use_container_width=True)
st.sidebar.markdown("<h1 style='text-align: center; color: white;'>HeartGaurd AI</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: center; color: white;'>Your Personal Heart Health Assistant</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")  # Adds a separator line


# Sidebar navigation

page = st.sidebar.radio("Features", ["Heart Disease prediction model" , "Precautions for Heart Disease"])

if page == "Heart Disease prediction model":
    st.title("Heart Disease Prediction using ML")
    
    # Input fields
    # age = st.number_input("Age", min_value=1, max_value=120)

    col1 , col2  = st.columns(2)
    with col2:
        sex = st.selectbox("Please select your gender" , ["Male" , "Female"])
    with col1:
        age = st.number_input("Age", min_value=15, max_value=120)
    col3 , col4  = st.columns(2)
    with col3:
        excercise = st.selectbox("Exercise Level" , ["High" , "Moderate" , "Low"])
    with col4:
        Family_history = st.selectbox("Family History" , ["Yes" , "No"])
        
    col5 , col6  = st.columns(2)
    with col5:
        height = st.number_input("Height(inches)", min_value=0.0, max_value=12.0, value=3.0)
    with col6:
        weight = st.number_input("Weight (Kg)", min_value=30, max_value=300)
    col7 , col8  = st.columns(2)
    with col7:
        blood = st.selectbox("High Blood Pressure" , ["yes" , "no"])
    with col8:
        alchohol = st.selectbox("Alchohol Consumption" , ["yes" , "no"])
    col9 , col10  = st.columns(2)
    with col10:
        sleep = st.number_input("Sleep Hours", min_value=1, max_value=24)
    with col9:
        Stress= st.selectbox("Stress Level" , ["High" , "Moderate" , "Low"])
        sugar = "Moderate"
    col10 , col11  = st.columns(2)
    with col10:
        bloodp = st.number_input("Blood Pressure Level", min_value=80, max_value=300)
    with col11:
        cholesterol = st.number_input("Cholestrol Level", min_value=80, max_value=300)
    col12 , col13  = st.columns(2)
    with col12:
        smoke = st.selectbox("Smoking" , ["yes" , "no"])
    with col13:
        diabetes = st.selectbox("Diabetes" , ["yes" , "no"])
    bmi = weight / (height ** 2)

    col14 , col15  = st.columns(2)
    with col14:
        Ldl = st.number_input("LDL Level", min_value=80, max_value=300)
    with col15:
        Hdl = st.number_input("HDL Level", min_value=80, max_value=300)
    
   
    # Predict button

    if st.button("Heart Disease Test Result"):
        input_data = pd.DataFrame({
            "Age": [age],
            "Gender" : [sex],
            "Blood Pressure" : [bloodp],
            "Cholestrol" : [cholesterol],
            "Exercise Habits" : [excercise],
            "Smoking" : [smoke],
            "Family Heart Disease" : [Family_history],
            "Diabetes" : [diabetes],
            "BMI" : [bmi],
            "High Blood Pressure" : [blood],
            "Low HDL Cholesterol" : [Ldl],
            "High LDL Cholesterol" : [Hdl],
            "Alcohol Consumption" : [alchohol],
            "Stress Level" : [Stress],
            "Sleep Hours" : [sleep],
            "Sugar Consumption" : [sugar],
        })
        categorical_features = ["Gender", "Exercise Habits", "Smoking", "Family Heart Disease", 
                        "Diabetes", "Alcohol Consumption", "Stress Level" , "Sugar Consumption" , "High Blood Pressure"]
        for col in categorical_features:
            le = LabelEncoder()
            input_data[col] = le.fit_transform(input_data[col]) 
        
    
        input_data = input_data.values
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)

        prediction = model.predict(input_data).flatten()

        if Ldl >= 200 and Hdl  >= 200 and cholesterol >= 120 and bloodp >= 150:
            prediction = [1]

        if prediction[0] == 1:
            st.write("### **Prediction: Positive**")
            st.write("You are at risk of heart disease. Please consult a doctor for professional medical advice.")
        else:
            st.write("### **Prediction: Negative**")
            st.write("You are not at risk of heart disease. Keep up the good work!")


if page == "Chat with Model":
    st.title("Chat with Model")
    st.write("Chat with the model to get the answers to your queries")
    user_input = st.text_input("Enter your query")
    if st.button("Send"):
        st.write("Model's response")
if page == "Precautions for Heart Disease":
    

    # Title
    st.title("❤️ Heart Health Precautions")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .precaution-card {
                background-color: white;
                color: #333;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                font-size: 16px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    # List of precautions
    precautions = [
        "✅ Eat a healthy diet rich in fruits, vegetables, and whole grains.",
        "✅ Exercise regularly—at least 30 minutes most days of the week.",
        "✅ Avoid smoking and limit alcohol consumption.",
        "✅ Manage stress with relaxation techniques like meditation.",
        "✅ Maintain a healthy weight and monitor blood pressure regularly.",
        "✅ Get regular health check-ups and follow doctor’s advice.",
    ]

    # Display precautions
    for precaution in precautions:
        st.markdown(f"<div class='precaution-card'>{precaution}</div>", unsafe_allow_html=True)


# Sidebar Footer
st.sidebar.markdown("---")  # Adds a separator line
st.sidebar.markdown("### ⚠️ Warning:")
st.sidebar.markdown("This prediction is not a medical diagnosis. Please consult a doctor for professional medical advice.")
st.sidebar.markdown("##### Made by **Cognitive Coders** 🚀")
