import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')

st.title("Employee Promotion Prediction")

df = pd.read_csv("train.csv")

department = st.selectbox("Department", pd.unique(df['department']))
region = st.selectbox("Region", pd.unique(df['region']))
education = st.selectbox("Education", pd.unique(df['education']))
gender = st.selectbox("Gender", ["Female","Male"])
recruitment_channel = st.selectbox("Recruitment Channel", pd.unique(df['recruitment_channel']))
previous_year_rating = st.selectbox("Previous Year Rating", pd.unique(df['previous_year_rating']))

no_of_trainings = st.number_input("No of Trainings", min_value=1, step=1)
age = st.number_input("Age", min_value=20, step=1)
length_of_service = st.number_input("Length of Service", min_value=1, step=1)
KPIs_met = st.selectbox("KPI met >80%", ["Yes", "No"])
awards_won = st.selectbox("Awards Won?", ["Yes", "No"])
avg_training_score = st.number_input("Average Training Score", min_value=0, step=1)

KPIs_met = 1 if KPIs_met == "Yes" else 0
awards_won = 1 if awards_won == "Yes" else 0

input_data = {
    "department": department,
    "region": region,
    "education": education,
    "gender": gender,
    "recruitment_channel": recruitment_channel,
    "previous_year_rating": previous_year_rating,
    "no_of_trainings": no_of_trainings,
    "age": age,
    "length_of_service": length_of_service,
    "KPIs_met >80%": KPIs_met,
    "awards_won?": awards_won,
    "avg_training_score": avg_training_score,
}

try:
    model = joblib.load("Promotion_prediction_model.pkl")
except Exception as e:
    st.error(f"Error loading the model: {e}")

if st.button("Predict"):
    try:
        X_input = pd.DataFrame([input_data])  
        prediction = model.predict(X_input)
        if prediction[0]==1:
            st.markdown(
                "<h3 style='color: green;'>Employee Promoted.</h3>",unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='color: red;'>Employee not promoted.</h3>",unsafe_allow_html=True
            )
    except Exception as e:
        st.error(f"Error during prediction: {e}")
