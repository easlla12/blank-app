import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Sidebar panel
st.sidebar.title("Breast Cancer Predictions")
st.sidebar.write("This application has been designed to give you an indication of whether you are likely to have a loan request accepted or rejected. 
This result is indicative, and the actual outcome will depend on your own personal 
circumstances and lender checks, as well as the amount borrowed and the terms of the loan.\n

def main():
    st.title("Breast Cancer Survival Prediction")
    st.write("Please enter the patient's features to predict the survival outcome.")
    
# --- Define the order and names of your features EXACTLY as your model expects them ---
survival_features = [
    'Age', 'Race', 'Marital Status', '7th Stage', '8th Stage', '6th Stage', 'Grade', 'A Stage',
    'Tumor Size', 'Estrogen Status', 'Progesterone Status', 'Regional Node Examined',
    'Reginol Node Positive'
]

categorical_features = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']
numerical_features = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive']
st.subheader('Please enter your information below:')

# User inputs
Race = st.selectbox("Race: ", ['Other','White','Black'])
Tstage = st.selectbox("T Stage: ", ['T2', 'T1', 'T3', 'T4'])
Nstage = st.selectbox("N Stage: ", ['N3', 'N2', 'N1'])
Sixstage = st.selectbox("6 Stage: ", ['IIIC', 'IIIA', 'IIB', 'IIA', 'I'])
Grade = st.selectbox ("Grade:",['Moderately differentiated', 'Poorly differentiated', 'Well differentiated'])
Astage= st.selectbox("A Stage: ", ['Regional', 'Distant', 'Localized'])
EstrogenStatus = st.selectbox("Estrogen Status: ", ['Positive', 'Negative'])
ProgesteroneStatus = st.selectbox("Interest Rate: ", ['Positive', 'Negative'])

# Create dataframe with user input values
user_input = pd.DataFrame({
    'person_race': [Race],
    'Tstage_level': [Tstage],
    'Nstage_level': [Nstage],
    'Sixstage_level': [Sixstage],
    'Grade_level': [Grade],
    'Astage_level': [Astage],
    'EstrogenStatus_level': [EstrogenStatus],
    'ProgesteroneStatus_level': [ProgesteroneStatus],

})

# Display user input into a table
#st.table(user_input)

if st.button("Predict"):
    result = predict(user_input)
    if result[0] == 0:
        st.success("Accepted")
    else:
        st.error("Rejected")
