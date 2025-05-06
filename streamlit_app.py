import streamlit as st
import pandas as pd
import pickle
import numpy as np


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
ProgesteroneStatus = st.selectbox("Interest rate: ", ['Positive', 'Negative'])


    patient_data = {}
    for feature in survival_features:
        if feature in categorical_features:
            # --- Use the fitted classes from the LabelEncoder for selectbox options ---
            patient_data[feature] = st.selectbox(f"{feature.capitalize()}", list(label_encoders[feature].classes_))
        else:
            patient_data[feature] = st.number_input(f"{feature.capitalize()}")

    if st.button("Predict Survival"):
        processed_input = preprocess_data(patient_data)

        # Make the prediction
        survival_prediction, probability = predict_survival(survival_model, processed_input)

        st.subheader("Survival Prediction:")
        # --- Adjust these labels based on how your target variable 'Status' was encoded (e.g., 0 for Alive, 1 for Dead) ---
        if survival_prediction == 1:
            st.error(f"The model predicts the patient is **likely to die** with a probability of {probability:.2f}.")
        else:
            st.success(f"The model predicts the patient is **likely to survive** with a probability of {1 - probability:.2f}.")

if __name__ == '__main__':
    main()
