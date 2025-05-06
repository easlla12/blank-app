import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load saved model
model = load_model('seer_model')  # Make sure this file exists in your working directory

# Title
st.set_page_config(page_title="SEER Breast Cancer Predictor", layout="centered")
st.title("🩺 Breast Cancer Outcome Prediction (SEER-based)")
st.write("This interface allows medical professionals to predict patient outcomes based on clinical features from the SEER dataset.")

# Sidebar Inputs
st.sidebar.header("Enter Patient Details")

# Replace these with actual features used in your model
age = st.sidebar.number_input("Age at Diagnosis", 20, 100, 50)
tumor_size = st.sidebar.number_input("Tumor Size (mm)", 0, 200, 25)
tumor_grade = st.sidebar.selectbox("Tumor Grade", ['Low', 'Intermediate', 'High'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian or Pacific Islander', 'Other'])
histology = st.sidebar.selectbox("Histology", ['Infiltrating Ductal Carcinoma', 'Lobular Carcinoma', 'Other'])
surgery = st.sidebar.selectbox("Surgery Type", ['None', 'Lumpectomy', 'Mastectomy'])

# Collect data into a dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'TumorSize': [tumor_size],
    'TumorGrade': [tumor_grade],
    'Race': [race],
    'Histology': [histology],
    'Surgery': [surgery]
})

# Display prediction
st.subheader("Prediction Result")
if st.button("Predict"):
    prediction = predict_model(model, data=input_data)
    predicted_class = prediction['prediction_label'][0]
    probability = prediction['prediction_score'][0]

    st.success(f"Predicted Outcome: **{predicted_class}**")
    st.info(f"Model Confidence: **{probability:.2f}**")

    with st.expander("View Input Summary"):
        st.dataframe(input_data)

