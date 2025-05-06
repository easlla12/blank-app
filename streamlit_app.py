import streamlit as st

st.title("Breast Cancer Prediction App")
st.write("Enter patient details to predict the survival/class of breast cancer.")

# User inputs
age = st.number_input("Age at Diagnosis", min_value=0, max_value=120, step=1)
tumor_size = st.number_input("Tumor Size (mm)", min_value=0, max_value=200)
tumor_grade = st.selectbox("Tumor Grade", ['Low', 'Intermediate', 'High'])
race = st.selectbox("Race", ['White', 'Black', 'Asian or Pacific Islander', 'Other'])
histology = st.selectbox("Histology", ['Infiltrating Ductal Carcinoma', 'Lobular Carcinoma', 'Other'])
surgery = st.selectbox("Surgery Type", ['None', 'Lumpectomy', 'Mastectomy'])

# Create a DataFrame with input values
input_df = pd.DataFrame({
    'Age': [age],
    'TumorSize': [tumor_size],
    'TumorGrade': [tumor_grade],
    'Race': [race],
    'Histology': [histology],
    'Surgery': [surgery],
})

# Make prediction
if st.button("Predict"):
    prediction = predict_model(model, data=input_df)
    result = prediction['prediction_label'][0]
    st.success(f"Prediction: {result}")
