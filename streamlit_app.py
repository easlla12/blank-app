import streamlit as st
st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("Breast Cancer Survival Prediction")
st.markdown("""
This tool assists clinicians in predicting **patient outcomes** using SEER data and machine learning.  
Please enter the patient data below.
""")

# Sidebar for inputs
st.sidebar.header("Patient Information")

# Example inputs — adjust these to match your SEER dataset
age = st.sidebar.number_input("Age at Diagnosis", min_value=0, max_value=120, help="Enter the patient's age at the time of diagnosis")
tumor_size = st.sidebar.number_input("Tumor Size (in mm)", min_value=0, max_value=200, help="Enter the tumor size in millimeters")
tumor_grade = st.sidebar.selectbox("Tumor Grade", ['Low', 'Intermediate', 'High'], help="Pathologic tumor grade")
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian or Pacific Islander', 'Other'])
histology = st.sidebar.selectbox("Histology Type", ['Infiltrating Ductal Carcinoma', 'Lobular Carcinoma', 'Other'])
surgery = st.sidebar.selectbox("Surgical Procedure", ['None', 'Lumpectomy', 'Mastectomy'])

# Collect data into DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'TumorSize': [tumor_size],
    'TumorGrade': [tumor_grade],
    'Race': [race],
    'Histology': [histology],
    'Surgery': [surgery],
})

# Prediction section
st.subheader("Prediction Output")
if st.button("Run Prediction"):
    prediction = predict_model(model, data=input_df)
    result = prediction['prediction_label'][0]
    score = prediction['prediction_score'][0]

    st.success(f"**Predicted Outcome:** {result}")
    st.info(f"**Confidence Score:** {score:.2f}")

    # Optional: Show patient summary
    with st.expander("Patient Summary"):
        st.dataframe(input_df)


