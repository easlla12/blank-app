pip install pycaret
streamlit
pycaret==3.2.0
shap
matplotlib
pandas
scikit-learn
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model, get_config
import shap
import matplotlib.pyplot as plt

# Streamlit App Layout
st.set_page_config(page_title="SEER Breast Cancer Predictor", layout="centered")
st.title("🩺 SEER Breast Cancer Outcome Predictor")
st.markdown("This tool helps clinicians assess breast cancer outcomes based on SEER dataset features.")

# Sidebar Inputs
st.sidebar.header("Patient Clinical Information")

# Input fields - adjust according to your trained model
age = st.sidebar.number_input("Age at Diagnosis", 20, 100, 50)
tumor_size = st.sidebar.number_input("Tumor Size (mm)", 0, 200, 25)
tumor_grade = st.sidebar.selectbox("Tumor Grade", ['Low', 'Intermediate', 'High'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian or Pacific Islander', 'Other'])
histology = st.sidebar.selectbox("Histology", ['Infiltrating Ductal Carcinoma', 'Lobular Carcinoma', 'Other'])
surgery = st.sidebar.selectbox("Surgery Type", ['None', 'Lumpectomy', 'Mastectomy'])

# Collect inputs into a DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'TumorSize': [tumor_size],
    'TumorGrade': [tumor_grade],
    'Race': [race],
    'Histology': [histology],
    'Surgery': [surgery]
})

st.subheader("Prediction Results")
if st.button("Predict"):
    # Make prediction
    prediction = predict_model(model, data=input_df)
    predicted_class = prediction['prediction_label'][0]
    probability = prediction['prediction_score'][0]

    # Display result
    st.success(f"Predicted Outcome: **{predicted_class}**")
    st.info(f"Model Confidence: **{probability:.2f}**")

    # Show input summary
    with st.expander("Patient Summary"):
        st.dataframe(input_df)

    # Show feature importance using SHAP
    st.subheader("Feature Contribution (SHAP)")
    try:
        # Get preprocessing pipeline and model
        prep_pipe = get_config('prep_pipe')
        model_object = get_config('trained_model')

        # Apply pipeline transform
        transformed_input = prep_pipe.transform(input_df)

        # Use SHAP to explain the model
        explainer = shap.Explainer(model_object)
        shap_values = explainer(transformed_input)

        # Plot SHAP force plot or summary bar
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning("Feature explanation unavailable.")
        st.text(f"Details: {e}")

