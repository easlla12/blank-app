import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained survival prediction model


# Assuming your model was trained on these features. Adjust as needed.
survival_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def predict_survival(model, data):
    """Makes a survival prediction using the loaded model."""
    input_df = pd.DataFrame([data], columns=survival_features)
    prediction_probability = model.predict_proba(input_df)[:, 1][0]  # Assuming class 1 represents 'die'
    prediction_class = model.predict(input_df)[0]
    return prediction_class, prediction_probability

def main():
    st.title("Breast Cancer Survival Prediction")
    st.write("Please enter the patient's features to predict the survival outcome.")

    patient_data = {}
    for feature in survival_features:
        patient_data[feature] = st.number_input(f"{feature.capitalize()}", format="%.3f")

    if st.button("Predict Survival"):
        input_values = list(patient_data.values())

        # Make the prediction
        survival_prediction, probability = predict_survival(survival_model, input_values)

        st.subheader("Survival Prediction:")
        if survival_prediction == 1:
            st.error(f"The model predicts the patient is **less likely to survive** with a probability of {probability:.2f}.")
        else:
            st.success(f"The model predicts the patient is **likely to survive** with a probability of {1 - probability:.2f}.")

if __name__ == '__main__':
    main()
