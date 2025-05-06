import streamlit as st
import pandas as pd
import pickle
import numpy as np


# --- Define the order and names of your features EXACTLY as your model expects them ---
survival_features = [
    'Age', 'Race', 'Marital Status', '7th Stage', '8th Stage', '6th Stage', 'Grade', 'A Stage',
    'Tumor Size', 'Estrogen Status', 'Progesterone Status', 'Regional Node Examined',
    'Reginol Node Positive'
]

# --- If your categorical features need specific encoding (e.g., one-hot encoding),
# --- you'll need to handle that before making the prediction. ---
categorical_features = ['Race', 'Marital Status', '7th Stage', '8th Stage', '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']

def predict_survival(model, data):
    """Makes a survival prediction using the loaded model."""
    input_df = pd.DataFrame([data], columns=survival_features)
    # --- If you did any preprocessing (e.g., scaling, one-hot encoding) before training,
    # --- you MUST apply the SAME preprocessing here before making the prediction. ---
    prediction_probability = model.predict_proba(input_df)[:, 1][0] # Assuming class 1 represents 'Die'
    prediction_class = model.predict(input_df)[0]
    return prediction_class, prediction_probability

def main():
    st.title("Breast Cancer Survival Prediction")
    st.write("Please enter the patient's features to predict the survival outcome.")

    patient_data = {}
    for feature in survival_features:
        if feature in categorical_features:
            # --- Replace with the actual unique categories for each feature ---
            unique_categories = ["Category 1", "Category 2", "Category 3"] # Placeholder
            patient_data[feature] = st.selectbox(f"{feature.capitalize()}", unique_categories)
        else:
            patient_data[feature] = st.number_input(f"{feature.capitalize()}")

    if st.button("Predict Survival"):
        input_values = list(patient_data.values())

        # --- You might need to perform preprocessing on 'input_values' here
        # --- to match the format your model was trained on (e.g., one-hot encode
        # --- categorical features). ---

        # Make the prediction
        survival_prediction, probability = predict_survival(survival_model, input_values)

        st.subheader("Survival Prediction:")
        # --- Adjust these labels based on how your target variable is encoded ---
        if survival_prediction == 1:
            st.error(f"The model predicts the patient is **less likely to survive** with a probability of {probability:.2f}.")
        else:
            st.success(f"The model predicts the patient is **likely to survive** with a probability of {1 - probability:.2f}.")

if __name__ == '__main__':
    main()
