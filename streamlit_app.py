import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
pip install scikit-learn

# --- Define the order and names of your features EXACTLY as your model expects them ---
survival_features = [
    'Age', 'Race', 'Marital Status', '7th Stage', '8th Stage', '6th Stage', 'Grade', 'A Stage',
    'Tumor Size', 'Estrogen Status', 'Progesterone Status', 'Regional Node Examined',
    'Reginol Node Positive'
]

categorical_features = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'Grade', 'A Stage', 'Estrogen Status', 'Progesterone Status']
numerical_features = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive']

# --- Initialize LabelEncoders for categorical features ---
label_encoders = {}
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    if feature == 'Race':
        label_encoders[feature].fit(['Other', 'White', 'Black'])
    elif feature == 'Marital Status':
        label_encoders[feature].fit(['Married(including common law)', 'Divorced', 'Single (never married)', 'Widowed'])
    elif feature == 'T Stage':
        label_encoders[feature].fit(['T2', 'T1', 'T3', 'T4'])
    elif feature == 'N Stage':
        label_encoders[feature].fit(['N3', 'N2', 'N1'])
    elif feature == '6th Stage':
        label_encoders[feature].fit(['IIIC', 'IIIA', 'IIB', 'IIA', 'I'])
    elif feature == 'Grade':
        label_encoders[feature].fit(['Moderately differentiated', 'Poorly differentiated', 'Well differentiated'])
    elif feature == 'A Stage':
        label_encoders[feature].fit(['Regional', 'Distant', 'Localized'])
    elif feature == 'Estrogen Status':
        label_encoders[feature].fit(['Positive', 'Negative'])
    elif feature == 'Progesterone Status':
        label_encoders[feature].fit(['Positive', 'Negative'])

# --- Initialize StandardScaler for numerical features if your model used scaling ---
scaler = None
# if preprocessor is None: # Only initialize if a preprocessor wasn't loaded
#     scaler = StandardScaler()
#     # --- You would typically fit the scaler on your training data ---
#     # --- For prediction, you would only transform the new data ---

def preprocess_data(data):
    processed_data = {}
    for feature, value in data.items():
        if feature in categorical_features:
            processed_data[feature] = label_encoders[feature].transform([value])[0]
        elif feature in numerical_features:
            processed_data[feature] = value
    return list(processed_data.values())

def predict_survival(model, processed_data):
    """Makes a survival prediction using the loaded model."""
    input_array = np.array(processed_data).reshape(1, -1)
    # --- If you used scaling, apply it here ---
    # if scaler:
    #     input_scaled = scaler.transform(input_array)
    #     prediction_probability = model.predict_proba(input_scaled)[:, 1][0]
    #     prediction_class = model.predict(input_scaled)[0]
    # else:
    prediction_probability = model.predict_proba(input_array)[:, 1][0] # Assuming class 1 represents 'Dead'
    prediction_class = model.predict(input_array)[0]
    return prediction_class, prediction_probability

def main():
    st.title("Breast Cancer Survival Prediction")
    st.write("Please enter the patient's features to predict the survival outcome.")

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
