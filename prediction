import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example breast cancer-specific features (customize based on your dataset)
numerical_cols = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
categorical_cols = ['diagnosis_stage', 'tumor_grade']

def predict_cancer(data):
    # Load trained cancer prediction model
    model = joblib.load("cancer_model.sav")
    # Load scaler
    scaler = joblib.load("cancer_scaler.sav")
    
    # Preprocess categorical columns: uppercase & strip spaces
    data[categorical_cols] = data[categorical_cols].applymap(lambda x: str(x).upper().replace(" ", ""))
    
    # Example encoding (customize as needed)
    encoding_map = {
        'diagnosis_stage': {'STAGEI': 0, 'STAGEII': 1, 'STAGEIII': 2, 'STAGEIV': 3},
        'tumor_grade': {'GRADE1': 0, 'GRADE2': 1, 'GRADE3': 2}
    }
    for col, mapping in encoding_map.items():
        data[col] = data[col].replace(mapping)
    
    # Scale numerical features
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    # Make prediction
    prediction = model.predict(data)
    
    return prediction
