"""
This script demonstrates how to load a pre-trained machine learning model to predict medical costs 
based on input data that includes patient characteristics and diagnosis information.

Key Steps:
1. Load the Pre-trained Model:
   The model is loaded using `joblib` from the specified path (`../models/medical_cost_predictor_model_v3.pkl`), 
   which is a pre-trained pipeline containing preprocessing steps (such as encoding and scaling) and a regression 
   model (e.g., `LinearRegression`).
   
2. Prepare Input Data:
   The input data is structured as a pandas DataFrame containing five features:
   - Age: The age of the patient (integer).
   - Sex: The gender of the patient (categorical: 'M' or 'F').
   - Zipcode: A 5-digit numeric value representing the patient's location.
   - Image_Type: Specifies whether the input is based on X-ray or skin image data (categorical: 'Xray' or 'Skin').
   - Diagnosis: The specific medical diagnosis derived from the image. For example, 'melanoma' in the case of skin 
     images or 'pneumonia' for X-ray images.

# These need to be mapped from the CV model predictions / labels
diagnosis_types = ['pneumonia', 'actinic keratoses and intraepithelial carcinoma', 
                   'basal cell carcinoma', 'benign keratosis-like lesions', 'dermatofibroma', 
                   'melanoma', 'melanocytic nevi', 'vascular lesions', 'Normal Lungs']


3. Make Predictions:
   The input data is passed into the pre-trained model, which predicts the medical cost associated with the given 
   patient information and diagnosis. The prediction is printed out as the estimated medical cost.

This structure ensures that the input data matches the expected format used during the model training, including 
both numeric and categorical variables. The pipeline inside the loaded model handles the necessary preprocessing 
steps such as one-hot encoding and scaling before making the prediction.
"""


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

input_data = pd.DataFrame({
    'Age': [34],
    'Sex': ['M'],
    'Zipcode': [77231],
    'Image_Type': ['Skin'],
    'Diagnosis': ['melanoma']
})

loaded_model = joblib.load('../models/medical_cost_predictor_model_v3.pkl')
#print("Model loaded successfully.")

# Make predictions with the loaded model
predicted_cost = loaded_model.predict(input_data)
print(f"Predicted medical cost: {predicted_cost[0]}")