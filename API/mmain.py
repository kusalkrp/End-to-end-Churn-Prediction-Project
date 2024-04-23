import random
from typing import List
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Import Packages
import joblib # Model Elements
import uvicorn # Testing
import pandas as pd # Dataframe
from pydantic import BaseModel # Data Validation
from fastapi import FastAPI # API Library
import numpy as np

#--------------------------#

# Instantiate an API Instance
app = FastAPI(title="Churn Prediction API", version='1.0')

# Initialize FastAPI app
app = FastAPI()

# Load PCA object
pca_loaded = joblib.load('../Exported/pca_transformer.pkl')

# Load StandardScaler object
scaler_loaded = joblib.load('../Exported/standard_scaler.pkl')

# Load the saved ColumnTransformer object
preprocessor_loaded = joblib.load('../Exported/preprocessor.pkl')

# Load trained RandomForestClassifier model
rf_classifier_loaded = joblib.load('../Exported/rf_classifier_pca_model.pkl')

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")


'''
Data Validation
1. Read the request body as JSON
2. Validate if the data has correct types
'''

class ChurnPredictionInput(BaseModel):
    
    gender: str
    senior_citizen: str
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float
    
    

#--------------------------#

# API End point
@app.get('/')
@app.get('/home')
def status():
    # API Status
    return {'message' : 'System is up and running'}

#--------------------------#

def generate_retention_message(prediction, input_data):
    if prediction == 1:
        prompt = "The customer is at risk of churning. We should offer personalized incentives to retain them."
    else:
        prompt = "The customer is loyal and staying with us. Let's thank them for their loyalty and suggest ways to enhance their experience."

    # Encode the prompt (exclude customer details)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text using GPT-2 model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=150, num_return_sequences=1, temperature=0.7)

    # Decode the generated response only
    retention_message = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    return retention_message




@app.post("/predict/")
def predict_churn(data: ChurnPredictionInput):
    input_data = pd.DataFrame([data.dict()])
    
    # Transform the input data using the loaded ColumnTransformer
    X_input_preprocessed = preprocessor_loaded.transform(input_data)
    
    # Apply PCA transformation
    X_input_pca = pca_loaded.transform(X_input_preprocessed)
    
    # Make prediction
    y_pred = rf_classifier_loaded.predict(X_input_pca)
    
    # Convert prediction to class label
    predicted_churn = 'The customer is likely to churn' if y_pred[0] == 1 else 'The customer is staying'
    
    # Generate retention message
    retention_message = generate_retention_message(y_pred[0], input_data )
    
    return {"Predicted_Churn": predicted_churn, "Retention_Message": retention_message}


feature_order = [
    'gender', 'senior_citizen', 'partner',
    'dependents', 'tenure', 'phone_service', 'multiple_lines', 'internet_service',
    'online_security', 'online_backup', 'device_protection', 'tech_support',
    'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
    'payment_method', 'monthly_charges', 'total_charges'
]

@app.post('/predict_batch/')
async def predict_batch(data: List[ChurnPredictionInput]):
    # Convert incoming data to DataFrame
    data_dicts = [item.dict() for item in data]
    churn_df = pd.DataFrame(data_dicts)
    
    # Rearrange columns
    churn_df = churn_df[feature_order]
    
    # Transform input data
    X_input_preprocessed = preprocessor_loaded.transform(churn_df)
    X_input_pca = pca_loaded.transform(X_input_preprocessed)
    
    # Make predictions
    y_pred = rf_classifier_loaded.predict(X_input_pca)
    
    # Churn Label
    churn_labels = ['Churn Customer' if i==1 else 'Staying Customer' for i in y_pred]
    
    return {'predictions': churn_labels}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)