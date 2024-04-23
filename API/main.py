import tempfile
from typing import List
from fastapi import File, UploadFile
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load PCA object
pca_loaded = joblib.load('../Exported/pca_transformer.pkl')

# Load the saved ColumnTransformer object
preprocessor_loaded = joblib.load('../Exported/preprocessor.pkl')

# Load trained RandomForestClassifier model
xgb_classifier_loaded = joblib.load('../Exported/xgb_classifier.pkl')

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

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

    # Define input schema for batch prediction
class BatchPredictionInput(BaseModel):
    records: List[ChurnPredictionInput]

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
    y_pred = xgb_classifier_loaded.predict(X_input_pca)
    
    # Convert prediction to class label
    predicted_churn = 'The customer is likely to churn' if y_pred[0] == 1 else 'The customer is staying'
    
    # Generate retention message
    retention_message = generate_retention_message(y_pred[0], input_data )
    
    return {"Predicted_Churn": predicted_churn, "Retention_Message": retention_message}



@app.post("/predict_batch/")
def predict_batch(data: BatchPredictionInput):
    # Convert incoming data to DataFrame
    input_data = pd.DataFrame([item.dict() for item in data.records])
    
    # Transform the input data using the loaded ColumnTransformer
    X_input_preprocessed = preprocessor_loaded.transform(input_data)
    
    # Apply PCA transformation
    X_input_pca = pca_loaded.transform(X_input_preprocessed)
    
    # Make predictions
    y_pred = xgb_classifier_loaded.predict(X_input_pca)
    
    # Convert predictions to class labels
    churn_labels = ['The customer is likely to churn' if pred == 1 else 'The customer is staying' for pred in y_pred]
    
    # Generate retention messages
    retention_messages = [generate_retention_message(pred, row) for pred, row in zip(y_pred, input_data.iterrows())]
    
    return {"predictions": churn_labels, "retention_messages": retention_messages}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
