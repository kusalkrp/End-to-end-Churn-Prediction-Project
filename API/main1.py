import tempfile
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load PCA object
pca_loaded = joblib.load('../Exported/pca_transformer.pkl')

# Load the saved ColumnTransformer object
preprocessor_loaded = joblib.load('../Exported/preprocessor.pkl')

# Load trained RandomForestClassifier model
rf_classifier_loaded = joblib.load('../Exported/rf_classifier_pca_model.pkl')

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

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

def generate_retention_message(prediction, input_data):
    if prediction == 1:
        prompt = "The customer is at risk of churning. We should offer personalized incentives to retain them."
    else:
        prompt = "The customer is loyal and staying with us. Let's thank them for their loyalty and suggest ways to enhance their experience."

    # Encode the prompt (exclude customer details)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text using the model
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
