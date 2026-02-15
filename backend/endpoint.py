import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import predict_new_data

app = FastAPI(title='Credit Risk Prediction API')

class InputData(BaseModel):
    data: list

@app.post('/predict/')
async def predict(X_new: InputData):
    X_new = pd.DataFrame(X_new.data)

    y_pred = predict_new_data(X_new)

    return {'predictions': y_pred.tolist()}