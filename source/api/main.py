import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from load_and_training import load_data
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from database import init_db, load_model_from_db
from typing import List

app = FastAPI()

POSTGRES_USERNAME = os.environ.get("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
POSTGRES_DB = os.environ.get("POSTGRES_DB")

# PostgreSQL database connection string
DB_URL = f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


class CollectAndTrainInput(BaseModel):
    startDate: str
    company: str

@app.get("/")
def hello():
    return {"status": "ok"}


@app.post("/collect-and-train")
def collect_and_train(input_data: CollectAndTrainInput):
    load_data(company=input_data.company, start_date=input_data.startDate)
    return {"status": "ok", "message": "Data collected and trained successfully."}


class PredictionInput(BaseModel):
    company: str
    daysToPredict: int = 1

class PredictionOutput(BaseModel):
    company: str
    predictions: List[float]
    dates: List[str]

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    engine = init_db()
    
    # Loading model, scaler, and last sequence from db
    result = load_model_from_db(input_data.company, engine)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No model found for company {input_data.company}")
    
    model, scaler, last_sequence = result
    
    try:
        end_date = pd.Timestamp.today()
        predictions = []
        dates = []

        for i in range(input_data.daysToPredict):
            next_pred = model.predict(last_sequence.reshape(1, 60, 1))
            
            next_pred_price = scaler.inverse_transform(next_pred)[0][0]
            predictions.append(float(next_pred_price))
            
            next_date = end_date + pd.Timedelta(days=i+1)
            print(next_date)
            while next_date.weekday() > 4:  # Skip weekends
                next_date += pd.Timedelta(days=1)
                print(next_date)
            dates.append(next_date.strftime('%Y-%m-%d'))
            
            last_sequence = np.append(last_sequence[1:], next_pred)

        return PredictionOutput(
            company=input_data.company,
            predictions=predictions,
            dates=dates
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")