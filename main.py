from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import os
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(BASE_DIR, "index3.html")
MODEL_PATH = os.path.join(BASE_DIR, "model_2.pkl")

model = joblib.load(MODEL_PATH)

class LoanRequest(BaseModel):
    person_age: int
    person_income: float
    loan_amnt: float
    loan_int_rate: float
    person_education: str
    person_home_ownership: str


@app.get("/")
def index():
    return FileResponse(HTML_FILE)
    
def preprocess(data: LoanRequest) -> pd.DataFrame:

    education_map = {
        "Студент": 1,
        "Бакалавр": 2,
        "Магистр": 3
    }

    home_map = {
        "Арендованое": 1,
        "Собственное": 2,
        "Ипотечное": 3
    }

    data = pd.DataFrame([{
        "person_age": data.person_age,
        "person_education_encoded": education_map[data.person_education],
        "person_income": data.person_income,
        "person_home_ownership_encoded": home_map[data.person_home_ownership],
        "loan_amnt": data.loan_amnt,
        "loan_int_rate": data.loan_int_rate
    }])

    return data

@app.post("/score")
def score(data: LoanRequest):

    if data.person_age < 20 or data.person_age > 83:
        return {"approved": False}

    if data.person_income < 30000:
        return {"approved": False}

    if data.loan_int_rate < 7:
        return {"approved": False}

    if data.loan_amnt > data.person_income * 200:
        return {"approved": False}
    
    

    X = preprocess(data)
    model_result = model.predict(X)[0]
    approved = not bool(model_result)

    return {"approved": approved }




