import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

class ClientData(BaseModel):
    persone_age: int
    person_income: float         
    loan_amnt: float              
    loan_int_rate: float
    person_education: str
    person_home_ownership: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

model = joblib.load("model_2.pkl")

@app.get("/")
def root():
    return FileResponse("frontend/index3.html")

def preprocess(data: ClientData) -> pd.DataFrame:
    FEATURE_ORDER = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "person_education_encoded",
    "person_home_ownership_encoded",
    "amnt_imcome",
    "age_inc",
    "log_income",
    "log_loan_amnt",
    "interest_burden",
    "estimated_payment",
    "income_after_payment",
    "large_loan"
    ]

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
        "person_age": data.age,
        "person_education_encoded": education_map[data.person_education],
        "person_income": data.person_income,
        "person_home_ownership_encoded": home_map[data.person_home_ownership],
        "loan_amnt": data.loan_amnt,
        "loan_int_rate": data.loan_int_rate
    }])

    data['amnt_imcome'] = data['loan_amnt']/data['person_income']
    data['age_inc'] = data['person_age'] * data['person_income']
    data['log_income'] = np.log1p(data['person_income'])
    data['log_loan_amnt'] = np.log1p(data['loan_amnt'])
    data['interest_burden'] = data['loan_amnt'] * data['loan_int_rate']
    data['estimated_payment'] = data['loan_amnt'] * (data['loan_int_rate'] / 100) / 12
    data['income_after_payment'] = data['person_income'] - data['estimated_payment']
    data['large_loan'] = (data['loan_amnt'] > data['person_income'] * 5).astype(int)
    return data[FEATURE_ORDER]



@app.post("/score")
def score(data: ClientData):
    X = preprocess(data)

    approved = model.predict_proba(X)[0][1] <= 0.25   

    return {
        "approved": approved
    }
