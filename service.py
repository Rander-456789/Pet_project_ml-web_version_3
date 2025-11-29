import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

class ClientData(BaseModel):
    age: int
    person_income: float
    loan_amnt: float
    loan_int_rate: float
    person_education_encoded: str
    person_home_ownership_encoded: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

model = joblib.load('model_2.pkl')

@app.get("/")
def root():
    return FileResponse("frontend/index3.html")

@app.post('/score')
def score(data: ClientData):
    features = [data.age, data.person_income, data.loan_amnt, data.loan_int_rate, data.person_education_encoded, data.person_home_ownership_encoded]
    features[-2] = 1 if features[-2] == 'Студент' else ( 2 if features[-2] == 'Бакалавр' else 3)
    features[-1] = 1 if features[-1] == 'Арендованое' else ( 2 if features[-1] == 'Собственное' else 3)
    approved = not model.predict([features])[0].item()
    return {'approved': approved}

