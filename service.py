import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ====== ВХОДНЫЕ ДАННЫЕ ======
class ClientData(BaseModel):
    age: int
    person_income: float          
    loan_amnt: float             
    loan_int_rate: float
    person_education: str
    person_home_ownership: str

# ====== APP ======
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
    return FileResponse("frontend/index.html")

# ====== PREPROCESSING ======
def preprocess(data: ClientData) -> pd.DataFrame:
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

    X = pd.DataFrame([{
        "person_age": data.age,
        "person_income": data.person_income,
        "loan_amnt": data.loan_amnt,
        "loan_int_rate": data.loan_int_rate,
        "person_education": education_map[data.person_education],
        "person_home_ownership": home_map[data.person_home_ownership]
    }])

    return X

# ====== SCORING ======
@app.post("/score")
def score(data: ClientData):
    X = preprocess(data)
    pred = model.predict(X)[0]    
    approved = bool(pred == 0)

    return {
        "approved": approved
    }

