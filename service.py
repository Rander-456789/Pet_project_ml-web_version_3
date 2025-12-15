import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# =======================
# Input schema
# =======================
class ClientData(BaseModel):
    age: int
    person_income: float          # рубли
    loan_amnt: float              # рубли
    loan_int_rate: float
    person_education: str
    person_home_ownership: str


# =======================
# App init
# =======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# Static frontend
# =======================
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# =======================
# Load model
# =======================
model = joblib.load("model_2.pkl")


# =======================
# Root
# =======================
@app.get("/")
def root():
    return FileResponse("frontend/index3.html")


# =======================
# Preprocessing
# =======================
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

    # ⚠️ СТРОГО ТОТ ЖЕ ПОРЯДОК, ЧТО ПРИ ОБУЧЕНИИ
    return pd.DataFrame([{
        "person_age": data.age,
        "person_education": education_map[data.person_education],
        "person_income": data.person_income,
        "person_home_ownership": home_map[data.person_home_ownership],
        "loan_amnt": data.loan_amnt,
        "loan_int_rate": data.loan_int_rate
    }])


# =======================
# Scoring endpoint
# =======================
@app.post("/score")
def score(data: ClientData):
    X = preprocess(data)

    pred = model.predict(X)[0]    # 0 — одобрено, 1 — отказ
    approved = bool(pred == 0)

    return {
        "approved": approved
    }
