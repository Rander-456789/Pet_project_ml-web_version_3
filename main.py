from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
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

# --------------------
# Загружаем модель ОДИН РАЗ
# --------------------

model = joblib.load(MODEL_PATH)

# --------------------
# Маппинги как в обучении
# --------------------

EDUCATION_MAP = {
    "Студент": 1,
    "Бакалавр": 2,
    "Магистр": 3
}

HOME_MAP = {
    "Арендованое": 1,
    "Собственное": 2,
    "Ипотечное": 3
}


# --------------------
# Модель входных данных
# --------------------

class LoanRequest(BaseModel):
    age: int
    person_income: float
    loan_amnt: float
    loan_int_rate: float
    person_education: str
    person_home_ownership: str


@app.get("/")
def index():
    return FileResponse(HTML_FILE)


@app.post("/score")
def score(data: LoanRequest):
    # --------------------
    # 1. ЖЁСТКИЕ ОТСЕЧКИ
    # --------------------

    if data.age < 18 or data.age > 100:
        return {"approved": False}

    if data.person_income <= 0 or data.loan_amnt <= 0:
        return {"approved": False}

    if data.person_income < 30000:
        return {"approved": False}

    if data.loan_int_rate > 18:
        return {"approved": False}

    # --------------------
    # 2. ПРЕПРОЦЕССИНГ (1 в 1)
    # --------------------

    try:
        education = EDUCATION_MAP[data.person_education]
        home = HOME_MAP[data.person_home_ownership]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid categorical value")

    age = data.age
    rate = data.loan_int_rate

    # слишком большой кредит — мгновенный отказ
    if loan_amnt > person_income * 200:
        return {"approved": False}

    # --------------------
    # 3. ЕСЛИ ДОШЛИ СЮДА → МОДЕЛЬ
    # --------------------

    features = np.array([[
        age,
        person_income,
        loan_amnt,
        rate,
        education,
        home
    ]])

    prediction = model.predict(features)[0]

    return {
        "approved": bool(prediction)
    }

