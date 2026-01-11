from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_FILE = os.path.join(BASE_DIR, "index3.html")


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
    # 1. Базовая валидация
    # --------------------

    if data.age < 18 or data.age > 100:
        return {"approved": False}

    if data.person_income <= 0 or data.loan_amnt <= 0:
        return {"approved": False}

    # --------------------
    # 2. Препроцессинг (1 в 1 как в обучении)
    # --------------------

    try:
        education = EDUCATION_MAP[data.person_education]
        home = HOME_MAP[data.person_home_ownership]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid categorical value")

    # Преобразования валюты (как в обучении)
    loan_amnt = data.loan_amnt * 92.93
    person_income = ((data.person_income * 92.93) // 12) / 5.051965139984243

    age = data.age
    rate = data.loan_int_rate

    # --------------------
    # 3. ЛОГИКА ВМЕСТО МОДЕЛИ
    # --------------------
    # Имитируем здравый смысл модели

    approved = True

    # слишком молодой
    if age < 21:
        approved = False

    # слишком большой кредит относительно дохода
    if loan_amnt > person_income * 20:
        approved = False

    # высокий процент
    if rate > 18:
        approved = False

    # образование + жильё как бонус
    if education == 1 and home == 1:
        approved = False

    if person_income < 30000:
        approved = False

    return {
        "approved": approved
    }
