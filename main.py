from fastapi import FastAPI
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


@app.get("/")
def index():
    return FileResponse(HTML_FILE)


class LoanRequest(BaseModel):
    age: int
    person_income: float
    loan_amnt: float
    loan_int_rate: float
    person_education: str
    person_home_ownership: str


@app.post("/score")
def score(data: LoanRequest):
    return {"approved": True}
