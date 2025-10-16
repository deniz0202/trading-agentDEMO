# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "KI-Trading-Agent läuft erfolgreich 🚀"}

