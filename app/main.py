from fastapi import FastAPI
from sqlalchemy import text

from app.db import engine

app = FastAPI(title="Doc QA Assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": "0.1.0"}


@app.get("/db-health")
def db_health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"database": "ok"}