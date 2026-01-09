from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from routers import features

assert load_dotenv(dotenv_path=Path(".env"))

app = FastAPI()


@app.get("/ping")
def ping():
    """Liveliness validation endpoint"""
    return "pong"


app.include_router(features.router)
