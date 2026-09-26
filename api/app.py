# https://fastapi.tiangolo.com/#example
# uvicorn api.app:app --reload
# http://127.0.0.1:8000/docs

from fastapi import FastAPI
from text_sentiment.main import sentiment_func

app = FastAPI()

@app.get("/analyzetext/{text}")
def analyze_text(text: str):
    return sentiment_func(text)