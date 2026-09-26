# https://fastapi.tiangolo.com/ru/tutorial/testing/#using-testclient
# python -m pytest

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_text_sentiment():
    response = client.get("/analyzetext/{Hello world}")
    
    assert response.status_code == 200