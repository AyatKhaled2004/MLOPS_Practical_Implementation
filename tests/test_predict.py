import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.predict import predict_price


def test_predict_price_valid_input():
    result = predict_price(location="1st Phase JP Nagar", sqft=1000.0, bath=2, bhk=2)
    assert isinstance(result, float)
    assert result > 0.0


def test_predict_price_unknown_location():
    result = predict_price(location="non_existent_location_123", sqft=1000.0, bath=2, bhk=2)
    assert isinstance(result, float)
    assert result > 0.0


client = TestClient(app)


def test_predict_endpoint_valid_request():
    response = client.post("/predict", json={
        "location": "1st Phase JP Nagar",
        "sqft": 1000.0,
        "bath": 2,
        "bhk": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)


def test_predict_endpoint_invalid_sqft():
    response = client.post("/predict", json={
        "location": "1st Phase JP Nagar",
        "sqft": 50.0,
        "bath": 2,
        "bhk": 2
    })
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_invalid_bhk():
    response = client.post("/predict", json={
        "location": "1st Phase JP Nagar",
        "sqft": 1000.0,
        "bath": 2,
        "bhk": 15
    })
    assert response.status_code == 422


def test_predict_endpoint_missing_field():
    response = client.post("/predict", json={
        "sqft": 1000.0,
        "bath": 2,
        "bhk": 2
    })
    assert response.status_code == 422


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"