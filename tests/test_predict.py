import pytest
from src.predict import predict_price

def test_predict_price_valid_input():
    result = predict_price(location="1st Phase JP Nagar", sqft=1000.0, bath=2, bhk=2)
    assert isinstance(result, float)
    assert result > 0.0

def test_predict_price_unknown_location():
    result = predict_price(location="non_existent_location_123", sqft=1000.0, bath=2, bhk=2)
    assert isinstance(result, float)
    assert result > 0.0