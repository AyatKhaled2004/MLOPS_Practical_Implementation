from fastapi import APIRouter, HTTPException
from src.schemas import PricePredictionRequest, PricePredictionResponse
from src.predict import predict_price
from src.logger import logger

router = APIRouter()


@router.get("/")
def read_root():
    return {"status": "ok", "message": "Bangalore Real Estate API is running"}


@router.post("/predict", response_model=PricePredictionResponse)
def predict(request: PricePredictionRequest):
    try:
        price = predict_price(
            location=request.location,
            sqft=request.sqft,
            bath=request.bath,
            bhk=request.bhk
        )
        return PricePredictionResponse(predicted_price=price)

    except Exception as e:
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server prediction error")