from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.schemas import PricePredictionRequest, PricePredictionResponse
from src.predict import model_service, predict_price
from src.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # بيتنفذ مرة واحدة بس، لحظة ما السيرفر يبدأ يشتغل
    model_service.load()
    logger.info("app_startup_model_loaded")
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="Bangalore Housing Price Prediction API",
    description="Production API for predicting property prices using Machine Learning",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Bangalore Real Estate API is running"}


@app.post("/predict", response_model=PricePredictionResponse)
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