from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.routes import router
from src.predict import model_service
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

app.include_router(router)