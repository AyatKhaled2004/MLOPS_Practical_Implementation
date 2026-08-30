import joblib
import json
import numpy as np
from src.config import MODEL_PATH, COLUMNS_PATH
from src.logger import logger

# تحميل الموديل وأسماء الأعمدة مرة واحدة عند التشغيل لتسريع الأداء
try:
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH, "r") as f:
        data_columns = json.load(f)["data_columns"]
    logger.info("artifacts_loaded_successfully")
except Exception as e:
    logger.error("failed_to_load_artifacts", error=str(e))

def predict_price(location: str, sqft: float, bath: int, bhk: int) -> float:
    try:
        # البحث عن مؤشر المنطقة في الأعمدة
        loc_lower = location.strip().lower()
        x = np.zeros(len(data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        
        # إذا كانت المنطقة موجودة في الـ columns المتاحة، نفعّل الـ One-Hot Encoding بتاعها
        if loc_lower in [c.lower() for c in data_columns]:
            loc_index = [c.lower() for c in data_columns].index(loc_lower)
            x[loc_index] = 1
            
        # توقع السعر
        prediction = model.predict([x])[0]
        logger.info("prediction_success", location=location, sqft=sqft, predicted_price=round(float(prediction), 2))
        return round(float(prediction), 2)
        
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise e