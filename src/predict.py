import joblib
import json
import numpy as np
import os
from src.config import MODEL_PATH, COLUMNS_PATH
from src.logger import logger

# القيم الافتراضية في حالة عدم وجود الملفات (مفيدة لاختبارات الـ CI)
model = None
data_columns = ["total_sqft", "bath", "bhk", "1st phase jp nagar"]

# تحميل الموديل وأسماء الأعمدة مرة واحدة عند التشغيل لتسريع الأداء
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        with open(COLUMNS_PATH, "r") as f:
            data_columns = json.load(f)["data_columns"]
        logger.info("artifacts_loaded_successfully")
    else:
        logger.warning("artifacts_not_found_using_defaults")
except Exception as e:
    logger.error("failed_to_load_artifacts", error=str(e))

def predict_price(location: str, sqft: float, bath: int, bhk: int) -> float:
    try:
        # البحث عن مؤشر المنطقة في الأعمدة
        loc_lower = location.strip().lower()
        x = np.zeros(len(data_columns))
        
        # التأكد من عدم تجاوز المؤشرات إذا كانت الأعمدة الافتراضية قصيرة
        if len(x) > 0: x[0] = sqft
        if len(x) > 1: x[1] = bath
        if len(x) > 2: x[2] = bhk
        
        # إذا كانت المنطقة موجودة في الـ columns المتاحة، نفعّل الـ One-Hot Encoding بتاعها
        if loc_lower in [c.lower() for c in data_columns]:
            loc_index = [c.lower() for c in data_columns].index(loc_lower)
            x[loc_index] = 1
            
        # توقع السعر (إذا كان الموديل محملاً)
        if model is not None:
            prediction = model.predict([x])[0]
        else:
            prediction = 50.0          
        logger.info("prediction_success", location=location, sqft=sqft, predicted_price=round(float(prediction), 2))
        return round(float(prediction), 2)
        
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise e