from pathlib import Path

# تحديد المسار الرئيسي للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent

# تحديد مسارات المجلد والملفات جوه artifacts
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "banglore_home_prices_model.pickle"
COLUMNS_PATH = ARTIFACTS_DIR / "columns.json"