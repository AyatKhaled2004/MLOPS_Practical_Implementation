import os
import json
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train():
    # 1. استخدام SQLite كـ Backend Store محلي لـ MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Bangalore_House_Price_Prediction")

    # 2. تحميل البيانات وملف الأعمدة
    data_path = os.path.join("artifacts", "bhp.csv")
    columns_path = os.path.join("artifacts", "columns.json")
    model_save_path = os.path.join("artifacts", "banglore_home_prices_model.pickle")

    if not os.path.exists(data_path) or not os.path.exists(columns_path):
        raise FileNotFoundError("Data or columns.json missing in artifacts directory. Run 'dvc pull' first.")

    df = pd.read_csv(data_path)
    
    with open(columns_path, "r") as f:
        data_columns = json.load(f)['data_columns']

    if 'location' in df.columns:
        df = pd.get_dummies(df, columns=['location'], drop_first=False)

    X = pd.DataFrame(0, index=np.arange(len(df)), columns=data_columns)
    for col in data_columns:
        if col in df.columns:
            X[col] = df[col]

    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. بدء تسجيل الجلسة في MLflow
    with mlflow.start_run(run_name="Linear_Regression_Run"):
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # تسجيل الـ Parameters والـ Metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)

        # تسجيل الموديل
        mlflow.sklearn.log_model(model, "house_price_model")

        # حفظ الموديل في artifacts
        with open(model_save_path, "wb") as f:
            pickle.dump(model, f)

        print(f"✅ Training completed! R2: {r2:.4f} | RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train()