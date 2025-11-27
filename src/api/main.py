import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

# Import Pydantic schemas
from.schemas import AQIInputFeature, AQIPrediction

# Biến toàn cục để giữ mô hình trong bộ nhớ (cache)
model_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # === KHỞI ĐỘNG ===
    print("API starting up...")
    
    # 1. Tải mô hình MLflow
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        print("CẢNH BÁO: MLFLOW_TRACKING_URI chưa được set. Không thể tải mô hình.")
    else:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        model_name = "hcmc-aqi-predictor"
        model_stage = "Production"
        model_uri = f"models:/{model_name}/{model_stage}"
        
        print(f"Loading model '{model_name}' (Stage: {model_stage}) from {model_uri}")
        try:
            # Tải mô hình "Production" mới nhất từ MLflow Registry [23, 21]
            model = mlflow.pyfunc.load_model(model_uri)
            model_cache["model"] = model
            print(f"Successfully loaded model version: {model.metadata.version}")
        except Exception as e:
            print(f"CRITICAL: Failed to load model on startup. Error: {e}")
    
    yield # API sẵn sàng phục vụ request

    # === TẮT ===
    print("API shutting down...")
    model_cache.clear()

app = FastAPI(
    title="HCMC AQI Forecasting API", 
    description="Dự án MLOps End-to-End",
    version="1.0.0",
    lifespan=lifespan
)

# Gắn công cụ giám sát Prometheus sau khi app được tạo
Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health_check():
    """Kiểm tra sức khoẻ của API và trạng thái mô hình"""
    model_ready = "model" in model_cache and model_cache["model"] is not None
    return {
        "status": "ok", 
        "model_loaded": model_ready,
        "model_version": model_cache["model"].metadata.version if model_ready else None
    }

@app.post("/predict", response_model=AQIPrediction)
async def predict(data: AQIInputFeature):
    """
    Nhận dữ liệu đầu vào và trả về dự đoán PM2.5
    """
    if "model" not in model_cache:
        raise HTTPException(status_code=503, detail="Model is not loaded or unavailable.")

    model = model_cache["model"]
    
    try:
        # Chuyển Pydantic model thành DataFrame [24, 25]
        input_df = pd.DataFrame([data.model_dump()])
        
        # Thực hiện dự đoán
        prediction = model.predict(input_df)
        
        # Đảm bảo kết quả trả về là một float
        predicted_value = float(prediction)

        return AQIPrediction(
            pm25_prediction=predicted_value,
            model_version_used=str(model.metadata.version)
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")