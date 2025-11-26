import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Thêm `src` vào path để pytest có thể import `app`
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app, model_cache

# Dữ liệu test hợp lệ (phải khớp `AQIInputFeature`)
valid_input = {
    "temperature": 30.5, "humidity": 80.0, "co_lag_1h": 0.5,
    "no2_lag_1h": 25.0, "o3_lag_1h": 40.0, "pm25_lag_1h": 85.2,
    "pm25_rolling_3h_mean": 80.1, "hour_of_day": 14,
    "day_of_week": 3, "month_of_year": 10
}

@pytest.fixture(scope="module")
def test_client():
    """Tạo TestClient và mock model loading"""
    
    # Tạo một model giả (mock)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([123.45])
    mock_model.metadata = MagicMock()
    mock_model.metadata.version = "v1-mock"
    
    # Patch (ghi đè) hàm `mlflow.pyfunc.load_model`
    with patch("src.api.main.mlflow.pyfunc.load_model", return_value=mock_model):
        # Ghi đè cache của app
        model_cache["model"] = mock_model
        
        # Khởi động client
        client = TestClient(app)
        yield client # Trả về client đã khởi động
        
    # Dọn dẹp
    model_cache.clear()

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert json_data["model_loaded"] == True
    assert json_data["model_version"] == "v1-mock"

def test_predict_success(test_client):
    response = test_client.post("/predict", json=valid_input)
    assert response.status_code == 200
    result = response.json()
    assert result["pm25_prediction"] == 123.45
    assert result["model_version_used"] == "v1-mock"

def test_predict_invalid_input(test_client):
    # Dữ liệu bị lỗi: 'temperature' là string
    invalid_data = valid_input.copy()
    invalid_data["temperature"] = "hot"
    
    response = test_client.post("/predict", json=invalid_data)
    
    # FastAPI/Pydantic sẽ tự động bắt lỗi
    assert response.status_code == 422 # Unprocessable Entity