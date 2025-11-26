from pydantic import BaseModel, Field

# Các đặc trưng (features) này PHẢI khớp với
# các đặc trưng được tạo ra trong `src/pipeline/preprocess.py`
class AQIInputFeature(BaseModel):
    temperature: float = Field(..., example=30.5)
    humidity: float = Field(..., example=80.0)
    co_lag_1h: float = Field(..., example=0.5)
    no2_lag_1h: float = Field(..., example=25.0)
    o3_lag_1h: float = Field(..., example=40.0)
    pm25_lag_1h: float = Field(..., example=85.2)
    pm25_rolling_3h_mean: float = Field(..., example=80.1)
    hour_of_day: int = Field(..., example=14)
    day_of_week: int = Field(..., example=3)
    month_of_year: int = Field(..., example=10)

class AQIPrediction(BaseModel):
    pm25_prediction: float
    model_version_used: str