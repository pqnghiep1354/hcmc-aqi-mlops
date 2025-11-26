import mlflow
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import os

PROCESSED_DATA_PATH = "data/processed/processed_aqi.csv"
PARAMS_PATH = "params.yaml"

def load_params():
    """Tải tham số từ params.yaml"""
    with open(PARAMS_PATH, 'r') as f:
        return yaml.safe_load(f)

def run_training(mlflow_tracking_uri: str, experiment_name: str) -> str | None:
    """
    Hàm chính để chạy pipeline đào tạo và log bằng MLflow.
    Trả về run_id nếu thành công.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    # Bật tự động log cho Scikit-learn (bao gồm XGBoost) [28]
    mlflow.sklearn.autolog(log_models=True)

    # Tải dữ liệu và tham số
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    params = load_params().get("train", {})
    
    # Xác định features (X) và target (y)
    # Các features PHẢI khớp với `AQIInputFeature` trong schemas.py
    features = [
        'temperature', 'humidity', 'co_lag_1h', 'no2_lag_1h', 'o3_lag_1h',
        'pm25_lag_1h', 'pm25_rolling_3h_mean', 'hour_of_day', 
        'day_of_week', 'month_of_year'
    ]
    target = 'target_pm25_next_1h'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # Time-series data không nên xáo trộn

    with mlflow.start_run() as run:
        print(f"Starting MLflow run: {run.info.run_id}")
        
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
            early_stopping_rounds=params.get("early_stopping_rounds", 10)
        )
        
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred = model.predict(X_test)
        
        # Log metrics thủ công (dù autolog đã làm)
        # Các metrics này rất quan trọng cho dự báo [31, 32, 13]
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        print(f"Model trained. RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        return run.info.run_id