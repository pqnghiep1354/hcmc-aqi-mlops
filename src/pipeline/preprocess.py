import pandas as pd
import os

RAW_DATA_PATH = "data/raw/hcmc_aqi.csv"
PROCESSED_DATA_PATH = "data/processed/processed_aqi.csv"

def create_features(df):
    """Tạo các đặc trưng time-series."""
    df_out = df.copy()
    
    # Chuyển đổi thời gian
    # SỬA LỖI: Thay 'timestamp' bằng tên cột ngày tháng thực tế trong file CSV của bạn. Ví dụ: 'Date'
    df_out['timestamp'] = pd.to_datetime(df_out['date'], dayfirst=True)
    df_out = df_out.set_index('timestamp').sort_index()
    
    # Tạo đặc trưng từ thời gian
    df_out['hour_of_day'] = df_out.index.hour
    df_out['day_of_week'] = df_out.index.dayofweek
    df_out['month_of_year'] = df_out.index.month
    
    # SỬA LỖI: Cập nhật tên cột cho khớp với file CSV của bạn
    pollutants = ['PM2.5', 'CO', 'NO2', 'O3']
    weather = ['Temperature', 'Humidity']
    
    # Tạo đặc trưng trễ (Lag Features)
    for col in pollutants + weather:
        # Giữ tên nhất quán (ví dụ: 'pm2.5' -> 'pm25_lag_1h')
        col_name_lower = col.lower().replace('.', '')
        df_out[f'{col_name_lower}_lag_1h'] = df_out[col].shift(1)

    # Tạo đặc trưng trượt (Rolling Features)
    df_out['pm25_rolling_3h_mean'] = df_out['PM2.5'].shift(1).rolling(window=3).mean()
    
    # Tạo biến mục tiêu: dự đoán PM2.5 1 giờ sau
    df_out['target_pm25_next_1h'] = df_out['PM2.5'].shift(-1)
    
    # Xóa các dòng NaN (do lag/rolling/target)
    df_out = df_out.dropna()
    
    return df_out

def run_preprocessing():
    """Chạy toàn bộ pipeline tiền xử lý."""
    print("Loading raw data...")
    # Tải dữ liệu thô (từ Mendeley )
    # Đảm bảo file CSV có cột 'timestamp' và các cột chỉ số (PM25, CO,...)
    df_raw = pd.read_csv(RAW_DATA_PATH)

    # (Tùy chọn) Thêm dòng này để kiểm tra tên các cột trong file CSV của bạn
    print("Columns in the raw data:", df_raw.columns.tolist())
    
    print("Creating features...")
    df_processed = create_features(df_raw)
    
    print(f"Saving processed data to {PROCESSED_DATA_PATH}")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_PATH)
    print("Preprocessing finished.")

if __name__ == "__main__":
    run_preprocessing()