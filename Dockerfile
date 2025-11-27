# Sử dụng base image chính thức của FastAPI cho production [37]
# Image này đã bao gồm Gunicorn, Uvicorn và các tối ưu hóa
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# Thiết lập thư mục làm việc
WORKDIR /app

# Tối ưu Layer Caching [38, 39]
# 1. Sao chép file requirements trước
COPY ./requirements.txt /app/requirements.txt

# 2. Cài đặt thư viện
# --no-cache-dir để giữ image nhỏ gọn [40]
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 3. Sao chép toàn bộ mã nguồn `src` vào thư mục `/app/src`
COPY ./src /app/src
# Sao chép các file config mà API có thể cần
COPY ./params.yaml /app/params.yaml

# Base image này [37] sẽ tự động tìm `app` trong `src/api/main.py`
# và chạy bằng Gunicorn + Uvicorn
# Biến môi trường (như MLFLOW_TRACKING_URI) sẽ được inject bởi Cloud Run