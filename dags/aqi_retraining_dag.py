from __future__ import annotations
import sys
import os
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

# Thêm `src` vào Python path để Airflow có thể import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.train import run_training
from src.pipeline.evaluate import compare_and_promote_model

# Cấu hình các đường dẫn và biến
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DVC_REMOTE_NAME = "gdrive"
MLFLOW_MODEL_NAME = "hcmc-aqi-predictor"

@dag(
    dag_id="hcmc_aqi_retraining_pipeline",
    default_args={
        "owner": "mlops_team",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "mlflow_tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000"),
    },
    schedule_interval="@daily", # Chạy hàng ngày
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "aqi", "retraining"],
)
def aqi_retraining_dag():
    """
    DAG tự động đào tạo lại, đánh giá và promote mô hình dự báo AQI.
    """

    @task
    def check_new_data():
        """(Mô phỏng) Kiểm tra xem có dữ liệu mới không."""
        print("Checking for new data... (Simulating success)")
        return True # Giả định luôn có dữ liệu mới cho demo

    # Task 1: Pull dữ liệu mới nhất từ DVC
    # Sử dụng BashOperator vì DVC là CLI
    pull_data = BashOperator(
        task_id="pull_latest_data",
        bash_command=f"cd {PROJECT_ROOT} && dvc pull -r {DVC_REMOTE_NAME}",
        doc_md="Kéo dữ liệu mới nhất (đã được cập nhật ở pipeline khác) từ DVC remote."
    )

    @task(task_id="run_model_training")
    def train_model_task(mlflow_tracking_uri: str):
        """Chạy script đào tạo MLflow."""
        print(f"Running training with MLflow server: {mlflow_tracking_uri}")
        # `run_training` là hàm trong `src/pipeline/train.py`
        # Nó sẽ tự động log vào MLflow (autolog)
        new_run_id = run_training(
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name="hcmc-aqi-forecasting"
        )
        if not new_run_id:
            raise ValueError("Training failed or did not return a run_id")
        return new_run_id

    @task(task_id="evaluate_and_promote_model")
    def evaluate_and_promote_task(new_run_id: str, mlflow_tracking_uri: str):
        """So sánh mô hình mới với mô hình 'Production' và promote nếu tốt hơn."""
        print(f"Evaluating new run {new_run_id} against Production model.")
        
        # `compare_and_promote_model` là hàm trong `src/pipeline/evaluate.py`
        promoted = compare_and_promote_model(
            mlflow_tracking_uri=mlflow_tracking_uri,
            model_name=MLFLOW_MODEL_NAME,
            new_run_id=new_run_id,
            comparison_metric="rmse", # Chỉ số để so sánh
            lower_is_better=True # Vì là RMSE
        )
        print(f"Model promotion status: {promoted}")

    # TODO: Thêm task `check_data_drift` (Evidently AI)
    # Tương tự như trên, gọi một hàm Python đã được định nghĩa trong `src/`

    # Định nghĩa luồng chạy
    data_checked = check_new_data()
    data_pulled = pull_data
    model_trained_run_id = train_model_task(
        mlflow_tracking_uri="{{ dag.default_args.mlflow_tracking_uri }}"
    )
    model_promoted = evaluate_and_promote_task(
        new_run_id=model_trained_run_id,
        mlflow_tracking_uri="{{ dag.default_args.mlflow_tracking_uri }}"
    )

    data_checked >> data_pulled >> model_trained_run_id >> model_promoted

# Gọi hàm để khởi tạo DAG
aqi_retraining_dag()