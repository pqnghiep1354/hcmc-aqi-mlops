import mlflow
from mlflow.tracking import MlflowClient

def compare_and_promote_model(
    mlflow_tracking_uri: str,
    model_name: str,
    new_run_id: str,
    comparison_metric: str = "rmse",
    lower_is_better: bool = True
) -> bool:
    """
    So sánh mô hình mới (từ new_run_id) với mô hình "Production" hiện tại.
    Promote (thăng hạng) mô hình mới nếu nó tốt hơn.
    """
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    # 1. Lấy metric của mô hình mới
    try:
        new_run = client.get_run(new_run_id)
        new_metric = new_run.data.metrics[comparison_metric]
    except Exception as e:
        print(f"Lỗi khi lấy metric cho run mới {new_run_id}: {e}")
        return False

    # 2. Lấy metric của mô hình "Production" hiện tại
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            print(f"Không có mô hình 'Production' nào. Tự động promote mô hình đầu tiên.")
            current_metric = float('inf') if lower_is_better else float('-inf')
        else:
            prod_run_id = prod_versions.run_id
            prod_run = client.get_run(prod_run_id)
            current_metric = prod_run.data.metrics[comparison_metric]
            
    except Exception as e:
        print(f"Lỗi khi lấy mô hình 'Production' hiện tại (tên: {model_name}): {e}")
        # Nếu model_name chưa tồn tại, cũng promote
        current_metric = float('inf') if lower_is_better else float('-inf')

    print(f"So sánh mô hình: Mới ({comparison_metric}={new_metric:.4f}) vs. Production ({comparison_metric}={current_metric:.4f})")

    # 3. Logic so sánh
    is_better = (new_metric < current_metric) if lower_is_better else (new_metric > current_metric)
    
    if is_better:
        print("Mô hình mới tốt hơn. Đang đăng ký và promote...")
        
        # 1. Đăng ký (register) mô hình mới từ run
        model_uri = f"runs:/{new_run_id}/model"
        # Đảm bảo model_name tồn tại (create=True)
        try:
            client.get_registered_model(model_name)
        except:
            client.create_registered_model(model_name)
            
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=new_run_id
        )
        
        # 2. Thêm mô tả
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Auto-promoted by Airflow. Metric ({comparison_metric}): {new_metric:.4f}"
        )
        
        # 3. Chuyển sang "Production" (và lưu trữ version cũ)
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Promoted Version {model_version.version} lên 'Production'.")
        return True
    else:
        print("Mô hình mới không tốt hơn. Không promote.")
        # Xóa version vừa đăng ký (nếu muốn)
        return False