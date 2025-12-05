# 🌤️ End-to-End MLOps: HCMC Air Quality Forecasting System

> **A production-grade MLOps system designed to predict PM2.5 air quality indices in Ho Chi Minh City.**  
> *Fully containerized, automated, and monitored.*

-----

## 📖 Project Overview

Ho Chi Minh City faces significant environmental challenges, particularly with fine particulate matter (PM2.5) pollution. This project is not just a machine learning model; it is a **complete MLOps platform** that automates the lifecycle of air quality forecasting.

By leveraging a microservices architecture, this system ensures:

  * **Reproducibility:** From data ingestion to model deployment.
  * **Automation:** Daily retraining pipelines via Apache Airflow.
  * **Observability:** Real-time monitoring of model health and API performance.
  * **Scalability:** Container-based deployment using Docker.

## 🏗️ System Architecture

The system follows a modular architecture where each component is isolated in its own container:mermaid
graph TD
Data --\>|Ingest| Airflow
Airflow --\>|Trigger| Training
Training --\>|Log Metrics| MLflow
Training --\>|Save Artifacts| MinIO
MLflow --\>|Register Model| Registry
Registry --\>|Load Prod Model| FastAPI
User --\>|Request| FastAPI
FastAPI --\>|Metrics| Prometheus
Prometheus --\>|Visualize| Grafana

````

### 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | **Apache Airflow** | Schedules ETL & Training pipelines (DAGs). |
| **Experiment Tracking** | **MLflow** | Tracks parameters, metrics, and manages model versions. |
| **Model Serving** | **FastAPI** | High-performance Async REST API for real-time predictions. |
| **Artifact Store** | **MinIO** | S3-compatible object storage for model artifacts & data. |
| **Metadata Store** | **PostgreSQL** | Backend database for Airflow and MLflow. |
| **Monitoring** | **Prometheus & Grafana** | System health and API latency visualization. |
| **Data Versioning** | **DVC** | Versions large datasets and ensures reproducibility. |
| **Containerization** | **Docker & Compose** | Encapsulates the entire stack for "One-Command" deployment. |

---

## 📂 Project Structure

```plaintext
├──.github/workflows/   # CI/CD Pipelines (GitHub Actions)
├── dags/                # Airflow DAGs (Pipeline definitions)
├── docker/              # Dockerfiles for specific services
├── monitoring/          # Prometheus & Grafana configurations
├── src/                 # Core Source Code
│   ├── api/             # FastAPI application logic
│   ├── data/            # Data ingestion & preprocessing scripts
│   ├── models/          # Training & Evaluation logic
├── tests/               # Unit & Integration tests (Pytest)
├── docker-compose.yml   # Main infrastructure definition
├── requirements.txt     # Python dependencies
└── README.md            # Project Documentation
````

-----

## 🚀 Getting Started

Follow these steps to get the system running on your local machine.

### Prerequisites

  * **Docker Desktop** (Allocated at least 4GB RAM)
  * **Git**
  * **Python 3.9+** (Optional, for local development)

### 1\. Clone the Repository

```bash
git clone [https://github.com/your-username/hcmc-air-quality-mlops.git](https://github.com/your-username/hcmc-air-quality-mlops.git)
cd hcmc-air-quality-mlops
```

### 2\. Setup Data (DVC)

Pull the tracked dataset from the remote storage (or local cache).

```bash
# If you have configured DVC remote
dvc pull
```

*(Note: For this demo, sample data is included in `data/raw` if DVC is not configured).*

### 3\. Launch the System

Build and start all services using Docker Compose.

```bash
docker-compose up -d --build
```

*Wait for a few minutes for all containers to initialize (especially Postgres and Airflow).*

### 4\. Initialization (First Run Only)

Ensure the MLflow bucket exists in MinIO.

```bash
# This is usually handled automatically by the 'create-buckets' service,
# but can be verified manually if needed.
docker exec -it mlops_minio mc mb local/mlflow-bucket |

| true
```

-----

## 🖥️ Accessing the Services

Once the system is up, access the interfaces via your browser:

| Service | URL | Default Credentials | Description |
| :--- | :--- | :--- | :--- |
| **Airflow UI** | `http://localhost:8080` | `admin` / `adminparams` | Manage pipelines & DAGs |
| **MLflow UI** | `http://localhost:5000` | N/A | View experiments & models |
| **FastAPI Docs** | `http://localhost:8000/docs` | N/A | Test prediction API |
| **MinIO UI** | `http://localhost:9001` | `minioadmin` / `minioadmin` | View stored artifacts |
| **Grafana** | `http://localhost:3000` | `admin` / `admin` | Monitor dashboards |
| **Prometheus** | `http://localhost:9090` | N/A | Raw metrics query |

-----

## 🧪 Usage Guide

### 1\. Trigger the Pipeline

1.  Go to **Airflow UI** (`http://localhost:8080`).
2.  Enable the `hcmc_air_quality_training` DAG.
3.  Click the **Trigger DAG** (Play button) to start the pipeline manually.
4.  Watch the pipeline move from `Ingest` -\> `Preprocess` -\> `Train` -\> `Register`.

### 2\. Check Model Registry

1.  Go to **MLflow UI** (`http://localhost:5000`).
2.  Navigate to the **Models** tab.
3.  You should see `HCMC_PM25_RF_Model` registered.
4.  Ensure the latest version has the **Production** tag (or manually transition it).

### 3\. Make a Prediction

Use the FastAPI Swagger UI or `curl`:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "temperature": 32.5,
  "humidity": 75.0,
  "wind_speed": 3.5,
  "hour_of_day": 14,
  "month": 5
}'
```

### 4\. Monitor Health

Go to **Grafana** (`http://localhost:3000`) and open the **FastAPI Monitoring Dashboard** to see Request Rate and Latency charts updating in real-time.

-----

## 🛡️ Quality Assurance

### Running Tests

Unit tests are integrated into the Docker build, but you can run them locally:

```bash
pip install -r requirements.txt
pytest tests/
```

### CI/CD

This project uses **GitHub Actions** for Continuous Integration. Every push to the `main` branch triggers:

1.  Environment setup.
2.  Linter checks (Flake8).
3.  Unit tests execution.
4.  Docker image build verification.

-----

## 🔮 Future Improvements

  * **Drift Detection:** Integrate *Evidently AI* to detect data drift and trigger retraining automatically.
  * **Kubernetes:** Migrate from Docker Compose to Helm Charts for K8s deployment.
  * **Feature Store:** Implement *Feast* for managing features consistency between training and serving.

-----

## 🤝 Contributing

Contributions are welcome\! Please fork the repository and create a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

-----

*Built with ❤️ for a cleaner Ho Chi Minh City.*

```

3.  **Bảng (Table):** Dùng bảng để liệt kê Tech Stack và Access URL giúp thông tin rõ ràng, dễ tra cứu.
4.  **One-Command Setup:** Nhấn mạnh vào tính dễ dàng khi triển khai (`docker-compose up`).
5.  **Context:** Có phần giải thích bối cảnh TP.HCM để dự án có ý nghĩa thực tế (AI for Good).
```
