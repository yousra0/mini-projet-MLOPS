# 🚀 Mini-Projet MLOps - Churn Prediction

A complete MLOps project for predicting customer churn using machine learning, featuring model tracking with MLflow, API deployment with FastAPI, monitoring with Elasticsearch and Kibana, and containerization with Docker.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Monitoring](#monitoring)
- [Docker Deployment](#docker-deployment)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project implements a complete MLOps pipeline for customer churn prediction using Random Forest classifier. It includes:
- Machine learning model training and evaluation
- Model versioning and tracking with MLflow
- REST API for predictions using FastAPI
- Logging and monitoring with Elasticsearch and Kibana
- Container orchestration with Docker Compose
- System monitoring with cAdvisor

## ✨ Features

- **Machine Learning Pipeline**: Data preparation, model training, and evaluation
- **Model Registry**: Version control and staging with MLflow
- **REST API**: FastAPI-based prediction service
- **Experiment Tracking**: MLflow for tracking experiments, parameters, and metrics
- **Logging**: Centralized logging with Elasticsearch
- **Monitoring Dashboard**: Kibana for log visualization
- **Containerization**: Docker and Docker Compose for easy deployment
- **System Monitoring**: cAdvisor for container resource monitoring

## 📁 Project Structure

```
mini-projet-MLOPS/
├── app.py                    # FastAPI application for predictions
├── main.py                   # Main training pipeline with MLflow integration
├── model_pipeline.py         # Model training, evaluation, and data preparation
├── logger_config.py          # Logging configuration with Elasticsearch
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker image for the application
├── Dockerfilemlflow         # Docker image for MLflow server
├── docker-compose.yml       # Multi-container orchestration
├── makefile                 # Build and run commands
├── .env                     # Environment variables
├── churn-bigml-80.csv      # Training dataset
├── churn-bigml-20.csv      # Test dataset
└── tests/                   # Unit tests
```

## 🔧 Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Git

## 💻 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yousra0/mini-projet-MLOPS.git
cd mini-projet-MLOPS
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file with necessary configuration variables.

## 🚀 Usage

### Start the MLflow server
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

### Train the model
```bash
python main.py --data churn-bigml-80.csv --train --evaluate --save random_forest_model.joblib
```

### Run the FastAPI application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or simply:
```bash
python app.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns the health status of the API and model loading status.

### Hello World
```bash
GET /
```
Welcome endpoint.

### Make Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": [value1, value2, value3, ...]
}
```

Example using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}'
```

## 🎓 Model Training

### Training with MLflow tracking
```bash
python main.py --data churn-bigml-80.csv --train --evaluate --stage Production
```

### Available arguments:
- `--data`: Path to the training CSV file (required)
- `--train`: Flag to train the model
- `--evaluate`: Flag to evaluate the model
- `--save`: Path to save the trained model (default: random_forest_model.joblib)
- `--load`: Path to load an existing model
- `--stage`: MLflow model stage (Production or Staging)

### View experiments
Open your browser and navigate to:
```
http://localhost:5000
```

## 📊 Monitoring

### Start monitoring stack with Docker Compose
```bash
docker-compose up -d
```

This will start:
- **Elasticsearch** (port 9200): Log storage and indexing
- **Kibana** (port 5601): Log visualization dashboard
- **cAdvisor** (port 8080): Container resource monitoring

### Access monitoring dashboards:
- Kibana: http://localhost:5601
- cAdvisor: http://localhost:8080
- Elasticsearch: http://localhost:9200

## 🐳 Docker Deployment

### Build Docker images
```bash
docker build -t churn-prediction-app -f Dockerfile .
docker build -t mlflow-server -f Dockerfilemlflow .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

### Stop services
```bash
docker-compose down
```

## 🛠️ Technologies Used

- **Python 3.12**: Programming language
- **FastAPI**: Modern web framework for building APIs
- **MLflow**: Experiment tracking and model registry
- **scikit-learn**: Machine learning library
- **pandas & numpy**: Data manipulation and analysis
- **Elasticsearch**: Log storage and search engine
- **Kibana**: Data visualization and exploration
- **Docker**: Containerization platform
- **cAdvisor**: Container resource monitoring
- **Uvicorn**: ASGI server for FastAPI
- **Joblib**: Model serialization

## 📈 Model Details

- **Algorithm**: Random Forest Classifier
- **Dataset**: Customer churn data (churn-bigml)
- **Features**: 20 input features
- **Target**: Binary classification (churn/no churn)
- **Metrics tracked**: Accuracy, Precision, Recall, F1-Score

## 🧪 Testing

Run tests using pytest:
```bash
pytest tests/
```

## 📝 Logging

The application logs are sent to Elasticsearch for centralized monitoring. Logs include:
- Prediction requests and responses
- Model loading status
- Error messages and exceptions
- MLflow experiment metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Yousra - [@yousra0](https://github.com/yousra0)

Project Link: [https://github.com/yousra0/mini-projet-MLOPS](https://github.com/yousra0/mini-projet-MLOPS)

## 📄 License

This project is open source and available for educational purposes.

---

**Made with ❤️ for MLOps learning and practice**
