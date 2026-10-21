<<<<<<< HEAD
# 🚗 FlowCast: Modular Traffic Prediction Pipeline

FlowCast is an end-to-end Machine Learning and Deep Learning pipeline for real-time traffic volume forecasting. It bridges classical ensemble algorithms (XGBoost, Random Forest, Ridge, Linear Regression, SVR) with sequence-based deep learning architectures (LSTM, GRU) inside a dynamic Streamlit web dashboard.

## 🛠️ Project Architecture
FlowCast/
├── config.yaml                    # Global pipeline configuration
├── data/                          # Raw & processed data storage
├── models/                        # Saved model artifacts (.joblib, .pth)
├── src/
│   ├── utils.py                   # Config loaders and logging utilities
│   ├── ingest.py                  # Raw data ingestion pipeline
│   ├── clean.py                   # Data cleaning and missing value handling
│   ├── features.py                # Feature engineering & spatial-temporal transformations
│   ├── ml_models.py               # Classical ML GridSearch training pipeline
│   ├── dl_models.py               # PyTorch LSTM & GRU model architectures
│   └── evaluate.py                # Model evaluation and leaderboard generation
└── dashboard/
└── app.py                     # Streamlit multi-tab web application

## ⚙️ Quickstart & Reproduction

 1. Environment Setup
bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

2. Run Data Ingestion, Cleaning & Feature Engineering
Bash
python -m src.ingest
python -m src.clean
python -m src.features


3. Train Classical ML Models & Hyperparameter Tuning
Train XGBoost, Random Forest, Ridge, Linear Regression, and SVR using 3-fold cross-validation:

python -m src.ml_models

4. Evaluate & Generate Benchmark Leaderboard
Run evaluation across all models on the test set split:

python -m src.evaluate

5. Launch Interactive Streamlit Dashboard
streamlit run dashboard/app.py

📊 Features & Dashboard Views
⚡ Live Inference: Interactive sidebar controls for feature sliders, coordinates, and real-time model switching.

📁 Data Upload Module: Batch prediction support for uploaded CSV and Parquet traffic data files.

📊 Model Leaderboard: Head-to-head metric benchmarking (RMSE, MAE, R²) and artifact status checks.

📝 Reporting: System summary reporting and file exports.

🏆 Model Leaderboard Highlights
Classical Champion: XGBoost / Random Forest (Low RMSE, high non-linear sensitivity).

Deep Sequence Model: PyTorch LSTM (Captures temporal historical lag dynamics).
=======
# FlowCast-AI
>>>>>>> 92e2dce133705f585add1bd8e353c290fd8d619f
