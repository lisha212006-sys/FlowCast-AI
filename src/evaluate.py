from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.utils import load_config, setup_logger
from src.dl_models import TrafficLSTM, TrafficGRU, TrafficSequenceDataset
from torch.utils.data import DataLoader

logger = setup_logger("FlowCast.Evaluation")

def evaluate_pipeline(config_path: str = "config.yaml"):
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_data_dir"])
    data_path = processed_dir / "flowcast_analysis_ready.parquet"

    logger.info(f"Loading dataset for evaluation from {data_path}")
    df = pd.read_parquet(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    target_col = "traffic_volume"
    drop_cols = [
        target_col, "timestamp", "road_name", "date", 
        "date_x", "date_y", "time_x", "time_y", 
        "weather_condition", "congestion_level"
    ]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X_raw = df[feature_cols].select_dtypes(include=[np.number]).values
    y_raw = df[target_col].values

    split_idx = int(len(df) * 0.8)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[:split_idx])
    X_test = scaler.transform(X_raw[split_idx:])
    
    y_train = y_raw[:split_idx]
    y_test = y_raw[split_idx:]

    models_dir = Path("models")
    results = []

    model_filenames = {
        "Linear Regression": "linear_regression.pkl",
        "Ridge": "ridge.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }

    for name, filename in model_filenames.items():
        model_path = models_dir / filename
        if not model_path.exists():
            logger.warning(f"Model artifact not found for {name} at {model_path}. Skipping.")
            continue
            
        logger.info(f"Evaluating Classical ML Model: {name}")
        model = joblib.load(model_path)
         
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results.append({
            "Model": name,
            "Type": "Classical ML",
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    lookback = 24
    batch_size = 64
    test_dataset = TrafficSequenceDataset(X_test, y_test, lookback=lookback)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_test.shape[1]

    dl_models = {
        "LSTM": ("flowcast_lstm.pth", TrafficLSTM(input_dim=input_dim)),
        "GRU": ("flowcast_gru.pth", TrafficGRU(input_dim=input_dim))
    }

    for name, (filename, net) in dl_models.items():
        model_path = models_dir / filename
        if not model_path.exists():
            logger.warning(f"DL Model artifact not found for {name} at {model_path}. Skipping.")
            continue

        logger.info(f"Evaluating Deep Learning Model: {name}")
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()

        preds_list, targets_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = net(X_batch)
                preds_list.extend(preds.cpu().numpy().flatten())
                
        y_test_aligned = y_test[lookback:]
        preds_arr = np.array(preds_list)
        
        if len(preds_arr) != len(y_test_aligned):
            y_test_aligned = y_test_aligned[:len(preds_arr)]
            preds_arr = preds_arr[:len(y_test_aligned)]

        rmse = np.sqrt(mean_squared_error(y_test_aligned, preds_arr))
        mae = mean_absolute_error(y_test_aligned, preds_arr)
        r2 = r2_score(y_test_aligned, preds_arr)

        results.append({
            "Model": name,
            "Type": "Deep Learning",
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

    if results:
        results_df = pd.DataFrame(results).sort_values(by="RMSE")
        print("\n" + "="*65)
        print("🏆 FLOWCAST MODEL PERFORMANCE LEADERBOARD 🏆")
        print("="*65)
        print(results_df.to_string(index=False))
        print("="*65 + "\n")
        results_df.to_csv("models/model_leaderboard_summary.csv", index=False)
        logger.info("Saved leaderboard summary to 'models/model_leaderboard_summary.csv'")
    else:
        logger.warning("No model results found to display.")
def save_evaluation_report(results_dict):
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_df = pd.DataFrame(results_dict)
    report_path = reports_dir / "model_leaderboard_summary.csv"
    leaderboard_df.to_csv(report_path, index=False)
    print(f"✅ Evaluation leaderboard saved successfully to {report_path}")

if __name__ == "__main__":
    sample_results = [
        {"Model": "XGBoost", "RMSE": 42.15, "MAE": 31.02, "R2": 0.88},
        {"Model": "Random Forest", "RMSE": 44.30, "MAE": 32.50, "R2": 0.85},
        {"Model": "LSTM", "RMSE": 39.80, "MAE": 29.40, "R2": 0.91}
    ]
    save_evaluation_report(sample_results)

if __name__ == "__main__":
    evaluate_pipeline()