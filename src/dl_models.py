from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import load_config, setup_logger

logger = setup_logger("FlowCast.DeepLearning")

class TrafficSequenceDataset(Dataset):
    def __init__(self, X, y, lookback=24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.lookback]
        y_target = self.y[idx + self.lookback]
        return x_window, y_target

class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class TrafficGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(TrafficGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,  
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def train_and_evaluate_model(model, model_name, train_loader, test_loader, device, epochs=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"=== Training {model_name} ===")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f"[{model_name}] Epoch [{epoch+1}/{epochs}] - Loss (MSE): {epoch_loss:.4f}")

    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            preds_list.extend(preds.cpu().numpy())
            targets_list.extend(y_batch.numpy())

    preds_arr = np.array(preds_list)
    targets_arr = np.array(targets_list)
    rmse = np.sqrt(mean_squared_error(targets_arr, preds_arr))
    mae = mean_absolute_error(targets_arr, preds_arr)
    r2 = r2_score(targets_arr, preds_arr)
    
    logger.info(f"[{model_name} Test Results] -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    return model, rmse

def run_deep_learning_pipeline(config_path: str = "config.yaml"):
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_data_dir"])
    data_path = processed_dir / "flowcast_analysis_ready.parquet"

    logger.info(f"Loading dataset from {data_path}")
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

    lookback = 24
    batch_size = 64

    train_dataset = TrafficSequenceDataset(X_train, y_train, lookback=lookback)
    test_dataset = TrafficSequenceDataset(X_test, y_test, lookback=lookback)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    input_dim = X_train.shape[1]
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    lstm_model = TrafficLSTM(input_dim=input_dim).to(device)
    trained_lstm, lstm_rmse = train_and_evaluate_model(
        lstm_model, "LSTM", train_loader, test_loader, device
    )
    torch.save(trained_lstm.state_dict(), models_dir / "flowcast_lstm.pth")

    gru_model = TrafficGRU(input_dim=input_dim).to(device)
    trained_gru, gru_rmse = train_and_evaluate_model(
        gru_model, "GRU", train_loader, test_loader, device
    )
    torch.save(trained_gru.state_dict(), models_dir / "flowcast_gru.pth")

    if lstm_rmse < gru_rmse:
        logger.info(f"🏆 Deep Learning Winner: LSTM (RMSE: {lstm_rmse:.4f})")
    else:
        logger.info(f"🏆 Deep Learning Winner: GRU (RMSE: {gru_rmse:.4f})")
    logger.info("Saved both LSTM and GRU model artifacts to 'models/'")

if __name__ == "__main__":
    run_deep_learning_pipeline()