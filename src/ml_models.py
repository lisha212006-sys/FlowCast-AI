from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import load_config, setup_logger
import joblib
import numpy as np
import pandas as pd

logger = setup_logger("FlowCast.MLTuning")

def load_data(config_path: str = "config.yaml"):
  config = load_config(config_path)
  processed_dir = Path(config["paths"]["processed_data_dir"])
  data_path = processed_dir / "flowcast_analysis_ready.parquet"

  logger.info(f"Loading dataset from {data_path}")
  df = pd.read_parquet(data_path)
  
  if len(df) == 0:
    raise ValueError(f"The dataset at {data_path} contains 0 rows! Please re-run your feature engineering pipeline.")

  logger.info(f"Successfully loaded dataset with {len(df):,} rows and {len(df.columns)} columns.")

  df["timestamp"] = pd.to_datetime(df["timestamp"])
  df = df.sort_values("timestamp")

  # Ensure latitude and longitude are explicitly numeric if present in dataframe
  for coord_col in ["latitude", "longitude"]:
      if coord_col in df.columns:
          df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce").fillna(df[coord_col].mean())

  target_col = "traffic_volume"
  drop_cols = [
      target_col,
      "timestamp",
      "road_name",
      "date",
      "date_x",
      "date_y",
      "time_x",
      "time_y",
      "weather_condition",
      "congestion_level",
  ]
  feature_cols = [col for col in df.columns if col not in drop_cols]

  X = df[feature_cols].select_dtypes(include=[np.number])
  y = df[target_col]
  
  logger.info(f"Filtered to {X.shape[1]} numeric feature columns for training (including coordinates if present).")
  split_idx = int(len(df) * 0.8)
  logger.info(f"Train split size: {split_idx}, Test split size: {len(df) - split_idx}")
  
  return (
      X.iloc[:split_idx],
      X.iloc[split_idx:],
      y.iloc[:split_idx],
      y.iloc[split_idx:],
  )

def run_training_and_tuning():
  X_train, X_test, y_train, y_test = load_data()

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  model_configs = {
      "Linear Regression": {
          "model": LinearRegression(),
          "params": {},
          "data_scaled": True,
      },
      "Ridge": {
         "model": Ridge(),
         "params": {
            "alpha": [0.1, 1.0, 10.0]},
            "data_scaled": True,
      },
      "Decision Tree": {
          "model": DecisionTreeRegressor(random_state=42),
          "params": {
              "max_depth": [4, 6, 8, None],
              "min_samples_split": [2, 5],
          },
          "data_scaled": False,
      },
      "Random Forest": {
          "model": RandomForestRegressor(random_state=42, n_jobs=-1),
          "params": {
              "n_estimators": [50, 100],
              "max_depth": [6, 10, None],
          },
          "data_scaled": False,
      },
      "XGBoost": {
          "model": XGBRegressor(random_state=42, n_jobs=-1),
          "params": {
              "n_estimators": [50, 100],
              "learning_rate": [0.05, 0.1],
              "max_depth": [4, 6],
          },
          "data_scaled": False,
      },
      "Support Vector Machine (SVR)": {
          "model": SVR(),
          "params": {"C": [1.0], "epsilon": [0.1]},
          "data_scaled": True,
      }, 
  }

  best_overall_rmse = float("inf")
  best_overall_model = None
  best_model_name = ""
  models_dir = Path("models")
  models_dir.mkdir(parents=True, exist_ok=True)

  logger.info("=== Starting Grid Search Tuning Across All ML Models ===")

  for name, config in model_configs.items():
    logger.info(f"Tuning {name}...")

    xtr = X_train_scaled if config["data_scaled"] else X_train
    xte = X_test_scaled if config["data_scaled"] else X_test

    if name == "Support Vector Machine (SVR)":
        logger.info("Subsampling training data to 5,000 rows for SVR...")
        sample_idx = np.random.choice(len(xtr), size=min(5000, len(xtr)), replace=False)
        if hasattr(xtr, "iloc"):
            xtr_fit = xtr.iloc[sample_idx]
        else:
            xtr_fit = xtr[sample_idx]
        y_train_fit = y_train.iloc[sample_idx] if hasattr(y_train, "iloc") else y_train[sample_idx]
    else:
        xtr_fit, y_train_fit = xtr, y_train

    if config["params"]:
      grid_search = GridSearchCV(
          estimator=config["model"],
          param_grid=config["params"],
          cv=3,
          scoring="neg_root_mean_squared_error",
          n_jobs=-1,
      )
      grid_search.fit(xtr_fit, y_train_fit)
      best_model = grid_search.best_estimator_
      logger.info(f"[{name}] Best Params: {grid_search.best_params_}")
    else:
      best_model = config["model"]
      best_model.fit(xtr_fit, y_train_fit)

    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_filename = f"{safe_name}.joblib"
    joblib.dump(best_model, models_dir / model_filename)
    logger.info(f"Saved {name} artifact to {models_dir / model_filename}")

    y_pred = best_model.predict(xte)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"[{name} Results] -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    if rmse < best_overall_rmse:
      best_overall_rmse = rmse
      best_overall_model = best_model
      best_model_name = name

  joblib.dump(best_overall_model, models_dir / "best_tuned_ml_model.joblib")
  joblib.dump(scaler, models_dir / "flowcast_scaler.joblib")
  logger.info(f"🏆 Overall Champion: {best_model_name} (RMSE: {best_overall_rmse:.4f})")
  logger.info("Saved tuned champion model artifact to 'models/'")

if __name__ == "__main__":
  run_training_and_tuning()