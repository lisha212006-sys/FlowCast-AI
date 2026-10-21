from pathlib import Path
import numpy as np
import pandas as pd

from src.utils import load_config, setup_logger

logger = setup_logger("FlowCast.Features")


def engineer_features(config_path: str = "config.yaml"):
  config = load_config(config_path)
  processed_dir = Path(config["paths"]["processed_data_dir"])

  data_path = processed_dir / "flowcast_analysis_ready.parquet"
  logger.info(f"Loading dataset from {data_path}")
  df = pd.read_parquet(data_path)
  
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  df = df.sort_values(by=["road_name", "timestamp"])

  # 1. Temporal cyclical encodings
  hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
  df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
  df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

  day_of_week = df["timestamp"].dt.dayofweek
  df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
  df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

  df["is_weekend"] = day_of_week >= 5

  # 2. Lag Features (Strictly per road segment)
  lag_steps = config["features"]["lag_windows"]
  for lag in lag_steps:
    df[f"volume_lag_{lag}"] = df.groupby("road_name")["traffic_volume"].shift(lag)
    df[f"speed_lag_{lag}"] = df.groupby("road_name")["avg_speed"].shift(lag)

  # 3. Rolling Statistics (Using .transform to stay within road groups safely)
  rolling_windows = config["features"]["rolling_windows"]
  for w in rolling_windows:
    df[f"volume_rolling_mean_{w}"] = (
        df.groupby("road_name")["traffic_volume"]
        .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
    )
    df[f"volume_rolling_std_{w}"] = (
        df.groupby("road_name")["traffic_volume"]
        .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).std())
        .fillna(0.0)  # Fill std dev NaNs for single-row windows
    )

  # 4. Volume-to-Capacity Ratio & Headroom
  if "road_capacity" in df.columns:
    df["vcr"] = df["traffic_volume"] / df["road_capacity"]

  # Instead of a destructive global dropna(), fill initial lag NaNs gracefully
    df = df.bfill().ffill().fillna(0)
  # Save final analysis-ready dataset
  output_path = processed_dir / "flowcast_analysis_ready.parquet"
  df.to_parquet(output_path)
  logger.info(f"Successfully saved feature-engineered dataset with {len(df):,} rows to {output_path}")


if __name__ == "__main__":
  engineer_features()