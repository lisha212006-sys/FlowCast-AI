from datetime import datetime
import os
from pathlib import Path
import pandas as pd
from src.utils import load_config, setup_logger

logger = setup_logger("FlowCast.Ingest")


def parse_flexible_dates(series: pd.Series) -> pd.Series:
  """Parses mixed date formats across source files."""
  parsed = pd.to_datetime(series, format="mixed", errors="coerce")
  return parsed


def load_and_validate_raw_data(config_path: str = "config.yaml"):
  config = load_config(config_path)
  raw_dir = Path(config["paths"]["raw_data_dir"])
  interim_dir = Path(config["paths"]["interim_data_dir"])
  interim_dir.mkdir(parents=True, exist_ok=True)

  traffic_path = raw_dir / config["data_sources"]["traffic_file"]
  logger.info(f"Loading traffic telemetry from {traffic_path}")
  df_traffic = pd.read_csv(traffic_path)

  df_traffic["date"] = parse_flexible_dates(df_traffic["date"])
  df_traffic["timestamp"] = pd.to_datetime(
      df_traffic["date"].astype(str) + " " + df_traffic["time"].astype(str),
      errors="coerce",
  )

  initial_len = len(df_traffic)
  df_traffic = df_traffic.dropna(subset=["road_id", "timestamp"])
  logger.info(
      f"Traffic log validation: dropped {initial_len - len(df_traffic)} records"
      " with null keys."
  )

  weather_path = raw_dir / config["data_sources"]["weather_file"]
  logger.info(f"Loading weather observations from {weather_path}")
  df_weather = pd.read_csv(weather_path)

  if "timestamp" not in df_weather.columns:
    if "date" in df_weather.columns and "time" in df_weather.columns:
      df_weather["timestamp"] = pd.to_datetime(
          df_weather["date"].astype(str) + " " + df_weather["time"].astype(str),
          errors="coerce",
      )
    else:
      raise KeyError(
          f"Weather file columns {df_weather.columns.tolist()} must contain"
          " either 'timestamp' or 'date'/'time'."
      )
  else:
    df_weather["timestamp"] = pd.to_datetime(
        df_weather["timestamp"], errors="coerce"
    )

  calendar_path = raw_dir / config["data_sources"]["calendar_file"]
  logger.info(f"Loading calendar events from {calendar_path}")
  df_calendar = pd.read_csv(calendar_path)
  df_calendar["date"] = pd.to_datetime(
      df_calendar["date"], format="mixed", errors="coerce"
  ).dt.date

  df_traffic.to_parquet(interim_dir / "traffic_validated.parquet")
  df_weather.to_parquet(interim_dir / "weather_validated.parquet")
  df_calendar.to_parquet(interim_dir / "calendar_validated.parquet")


if __name__ == "__main__":
  load_and_validate_raw_data()