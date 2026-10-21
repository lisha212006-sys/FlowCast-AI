from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import load_config, setup_logger

logger = setup_logger("FlowCast.Clean")


# Check raw traffic log
df_raw = pd.read_csv("data/raw/traffic_sensor_log.csv")
print("Raw Traffic Shape:", df_raw.shape)
print(df_raw.head(2))

def clean_and_merge_data(config_path: str = "config.yaml"):
  config = load_config(config_path)
  interim_dir = Path(config["paths"]["interim_data_dir"])
  processed_dir = Path(config["paths"]["processed_data_dir"])
  processed_dir.mkdir(parents=True, exist_ok=True)


  df_traffic = pd.read_parquet(interim_dir / "traffic_validated.parquet")
  df_weather = pd.read_parquet(interim_dir / "weather_validated.parquet")
  df_calendar = pd.read_parquet(interim_dir / "calendar_validated.parquet")

  df_traffic = df_traffic.sort_values(
      by=["road_id", "timestamp", "traffic_volume"], ascending=[True, True, False]
  )
  df_traffic = df_traffic.drop_duplicates(
      subset=["road_id", "timestamp"], keep="first"
  )
  logger.info(
      f"Deduplicated traffic logs. Current row count: {len(df_traffic)}"
  )

  max_speed = config["validation"]["max_speed_kmh"]
  max_occ = config["validation"]["max_occupancy_pct"]

  df_traffic.loc[df_traffic["avg_speed"] > max_speed, "avg_speed"] = np.nan
  df_traffic.loc[df_traffic["occupancy"] > max_occ, "occupancy"] = np.nan
  df_traffic.loc[df_traffic["traffic_volume"] < 0, "traffic_volume"] = 0

  df_traffic = df_traffic.sort_values(by=["road_id", "timestamp"])

  def impute_group(group):
    group = group.set_index("timestamp")
    group["traffic_volume"] = (
        group["traffic_volume"].interpolate(method="time").ffill().bfill()
    )
    group["avg_speed"] = (
        group["avg_speed"].interpolate(method="time").ffill().bfill()
    )
    group["occupancy"] = group["occupancy"].fillna(group["occupancy"].median())
    return group.reset_index()

  df_traffic = df_traffic.groupby("road_id", group_keys=False).apply(
      impute_group
  )

  df_weather["weather_condition"] = (
      df_weather["weather_condition"].str.strip().str.capitalize()
  )
 
  df_traffic = df_traffic.dropna(subset=["timestamp"]).sort_values("timestamp")
  df_weather = df_weather.dropna(subset=["timestamp"]).sort_values("timestamp")

  merge_by = (
      "weather_station_id"
      if (
          "weather_station_id" in df_traffic.columns
          and "weather_station_id" in df_weather.columns
      )
      else None
  )

  if merge_by:
    df_merged = pd.merge_asof(
        df_traffic,
        df_weather,
        on="timestamp",
        by=merge_by,
        direction="backward",
    )
  else:
    df_merged = pd.merge_asof(
        df_traffic, df_weather, on="timestamp", direction="backward"
    )
  df_merged["date_only"] = pd.to_datetime(df_merged[
    "timestamp"]).dt.date
  df_merged = pd.merge(
      df_merged,
      df_calendar,
      left_on="date_only",
      right_on="date",
      how="left",
      suffixes=("", "_cal"),
  )
  df_merged.drop(columns=["date_only", "date_cal"], inplace=True, errors="ignore")
  df_merged["public_holiday"] = (
      df_merged["public_holiday"].fillna(False).astype(bool)
  )
  df_merged["event_flag"] = df_merged["event_flag"].fillna(False).astype(bool)

  if (
      "road_capacity" in df_merged.columns
      and "congestion_level" in df_merged.columns
  ):
    vcr = df_merged["traffic_volume"] / df_merged["road_capacity"].replace(
        0, np.nan
    )
    conditions = [
        vcr < 0.5,
        (vcr >= 0.5) & (vcr < 0.75),
        (vcr >= 0.75) & (vcr < 1.0),
        vcr >= 1.0,
    ]
    choices = ["Free-flow", "Moderate", "Heavy", "Severe"]
    calculated_congestion = pd.Series(
        np.select(conditions, choices, default="Moderate"),
        index=df_merged.index,
    )
    df_merged["congestion_level"] = df_merged["congestion_level"].fillna(
        calculated_congestion
    )

  output_path = processed_dir / "flowcast_analysis_ready.parquet"
  df_merged.to_parquet(output_path)



if __name__ == "__main__":
  clean_and_merge_data()