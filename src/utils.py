import logging
from pathlib import Path
import yaml
def setup_logger(name: str) -> logging.Logger:
  """Configure standard logger for FlowCast pipeline stages."""
  logger = logging.getLogger(name)
  if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
  return logger
def load_config(config_path: str = "config.yaml") -> dict:
  """Load YAML configuration file."""
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)
  return config