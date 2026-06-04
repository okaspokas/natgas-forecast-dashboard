import os
from dotenv import load_dotenv

load_dotenv()

# API
EIA_API_KEY = os.getenv("EIA_API_KEY", "")
EIA_BASE_URL = "https://api.eia.gov/v2"

# EIA series for Henry Hub Natural Gas Spot Price ($/MMBtu)
EIA_SERIES_ID = "NG.RNGWHHD.D"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_prices.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_prices.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "prophet_model.pkl")

# Forecast settings
FORECAST_DAYS = 90
