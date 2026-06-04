# Natural Gas Price Forecasting Dashboard
**Prepared by:** Ojas Khetarpal — Summer Trainee

Machine learning-based time-series forecasting for natural gas prices using Facebook Prophet and Streamlit.

---

## Quick Start

### Local (Python)
```bash
# 1. Clone and enter the project
cd gas_forecast

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env and add your EIA API key

# 5. Fetch data and run dashboard
python src/ingest.py
streamlit run src/dashboard.py
```

### Docker
```bash
cp .env.example .env
# Edit .env and add your EIA API key

docker-compose up --build
# Open http://localhost:8501
```

---

## Project Structure
```
gas_forecast/
├── src/
│   ├── config.py       # Central configuration
│   ├── ingest.py       # EIA API data fetching
│   ├── process.py      # Data cleaning & feature engineering
│   ├── model.py        # Prophet forecasting model
│   └── dashboard.py    # Streamlit dashboard
├── data/               # Raw and processed CSV files (git-ignored)
├── models/             # Saved model files (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Data Source
Henry Hub Natural Gas Spot Price — [EIA Open Data](https://www.eia.gov/opendata/)
Free API key required.
