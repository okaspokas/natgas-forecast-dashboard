# 🔥 Natural Gas Price Forecasting Dashboard

> ML-powered time-series forecasting for Henry Hub natural gas spot prices, built with Facebook Prophet and deployed via Streamlit + Docker.

**Prepared by:** Ojas Khetarpal — Summer Trainee

---

## Overview

This project implements an end-to-end machine learning pipeline that fetches real-time natural gas price data from the U.S. Energy Information Administration (EIA), trains a Facebook Prophet forecasting model, and surfaces predictions through an interactive Streamlit dashboard.

**Key highlights:**
- Automated data ingestion from EIA Open Data API (Henry Hub, series `RNGWHHD`)
- Facebook Prophet model with multiplicative seasonality and 90-day forecast horizon
- Interactive dashboard with trend visualization, seasonality decomposition, and historical analysis
- Fully containerized with Docker for one-command deployment

---

## Model Performance

| Metric | Value |
|--------|-------|
| MAE    | $1.034 / MMBtu |
| RMSE   | $1.552 / MMBtu |
| MAPE   | 35.5% |

---

## Dashboard Features

**Tab 1 — Forecast**
Historical prices overlaid with a 90-day Prophet forecast, confidence interval toggle, rolling average toggle, and adjustable forecast horizon slider.

**Tab 2 — Historical Analysis**
Yearly average bar chart, price distribution histogram, and year-on-year comparison.

**Tab 3 — Seasonality**
Prophet component plots and average price by month bar chart.

**Tab 4 — Data Table**
Full dataset filterable by date range with CSV download.

---

## Project Structure

```
gas_forecast/
├── src/
│   ├── config.py         # Central configuration & path management
│   ├── ingest.py         # EIA API data fetching → data/raw_prices.csv
│   ├── process.py        # Data cleaning, outlier removal, gap-filling
│   ├── model.py          # Prophet training → models/prophet_model.pkl
│   └── dashboard.py      # Streamlit dashboard application
├── data/
│   ├── raw_prices.csv    # (git-ignored)
│   ├── processed_prices.csv
│   └── forecast.csv
├── models/
│   └── prophet_model.pkl # (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .env                  # Your API key — never commit this
```

---

## Quick Start

### Prerequisites

- Python 3.10+ or Docker
- Free EIA API key → [https://www.eia.gov/opendata/](https://www.eia.gov/opendata/)

---

### Option A — Docker (Recommended)

```bash
# 1. Copy and fill in your API key
cp .env.example .env
# Edit .env: EIA_API_KEY=your_key_here

# 2. Run the pipeline
python src/ingest.py
python src/process.py
python src/model.py

# 3. Launch the dashboard
docker-compose up --build
```

Open [http://localhost:8501](http://localhost:8501)

---

### Option B — Local Python

```bash
# 1. Clone and enter the project
cd gas_forecast

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
cp .env.example .env
# Edit .env: EIA_API_KEY=your_key_here

# 5. Run the full pipeline
python src/ingest.py
python src/process.py
python src/model.py

# 6. Launch the dashboard
streamlit run src/dashboard.py
```

---

## Prophet Model Configuration

```python
Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.1,
    interval_width=0.95
)
```

Forecast horizon: **90 days**

---

## Data Source

**Henry Hub Natural Gas Spot Price** — EIA Open Data API v2

- Series: `RNGWHHD`
- Endpoint: `https://api.eia.gov/v2/natural-gas/pri/fut/data/`
- Frequency: Daily, $/MMBtu
- API key required (free): [https://www.eia.gov/opendata/](https://www.eia.gov/opendata/)

---

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
EIA_API_KEY=your_api_key_here
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore` by default.

---

## Pipeline Order

The scripts must be run in sequence — each step depends on the output of the previous:

```
ingest.py → process.py → model.py → dashboard.py
```

Docker handles the dashboard only. Run the first three scripts manually before `docker-compose up`.
