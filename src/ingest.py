import os
import time
import requests
import pandas as pd
from config import EIA_API_KEY, EIA_BASE_URL, RAW_DATA_PATH, DATA_DIR

def fetch_eia_gas_prices(api_key, start='2015-01-01'):
    url = f"{EIA_BASE_URL}/natural-gas/pri/fut/data/"
    params = {
        'api_key': api_key,
        'frequency': 'daily',
        'data[]': 'value',
        'facets[series][]': 'RNGWHHD',
        'start': start,
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000
    }

    all_data = []
    while True:
        response = requests.get(url, params=params)
        response.raise_for_status()
        result = response.json()
        data = result.get('response', {}).get('data', [])
        if not data:
            break
        all_data.extend(data)
        print(f'  Fetched {len(all_data):,} rows...')
        if len(data) < 5000:
            break
        params['offset'] += 5000
        time.sleep(0.3)

    df = pd.DataFrame(all_data)
    df = df[['period', 'value']].copy()
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna().sort_values('date').reset_index(drop=True)
    return df

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    print('Fetching Henry Hub prices from EIA...')
    df = fetch_eia_gas_prices(EIA_API_KEY)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f'✅ Saved {len(df):,} rows to {RAW_DATA_PATH}')
    print(f'   Date range: {df.date.min().date()} → {df.date.max().date()}')
