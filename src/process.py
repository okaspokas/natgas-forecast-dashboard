import numpy as np
import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove outliers beyond 3 standard deviations
    mean, std = df['price'].mean(), df['price'].std()
    before = len(df)
    df = df[np.abs(df['price'] - mean) < 3 * std].copy()
    print(f'  Removed {before - len(df)} outliers')

    # Rename for Prophet
    df = df[['date', 'price']].copy()
    df.columns = ['ds', 'y']

    # Fill date gaps with forward fill
    df = df.set_index('ds').resample('D').ffill().reset_index()

    return df

if __name__ == '__main__':
    print('Processing raw prices...')
    df_raw = pd.read_csv(RAW_DATA_PATH, parse_dates=['date'])
    df_processed = process(df_raw)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f'✅ Saved {len(df_processed):,} rows to {PROCESSED_DATA_PATH}')
