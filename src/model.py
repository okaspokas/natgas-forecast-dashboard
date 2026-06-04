import pickle
import os
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from config import PROCESSED_DATA_PATH, RAW_DATA_PATH, MODEL_PATH, MODEL_DIR, FORECAST_DAYS

def train(df: pd.DataFrame) -> Prophet:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        interval_width=0.95
    )
    print('  Training Prophet model...')
    model.fit(df)
    return model

def forecast(model: Prophet, days: int = FORECAST_DAYS) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=days, freq='D')
    fc = model.predict(future)
    fc_out = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    fc_out.columns = ['date', 'forecast', 'lower', 'upper']
    return fc_out

def evaluate(model: Prophet) -> dict:
    print('  Running cross-validation...')
    df_cv = cross_validation(
        model,
        initial='730 days',
        period='90 days',
        horizon='90 days',
        parallel='processes'
    )
    df_perf = performance_metrics(df_cv)
    return {
        'mae':  df_perf['mae'].mean(),
        'rmse': df_perf['rmse'].mean(),
        'mape': df_perf['mape'].mean() * 100
    }

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('Loading processed data...')
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['ds'])

    model = train(df)

    print('Generating forecast...')
    fc = forecast(model)

    # Save forecast alongside raw prices
    import os as _os
    data_dir = _os.path.dirname(RAW_DATA_PATH)
    fc.to_csv(_os.path.join(data_dir, 'forecast.csv'), index=False)
    print(f'✅ Forecast saved')

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'✅ Model saved to {MODEL_PATH}')

    metrics = evaluate(model)
    print(f'\n📊 Model performance:')
    print(f'   MAE  : ${metrics["mae"]:.3f}/MMBtu')
    print(f'   RMSE : ${metrics["rmse"]:.3f}/MMBtu')
    print(f'   MAPE : {metrics["mape"]:.1f}%')
