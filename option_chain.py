import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import yfinance as yf
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

def generate_price_paths(ticker, start_date, num_paths, forecast_days, T, use_risk_neutral=False, r=None, vol=None):
    delta_t = 1 / T
    rfr = r
    data = yf.download(ticker, start=start_date, progress=False)['Adj Close']
    
    log_rets = np.log(1 + data.pct_change())
    var_ret = log_rets.var() * T
    std = np.sqrt(var_ret) if vol == None else vol
    last_price = data.iloc[-1]
    cumulative_returns = np.exp(((rfr - 0.5 * (std ** 2)) * delta_t) +
                                 (std * np.sqrt(delta_t) * norm.ppf(np.random.rand(num_paths, forecast_days))))
    price_paths = np.zeros_like(cumulative_returns)
    price_paths[:, 0] = last_price * cumulative_returns[:, 0]
    for i in range(1, cumulative_returns.shape[1]):
        price_paths[:, i] = price_paths[:, i - 1] * cumulative_returns[:, i]
        
    us_calendar = USFederalHolidayCalendar()
    holidays = us_calendar.holidays(
        start=data.index[-1],
        end=data.index[-1] + pd.Timedelta(days=365)
    )
    custom_bday = CustomBusinessDay(holidays=holidays.to_list())
    
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq=custom_bday
    )
    
    paths_df = pd.DataFrame(
        price_paths.T,
        index=future_dates,
        columns=[f'path_{i}' for i in range(num_paths)]
    )
    
    return paths_df, {
        'volatility': std,
        'last_price': last_price,
        'start_date': data.index[0],
        'end_date': future_dates[-1],
        'data': data
    }

def generate_option_prices(paths_df, strikes, r, curr_date):
    """
    Generate call and put option prices for multiple strikes.

    Parameters:
    -----------
    paths_df : DataFrame
        Simulated price paths
    strikes : list
        List of strike prices
    r : float
        Risk-free rate
    curr_date : datetime
        Current date
    
    Returns:
    --------
    option_chain : DataFrame
        Option prices for each strike and future day
    """
    option_chain = []
    curr_datetime = pd.Timestamp(curr_date).replace(hour=16, minute=0, second=0)

    for K in strikes:
        call_prices_df = (paths_df - K).clip(lower=0)
        put_prices_df = (K - paths_df).clip(lower=0)

        # Calculate time-to-expiry in seconds relative to 4 PM
        call_prices_df['time_to_expiry'] = (
            (call_prices_df.index + pd.Timedelta(hours=16)) - curr_datetime
        ).total_seconds() / (365 * 24 * 3600)
        
        put_prices_df['time_to_expiry'] = call_prices_df['time_to_expiry']

        call_prices_df['call_expected_price'] = call_prices_df.mean(axis=1) * np.exp(-r * call_prices_df['time_to_expiry'])
        put_prices_df['put_expected_price'] = put_prices_df.mean(axis=1) * np.exp(-r * put_prices_df['time_to_expiry'])
        
        option_chain.append({
            'strike': K,
            'call_prices': call_prices_df['call_expected_price'],
            'put_prices': put_prices_df['put_expected_price']
        })
    
    option_chain_df = pd.concat(
        {f"Strike {item['strike']}": pd.DataFrame({
            'Call': item['call_prices'],
            'Put': item['put_prices']
        }) for item in option_chain},
        axis=1
    )
    
    return option_chain_df

if __name__ == "__main__":
    start_time = time.time()

    paths_df, params = generate_price_paths(
        ticker='AAPL',
        start_date='2019-01-01',
        num_paths=100000,
        forecast_days=3,
        T=365,
        r = 0.044686517622406094,
        vol = 0.2498
    )

    start_time_formatted = time.strftime("%H:%M:%S", time.gmtime(start_time))
    print(f"Start time: {start_time_formatted} (UTC)")

    last_price = params['last_price']
    lower = int(round(last_price * 0.98, 0))
    upper = int(round(last_price * 1.02, 0))
    strikes = np.arange(lower, upper + 1, 1)

    option_chain_df = generate_option_prices(
        paths_df=paths_df,
        strikes=strikes,
        r=0.044686517622406094,
        curr_date=params['data'].index[-1]
    )

    print(option_chain_df)

    paths_df.loc[:, :'path_10'].plot(title="Simulated Price Paths")
    plt.show()

    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s")
    print("Memory usage:", paths_df.memory_usage().sum() / 1024**2, "MB")
