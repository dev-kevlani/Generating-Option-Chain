import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import yfinance as yf
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

def generate_price_paths(ticker, start_date, num_paths, forecast_days, use_risk_neutral=False, r=None):
    """
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Historical data start date
    num_paths : int
        Number of Monte Carlo paths to simulate
    forecast_days : int
        Number of days to forecast
    use_risk_neutral : bool
        If True, uses risk-neutral drift instead of historical drift
    r : float
        Risk-free rate (required if use_risk_neutral=True)
    """
    T = 252
    delta_t = 1 / T
    rfr = 0.04 if r is None else r
    data = yf.download(ticker, start=start_date, progress=False)['Adj Close']
    
    log_rets = np.log(1 + data.pct_change())
    var_ret = log_rets.var() * T
    std = np.sqrt(var_ret)
    last_price = data.iloc[-1]
    cumulative_returns = np.exp(((rfr - 0.5 * (std ** 2)) * delta_t) +
                                 (std * np.sqrt(delta_t) * norm.ppf(np.random.rand(num_paths, T))))
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

    for K in strikes:
        call_prices_df = (paths_df - K).clip(lower=0)
        call_prices_df['time_to_expiry'] = (call_prices_df.index - curr_date).days / 252
        call_prices_df['call_expected_price'] = call_prices_df.mean(axis=1) * np.exp(-r * call_prices_df['time_to_expiry'])
        
        put_prices_df = (K - paths_df).clip(lower=0)
        put_prices_df['time_to_expiry'] = (put_prices_df.index - curr_date).days / 252
        put_prices_df['put_expected_price'] = put_prices_df.mean(axis=1) * np.exp(-r * put_prices_df['time_to_expiry'])
        
        option_chain.append({
            'strike': K,
            'call_prices': call_prices_df['call_expected_price'],
            'put_prices': put_prices_df['put_expected_price']
        })
    
    # Combine all strike prices into a single DataFrame
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
        num_paths=10000,
        forecast_days=252
    )

    last_price = params['last_price']
    lower = int(round(last_price * 0.95, 0))
    upper = int(round(last_price * 1.05, 0))
    strikes = np.arange(lower, upper + 1, 1)

    option_chain_df = generate_option_prices(
        paths_df=paths_df,
        strikes=strikes,
        r=0.04,
        curr_date=params['data'].index[-1]
    )

    print(option_chain_df)

    paths_df.loc[:, :'path_10'].plot(title="Simulated Price Paths")
    plt.show()

    print(f"\nExecution time: {time.time() - start_time:.2f} seconds")
    print("Memory usage:", paths_df.memory_usage().sum() / 1024**2, "MB")
