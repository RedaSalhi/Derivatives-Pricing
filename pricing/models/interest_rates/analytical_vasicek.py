from datetime import datetime
import numpy as np
import pandas as pd
import datetime
from pandas_datareader import data as web
import yfinance as yf
from scipy.optimize import minimize

# ----------------------
# 1. Auto-Detect Data Loader (FRED → yfinance fallback)
# ----------------------

def load_data_auto(ticker, start='1990-01-01', end=None):
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    try:
        df = web.DataReader(ticker, 'fred', start_dt, end_dt)
        df = df.rename(columns={ticker: 'Rate'})
        df = df / 100 # Convert FRED percentage values
        print(f"Loaded data from FRED for {ticker}")
    except Exception as e_fred:
        try:
            df = yf.download(ticker, start=start, end=end)
            df = df[['Close']].rename(columns={'Close': 'Rate'})
            df = df / 100
            print(f"FRED failed. Loaded data from Yahoo Finance for {ticker}")
        except Exception as e_yf:
            raise RuntimeError(f"Failed to load data from both FRED and Yahoo for {ticker}") from e_yf
    return df

# ----------------------
# 2. Preprocessing Function
# ----------------------

def preprocess(df, freq='ME'):
    df = df.resample(freq).last().dropna()
    df['dt'] = df.index.to_series().diff().dt.days / 365.0
    return df['Rate'], df['dt'].iloc[1]

# ----------------------
# 3. OU Process: Negative Log-Likelihood
# ----------------------

def neg_log_likelihood(params, r, dt):
    a, lam, sigma = params
    r_t, r_tp = r[:-1], r[1:]
    mu = r_t + a * (lam - r_t) * dt
    var = sigma**2 * dt
    return 0.5 * np.sum(np.log(2 * np.pi * var) + ((r_tp - mu)**2) / var)

# ----------------------
# 4. Parameter Estimation Function
# ----------------------

def estimate_ou_parameters(rates, dt):
    r_values = rates.values
    init = [1.0, np.mean(r_values), np.std(r_values)]
    bounds = [(1e-4, 10), (None, None), (1e-6, None)]
    res = minimize(neg_log_likelihood, init, args=(r_values, dt), bounds=bounds)
    return res.x  # a, lam, sigma

# ----------------------
# 5. Main Wrapper Function
# ----------------------

def run_ou_estimation(ticker, start='1990-01-01', end=None, freq='ME'):
    df = load_data_auto(ticker, start, end)
    r_series, dt = preprocess(df, freq=freq)
    a, lam, sigma = estimate_ou_parameters(r_series, dt)
    return a, lam, sigma, dt, r_series


# ------------------------------------------------------------------------------
# 1. Simulate Short Rate Path (Euler–Maruyama for Vasicek)
# ------------------------------------------------------------------------------

def simulate_vasicek_path(r0, a, lam, sigma, T, dt):
    N = int(T / dt) + 1
    time = np.linspace(0, T, N)
    r = np.zeros(N)
    r[0] = r0
    for t in range(1, N):
        du = np.random.normal()
        r[t] = r[t - 1] + a * (lam - r[t - 1]) * dt + sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a)) * du
    return time, r

# ------------------------------------------------------------------------------
# 2. Vasicek Zero-Coupon Bond Price Formula
# ------------------------------------------------------------------------------

def vasicek_zero_coupon_price(r_t, t, T, a, lam, sigma, face_value=1.0):
    B = (1 - np.exp(-a * (T - t))) / a
    A = np.exp((lam - sigma**2 / (2 * a**2)) * (B - (T - t)) - (sigma**2 / (4 * a)) * B**2)
    return face_value * A * np.exp(-B * r_t)

