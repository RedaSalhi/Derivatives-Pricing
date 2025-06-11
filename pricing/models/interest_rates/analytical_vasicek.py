# pricing/models/interest_rates/analytical_vasicek.py



from datetime import datetime
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import yfinance as yf
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# ------------------------------------------------------------------------------
# 3. Generate Yield Curves at Different Snapshot Times
# ------------------------------------------------------------------------------
def generate_yield_curves(r_path, snapshot_times, maturities, a, lam, sigma, dt):
    yield_curves = {}
    for t_snap in snapshot_times:
        idx = int(t_snap / dt)
        r_t = r_path[idx]
        yields = []
        for m in maturities:
            T = t_snap + m
            P = vasicek_zero_coupon_price(r_t, t_snap, T, a, lam, sigma)
            y = -np.log(P) / m
            yields.append(y)
        yield_curves[t_snap] = yields
    return yield_curves

# ------------------------------------------------------------------------------
# 4. Plot Yield Curves
# ------------------------------------------------------------------------------
def plot_yield_curves(yield_curves, maturities):
    plt.figure(figsize=(10, 6))
    for t_snap, yields in yield_curves.items():
        plt.plot(maturities, yields, label=f'Time {t_snap}y')
    plt.title('Simulated Yield Curves under Vasicek Model')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (continuously compounded)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# 5. Price a Coupon Bond Using the Simulated Short Rate Path
# ------------------------------------------------------------------------------
def price_coupon_bond(rates, a, lam, sigma, maturity=5, coupon=0.05, face_value=1.0, dt=0.5):
    cashflow_dates = np.arange(dt, maturity + dt, dt)
    price = 0
    for T in cashflow_dates:
        coupon_payment = coupon * face_value * dt
        if np.isclose(T, maturity):
            coupon_payment += face_value
        P = vasicek_zero_coupon_price(rates[0], 0, T, a, lam, sigma, face_value)
        price += coupon_payment * P
    return price


def vasicek_bond_option_price(r_t, t, T1, T2, K, a, lam, sigma, face=1.0, option_type='call'):
    P_t_T1 = vasicek_zero_coupon_price(r_t, t, T1, a, lam, sigma)
    P_t_T2 = vasicek_zero_coupon_price(r_t, t, T2, a, lam, sigma)

    B = (1 - np.exp(-a * (T2 - T1))) / a
    sigma_P_sq = (sigma**2 / (2 * a**3)) * (1 - np.exp(-a * (T2 - T1)))**2 * (1 - np.exp(-2 * a * (T1 - t)))

    sigma_P = np.sqrt(sigma_P_sq)

    d1 = (np.log( face * P_t_T2 / (K * P_t_T1)) / sigma_P) + 0.5 * sigma_P
    d2 = d1 - sigma_P
    if option_type == 'call':
        call_price = face * P_t_T2 * norm.cdf(d1) - K * P_t_T1 * norm.cdf(d2)
    elif option_type == 'put':
        call_price = K * P_t_T1 * norm.cdf(-d2) - face * P_t_T2 * norm.cdf(-d1) 


    #call_price = face * P_t_T2 * norm.cdf(d1) - K * P_t_T1 * norm.cdf(d2)
    return call_price
