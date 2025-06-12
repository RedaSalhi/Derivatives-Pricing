# pricing/models/interest_rates/analytical_vasicek.py

# from pricing.models.interest_rates.analytical_vasicek import price_coupon_bond, run_ou_estimation, simulate_vasicek_path, vasicek_zero_coupon_price, plot_yield_curves, generate_yield_curves

from datetime import datetime
import numpy as np
import pandas as pd
from pandas_datareader import data as web
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------
# Auto-Detect Data Loader (FRED → yfinance fallback)
# ----------------------

def load_data_auto(ticker, start='1990-01-01', end=None):
    if end is None:
        end = datetime.today().date()
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
# Preprocessing Function
# ----------------------

def preprocess(df, freq='ME'):
    df = df.resample(freq).last().dropna()
    df['dt'] = df.index.to_series().diff().dt.days / 365.0
    return df['Rate'], df['dt'].iloc[-1]


#-----------------
# Calibrage (Likelihood)
#-----------------

import numpy as np
import pandas as pd

def compute_sufficient_stats(r):
    r_prev = r[:-1]
    r_next = r[1:]
    n = len(r_prev)

    S0 = np.mean(r_prev)
    S1 = np.mean(r_next)
    S00 = np.mean(r_prev * r_prev)
    S01 = np.mean(r_prev * r_next)

    return S0, S1, S00, S01, r_prev, r_next, n

def estimate_ab(S0, S1, S00, S01, dt):
    numerator = S1 * S00 - S0 * S01
    denominator = S1 * S0 - S0**2 + S00 - S01
    b_hat = numerator / denominator

    ratio = (S0 - b_hat) / (S1 - b_hat)
    a_hat = (1 / dt) * np.log(ratio)

    return a_hat, b_hat

def estimate_sigma(a, b, r_prev, r_next, dt):
    beta = (1 / a) * (1 - np.exp(-a * dt))
    m = b * a * beta + r_prev * (1 - a * beta)
    n = len(r_prev)

    numerator = np.sum((r_next - m) ** 2)
    sigma_squared = (1 / (n * beta * (1 - 0.5 * a * beta))) * numerator

    return np.sqrt(sigma_squared)

def estimate_ou_parameters(rates, dt):
    S0, S1, S00, S01, r_prev, r_next, n = compute_sufficient_stats(rates.values)
    a, b = estimate_ab(S0, S1, S00, S01, dt)
    sigma = estimate_sigma(a, b, r_prev, r_next, dt)
    return a, b, sigma

def run_ou_estimation(ticker, start='1990-01-01', end=None, freq='ME'):  #need extraction from pricing.models.interest_rates.analytical_vasicek import run_ou_estimation
    df = load_data_auto(ticker, start, end)
    r_series, dt = preprocess(df, freq=freq)
    a, lam, sigma = estimate_ou_parameters(r_series, dt)
    a, lam, sigma = float(a), float(lam), float(sigma)
    print(f"\nEstimated parameters for {ticker}:")
    print(f"  a (speed of mean reversion):   {a:.4f}")
    print(f"  lambda (long-term mean level): {lam:.4f}")
    print(f"  sigma (volatility):            {sigma:.4f}")
    return a, lam, sigma, dt, float(r_series.iloc[-1])



# ------------------------------------------------------------------------------
# Simulate Short Rate Path (Euler–Maruyama for Vasicek)
# ------------------------------------------------------------------------------

def simulate_vasicek_path(r0, a, lam, sigma, T, dt):  #need extraction from pricing.models.interest_rates.analytical_vasicek import simulate_vasicek_path
    N = int(T / dt) + 1
    time = np.linspace(0, T, N)
    r = np.zeros(N)
    r[0] = r0
    for t in range(1, N):
        du = np.random.normal()
        sigma_p = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        mu_t = r0 * np.exp(-a * dt) + lam * (1 - np.exp(-a * dt))
        r[t] = mu_t + sigma_p * du
    return time, r

# ------------------------------------------------------------------------------
# Vasicek Zero-Coupon Bond Price Formula
# ------------------------------------------------------------------------------

def vasicek_zero_coupon_price(r_t, t, T, a, lam, sigma, face_value=1.0):  #need extraction from pricing.models.interest_rates.analytical_vasicek import vasicek_zero_coupon_price
    B = (1 - np.exp(-a * (T - t))) / a
    A = np.exp((lam - sigma**2 / (2 * a**2)) * (B - (T - t)) - (sigma**2 / (4 * a)) * B**2)
    return face_value * A * np.exp(-B * r_t)


# ------------------------------------------------------------------------------
# Generate Yield Curves at Different Snapshot Times
# ------------------------------------------------------------------------------
def generate_yield_curves(r_path, snapshot_times, maturities, a, theta, sigma, dt): #need extraction from pricing.models.interest_rates.analytical_vasicek import generate_yield_curves
    yield_curves = {}
    n = len(r_path)
    for t_snap in snapshot_times:
        idx = int(t_snap / dt)
        if idx >= n:
            continue  # Avoid IndexError
        r_t = r_path[idx]
        yields = []
        for m in maturities:
            T = t_snap + m
            P = vasicek_zero_coupon_price(r_t, t_snap, T, a, theta, sigma)
            y = -np.log(P) / m
            yields.append(y)
        yield_curves[t_snap] = yields
    return yield_curves

# ------------------------------------------------------------------------------
# Plot Yield Curves
# ------------------------------------------------------------------------------
def plot_yield_curves(yield_curves, maturities): #need extraction from pricing.models.interest_rates.analytical_vasicek import plot_yield_curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for t_snap, yields in yield_curves.items():
        ax.plot(maturities, yields, label=f'Time {t_snap}y')
    ax.set_title('Simulated Yield Curves under Vasicek Model')
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Yield (continuously compounded)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    return fig



# ------------------------------------------------------------------------------
# Price a Coupon Bond Using the Simulated Short Rate Path
# ------------------------------------------------------------------------------
def price_coupon_bond(r0, t, a, lam, sigma, maturity=5, face=1.0, coupon=0.05, dt=0.5): #need extraction from pricing.models.interest_rates.analytical_vasicek import price_coupon_bond
    cashflow_dates = np.arange(t+dt, maturity + 1e-6, dt)
    price = 0
    coupon_payment = face * coupon * dt

    for t_i in cashflow_dates:
        P = vasicek_zero_coupon_price(r0, t, t_i, a, lam, sigma, face_value=1)
        price += coupon_payment * P

    f_price = vasicek_zero_coupon_price(r0, t, maturity, a, lam, sigma, face_value=1)
    price += face * f_price
    return price

# -----------------
# Bond Option Price
# -----------------

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
