import numpy as np
from scipy.stats import norm

def digital_cash_or_nothing(option_type, S, K, T, r, sigma, Q=1.0):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 -= sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return Q * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return Q * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def digital_asset_or_nothing(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == "call":
        return S * norm.cdf(d1)
    elif option_type.lower() == "put":
        return S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
