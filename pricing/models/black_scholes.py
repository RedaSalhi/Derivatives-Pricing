# pricing/models/black_scholes.py

from scipy.stats import norm
import numpy as np

def black_scholes_price(option_type, S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes formula for European option pricing.

    Parameters:
        S : Spot price
        K : Strike price
        T : Time to maturity (in years)
        r : Risk-free rate
        sigma : Volatility
        q : Dividend yield (default: 0.0)

    Returns:
        float: option price
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        return print("option_type must be 'Call' or 'Put'")
