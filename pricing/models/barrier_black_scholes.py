import numpy as np
from scipy.stats import norm

def _phi(x):
    return norm.cdf(x)

def _lambda(r, sigma):
    return (r + (sigma**2) / 2) / (sigma**2)

def barrier_price(S, K, H, T, r, sigma, option_type, barrier_type):
    """
    Barrier option pricing using Reiner-Rubinstein formula.
    Only valid for European-style options without dividends.

    Parameters:
        S : spot price
        K : strike
        H : barrier level
        T : time to maturity
        r : risk-free rate
        sigma : volatility
        option_type : 'call' or 'put'
        barrier_type : 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'

    Returns:
        float: barrier option price
    """
    if H <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Invalid parameters.")

    mu = _lambda(r, sigma)
    z = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + mu * sigma * np.sqrt(T)
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + mu * sigma * np.sqrt(T)
    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + mu * sigma * np.sqrt(T)
    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + mu * sigma * np.sqrt(T)
    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + mu * sigma * np.sqrt(T)

    A = S * _phi(x1) - K * np.exp(-r * T) * _phi(x1 - sigma * np.sqrt(T))
    B = S * (H / S)**(2 * mu) * _phi(y1) - K * np.exp(-r * T) * (H / S)**(2 * mu - 2) * _phi(y1 - sigma * np.sqrt(T))

    if option_type == "call":
        if barrier_type == "up-and-out":
            return A - B if S < H else 0.0
        elif barrier_type == "up-and-in":
            return B if S < H else A
        elif barrier_type == "down-and-out":
            return A if S > H else 0.0
        elif barrier_type == "down-and-in":
            return 0.0 if S > H else A
    elif option_type == "put":
        raise NotImplementedError("Put barrier options not yet implemented.")
    else:
        raise ValueError("Unknown option or barrier type.")
