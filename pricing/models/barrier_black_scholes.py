import numpy as np
from scipy.stats import norm

def _phi(x):
    return norm.cdf(x)

def _lambda(r, sigma):
    return (r + 0.5 * sigma**2) / sigma**2

def barrier_price(S, K, H, T, r, sigma, option_type, barrier_type):
    """
    Price a European barrier option using Reiner-Rubinstein formula (no dividends).

    Parameters:
        S : float – Spot price
        K : float – Strike price
        H : float – Barrier level
        T : float – Time to maturity (in years)
        r : float – Risk-free rate
        sigma : float – Volatility
        option_type : str – 'call' or 'put'
        barrier_type : str – 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'

    Returns:
        float: Option price
    """
    if S <= 0 or K <= 0 or H <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Invalid input parameters.")

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if barrier_type not in {"up-and-out", "down-and-out", "up-and-in", "down-and-in"}:
        raise ValueError("barrier_type must be one of the 4 standard types")

    # Knocked out immediately
    if (barrier_type.startswith("up") and S >= H) or (barrier_type.startswith("down") and S <= H):
        if "out" in barrier_type:
            return 0.0  # already knocked out
        # knocked in already: price vanilla option
        return vanilla_bs_price(S, K, T, r, sigma, option_type)

    mu = _lambda(r, sigma)
    vol_sqrt_t = sigma * np.sqrt(T)

    ln_SH = np.log(S / H)
    ln_HS = np.log(H / S)
    ln_SK = np.log(S / K)
    ln_H2_SK = np.log(H**2 / (S * K))

    x1 = (ln_SK / vol_sqrt_t) + mu * vol_sqrt_t
    x2 = x1 - vol_sqrt_t
    y1 = (ln_H2_SK / vol_sqrt_t) + mu * vol_sqrt_t
    y2 = y1 - vol_sqrt_t
    z = (ln_SH / vol_sqrt_t) + mu * vol_sqrt_t
    z2 = z - vol_sqrt_t

    eta = 1 if option_type == "call" else -1

    def A():
        return eta * (S * _phi(eta * x1) - K * np.exp(-r * T) * _phi(eta * x2))

    def B():
        power = 2 * mu
        return eta * (S * (H / S)**power * _phi(eta * y1) -
                      K * np.exp(-r * T) * (H / S)**(power - 2) * _phi(eta * y2))

    def C():
        return eta * (S * _phi(eta * z) - K * np.exp(-r * T) * _phi(eta * z2))

    def D():
        power = 2 * mu
        return eta * (S * (H / S)**power * _phi(eta * (z - vol_sqrt_t)) -
                      K * np.exp(-r * T) * (H / S)**(power - 2) * _phi(eta * (z2 - vol_sqrt_t)))

    # Reiner-Rubinstein closed-form logic
    if barrier_type == "up-and-out":
        return B() if option_type == "call" else B()
    elif barrier_type == "up-and-in":
        return A() - B() if option_type == "call" else A() - B()
    elif barrier_type == "down-and-out":
        return B() if option_type == "call" else B()
    elif barrier_type == "down-and-in":
        return A() - B() if option_type == "call" else A() - B()

    raise ValueError("Unknown combination.")
