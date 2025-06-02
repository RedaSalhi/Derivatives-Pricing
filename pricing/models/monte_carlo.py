# pricing/models/monte_carlo.py

import numpy as np

def monte_carlo_price(option_type, S, K, T, r, sigma, q=0.0, n_simulations=100_000):
    """
    Monte Carlo pricing for European call or put options.

    Parameters:
        option_type : 'call' or 'put'
        S : Spot price
        K : Strike price
        T : Time to maturity (in years)
        r : Risk-free rate
        sigma : Volatility
        q : Dividend yield
        n_simulations : Number of Monte Carlo paths

    Returns:
        float : Estimated option price
    """
    Z = np.random.standard_normal(n_simulations)

    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.exp(-r * T) * np.mean(payoff)
