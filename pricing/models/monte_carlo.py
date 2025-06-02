# pricing/models/monte_carlo.py

import numpy as np

def monte_carlo_price(option_type, S, K, T, r, sigma, q=0.0, n_simulations=100_000):
    """
    Monte Carlo pricing for European Call or Put options.

    Parameters:
        option_type : 'Call' or 'Put'
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

    if option_type == 'Call':
        payoff = np.maximum(ST - K, 0)
    elif option_type == 'Put':
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'Call' or 'Put'")

    return np.exp(-r * T) * np.mean(payoff)
