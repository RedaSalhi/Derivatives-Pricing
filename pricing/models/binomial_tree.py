# pricing/models/binomial_tree.py

import numpy as np

def binomial_tree_price(option_type, S, K, T, r, sigma, q=0.0, N=100, american=False):
    """
    Binomial Tree pricing for European or American vanilla options.

    Parameters:
        option_type : 'call' or 'put'
        S : Spot price
        K : Strike price
        T : Time to maturity (in years)
        r : Risk-free interest rate
        sigma : Volatility of the underlying asset
        q : Continuous dividend yield (default: 0.0)
        N : Number of time steps (default: 100)
        american : True for American option, False for European

    Returns:
        float : Option price
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))         # Up factor
    d = 1 / u                               # Down factor
    p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability

    # Discount factor per step
    disc = np.exp(-r * dt)

    # Initialize asset prices at maturity
    ST = np.array([S * (u**j) * (d**(N - j)) for j in range(N + 1)])

    # Initialize option values at maturity
    if option_type == 'call':
        option_values = np.maximum(ST - K, 0)
    elif option_type == 'put':
        option_values = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'Call' or 'Put'")

    # Backward induction
    for i in range(N - 1, -1, -1):
        option_values = disc * (p * option_values[1:] + (1 - p) * option_values[:-1])

        if american:
            ST = np.array([S * (u**j) * (d**(i - j)) for j in range(i + 1)])
            if option_type == 'call':
                option_values = np.maximum(option_values, ST - K)
            else:
                option_values = np.maximum(option_values, K - ST)

    return option_values[0]
