# pricing/models/lookback_monte_carlo.py

import numpy as np

def monte_carlo_lookback_option(S0, r, sigma, T, n_paths, n_steps, option_type="call", floating_strike=True):
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate price paths
    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        prices[:, t] = prices[:, t - 1] * np.exp(drift + diffusion * z)

    S_T = prices[:, -1]
    S_max = np.max(prices, axis=1)
    S_min = np.min(prices, axis=1)

    if floating_strike:
        if option_type == "call":
            payoffs = S_T - S_min
        else:
            payoffs = S_max - S_T
    else:
        # Fixed strike assumed as S0
        if option_type == "call":
            payoffs = np.maximum(S_max - S0, 0)
        else:
            payoffs = np.maximum(S0 - S_min, 0)

    discounted_payoff = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoff), np.std(discounted_payoff) / np.sqrt(n_paths)
