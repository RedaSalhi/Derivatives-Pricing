# pricing/models/lookback_binomial.py

import numpy as np

def binomial_lookback_fixed_european(S0, K, r, sigma, T, N, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Tree of asset prices
    asset_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            asset_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)

    # Tree of payoff values (path-dependent on min or max)
    payoff_tree = np.zeros_like(asset_tree)

    for j in range(N + 1):
        path_prices = [S0 * (u ** (N - j)) * (d ** j)]
        max_S = max(path_prices)
        min_S = min(path_prices)

        S_T = asset_tree[j, N]
        if option_type == "call":
            payoff_tree[j, N] = max(S_T - min_S, 0)
        else:
            payoff_tree[j, N] = max(max_S - S_T, 0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            payoff_tree[j, i] = discount * (
                p * payoff_tree[j, i + 1] + (1 - p) * payoff_tree[j + 1, i + 1]
            )

    return payoff_tree[0, 0]
