pricing/models/barrier_monte_carlo.py

import numpy as np

def monte_carlo_barrier(
    S0, K, H, T, r, sigma,
    option_type="call", barrier_type="up-and-out",
    n_simulations=10000, n_steps=100
):
    """
    Monte Carlo pricing of a European barrier option.

    Parameters:
        S0 : float – Spot price
        K : float – Strike price
        H : float – Barrier level
        T : float – Time to maturity
        r : float – Risk-free interest rate
        sigma : float – Volatility
        option_type : str – 'call' or 'put'
        barrier_type : str – 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        n_simulations : int – Number of simulation paths
        n_steps : int – Number of time steps per path

    Returns:
        tuple: (Estimated option price, Simulated paths)
    """
    dt = T / n_steps
    discount = np.exp(-r * T)

    # Initialize paths
    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = S0

    # Generate paths
    Z = np.random.randn(n_simulations, n_steps)
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])

    # Final asset prices
    S_T = S[:, -1]
    S_max = np.max(S, axis=1)
    S_min = np.min(S, axis=1)

    # Determine barrier hit status
    if barrier_type == "up-and-out":
        knocked_in = S_max < H
    elif barrier_type == "down-and-out":
        knocked_in = S_min > H
    elif barrier_type == "up-and-in":
        knocked_in = S_max >= H
    elif barrier_type == "down-and-in":
        knocked_in = S_min <= H
    else:
        raise ValueError("Invalid barrier type.")

    # Compute payoff
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("Invalid option type.")

    payoff[~knocked_in] = 0.0
    price = discount * np.mean(payoff)

    return price, S



