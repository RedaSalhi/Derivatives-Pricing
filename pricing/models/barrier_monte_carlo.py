import numpy as np

def monte_carlo_barrier(
    S0, K, H, T, r, sigma, option_type="call", barrier_type="up-and-out",
    n_simulations=10000, n_steps=100
):
    """
    Monte Carlo pricing of European barrier option.

    Parameters:
        S0 : float – Initial spot price
        K : float – Strike price
        H : float – Barrier level
        T : float – Time to maturity
        r : float – Risk-free rate
        sigma : float – Volatility
        option_type : 'call' or 'put'
        barrier_type : 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        n_simulations : int – Number of simulated paths
        n_steps : int – Number of time steps per path

    Returns:
        float: Estimated option price
    """
    dt = T / n_steps
    discount = np.exp(-r * T)

    # Simulate Brownian motion
    Z = np.random.randn(n_simulations, n_steps)
    S = np.zeros_like(Z)
    S[:, 0] = S0

    for t in range(1, n_steps):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    S_T = S[:, -1]
    S_min = np.min(S, axis=1)
    S_max = np.max(S, axis=1)

    # Check barrier conditions
    if barrier_type == "up-and-out":
        barrier_triggered = S_max >= H
        knocked_in = ~barrier_triggered
    elif barrier_type == "down-and-out":
        barrier_triggered = S_min <= H
        knocked_in = ~barrier_triggered
    elif barrier_type == "up-and-in":
        barrier_triggered = S_max >= H
        knocked_in = barrier_triggered
    elif barrier_type == "down-and-in":
        barrier_triggered = S_min <= H
        knocked_in = barrier_triggered
    else:
        raise ValueError("Unknown barrier type")

    # Compute terminal payoff
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError("Unknown option type")

    # Only keep payoff where option is active
    payoff[~knocked_in] = 0.0

    return discount * np.mean(payoff)
