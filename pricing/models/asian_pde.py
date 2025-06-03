# pricing/models/asian_pde.py

import numpy as np

def price_asian_option_pde(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    Smax: float,
    M: int,        # number of asset price steps
    N: int,        # number of time steps
    option_type: str,     # "call" or "put"
    asian_type: str       # "average_price" or "average_strike"
) -> float:
    """
    Price an Asian option using finite difference PDE methods.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility
        Smax (float): Maximum stock price to consider in the grid
        M (int): Number of stock price steps
        N (int): Number of time steps
        option_type (str): "call" or "put"
        asian_type (str): "average_price" or "average_strike"

    Returns:
        float: Estimated option price at S0
    """
    assert option_type in {"call", "put"}, "Invalid option type"
    assert asian_type in {"average_price", "average_strike"}, "Invalid Asian option type"

    dS = Smax / M
    dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.zeros(M + 1)

    # Terminal payoff
    if asian_type == "average_price":
        # Geometric approximation: G ≈ sqrt(S * S0)
        if option_type == "call":
            V[:] = np.maximum(np.sqrt(S * S0) - K, 0)
        else:  # put
            V[:] = np.maximum(K - np.sqrt(S * S0), 0)

    elif asian_type == "average_strike":
        # Standard payoff: max(S - average_strike, 0), approx strike = K
        if option_type == "call":
            V[:] = np.maximum(S - K, 0)
        else:  # put
            V[:] = np.maximum(K - S, 0)

    # Backward induction using explicit Euler
    for _ in range(N):
        V_new = V.copy()
        for i in range(1, M):
            delta = (V[i + 1] - V[i - 1]) / (2 * dS)
            gamma = (V[i + 1] - 2 * V[i] + V[i - 1]) / (dS ** 2)
            theta = -0.5 * sigma ** 2 * S[i] ** 2 * gamma - r * S[i] * delta + r * V[i]
            V_new[i] = V[i] - dt * theta
        V = V_new

    # Return interpolated value at S0
    return np.interp(S0, S, V)
