

from pricing.models.asian_monte_carlo import simulate_asian_paths
from pricing.models.asian_pde import price_asian_option_pde
import numpy as np

def price_asian_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int = 10000,
    method: str = "monte_carlo",       # or "pde"
    option_type: str = "call",         # "call" or "put"
    asian_type: str = "average_price"  # "average_price" or "average_strike"
) -> float:
    """
    Master pricing function for Asian options.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free rate
        sigma (float): Volatility
        n_steps (int): Number of time steps (for both MC and PDE)
        n_paths (int): Number of Monte Carlo paths
        method (str): "monte_carlo" or "pde"
        option_type (str): "call" or "put"
        asian_type (str): "average_price" or "average_strike"

    Returns:
        float: Estimated option price
    """
    assert method in {"monte_carlo", "pde"}, "Invalid method"
    assert option_type in {"call", "put"}, "Invalid option type"
    assert asian_type in {"average_price", "average_strike"}, "Invalid Asian option type"

    if method == "monte_carlo":
        paths = simulate_asian_paths(S0, T, r, sigma, n_steps, n_paths, option_type, asian_type)
        if asian_type == "average_price":
            averages = np.mean(paths, axis=1)
            payoffs = np.maximum(averages - K, 0) if option_type == "call" else np.maximum(K - averages, 0)
        else:  # average_strike
            averages = np.mean(paths, axis=1)
            S_T = paths[:, -1]
            payoffs = np.maximum(S_T - averages, 0) if option_type == "call" else np.maximum(averages - S_T, 0)
        return np.exp(-r * T) * np.mean(payoffs)

    elif method == "pde":
        return price_asian_option_pde(
            S0=S0, K=K, T=T, r=r, sigma=sigma,
            Smax=3*S0, M=200, N=n_steps,
            option_type=option_type,
            asian_type=asian_type
        )
