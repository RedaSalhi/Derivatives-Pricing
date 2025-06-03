# pricing/models/asian_monte_carlo.py

import numpy as np

def simulate_asian_paths(
    S0: float,               # Initial asset price
    T: float,                # Time to maturity (in years)
    r: float,                # Risk-free interest rate
    sigma: float,            # Volatility of the underlying asset
    n_steps: int,            # Number of time steps for averaging
    n_paths: int,            # Number of Monte Carlo simulation paths
    option_type: str,        # "call" or "put"
    asian_type: str          # "average_price" or "average_strike"
) -> np.ndarray:
    """
    Simulates Monte Carlo price paths for Asian options using geometric Brownian motion.

    Parameters:
        S0 (float): Initial stock price
        T (float): Time to maturity in years
        r (float): Risk-free rate
        sigma (float): Volatility
        n_steps (int): Number of time steps in the simulation
        n_paths (int): Number of simulated paths
        option_type (str): "call" or "put"
        asian_type (str): "average_price" or "average_strike"

    Returns:
        np.ndarray: Simulated asset paths (n_paths x n_steps), excluding initial S0
    """
    assert option_type in {"call", "put"}, "Invalid option type"
    assert asian_type in {"average_price", "average_strike"}, "Invalid Asian option type"
    
    dt = T / n_steps
    Z = np.random.normal(size=(n_paths, n_steps))  # Brownian increments
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # Generate log-price paths
    log_returns = drift + diffusion
    log_S_paths = np.cumsum(log_returns, axis=1)
    log_S_paths = np.insert(log_S_paths, 0, 0, axis=1)  # Start from log(S0)

    # Convert to price paths
    S_paths = S0 * np.exp(log_S_paths)

    # Remove initial S0 for averaging purposes
    return S_paths[:, 1:]
