#pricin

import numpy as np

def simulate_asian_paths(
    S0: float,         # Initial asset price
    T: float,          # Time to maturity (in years)
    r: float,          # Risk-free interest rate
    sigma: float,      # Volatility of the underlying asset
    n_steps: int,      # Number of time steps for averaging
    n_paths: int       # Number of Monte Carlo simulation paths
) -> np.ndarray:
    """
    Simulates Monte Carlo price paths for an Asian option using geometric Brownian motion.

    Returns:
        np.ndarray: Simulated asset price paths of shape (n_paths, n_steps), excluding initial S0.
    """
    dt = T / n_steps  # Time increment
    Z = np.random.normal(size=(n_paths, n_steps))  # Random shocks
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    # Log-price increments and cumulative sum
    log_returns = drift + diffusion
    log_S_paths = np.cumsum(log_returns, axis=1)
    log_S_paths = np.insert(log_S_paths, 0, 0, axis=1)  # Start at log(S0)

    # Convert log-prices to actual prices
    S_paths = S0 * np.exp(log_S_paths)
    
    # Drop initial price for averaging purposes
    return S_paths[:, 1:]
