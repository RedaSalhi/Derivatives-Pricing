#pricing/models/barrier_monte_carlo.py

import numpy as np

def monte_carlo_barrier(
    S0, K, H, T, r, sigma,
    option_type="call", 
    barrier_type="up-and-out",
    n_simulations=10000, 
    n_steps=100
):
    """
    Monte Carlo pricing of a European barrier option - Optimized version.

    Parameters:
        S0 : float - Spot price
        K : float - Strike price
        H : float - Barrier level
        T : float - Time to maturity
        r : float - Risk-free rate
        sigma : float - Volatility
        option_type : str - 'call' or 'put'
        barrier_type : str - 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        n_simulations : int - Number of paths
        n_steps : int - Number of time steps

    Returns:
        tuple: (price, paths) - Option price and simulated paths
    """
    # Validation rapide des inputs
    if not all(x > 0 for x in [S0, K, H, T]):
        raise ValueError("S0, K, H, T doivent être > 0")
    if sigma < 0:
        raise ValueError("sigma doit être >= 0")
    
    dt = T / n_steps
    discount = np.exp(-r * T)
    
    # Simulation plus efficace avec vectorisation complète
    Z = np.random.randn(n_simulations, n_steps)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_returns = np.cumsum(increments, axis=1)
    
    # Construction des paths
    S = np.empty((n_simulations, n_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(log_returns)
    
    # Détection de barrière optimisée
    if 'up' in barrier_type:
        barrier_crossed = np.any(S[:, 1:] >= H, axis=1)  # 'up' barriers
    else:
        barrier_crossed = np.any(S[:, 1:] <= H, axis=1)  # 'down' barriers
    
    # Payoff final
    S_T = S[:, -1]
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:  # put
        payoff = np.maximum(K - S_T, 0.0)
    
    # Application de la logique de barrière
    if 'out' in barrier_type:
        payoff[barrier_crossed] = 0.0  # Knock-out
    else:  # 'in'
        payoff[~barrier_crossed] = 0.0  # Knock-in
    
    price = discount * np.mean(payoff)
    
    return price, S
