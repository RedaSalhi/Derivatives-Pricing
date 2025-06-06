# pricing/models/longstaff_schwartz.py

import numpy as np

def longstaff_schwartz_price(option_type, S, K, T, r, sigma, q=0.0, n_simulations=100_000, n_steps=50, poly_degree=2):
    """
    Longstaff-Schwartz Monte Carlo pricing for American Call/Put options.

    Parameters:
        option_type : 'call' or 'put' (case-insensitive)
        S : Spot price
        K : Strike price
        T : Time to maturity (in years)
        r : Risk-free rate
        sigma : Volatility
        q : Dividend yield
        n_simulations : Number of Monte Carlo paths
        n_steps : Time steps for simulation
        poly_degree : Degree of polynomial for regression

    Returns:
        float : Estimated American option price
    """
    if option_type is None:
        raise ValueError("Missing option_type: expected 'Call' or 'Put'")
    option_type = option_type.lower()

    dt = T / n_steps
    df = np.exp(-r * dt)

    np.random.seed(0)
    Z = np.random.randn(n_simulations, n_steps)
    paths = np.zeros_like(Z)
    paths[:, 0] = S

    # Generate price paths
    for t in range(1, n_steps):
        paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    # Compute payoffs at maturity
    if option_type == 'call':
        payoff = np.maximum(paths[:, -1] - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - paths[:, -1], 0)
    else:
        raise ValueError("option_type must be 'call' or 'put' (case-insensitive)")

    cashflows = payoff.copy()

    # Backward induction using regression
    for t in range(n_steps - 2, 0, -1):
        if option_type == 'call':
            itm = paths[:, t] > K
        else:
            itm = paths[:, t] < K

        X = paths[itm, t]
        Y = cashflows[itm] * df

        if len(X) == 0:
            continue

        # Regression on ITM paths
        A = np.vander(X, N=poly_degree + 1, increasing=True)
        coeffs = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation_value = A @ coeffs

        # Exercise or continue
        if option_type == 'call':
            exercise_value = X - K
        else:
            exercise_value = K - X

        exercise = exercise_value > continuation_value
        idx = np.where(itm)[0][exercise]
        cashflows[idx] = exercise_value[exercise]

        cashflows *= df

    return np.mean(cashflows)

