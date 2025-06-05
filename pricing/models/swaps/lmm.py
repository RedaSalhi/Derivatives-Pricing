import numpy as np

def simulate_libor_paths(L0, vol, dt, n_steps, n_paths, drift=0.0):
    """
    Simulates forward LIBOR rates using a lognormal model:
        dL = mu * L * dt + sigma * L * dW
    """
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = L0
    for t in range(1, n_steps + 1):
        Z = np.random.normal(size=n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z)
    return paths

def price_interest_rate_swap_lmm(
    notional,
    fixed_rate,
    L0,
    vol,
    payment_times,
    discount_curve,
    n_steps=10,
    n_paths=10000,
    seed=42
):
    """
    Prices a payer IRS using a simplified LIBOR Market Model with lognormal dynamics.

    Args:
        notional: Notional amount
        fixed_rate: Fixed leg rate (K)
        L0: Initial forward rate (flat across all tenors)
        vol: Volatility of LIBOR
        payment_times: List of payment times
        discount_curve: Callable P(0, T)
        n_steps: Number of time steps (same as # of payment times ideally)
        n_paths: Monte Carlo paths
        seed: Random seed

    Returns:
        Estimated IRS NPV (payer: float - fixed)
    """
    np.random.seed(seed)
    dt = payment_times[1] - payment_times[0]
    year_fractions = np.diff([0] + list(payment_times))

    # Simulate LIBOR forward paths
    libor_paths = simulate_libor_paths(L0, vol, dt, n_steps=len(payment_times)-1, n_paths=n_paths)

    # Compute discounted floating leg cash flows (Monte Carlo mean)
    float_legs = np.zeros(n_paths)
    for i, (alpha, T) in enumerate(zip(year_fractions, payment_times)):
        if i == 0:
            continue
        float_legs += notional * alpha * libor_paths[:, i - 1] * discount_curve(T)
    pv_float = np.mean(float_legs)

    # Fixed leg (deterministic)
    pv_fixed = notional * sum(
        fixed_rate * alpha * discount_curve(T)
        for alpha, T in zip(year_fractions, payment_times)
    )

    return pv_float - pv_fixed
