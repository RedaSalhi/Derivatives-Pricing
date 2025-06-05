import numpy as np

def price_equity_swap_forward_replication(S0, K, r, q, T):
    """
    Approximate equity swap using forward replication.
    
    Args:
        S0: Spot price
        K: Fixed strike or equivalent
        r: Risk-free rate
        q: Dividend yield
        T: Maturity

    Returns:
        NPV of the equity leg minus fixed leg (in present value)
    """
    F = S0 * np.exp((r - q) * T)
    return (F - K) * np.exp(-r * T)
