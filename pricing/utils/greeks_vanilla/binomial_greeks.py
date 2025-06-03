import numpy as np
from pricing.models.binomial_tree import binomial_tree_price

def binomial_greeks(S, K, T, r, sigma, q=0.0, option_type="call", N=100, american=False, h=1e-2):
    def price(S_val, T_val=None, r_val=None, sigma_val=None):
        return binomial_tree_price(
            option_type=option_type,
            S=S_val,
            K=K,
            T=T if T_val is None else T_val,
            r=r if r_val is None else r_val,
            sigma=sigma if sigma_val is None else sigma_val,
            q=q,
            N=N,
            american=american
        )

    # Central differences
    delta = (price(S + h) - price(S - h)) / (2 * h)
    gamma = (price(S + h) - 2 * price(S) + price(S - h)) / (h ** 2)
    vega = (price(S, sigma_val=sigma + h) - price(S, sigma_val=sigma - h)) / (2 * h)
    rho = (price(S, r_val=r - h) - price(S, r_val=r)) / h
    theta = (price(S, T_val=T - h) - price(S)) / h  # Backward diff

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100,
        "theta": theta / 365,
        "rho": rho / 100
    }
