from pricing.utils.greeks_vanilla.black_scholes_greeks import bs_greeks
from pricing.utils.greeks_vanilla.binomial_greeks import binomial_greeks
from pricing.utils.greeks_vanilla.monte_carlo_greeks import monte_carlo_greeks
import numpy as np

def compute_greek(
    greek_name, model, option_type, S_values, K, T, r, sigma, q=0.0
):
    """
    Dispatches Greek computation for a given model.

    Returns:
        list of float values of the specified Greek over spot prices
    """
    greek_name = greek_name.lower()
    model = model.lower()
    option_type = option_type.lower()

    greek_values = []

    for S in S_values:
        if model == "black-scholes":
            g = bs_greeks(S, K, T, r, sigma, q, option_type)
        elif model == "binomial":
            g = binomial_greeks(S, K, T, r, sigma, q, option_type)
        elif model == "monte-carlo":
            g = monte_carlo_greeks(S, K, T, r, sigma, q, option_type)
        else:
            raise ValueError(f"Unsupported model: {model}")

        greek_values.append(g[greek_name])

    return greek_values
