# pricing/vanilla_options.py

from pricing.models.black_scholes import black_scholes_price
from pricing.models.binomial_tree import binomial_tree_price
from pricing.models.longstaff_schwartz import longstaff_schwartz_price
from pricing.models.monte_carlo import monte_carlo_price

def price_vanilla_option(option_type, exercise_style, model, **kwargs):
    """
    Unified interface for vanilla option pricing.
    Supports Black-Scholes, Binomial, and Monte Carlo.
    """
    option_type = option_type.lower()
    exercise_style = exercise_style.lower()
    model = model.lower()

    if exercise_style not in ["european", "american"]:
        raise ValueError("Exercise style must be 'European' or 'American'")

    if model == "black-scholes":
        if exercise_style == "american":
            raise ValueError("Black-Scholes does not support American options.")
        return black_scholes_price(option_type=option_type, **kwargs)

    elif model == "binomial":
        return binomial_tree_price(
            option_type=option_type,
            american=(exercise_style == "american"),
            **kwargs
        )

    elif model == "monte-carlo":
        if exercise_style == "american":
            return longstaff_schwartz_price(option_type=option_type, **kwargs)
        else:
            return monte_carlo_price(option_type=option_type, **kwargs)

    else:
        raise ValueError(f"Unknown pricing model: {model}")
