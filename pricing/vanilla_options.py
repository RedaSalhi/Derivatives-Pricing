# pricing/vanilla_options.py

from pricing.models.black_scholes import black_scholes_price
from pricing.models.binomial_tree import binomial_tree_price
from pricing.models.longstaff_schwartz import longstaff_schwartz_price
from pricing.models.monte_carlo import monte_carlo_price

def price_vanilla_option(option_type, exercise_style, model, **kwargs):
    """
    Unified interface for vanilla option pricing.
    """

    option_type = option_type.lower()
    exercise_style = exercise_style.lower()
    model = model.lower()

    if exercise_style not in ["European", "American"]:
        raise ValueError("Exercise style must be 'European' or 'American'")

    # Black-Scholes
    if model == "Black-Scholes":
        if exercise_style == "American":
            raise ValueError("Black-Scholes does not support American options.")
        return black_scholes_price(option_type=option_type, **kwargs)

    # Binomial Tree
    elif model == "Binomial":
        return binomial_tree_price(
            option_type=option_type,
            American=(exercise_style == "American"),
            **kwargs
        )

    # Monte Carlo
    elif model == "Monte-Carlo":
        if exercise_style == "American":
            return longstaff_schwartz_price(option_type=option_type, **kwargs)
        else:
            return monte_carlo_price(option_type=option_type, **kwargs)

    else:
        raise ValueError(f"Unknown pricing model: {model}")

