# pricing/vanilla_option.py

from .models.black_scholes import black_scholes_price
from .models.binomial_tree import binomial_tree_price

def price_vanilla_option(
    option_type,
    exercise_style,
    model,
    **kwargs
):
    """
    Master pricing function for vanilla options.

    Parameters:
        option_type (str): 'call' or 'put'
        exercise_style (str): 'european' or 'american'
        model (str): 'black-scholes' or 'binomial'
        kwargs: All pricing params (S, K, T, r, sigma, q, N...)

    Returns:
        float: option price
    """

    option_type = option_type.lower()
    exercise_style = exercise_style.lower()
    model = model.lower()

    if exercise_style not in ["european", "american"]:
        raise ValueError("exercise_style must be 'european' or 'american'")

    if model == "black-scholes":
        if exercise_style == "american":
            raise ValueError("Black-Scholes model does not support American options.")
        return black_scholes_price(option_type=option_type, **kwargs)

    elif model == "binomial":
        american_flag = exercise_style == "american"
        return binomial_tree_price(option_type=option_type, american=american_flag, **kwargs)

    else:
        raise ValueError(f"Unknown model: {model}")
