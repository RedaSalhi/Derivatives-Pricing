# pricing/vanilla_options.py

from pricing.models.black_scholes import black_scholes_price
from pricing.models.binomial_tree import binomial_tree_price
from pricing.models.longstaff_schwartz import longstaff_schwartz_price
from pricing.models.monte_carlo import monte_carlo_price
import numpy as np
import matplotlib.pyplot as plt

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




def plot_option_price_vs_param(
    option_type,
    exercise_style,
    model,
    param_name,
    param_range,
    fixed_params,
    n_points=50,
):
    """
    Plots the option price as a function of one varying parameter.

    Parameters:
        option_type : 'call' or 'put'
        exercise_style : 'european' or 'american'
        model : 'black-scholes', 'binomial', or 'monte-carlo'
        param_name : str, one of ['S', 'K', 'T', 'r', 'sigma', 'q']
        param_range : tuple, (min_value, max_value)
        fixed_params : dict, fixed values for other parameters
        n_points : int, number of points to evaluate

    Returns:
        matplotlib.figure.Figure
    """
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    prices = []

    for val in param_values:
        input_params = fixed_params.copy()
        input_params[param_name] = val

        if model == "binomial":
            input_params["N"] = fixed_params.get("N", 100)
        elif model == "monte-carlo":
            input_params["n_simulations"] = fixed_params.get("n_simulations", 1000)

        try:
            price = price_vanilla_option(
                option_type=option_type,
                exercise_style=exercise_style,
                model=model,
                **input_params
            )
        except:
            price = np.nan

        prices.append(price)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(param_values, prices, label=f"Price vs {param_name}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Option Price")
    ax.set_title(f"{model.title()} {exercise_style.title()} {option_type.title()} Option")
    ax.grid(True)
    ax.legend()
    return fig

