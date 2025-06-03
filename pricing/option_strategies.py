from .vanilla_options import price_vanilla_option
import numpy as np

def price_option_strategy(legs, exercise_style, model, **kwargs):
    """
    Price a multi-leg option strategy by summing the price of each leg.

    Parameters:
        legs (list): Each dict has "type" ('call'/'put'), "strike", and "qty"
        exercise_style (str): "european" or "american"
        model (str): Pricing model ("black-scholes", "binomial", etc.)
        kwargs: Common params like S, T, sigma, r...

    Returns:
        dict: {
            "strategy_price": float,
            "individual_prices": list of floats
        }
    """
    total_price = 0
    individual_prices = []

    for leg in legs:
        price = price_vanilla_option(
            option_type=leg["type"].lower(),
            exercise_style=exercise_style.lower(),
            model=model.lower(),
            K=leg["strike"],
            **kwargs
        )
        leg_price = leg["qty"] * price
        total_price += leg_price
        individual_prices.append(leg_price)

    return {
        "strategy_price": total_price,
        "individual_prices": individual_prices
    }

def compute_strategy_payoff(legs, spot_prices):
    """
    Compute total strategy payoff over a range of spot prices.

    Parameters:
        legs (list): Strategy legs as described above
        spot_prices (np.array): Range of spot prices

    Returns:
        np.array: Total payoff at each spot price
    """
    payoffs = np.zeros_like(spot_prices, dtype=float)

    for leg in legs:
        K = leg["strike"]
        qty = leg["qty"]
        opt_type = leg["type"].lower()

        if opt_type == "call":
            payoffs += qty * np.maximum(spot_prices - K, 0)
        elif opt_type == "put":
            payoffs += qty * np.maximum(K - spot_prices, 0)
        else:
            raise ValueError("Leg type must be 'call' or 'put'.")

    return payoffs

def get_predefined_strategy(name, strike1, strike2=None, strike3=None, strike4=None):
    """
    Return predefined multi-leg strategy.

    Parameters:
        name (str): Strategy name
        strike1..4: Strike prices used depending on strategy

    Returns:
        list of dicts OR str if error
    """
    name = name.strip().lower()

    if name == "straddle":
        return [
            {"type": "call", "strike": strike1, "qty": 1},
            {"type": "put", "strike": strike1, "qty": 1}
        ]

    elif name == "bull call spread":
        if strike2 is None:
            return "Bull Call Spread requires strike1 (Long) and strike2 (Short)."
        return [
            {"type": "call", "strike": strike1, "qty": 1},
            {"type": "call", "strike": strike2, "qty": -1}
        ]

    elif name == "bear put spread":
        if strike2 is None:
            return "Bear Put Spread requires strike1 (Long) and strike2 (Short)."
        return [
            {"type": "put", "strike": strike1, "qty": 1},
            {"type": "put", "strike": strike2, "qty": -1}
        ]

    elif name == "butterfly":
        if None in (strike2, strike3):
            return "Butterfly requires strike1 (low), strike2 (middle), and strike3 (high)."
        return [
            {"type": "call", "strike": strike1, "qty": 1},
            {"type": "call", "strike": strike2, "qty": -2},
            {"type": "call", "strike": strike3, "qty": 1}
        ]

    elif name == "iron condor":
        if None in (strike2, strike3, strike4):
            return "Iron Condor requires 4 strikes: strike1 (Put Long), strike2 (Put Short), strike3 (Call Short), strike4 (Call Long)."
        return [
            {"type": "put", "strike": strike1, "qty": 1},
            {"type": "put", "strike": strike2, "qty": -1},
            {"type": "call", "strike": strike3, "qty": -1},
            {"type": "call", "strike": strike4, "qty": 1}
        ]

    else:
        return f"Unknown strategy name: {name.capitalize()}"



import matplotlib.pyplot as plt
import numpy as np

def plot_strategy_price_vs_param(
    legs,
    exercise_style,
    model,
    param_name,
    param_range,
    fixed_params,
    n_points=50
):
    """
    Plot the total price of an option strategy as one parameter varies.

    Parameters:
        legs : list of dicts describing the strategy
        exercise_style : 'european' or 'american'
        model : pricing model string
        param_name : parameter to vary ('S', 'K', 'T', 'r', 'sigma', 'q')
        param_range : (min, max) tuple
        fixed_params : dict of fixed input parameters
        n_points : number of values to compute

    Returns:
        matplotlib.figure.Figure
    """
    x_vals = np.linspace(param_range[0], param_range[1], n_points)
    y_vals = []

    for val in x_vals:
        kwargs = fixed_params.copy()
        kwargs[param_name] = val

        # Pass extra parameters for binomial or monte-carlo
        if model == "binomial":
            kwargs["N"] = fixed_params.get("N", 100)
        elif model == "monte-carlo":
            kwargs["n_simulations"] = fixed_params.get("n_simulations", 1000)

        try:
            result = price_option_strategy(
                legs=legs,
                exercise_style=exercise_style,
                model=model,
                **kwargs
            )
            y_vals.append(result["strategy_price"])
        except:
            y_vals.append(np.nan)

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(x_vals, y_vals, label="Strategy Price")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Strategy Price")
    ax.set_title(f"Strategy Price vs {param_name} ({model}, {exercise_style})")
    ax.grid(True)
    ax.legend()
    return fig

