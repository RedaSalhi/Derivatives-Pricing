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
