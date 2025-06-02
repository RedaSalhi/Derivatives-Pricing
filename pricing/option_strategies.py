from .vanilla_options import price_vanilla_option
import numpy as np


def price_option_strategy(legs, exercise_style, model, **kwargs):
    """
    Price a multi-leg option strategy by summing the price of each leg.

    Parameters:
        legs (list): List of dictionaries, each containing:
            - "type": "call" or "put"
            - "strike": strike price
            - "qty": quantity (positive for Long, negative for Short)
        exercise_style (str): "european" or "american"
        model (str): Pricing model ("black-scholes", "binomial", etc.)
        kwargs: Common option parameters like S, T, sigma, r

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
            option_type=leg["type"],
            exercise_style=exercise_style,
            model=model,
            K=leg["strike"],
            **kwargs
        )
        leg_price = leg["qty"] * price
        total_price += leg_price
        individual_prices.append(leg_price)

    return {
        "Strategy price": total_price,
        "Individual prices": individual_prices
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
        if leg["type"].lower() == "call":
            payoffs += qty * np.maximum(spot_prices - K, 0)
        elif leg["type"].lower() == "put":
            payoffs += qty * np.maximum(K - spot_prices, 0)

    return payoffs


def get_predefined_strategy(name, strike1, strike2=None, strike3=None, strike4=None):
    """
    Return a predefined option strategy using explicit strike prices.

    Parameters:
        name (str): Strategy name: "Straddle", "Bull call Spread", "Bear put Spread", "Butterfly", etc.
        strike1, strike2, strike3 (float): Strike prices as needed by the strategy

    Returns:
        list of dicts: Each dict has "type", "strike", and "qty"
    """
    name = name.lower()

    if name == "Straddle":
        if strike2 is not None or strike3 is not None:
            return "Straddle only requires strike1 (ATM)."
        return [
            {"type": "call", "strike": strike1, "qty": 1},
            {"type": "put", "strike": strike1, "qty": 1}
        ]
    
    elif name == "Bull call Spread":
        if strike2 is None:
            return "Bull call Spread requires strike1 (Long) and strike2 (Short)."
        return [
            {"type": "call", "strike": strike1, "qty": 1},
            {"type": "call", "strike": strike2, "qty": -1}
        ]

    elif name == "Bear put Spread":
        if strike2 is None:
            return print("Bear put Spread requires strike1 (Long) and strike2 (Short).")
        return [
            {"type": "put", "strike": strike1, "qty": 1},
            {"type": "put", "strike": strike2, "qty": -1}
        ]
    
    elif name == "Butterfly":
        if strike2 is None or strike3 is None:
            return print("Butterfly requires strike1 (low), strike2 (middle), and strike3 (high).")
        return [
            {"type": "call", "strike": strike1, "qty": 1},
            {"type": "call", "strike": strike2, "qty": -2},
            {"type": "call", "strike": strike3, "qty": 1}
        ]

    elif name == "Iron Condor":
        if strike2 is None or strike3 is None:
            return print("Iron Condor requires 4 strikes: strike1 (put Long), strike2 (put Short), strike3 (call Short), strike4 (call Long)")
        if strike4 is None:
            return print("Iron Condor needs strike4 (call Long)")
        return [
            {"type": "put", "strike": strike1, "qty": 1},
            {"type": "put", "strike": strike2, "qty": -1},
            {"type": "call", "strike": strike3, "qty": -1},
            {"type": "call", "strike": strike4, "qty": 1}
        ]

    else:
        return print(f"Unknown strategy name: {name}")
