import numpy as np

def price_forward_contract(
    spot_price: float,
    interest_rate: float,
    time_to_maturity: float,
    storage_cost: float = 0.0,
    dividend_yield: float = 0.0,
    pricing_model: str = "cost_of_carry",
    sub_type: str = "plain"
) -> float:
    """
    Prices a forward contract using the specified pricing model.

    Parameters:
        spot_price (float): Current spot price of the underlying asset.
        interest_rate (float): Annualized continuous risk-free rate (r).
        time_to_maturity (float): Time to maturity in years (T).
        storage_cost (float): Annualized continuous storage cost (c), default 0.
        dividend_yield (float): Annualized continuous dividend yield (q), default 0.
        pricing_model (str): Pricing model to use (default "cost_of_carry").
        sub_type (str): Type of forward (default "plain").

    Returns:
        float: Forward price (F).
    """
    if pricing_model == "cost_of_carry":
        F = spot_price * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity)
        return F
    else:
        raise NotImplementedError(f"Pricing model '{pricing_model}' is not implemented for forward contracts.")
