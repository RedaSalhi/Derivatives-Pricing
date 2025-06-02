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

import matplotlib.pyplot as plt
import numpy as np

def plot_forward_payout_and_price(forward_price: float, strike_price: float, position: str = "long"):
    """
    Plots the payout and price of a forward contract at maturity.

    Parameters:
        forward_price (float): Fair value of the forward (from pricing model).
        strike_price (float): Agreed delivery price (K).
        position (str): "long" or "short" forward position.
    """
    # Define range of spot prices at maturity
    S_T = np.linspace(0.5 * strike_price, 1.5 * strike_price, 500)

    # Define payout depending on long or short
    if position == "long":
        payout = S_T - strike_price
        label_payout = "Long Forward Payout"
    elif position == "short":
        payout = strike_price - S_T
        label_payout = "Short Forward Payout"
    else:
        raise ValueError("Position must be 'long' or 'short'")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(S_T, payout, label=label_payout, color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axhline(forward_price - strike_price, color='red', linestyle='--', label="Fair Forward Price (Today)")

    plt.title(f"Forward Contract: {label_payout} and Fair Price")
    plt.xlabel("Spot Price at Maturity (S_T)")
    plt.ylabel("Payout / Forward Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
