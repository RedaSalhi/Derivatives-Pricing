import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Pricing Function
# -----------------------------

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

# -----------------------------
# Plot at time t<T
# -----------------------------



def plot_forward_mark_to_market(
    strike_price: float,
    time_to_maturity: float,
    interest_rate: float,
    storage_cost: float = 0.0,
    dividend_yield: float = 0.0,
    position: str = "Long"
):
    """
    Plots the mark-to-market (MtM) value of a forward contract at time t before maturity.

    Parameters:
        strike_price (float): Agreed delivery price (K).
        time_to_maturity (float): Remaining time to maturity in years (T - t).
        interest_rate (float): Annual continuous risk-free rate (r).
        storage_cost (float): Annual continuous storage cost rate (c).
        dividend_yield (float): Annual continuous dividend yield (q).
        position (str): 'Long' or 'Short'
    """
    # Range of spot prices S_t
    S_t = np.linspace(0.5 * strike_price, 1.5 * strike_price, 500)
    
    # Forward value at time t (not maturity)
    value_t = S_t * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity) \
              - strike_price * np.exp(-interest_rate * time_to_maturity)

    if position == "Short":
        value_t = -value_t
        label = "Short Forward Value (t < T)"
    else:
        label = "Long Forward Value (t < T)"

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(S_t, value_t, label=label, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Spot Price at Time t (S_t)")
    plt.ylabel("Forward Contract Value at Time t")
    plt.title("Mark-to-Market Value of Forward Contract Before Maturity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Plot at T payout
# -----------------------------

def plot_forward_payout_and_value(strike_price: float, position: str = "Long"):
    """
    Plots the payout and value of a forward contract at maturity.

    Parameters:
        strike_price (float): Agreed delivery price (K).
        position (str): 'Long' or 'Short'
    """
    # Simulated spot prices at maturity
    S_T = np.linspace(0.5 * strike_price, 1.5 * strike_price, 500)

    if position == "Long":
        payout = S_T - strike_price
        label_payout = "Long Forward Payout"
    elif position == "Short":
        payout = strike_price - S_T
        label_payout = "Short Forward Payout"
    else:
        raise ValueError("Position must be 'Long' or 'Short'")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(S_T, payout, label=label_payout, color='blue')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    plt.title(f"Forward Contract: {label_payout}")
    plt.xlabel("Spot Price at Maturity (S_T)")
    plt.ylabel("Payout at Maturity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

