import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
        spot_price : Current spot price of the underlying asset.
        interest_rate : Annualized continuous risk-free rate (r).
        time_to_maturity : Time to maturity in years (T).
        storage_cost : Annualized continuous storage cost (c), default 0.
        dividend_yield : Annualized continuous dividend yield (q), default 0.
        pricing_model : Pricing model to use (default "cost_of_carry").
        sub_type : Type of forward (default "plain").

    Returns:
        Forward price (F).
    """
    pricing_model = pricing_model.lower()

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
    position: str = "long"
):
    """
    Plots the mark-to-market (MtM) value of a forward contract at time t before maturity.

    Parameters:
        strike_price : Agreed delivery price (K).
        time_to_maturity : Remaining time to maturity in years (T - t).
        interest_rate : Annual continuous risk-free rate (r).
        storage_cost : Annual continuous storage cost rate (c).
        dividend_yield : Annual continuous dividend yield (q).
        position : 'long' or 'short'
    """
    position = position.lower()
    S_t = np.linspace(0.5 * strike_price, 1.5 * strike_price, 500)

    value_t = S_t * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity) \
              - strike_price * np.exp(-interest_rate * time_to_maturity)

    if position == "short":
        value_t = -value_t
        label = "Short Forward Value (t < T)"
    else:
        label = "Long Forward Value (t < T)"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_t, value_t, label=label, color='purple')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_xlabel("Spot Price at Time t (Sₜ)")
    ax.set_ylabel("Forward Contract Value at Time t")
    ax.set_title("Mark-to-Market Value of Forward Contract Before Maturity")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# -----------------------------
# Plot at T payout
# -----------------------------

def plot_forward_payout_and_value(strike_price: float, position: str = "long"):
    """
    Plots the payout and value of a forward contract at maturity.

    Parameters:
        strike_price : Agreed delivery price (K).
        position : 'long' or 'short'
    """
    position = position.lower()
    S_T = np.linspace(0.5 * strike_price, 1.5 * strike_price, 500)

    if position == "long":
        payout = S_T - strike_price
        label = "Long Forward Payout"
    elif position == "short":
        payout = strike_price - S_T
        label = "Short Forward Payout"
    else:
        raise ValueError("Position must be 'long' or 'short'")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_T, payout, label=label, color='blue')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_xlabel("Spot Price at Maturity (Sₜ)")
    ax.set_ylabel("Payout at Maturity")
    ax.set_title(f"Forward Contract: {label}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

