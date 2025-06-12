# pricing/digital_option.py


from pricing.models.digital_black_scholes import (
    digital_cash_or_nothing,
    digital_asset_or_nothing
)
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def price_digital_option(model="black_scholes", option_type="call", style="cash",
                         S=100, K=100, T=1, r=0.05, sigma=0.2, Q=1.0):
    """
    Price a digital option using the specified model.

    Parameters:
        model : pricing model (default 'black_scholes')
        option_type : 'call' or 'put'
        style : 'cash' or 'asset'
        S, K, T, r, sigma, Q : standard option parameters

    Returns:
        float: option price
    """
    if model == "black_scholes":
        if style == "cash":
            return digital_cash_or_nothing(option_type, S, K, T, r, sigma, Q)
        elif style == "asset":
            return digital_asset_or_nothing(option_type, S, K, T, r, sigma)
        else:
            raise ValueError("style must be 'cash' or 'asset'")
    else:
        raise NotImplementedError(f"Model '{model}' not implemented")


def plot_digital_payoff(K, option_type="call", style="cash", Q=1.0, S_min=0, S_max=200, num=500):
    """
    Affiche dans Streamlit le payoff à maturité d'une option digitale.

    Parameters:
        K : float – Strike
        option_type : str – 'call' or 'put'
        style : str – 'cash' or 'asset'
        Q : float – Payout amount if exercised
        S_min, S_max : float – Spot price bounds
        num : int – Discretization points
    """
    S_range = np.linspace(S_min, S_max, num)
    payoff = np.zeros_like(S_range)

    for i, S in enumerate(S_range):
        if option_type == "call" and S > K:
            payoff[i] = Q if style == "cash" else S
        elif option_type == "put" and S < K:
            payoff[i] = Q if style == "cash" else S

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(S_range, payoff, label="Payoff à maturité")
    ax.axvline(x=K, linestyle="--", color="gray", label="Strike K")
    ax.set_title(f"Payoff – Option Digitale {style.capitalize()} {option_type.upper()}")
    ax.set_xlabel("Prix du sous-jacent (S)")
    ax.set_ylabel("Payoff")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
