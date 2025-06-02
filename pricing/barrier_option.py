# pricing/barrier_option.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pricing.models.barrier_black_scholes import barrier_price


def price_barrier_option(
    S, K, H, T, r, sigma,
    option_type="call",
    barrier_type="up-and-out",
    model
):
    """
    Price a European barrier option using the specified model.

    Parameters:
        S : float
            Spot price
        K : float
            Strike price
        H : float
            Barrier level
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        barrier_type : str
            'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        model : str
            Pricing model, currently only 'black_scholes' supported

    Returns:
        float: barrier option price
    """
    if model == "Black-Scholes":
        return barrier_price(S, K, H, T, r, sigma, option_type.lower(), barrier_type.lower())
    elif model == "Monte-Carlo":
        return monte_carlo_barrier(
            S0=S, K=K, H=H, T=T, r=r, sigma=sigma,
            option_type=option_type.lower(), barrier_type=barrier_type.lower(),
            n_simulations=n_simulations, n_steps=n_steps
        )
    else:
        raise NotImplementedError(f"Model '{model}' not implemented.")


def plot_barrier_payoff(K, H, option_type="call", barrier_type="up-and-out", S_min=0, S_max=200, num=500):
    """
    Affiche dans Streamlit le payoff à maturité d'une option barrière européenne.
    """
    S_range = np.linspace(S_min, S_max, num)
    payoff = np.zeros_like(S_range)

    for i, S in enumerate(S_range):
        knocked_out = False
        if barrier_type == "up-and-out" and S >= H:
            knocked_out = True
        elif barrier_type == "down-and-out" and S <= H:
            knocked_out = True
        elif barrier_type == "up-and-in" and S < H:
            knocked_out = True
        elif barrier_type == "down-and-in" and S > H:
            knocked_out = True

        if not knocked_out:
            if option_type == "call":
                payoff[i] = max(S - K, 0)
            elif option_type == "put":
                payoff[i] = max(K - S, 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(S_range, payoff, label="Payoff à maturité")
    ax.axvline(x=K, linestyle="--", color="gray", label="Strike K")
    ax.axvline(x=H, linestyle="--", color="red", label="Barrière H")
    ax.set_title(f"Payoff – {option_type.upper()} {barrier_type.replace('-', ' ').title()}")
    ax.set_xlabel("Spot Price at Maturity (S)")
    ax.set_ylabel("Payoff")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
