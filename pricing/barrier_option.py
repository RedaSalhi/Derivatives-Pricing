# pricing/barrier_option.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pricing.models.barrier_monte_carlo import monte_carlo_barrier


def price_barrier_option(S, K, H, T, r, sigma, option_type, barrier_type, 
                        model='monte_carlo', n_simulations=10000, 
                        n_steps=252, rebate=0.0, payout_style='cash'):
                            
    # Input validation
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    if barrier_type not in ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']:
        raise ValueError("Invalid barrier_type")
    if payout_style not in ['cash', 'asset']:
        raise ValueError("payout_style must be 'cash' or 'asset'")
    
    dt = T / n_steps
    payoffs = []
    
    for _ in range(n_simulations):
        # Generate price path
        path = [S]
        crossed_barrier = False
        
        for _ in range(n_steps):
            z = np.random.normal()
            St = path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            path.append(St)
            
            # Check barrier crossing
            if not crossed_barrier:
                if ('up' in barrier_type and St >= H) or ('down' in barrier_type and St <= H):
                    crossed_barrier = True
        
        ST = path[-1]  # Final price
        
        # Determine payoff based on barrier type and crossing
        if 'out' in barrier_type:
            # Knock-out option
            if crossed_barrier:
                payoff = rebate
            else:
                # Option survived - calculate vanilla payoff
                if option_type == 'call':
                    if payout_style == 'asset':
                        payoff = ST if ST > K else 0
                    else:
                        payoff = max(ST - K, 0)
                else:  # put
                    if payout_style == 'asset':
                        payoff = ST if ST < K else 0
                    else:
                        payoff = max(K - ST, 0)
        else:
            # Knock-in option
            if crossed_barrier:
                # Option activated - calculate vanilla payoff
                if option_type == 'call':
                    if payout_style == 'asset':
                        payoff = ST if ST > K else 0
                    else:
                        payoff = max(ST - K, 0)
                else:  # put
                    if payout_style == 'asset':
                        payoff = ST if ST < K else 0
                    else:
                        payoff = max(K - ST, 0)
            else:
                payoff = rebate  # Option never activated
        
        payoffs.append(payoff)
    
    # Calculate present value
    discount_factor = np.exp(-r * T)
    option_price = discount_factor * np.mean(payoffs)
    standard_error = discount_factor * np.std(payoffs) / np.sqrt(n_simulations)
    
    return option_price, standard_error


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


def plot_sample_paths_barrier(S_paths, K, H, option_type, barrier_type):
    """
    Plot sample Monte Carlo paths and mark the barrier level.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in S_paths:  # plot first 20 paths
        ax.plot(path, alpha=0.7)
    
    ax.axhline(y=H, color='red', linestyle='--', label=f"Barrier H = {H}")
    ax.axhline(y=K, color='gray', linestyle='--', label=f"Strike K = {K}")
    ax.set_title(f"Sample Price Paths – {option_type.upper()} {barrier_type.replace('-', ' ').title()}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Simulated Spot Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


