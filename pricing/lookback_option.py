# pricing/lookback_option.py

from pricing.models.lookback_monte_carlo import monte_carlo_lookback_option
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def price_lookback_option(
    S0, K=None, r=0.05, sigma=0.2, T=1.0,
    model="monte_carlo", option_type="call", floating_strike=True,
    n_paths=100000, n_steps=252
):
    """
    General interface to price lookback options.
    
    Parameters:
    - S0: Initial asset price
    - K: Strike price (only used for fixed-strike options)
    - r: Risk-free rate
    - sigma: Volatility
    - T: Time to maturity
    - model: 'monte_carlo'
    - option_type: 'call' or 'put'
    - floating_strike: True for floating strike, False for fixed
    - n_paths, n_steps: Monte Carlo parameters

    Returns:
    - price, stderr (for Monte Carlo) or price (for binomial)
    """
    if model == "monte_carlo":
        return monte_carlo_lookback_option(
            S0=S0, r=r, sigma=sigma, T=T,
            n_paths=n_paths, n_steps=n_steps,
            option_type=option_type,
            floating_strike=floating_strike
        )

    else:
        raise ValueError(f"Unknown model: {model}")

def price_lookback_option(S0, K=None, r=0.05, sigma=0.2, T=1.0,
                          model="monte_carlo", option_type="call", floating_strike=True,
                          n_paths=100000, n_steps=252, N=100):
    if model == "monte_carlo":
        return monte_carlo_lookback_option(
            S0=S0, r=r, sigma=sigma, T=T,
            n_paths=n_paths, n_steps=n_steps,
            option_type=option_type,
            floating_strike=floating_strike
        )
    else:
        raise ValueError(f"Unknown model: {model}")

def plot_payoff(S0, option_type="call", K=None, floating_strike=True):
    S_range = np.linspace(0.5 * S0, 1.5 * S0, 200)
    if floating_strike:
        if option_type == "call":
            payoff = S_range - np.minimum.accumulate(S_range)
        else:
            payoff = np.maximum.accumulate(S_range) - S_range
    else:
        if K is None:
            raise ValueError("Strike K must be provided.")
        payoff = np.maximum(S_range - K, 0) if option_type == "call" else np.maximum(K - S_range, 0)

    fig, ax = plt.subplots()
    ax.plot(S_range, payoff)
    ax.set_title("Payoff Function")
    ax.set_xlabel("Underlying Price")
    ax.set_ylabel("Payoff")
    ax.grid(True)
    return fig

def plot_paths(S0, r, sigma, T, n_paths, n_steps=252):
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion * z)

    fig, ax = plt.subplots()
    for i in range(n_paths):
        ax.plot(np.linspace(0, T, n_steps + 1), paths[i], lw=0.7)
    ax.set_title("Simulated Asset Paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    return fig

def plot_price_distribution(S0, r, sigma, T, option_type="call", floating_strike=True, n_paths=10000, n_steps=252):
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S0
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        prices[:, t] = prices[:, t - 1] * np.exp(drift + diffusion * z)

    S_T = prices[:, -1]
    S_max = np.max(prices, axis=1)
    S_min = np.min(prices, axis=1)

    if floating_strike:
        payoffs = S_T - S_min if option_type == "call" else S_max - S_T
    else:
        payoffs = np.maximum(S_max - S0, 0) if option_type == "call" else np.maximum(S0 - S_min, 0)

    discounted = np.exp(-r * T) * payoffs

    fig, ax = plt.subplots()
    ax.hist(discounted, bins=50, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Discounted Payoffs")
    ax.set_xlabel("Payoff")
    ax.set_ylabel("Frequency")
    return fig

