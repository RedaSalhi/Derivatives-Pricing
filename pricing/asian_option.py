#princing/asian_option.py

from pricing.models.asian_monte_carlo import simulate_asian_paths
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def price_asian_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int = 10000,
    method: str = "monte_carlo",       # or "pde"
    option_type: str = "call",         # "call" or "put"
    asian_type: str = "average_price"  # "average_price" or "average_strike"
) -> float:
    """
    Master pricing function for Asian options.

    Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free rate
        sigma (float): Volatility
        n_steps (int): Number of time steps (for both MC and PDE)
        n_paths (int): Number of Monte Carlo paths
        method (str): "monte_carlo" or "pde"
        option_type (str): "call" or "put"
        asian_type (str): "average_price" or "average_strike"

    Returns:
        float: Estimated option price
    """
    assert method in {"monte_carlo"}, "Invalid method"
    assert option_type in {"call", "put"}, "Invalid option type"
    assert asian_type in {"average_price", "average_strike"}, "Invalid Asian option type"

    if method == "monte_carlo":
        paths = simulate_asian_paths(S0, T, r, sigma, n_steps, n_paths, option_type, asian_type)
        if asian_type == "average_price":
            averages = np.mean(paths, axis=1)
            payoffs = np.maximum(averages - K, 0) if option_type == "call" else np.maximum(K - averages, 0)
        else:  # average_strike
            averages = np.mean(paths, axis=1)
            S_T = paths[:, -1]
            payoffs = np.maximum(S_T - averages, 0) if option_type == "call" else np.maximum(averages - S_T, 0)
        return np.exp(-r * T) * np.mean(payoffs)


def plot_asian_option_payoff(K: float, option_type: str = "call", asian_type: str = "average_price"):
    """
    Display the payoff function of an Asian option using Streamlit.

    Parameters:
        K (float): Strike price
        option_type (str): "call" or "put"
        asian_type (str): "average_price" or "average_strike"
    """
    x = np.linspace(0.5 * K, 1.5 * K, 200)

    if asian_type == "average_price":
        x_label = "Average Price"
        payoff = np.maximum(x - K, 0) if option_type == "call" else np.maximum(K - x, 0)

    elif asian_type == "average_strike":
        x_label = "Final Price (S_T)"
        payoff = np.maximum(x - K, 0) if option_type == "call" else np.maximum(K - x, 0)

    else:
        st.error("Invalid asian_type")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, payoff, label="Payoff", linewidth=2)
    ax.set_title(f"{asian_type.replace('_', ' ').title()} Asian {option_type.title()} Payoff")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Payoff")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


def plot_monte_carlo_paths(paths: np.ndarray):
    """
    Display simulated Monte Carlo paths using Streamlit.

    Parameters:
        paths (np.ndarray): Simulated asset price paths (n_paths, n_steps)
        n_paths_to_plot (int): Number of paths to plot
    """
    n_paths, n_steps = paths.shape
    time_grid = np.linspace(0, 1, n_steps)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_paths):
        ax.plot(time_grid, paths[i], lw=0.8, alpha=0.7)

    ax.set_title(f"Monte Carlo Simulated Paths (n=n_paths))")
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Asset Price")
    ax.grid(True)
    st.pyplot(fig)

