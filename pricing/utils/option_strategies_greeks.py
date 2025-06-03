import numpy as np
import matplotlib.pyplot as plt
from pricing.utils.greeks_vanilla.greeks_interface import compute_greek

def plot_strategy_greek_vs_spot(
    greek_name,
    legs,
    model,
    S0,
    T,
    r,
    sigma,
    q=0.0,
    S_range=None,
    n_points=200
):
    """
    Plot a given Greek vs Spot Price for a multi-leg strategy.

    Parameters:
        greek_name : str, one of ['delta', 'gamma', 'vega', 'theta', 'rho']
        legs : list of dicts with 'type', 'strike', and 'qty'
        model : str, e.g., 'black-scholes', 'binomial', 'monte-carlo'
        S0 : float, current spot
        T : float, maturity
        r : float, risk-free rate
        sigma : float, volatility
        q : float, dividend yield
        S_range : np.array, optional spot price range
        n_points : int, number of spot steps

    Returns:
        matplotlib.figure.Figure
    """
    plt.style.use("bmh")

    K_list = [leg["strike"] for leg in legs]
    if S_range is None:
        S_range = np.linspace(min(K_list + [S0]) * 0.5, max(K_list + [S0]) * 1.5, n_points)

    y_vals = []

    for S in S_range:
        total_greek = 0.0
        for leg in legs:
            g_val = compute_greek(
                greek_name=greek_name,
                model=model,
                option_type=leg["type"],
                S_values=[S],
                K=leg["strike"],
                T=T,
                r=r,
                sigma=sigma,
                q=q
            )[0]
            total_greek += leg["qty"] * g_val
        y_vals.append(total_greek)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_range, y_vals, label=f"Strategy {greek_name.capitalize()}", color="darkred", linewidth=2)

    for K in K_list:
        ax.axvline(x=K, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=S0, color="blue", linestyle="--", label="Spot")

    ax.set_title(f"{greek_name.capitalize()} vs Spot Price", fontsize=20, weight="bold", color="black")
    ax.set_xlabel("Spot Price", fontsize=14, color="black")
    ax.set_ylabel(greek_name.capitalize(), fontsize=14, color="black")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")

    return fig

