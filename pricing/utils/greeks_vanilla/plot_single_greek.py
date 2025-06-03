import numpy as np
import matplotlib.pyplot as plt
from pricing.utils.greeks_vanilla.greeks_interface import compute_greek

def plot_single_greek_vs_spot(
    greek_name,
    model,
    option_type,
    S0,
    K,
    T,
    r,
    sigma,
    q=0.0,
    S_range=None,
    n_points=200,
):
    plt.style.use("seaborn-whitegrid")  # Try "ggplot", "bmh", or "default" too

    if S_range is None:
        S_range = np.linspace(0.5 * K, 1.5 * K, n_points)

    values = compute_greek(
        greek_name, model, option_type, S_range, K, T, r, sigma, q
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S_range, values, label=greek_name.capitalize(), color='darkblue', linewidth=2)

    # Vertical markers
    ax.axvline(x=K, color="green", linestyle="--", label="Strike")
    ax.axvline(x=S0, color="red", linestyle="--", label="Spot Initial")

    # Labels and styling
    ax.set_title(f"{greek_name.capitalize()} vs Spot Price", fontsize=20, weight='bold', color='black')
    ax.set_xlabel("Spot Price", fontsize=14, color='black')
    ax.set_ylabel(greek_name.capitalize(), fontsize=14, color='black')
    ax.legend()

    # Optional: reset face colors to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    return fig

