import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


def plot_vasicek_paths(r, T_vec, lam, sigma, a, dt, plot_paths=1000):
    """
    Affiche les chemins simulés du modèle de Vasicek.
    """
    std_asy = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
    
    plt.figure(figsize=(10, 5))
    plt.plot(T_vec, r[:, :plot_paths], lw=0.7)
    plt.axhline(lam + std_asy, color='black', linestyle='--', label='±1 std')
    plt.axhline(lam - std_asy, color='black', linestyle='--')
    plt.axhline(lam, color='red', label='Mean')
    plt.title("Vasicek Sample Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Interest Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
