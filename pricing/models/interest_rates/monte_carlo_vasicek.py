# pricing/models/interest_rates/monte_carlo_vasicek.py




import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.stats import norm


def simulate_vasicek_paths_mc(a, lam, sigma, r0, T=10, paths=10000, dt=1/12):
    """
    Simule les chemins de taux d'intérêt selon le modèle de Vasicek.
    Retourne les chemins simulés, le vecteur temps et la distribution terminale.
    """
    N = round(T / dt)
    T_vec = np.linspace(0, T, N)
    std_dt = np.sqrt(sigma**2 / (2 * a) * (1 - np.exp(-2 * a * dt)))
    
    r = np.zeros((N, paths))
    r[0, :] = r0

    for t in range(N - 1):
        W = np.random.normal(size=paths)
        r[t + 1, :] = lam + np.exp(-a * dt) * (r[t, :] - lam) + std_dt * W

    return r, T_vec, lam, sigma, a, dt


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


def vasicek_bond_option_price_mc(r0, a, lam, sigma, T1, T2, K, dt, n_paths=10000, face=1.0, option_type='call'):
    N1 = int(T1 / dt)
    N2 = int(T2 / dt)
    N_total = N2 + 1

    # Simulate short rate paths
    r_paths = np.zeros((n_paths, N_total))
    r_paths[:, 0] = r0
    sqrt_dt = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))

    for t in range(1, N_total):
        dr = a * (lam - r_paths[:, t-1]) * dt + sqrt_dt * np.random.normal(size=n_paths)
        r_paths[:, t] = r_paths[:, t-1] + dr

    # Compute zero-coupon prices P(T1, T2) for each path
    r_T1 = r_paths[:, N1]
    r_T2 = r_paths[:, N2]

    B_T1T2 = (1 - np.exp(-a * (T2 - T1))) / a
    A_T1T2 = np.exp((lam - sigma**2 / (2 * a**2)) * (B_T1T2 - (T2 - T1)) - (sigma**2 / (4 * a)) * B_T1T2**2)
    P_T1_T2 = A_T1T2 * np.exp(-B_T1T2 * r_T1)

    # Payoff: max(P(T1,T2) - K, 0)
    if option_type == 'call':
        payoff_T1 = np.maximum(face * P_T1_T2 - K, 0)
    elif option_type == 'put':
        payoff_T1 = np.maximum(K - face * P_T1_T2, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discount back to time 0 using P(0, T1)
    B_0T1 = (1 - np.exp(-a * T1)) / a
    A_0T1 = np.exp((lam - sigma**2 / (2 * a**2)) * (B_0T1 - T1) - (sigma**2 / (4 * a)) * B_0T1**2)
    P_0_T1 = A_0T1 * np.exp(-B_0T1 * r0)

    # Monte Carlo estimate
    option_price = np.mean(payoff_T1) * P_0_T1
    option_std = np.std(payoff_T1) * P_0_T1 / np.sqrt(n_paths)

    return option_price, option_std
