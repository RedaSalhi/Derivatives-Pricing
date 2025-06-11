import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

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
