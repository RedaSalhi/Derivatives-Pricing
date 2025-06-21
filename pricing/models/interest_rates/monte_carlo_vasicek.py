# pricing/models/interest_rates/monte_carlo_vasicek.py
# from pricing.models.interest_rates.monte_carlo_vasicek import simulate_vasicek_paths, plot_vasicek_paths, plot_yield_distribution, vasicek_bond_option_price_mc

from scipy.optimize import bisect
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate_vasicek_paths(a, lam, sigma, r0, T, dt, n_paths):
    N = int(round(T / dt)) + 1
    r = np.zeros((N, n_paths))
    r[0, :] = r0
    
    # Use exact discretization for better accuracy
    for t in range(N - 1):
        Z = np.random.normal(0, 1, n_paths)
        # Exact mean and variance for Vasicek process
        mu_t = r[t, :] * np.exp(-a * dt) + lam * (1 - np.exp(-a * dt))
        sigma_t = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        r[t + 1, :] = mu_t + sigma_t * Z
    
    return np.linspace(0, T, N), r

def plot_vasicek_paths(T_vec, r_paths, lam, n_paths_to_plot=100):
    """Plot sample Vasicek paths."""
    plt.figure(figsize=(10, 5))
    # Plot only a subset of paths for clarity
    n_plot = min(n_paths_to_plot, r_paths.shape[1])
    plt.plot(T_vec, r_paths[:, :n_plot], lw=0.6, alpha=0.6)
    plt.axhline(lam, color='red', linestyle='--', label='Mean Reversion Level (λ)')
    plt.title("Vasicek Sample Paths")
    plt.xlabel("Time (Years)")
    plt.ylabel("Short Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_yield_distribution(r):
    """Plot the terminal distribution of short rates."""
    r_T = r[-1, :]
    mean_yield = np.mean(r_T)
    std_yield = np.std(r_T)
    
    x = np.linspace(r_T.min(), r_T.max(), 200)
    pdf_fit = ss.norm.pdf(x, mean_yield, std_yield)
    
    plt.figure(figsize=(10, 5))
    plt.hist(r_T, bins=50, density=True, color='lightblue', edgecolor='black', 
             label='Simulated Rates', alpha=0.7)
    plt.plot(x, pdf_fit, color='red', lw=2, label='Normal Fit')
    plt.axvline(mean_yield, color='green', linestyle='--', 
                label=f'Mean = {mean_yield:.4f}')
    plt.title("Terminal Short Rate Distribution")
    plt.xlabel("Short Rate")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def vasicek_bond_option_price_mc(r0, a, lam, sigma, T1, T2, K, dt, n_paths=10000, face=1.0, option_type='call'):
    """
    Price a bond option using Monte Carlo simulation under Vasicek model.
    
    Parameters:
    - r0: Initial short rate
    - a: Speed of mean reversion  
    - lam: Long-term mean level
    - sigma: Volatility
    - T1: Option expiry time
    - T2: Bond maturity time
    - K: Strike price
    - dt: Time step
    - n_paths: Number of Monte Carlo paths
    - face: Face value of bond
    - option_type: 'call' or 'put'
    
    Returns:
    - option_price: Estimated option price
    - option_std: Standard error of the estimate
    """
    # Validation
    if T1 <= 0:
        raise ValueError("Option expiry T1 must be positive")
    if T2 <= T1:
        raise ValueError("Bond maturity T2 must be greater than option expiry T1")
    if option_type not in ['call', 'put']:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Time indices
    N1 = int(round(T1 / dt))
    N2 = int(round(T2 / dt))
    N_total = max(N1, N2) + 1
    
    # Simulate short rate paths using exact discretization
    r_paths = np.zeros((n_paths, N_total))
    r_paths[:, 0] = r0
    
    for t in range(1, N_total):
        Z = np.random.normal(size=n_paths)
        # Exact discretization
        mu_t = r_paths[:, t-1] * np.exp(-a * dt) + lam * (1 - np.exp(-a * dt))
        sigma_t = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        r_paths[:, t] = mu_t + sigma_t * Z
    
    # Extract short rates at option expiry
    r_T1 = r_paths[:, N1]
    
    # Compute zero-coupon bond prices P(T1, T2) using analytical formula
    tau = T2 - T1  # Time to maturity of bond at option expiry
    B_T1T2 = (1 - np.exp(-a * tau)) / a
    A_T1T2 = np.exp((lam - sigma**2 / (2 * a**2)) * (B_T1T2 - tau) - 
                    (sigma**2 / (4 * a)) * B_T1T2**2)
    P_T1_T2 = A_T1T2 * np.exp(-B_T1T2 * r_T1)
    
    # Calculate option payoffs
    bond_prices_at_expiry = face * P_T1_T2
    
    if option_type == 'call':
        payoff_T1 = np.maximum(bond_prices_at_expiry - K, 0)
    elif option_type == 'put':
        payoff_T1 = np.maximum(K - bond_prices_at_expiry, 0)
    
    # Discount back to time 0 using analytical zero-coupon bond formula
    B_0T1 = (1 - np.exp(-a * T1)) / a
    A_0T1 = np.exp((lam - sigma**2 / (2 * a**2)) * (B_0T1 - T1) - 
                   (sigma**2 / (4 * a)) * B_0T1**2)
    P_0_T1 = A_0T1 * np.exp(-B_0T1 * r0)
    
    # Monte Carlo estimate
    option_price = np.mean(payoff_T1) * P_0_T1
    option_std = np.std(payoff_T1) * P_0_T1 / np.sqrt(n_paths)
    
    return option_price, option_std
