# /pricing/utils/greeks_vasicek.py

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from pricing.models.interest_rates.monte_carlo_vasicek.py import vasicek_bond_option_price_mc
from pricing.models.interest_rates.analytical_vasicek.py import vasicek_bond_option_price, vasicek_zero_coupon_price

# Compute numerical Greeks
def compute_greek_vs_spot(greek: str, t, T1, T2, K, a, lam, sigma, face, option_type='call', n_paths=10000, r_min=0.01, r_max=0.2, n=100, model="Analytical"):
    r_vals = np.linspace(r_min, r_max, n)
    greek_vals = []
    bond_vals = []
    h = 1e-4

    for r in r_vals:
        bond_price = vasicek_zero_coupon_price(r, t, T1, a, lam, sigma, face)
        bond_vals.append(bond_price)
        if model == "Analytical":
            if greek == 'price':
                g = vasicek_bond_option_price(r, t, T1, T2, K, a, lam, sigma, face, option_type)
            elif greek == 'delta':
                up = vasicek_bond_option_price(r + h, t, T1, T2, K, a, lam, sigma, face, option_type)
                down = vasicek_bond_option_price(r - h, t, T1, T2, K, a, lam, sigma, face, option_type)
                g = (up - down) / (2 * h)
            elif greek == 'vega':
                up = vasicek_bond_option_price(r, t, T1, T2, K, a, lam, sigma+h, face, option_type)
                down = vasicek_bond_option_price(r, t, T1, T2, K, a, lam, sigma-h, face, option_type)
                g = (up - down) / (2 * h)
            elif greek == 'rho':
                up = vasicek_bond_option_price(r, t, T1, T2, K, a, lam+h, sigma, face, option_type)
                down = vasicek_bond_option_price(r, t, T1, T2, K, a, lam-h, sigma, face, option_type)
                g = (up - down) / (2 * h)
            else:
                g = np.nan
        elif model == "Monte Carlo":
            if greek == 'price':
                g, _ = vasicek_bond_option_price_mc(r, a, lam, sigma, T1, T2, K, dt, n_paths, face, option_type)
            elif greek == 'delta':
                up, _ = vasicek_bond_option_price_mc(r + h, a, lam, sigma, T1, T2, K, dt, n_paths, face, option_type)
                down, _ = vasicek_bond_option_price_mc(r - h, a, lam, sigma, T1, T2, K, dt, n_paths, face, option_type)
                g = (up - down) / (2 * h)
            elif greek == 'vega':
                up, _ = vasicek_bond_option_price_mc(r_t, a, lam, sigma + h, T1, T2, K, dt, n_paths, face, option_type)
                down, _ = vasicek_bond_option_price_mc(r_t, a, lam, sigma - h, T1, T2, K, dt, n_paths, face, option_type)
                g = (up - down) / (2 * h)
            elif greek == 'rho':
                up, _ = vasicek_bond_option_price_mc(r_t, a, lam + h, sigma, T1, T2, K, dt, n_paths, face, option_type)
                down, _ = vasicek_bond_option_price_mc(r_t, a, lam - h, sigma, T1, T2, K, dt, n_paths, face, option_type)
                g = (up - down) / (2 * h)
            else:
                g = np.nan
        else:
            raise ValueError("Invalid model specified. Use 'Analytical' or 'Monte Carlo'.")

        greek_vals.append(g)


    plt.figure(figsize=(8, 5))
    plt.plot(bond_vals, greek_vals)
    plt.title(f"{greek.capitalize()} vs Bond Price")
    plt.xlabel("Zero-Coupon Bond Price P(t,T1)")
    plt.ylabel(greek.capitalize())
    plt.grid(True)
    plt.tight_layout()
    plt.show()
