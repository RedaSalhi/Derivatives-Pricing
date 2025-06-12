# /pricing/utils/greeks_vasicek.py

# from pricing.utils.greeks_vasicek import compute_greek_vs_spot

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from pricing.models.interest_rates.monte_carlo_vasicek import vasicek_bond_option_price_mc
from pricing.models.interest_rates.analytical_vasicek import vasicek_bond_option_price, vasicek_zero_coupon_price
from scipy.optimize import bisect


def compute_greek_vs_spot(greek: str, t, T1, T2, K, a, lam, sigma, face, option_type='call', n_paths=10000, model="Analytical"):

    def find_r_for_price(target_price, t, T, a, lam, sigma, face, direction='low'):
        """Find the short rate r such that P(t, T) ≈ target_price"""
        def objective(r):
            return vasicek_zero_coupon_price(r, t, T, a, lam, sigma, face) - target_price

        # Use a wide interval for initial guess
        if direction == 'low':
            return bisect(objective, -10, 5)  # Low rate gives high price
        else:
            return bisect(objective, 0, 100)  # High rate gives low price
    x_min = 0
    x_max = face + face / 5

    r_min = find_r_for_price(x_max, t, T1, a, lam, sigma, face, direction='low')
    r_max = find_r_for_price(x_min + 1e-5, t, T1, a, lam, sigma, face, direction='high')

    r_vals = np.linspace(r_min, r_max, 2000)

    greek_vals = []
    bond_vals = []
    h = 1e-5

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

    # Filter values within this x-range
    filtered_greeks = [g for p, g in zip(bond_vals, greek_vals) if x_min <= p <= x_max]

    # Handle case where no values are in range
    if filtered_greeks:
        y_min = min(filtered_greeks)
        y_max = max(filtered_greeks)
    else:
        y_min, y_max = 0, 1  # fallback to a default range


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bond_vals, greek_vals)
    ax.set_title(f"{greek.capitalize()} vs Bond Price")
    ax.set_xlabel("Zero-Coupon Bond Price P(t,T1)")
    ax.set_ylabel(greek.capitalize())
    ax.grid(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()

    return fig
