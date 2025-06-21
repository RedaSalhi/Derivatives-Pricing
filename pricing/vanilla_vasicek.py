# pricing/vanilla_vasicek.py

from pricing.models.interest_rates.analytical_vasicek import (
    vasicek_zero_coupon_price, 
    price_coupon_bond as _price_coupon_bond,
    vasicek_bond_option_price
)

def price_zero_coupon(r_t, t, T, a, lam, sigma, face_value=1.0):
    """Price a zero-coupon bond using Vasicek model."""
    return vasicek_zero_coupon_price(r_t, t, T, a, lam, sigma, face_value)

def price_coupon_bond(r0, a, lam, sigma, maturity=5, coupon=0.05, face=1.0, dt=0.5, t=0):
    """Price a coupon bond using Vasicek model."""
    return _price_coupon_bond(r0, t, a, lam, sigma, maturity, face, coupon, dt)

def price_bond_option(r0, a, lam, sigma, T1, T2, K, dt, face=1.0, option_type='call', model="Analytical", n_paths=10000, t=0):
    if model == "Analytical":
        return vasicek_bond_option_price(r0, t, T1, T2, K, a, lam, sigma, face, option_type)
    elif model == "Monte Carlo":
        from pricing.models.interest_rates.monte_carlo_vasicek import vasicek_bond_option_price_mc
        price, std = vasicek_bond_option_price_mc(r0, a, lam, sigma, T1, T2, K, dt, n_paths, face, option_type)
        return price 
    else:
        raise ValueError("Model must be 'Analytical' or 'Monte Carlo'.")
