#pricing/vanilla_vasicek.py

# from pricing.vanilla_vasicek import price_zero_coupon, price_coupon_bond, price_bond_option

from pricing.models.interest_rates.analytical_vasicek import vasicek_zero_coupon_price, price_coupon_bond


def price_zero_coupon(r_t, t, T, a, lam, sigma, face_value=1.0):
        return vasicek_zero_coupon_price(r_t, t, T, a, lam, sigma, face_value)


def price_coupon_bond(rates, a, lam, sigma, maturity=5, coupon=0.05, face=1.0, dt=0.5):
        return price_coupon_bond(rates, a, lam, sigma, maturity, coupon, face, dt)


def price_bond_option(r0, a, lam, sigma, T1, T2, K, dt, face=1.0, option_type='call', model="Analytical", n_paths=10000):
    if model == "Analytical":
        from pricing.models.interest_rates.analytical_vasicek import vasicek_bond_option_price
        return vasicek_bond_option_price(r0, 0, T1, T2, K, a, lam, sigma, face, option_type)
    elif model == "Monte Carlo":
        from pricing.models.interest_rates.monte_carlo_vasicek import vasicek_bond_option_price_mc
        price, std = vasicek_bond_option_price_mc(r0, a, lam, sigma, T1, T2, K, dt, n_paths, face, option_type)
        return price 
    else:
        raise ValueError("Model must be 'Analytical' or 'Monte Carlo'.")
