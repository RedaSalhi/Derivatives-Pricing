from pricing.models.interest_rates.analytical_vasicek import vasicek_zero_coupon_price 

# ------------------------------------------------------------------------------
# 5. Price a Coupon Bond Using the Simulated Short Rate Path
# ------------------------------------------------------------------------------
def price_coupon_bond(rates, a, lam, sigma, maturity=5, coupon=0.05, face_value=1.0, dt=0.5):
    cashflow_dates = np.arange(dt, maturity + dt, dt)
    price = 0
    for T in cashflow_dates:
        coupon_payment = coupon * face_value * dt
        if np.isclose(T, maturity):
            coupon_payment += face_value
        P = vasicek_zero_coupon_price(rates[0], 0, T, a, lam, sigma, face_value)
        price += coupon_payment * P
    return price
