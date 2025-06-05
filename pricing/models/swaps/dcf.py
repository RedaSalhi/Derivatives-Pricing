import numpy as np

def price_interest_rate_swap_dcf(notional, fixed_rate, floating_rates, payment_times, discount_curve):
    year_fractions = np.diff([0] + list(payment_times))
    pv_fixed = notional * sum(
        fixed_rate * alpha * discount_curve(t)
        for alpha, t in zip(year_fractions, payment_times)
    )
    pv_float = notional * sum(
        fwd * alpha * discount_curve(t)
        for fwd, alpha, t in zip(floating_rates, year_fractions, payment_times)
    )
    return pv_float - pv_fixed


def price_currency_swap_dcf(notional_domestic, rate_domestic, rate_foreign,
                            payment_times, discount_domestic, discount_foreign, fx_forward_curve):
    year_fractions = np.diff([0] + list(payment_times))
    pv_dom = sum(
        notional_domestic * r_d * alpha * discount_domestic(t)
        for r_d, alpha, t in zip(rate_domestic, year_fractions, payment_times)
    )
    notional_foreign = notional_domestic / fx_forward_curve(0)  # approx
    pv_for = sum(
        notional_foreign * r_f * alpha * discount_foreign(t) * fx_forward_curve(t)
        for r_f, alpha, t in zip(rate_foreign, year_fractions, payment_times)
    )
    return pv_for - pv_dom


def price_equity_swap_dcf(notional, equity_start, equity_end, fixed_rate,
                          payment_times, discount_curve):
    total_return = (equity_end - equity_start) / equity_start
    pv_equity = notional * total_return * discount_curve(payment_times[-1])
    year_fractions = np.diff([0] + list(payment_times))
    pv_fixed = sum(
        notional * fixed_rate * alpha * discount_curve(t)
        for alpha, t in zip(year_fractions, payment_times)
    )
    return pv_equity - pv_fixed
