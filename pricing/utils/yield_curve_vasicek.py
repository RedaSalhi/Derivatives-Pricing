import numpy as np
import matplotlib.pyplot as plt
from pricing.models.interest_rates.analytical_vasicek import vasicek_zero_coupon_price

# ------------------------------------------------------------------------------
# 3. Generate Yield Curves at Different Snapshot Times
# ------------------------------------------------------------------------------
def generate_yield_curves(r_path, snapshot_times, maturities, a, lam, sigma, dt):
    yield_curves = {}
    for t_snap in snapshot_times:
        idx = int(t_snap / dt)
        r_t = r_path[idx]
        yields = []
        for m in maturities:
            T = t_snap + m
            P = vasicek_zero_coupon_price(r_t, t_snap, T, a, lam, sigma)
            y = -np.log(P) / m
            yields.append(y)
        yield_curves[t_snap] = yields
    return yield_curves

# ------------------------------------------------------------------------------
# 4. Plot Yield Curves
# ------------------------------------------------------------------------------
def plot_yield_curves(yield_curves, maturities):
    plt.figure(figsize=(10, 6))
    for t_snap, yields in yield_curves.items():
        plt.plot(maturities, yields, label=f'Time {t_snap}y')
    plt.title('Simulated Yield Curves under Vasicek Model')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (continuously compounded)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
