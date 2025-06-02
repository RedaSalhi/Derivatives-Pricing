from pricing.models.barrier_black_scholes import barrier_price

def price_barrier_option(
    S, K, H, T, r, sigma,
    option_type="call",
    barrier_type="up-and-out",
    model="black_scholes"
):
    """
    Price a European barrier option using the specified model.

    Parameters:
        S : float
            Spot price
        K : float
            Strike price
        H : float
            Barrier level
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        barrier_type : str
            'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        model : str
            Pricing model, currently only 'black_scholes' supported

    Returns:
        float: barrier option price
    """
    if model == "black_scholes":
        return barrier_price(S, K, H, T, r, sigma, option_type.lower(), barrier_type.lower())
    else:
        raise NotImplementedError(f"Model '{model}' not implemented.")


import matplotlib.pyplot as plt
import numpy as np

def plot_barrier_payoff(K, H, option_type="call", barrier_type="up-and-out", S_min=0, S_max=200, num=500):
    """
    Trace le payoff à maturité d'une option barrière européenne.

    Paramètres :
        K : float
            Prix d'exercice (strike)
        H : float
            Niveau de barrière
        option_type : str
            'call' ou 'put'
        barrier_type : str
            'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
        S_min : float
            Prix spot minimum pour l'axe x
        S_max : float
            Prix spot maximum pour l'axe x
        num : int
            Nombre de points de discrétisation

    Affiche :
        Un graphique matplotlib du payoff à maturité
    """
    S_range = np.linspace(S_min, S_max, num)
    payoff = np.zeros_like(S_range)

    for i, S in enumerate(S_range):
        knocked_out = False
        if barrier_type == "up-and-out" and S >= H:
            knocked_out = True
        elif barrier_type == "down-and-out" and S <= H:
            knocked_out = True
        elif barrier_type == "up-and-in" and S < H:
            knocked_out = True
        elif barrier_type == "down-and-in" and S > H:
            knocked_out = True

        if not knocked_out:
            if option_type == "call":
                payoff[i] = max(S - K, 0)
            elif option_type == "put":
                payoff[i] = max(K - S, 0)

    plt.figure(figsize=(8, 4.5))
    plt.plot(S_range, payoff, label="Payoff à maturité")
    plt.axvline(x=K, linestyle="--", color="gray", label="Strike K")
    plt.axvline(x=H, linestyle="--", color="red", label="Barrière H")
    plt.titl
