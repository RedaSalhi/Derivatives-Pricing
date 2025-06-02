from pricing.models.digital_black_scholes import (
    digital_cash_or_nothing,
    digital_asset_or_nothing
)
import numpy as np
import matplotlib.pyplot as plt

def price_digital_option(model="black_scholes", option_type="call", style="cash",
                         S=100, K=100, T=1, r=0.05, sigma=0.2, Q=1.0):
    """
    Dispatcher to price digital options under specified model.

    Parameters:
        model : pricing model to use (default 'black_scholes')
        option_type : 'call' or 'put'
        style : 'cash' or 'asset'
        S, K, T, r, sigma, Q : standard Black-Scholes parameters

    Returns:
        float: option price
    """
    if model == "black_scholes":
        if style == "cash":
            return digital_cash_or_nothing(option_type, S, K, T, r, sigma, Q)
        elif style == "asset":
            return digital_asset_or_nothing(option_type, S, K, T, r, sigma)
        else:
            raise ValueError("style must be 'cash' or 'asset'")
    else:
        raise NotImplementedError(f"Model {model} not implemented")



def plot_digital_payoff(K, option_type="call", style="cash", Q=1.0, S_min=0, S_max=200, num=500):
    """
    Trace le payoff à maturité d'une option digitale (cash ou asset).

    Paramètres :
        K : float
            Strike
        option_type : str
            'call' ou 'put'
        style : str
            'cash' ou 'asset'
        Q : float
            Montant versé si l'option paie (pour cash)
        S_min, S_max : float
            Borne inférieure et supérieure pour le prix du sous-jacent
        num : int
            Nombre de points de discrétisation

    Affiche :
        Un graphique matplotlib du payoff à maturité
    """
    S_range = np.linspace(S_min, S_max, num)
    payoff = np.zeros_like(S_range)

    for i, S in enumerate(S_range):
        if option_type == "call":
            if S > K:
                payoff[i] = Q if style == "cash" else S
        elif option_type == "put":
            if S < K:
                payoff[i] = Q if style == "cash" else S

    plt.figure(figsize=(8, 4.5))
    plt.plot(S_range, payoff, label="Payoff à maturité")
    plt.axvline(x=K, linestyle="--", color="gray", label="Strike K")
    plt.title(f"Payoff - Option Digitale {style.capitalize()} {option_type.upper()}")
    plt.xlabel("Prix du sous-jacent (S)")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
