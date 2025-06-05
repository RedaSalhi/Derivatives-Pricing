from pricing.models.swaps.dcf import (
    price_interest_rate_swap_dcf,
    price_currency_swap_dcf,
    price_equity_swap_dcf
)
from pricing.models.swaps.lmm import price_interest_rate_swap_lmm
from pricing.models.swaps.forward_replication import price_equity_swap_forward_replication
import numpy as np
import matplotlib.pyplot as plt

def price_swap(swap_type, model, **kwargs):
    """
    General interface for swap pricing.
    
    Args:
        swap_type: 'irs', 'currency', or 'equity'
        model: 'dcf', 'lmm', or 'replication'
        kwargs: dynamic parameters passed to models

    Returns:
        NPV of the swap at time t=0
    """
    swap_type = swap_type.lower()
    model = model.lower()

    if swap_type == "irs":
        if model == "dcf":
            return price_interest_rate_swap_dcf(**kwargs)
        elif model == "lmm":
            return price_interest_rate_swap_lmm(**kwargs)
        else:
            raise ValueError(f"Unsupported model '{model}' for IRS.")

    elif swap_type == "currency":
        if model == "dcf":
            return price_currency_swap_dcf(**kwargs)
        else:
            raise ValueError(f"Unsupported model '{model}' for currency swaps.")

    elif swap_type == "equity":
        if model == "dcf":
            return price_equity_swap_dcf(**kwargs)
        elif model == "replication":
            return price_equity_swap_forward_replication(**kwargs)
        else:
            raise ValueError(f"Unsupported model '{model}' for equity swaps.")

    else:
        raise ValueError(f"Unknown swap type '{swap_type}'.")



def plot_swap_price_vs_param(
    swap_type,
    model,
    param_name,
    param_range,
    fixed_params,
    n_points=50
):
    """
    Plots the swap price as a function of one varying parameter.

    Args:
        swap_type: 'irs', 'currency', or 'equity'
        model: 'dcf', 'lmm', or 'replication'
        param_name: name of parameter to vary (e.g., 'fixed_rate', 'notional', etc.)
        param_range: tuple (min_val, max_val)
        fixed_params: dictionary of fixed values for other params
        n_points: number of points in plot

    Returns:
        matplotlib.figure.Figure
    """
    values = np.linspace(param_range[0], param_range[1], n_points)
    prices = []

    for val in values:
        input_params = fixed_params.copy()
        input_params[param_name] = val

        try:
            price = price_swap(swap_type=swap_type, model=model, **input_params)
        except Exception as e:
            price = np.nan
        prices.append(price)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(values, prices, label=f"Swap Price vs {param_name}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Swap Price")
    ax.set_title(f"{model.upper()} {swap_type.capitalize()} Swap")
    ax.grid(True)
    ax.legend()
    return fig

