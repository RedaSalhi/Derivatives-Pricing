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
        else:
            raise ValueError(f"Unsupported model '{model}' for equity swaps.")

    else:
        raise ValueError(f"Unknown swap type '{swap_type}'.")

