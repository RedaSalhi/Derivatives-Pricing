from pricing.models.digital_black_scholes import (
    digital_cash_or_nothing,
    digital_asset_or_nothing
)

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
