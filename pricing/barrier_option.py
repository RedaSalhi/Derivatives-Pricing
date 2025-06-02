from pricing.models.barrier_black_scholes import barrier_price

def price_barrier_option(model="black_scholes", option_type="call",
                         barrier_type="up-and-out", S=100, K=100, H=110,
                         T=1.0, r=0.05, sigma=0.2):
    if model == "black_scholes":
        return barrier_price(S, K, H, T, r, sigma, option_type, barrier_type)
    else:
        raise NotImplementedError(f"Barrier model {model} not implemented.")
