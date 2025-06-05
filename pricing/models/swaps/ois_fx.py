def build_flat_discount_curve(rate):
    return lambda T: np.exp(-rate * T)

def build_flat_fx_forward_curve(spot, r_dom, r_for):
    return lambda T: spot * np.exp((r_dom - r_for) * T)
