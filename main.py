import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf

# main.py

import streamlit as st
from pricing.vanilla_option import price_vanilla_option
from pricing.european_options import plot_option_pnl_curve

# Optional: load styling
def apply_css():
    try:
        with open("plan.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

apply_css()

# ----- Sidebar inputs -----
st.sidebar.title("Option Parameters")

option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
exercise_style = st.sidebar.selectbox("Exercise Style", ["european", "american"])
model = st.sidebar.selectbox("Pricing Model", ["black-scholes", "binomial", "monte-carlo"])

S = st.sidebar.number_input("Spot Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
q = st.sidebar.number_input("Dividend Yield (q)", value=0.0)

# Model-specific inputs
kwargs = {
    "S": S,
    "K": K,
    "T": T,
    "r": r,
    "sigma": sigma,
    "q": q
}

if model == "binomial":
    kwargs["N"] = st.sidebar.slider("Time Steps (N)", min_value=10, max_value=500, value=200)
elif model == "monte-carlo":
    if exercise_style == "european":
        kwargs["n_simulations"] = st.sidebar.number_input("Simulations", value=100000)
    else:
        kwargs["n_simulations"] = st.sidebar.number_input("Simulations", value=100000)
        kwargs["n_steps"] = st.sidebar.slider("Time Steps", min_value=10, max_value=200, value=50)
        kwargs["poly_degree"] = st.sidebar.slider("Polynomial Degree", 1, 5, value=2)

# ----- Pricing Execution -----
st.title("Vanilla Option Pricing App")

try:
    price = price_vanilla_option(
        option_type=option_type,
        exercise_style=exercise_style,
        model=model,
        **kwargs
    )

    st.success(f"📈 Option Price: {price:.4f} €")

    # Plot P&L or Payoff
    st.subheader("Payoff / Return Diagram")
    plot_option_pnl_curve(
        option_type=option_type,
        S=S,
        K=K,
        price=price,
        r=r,
        T=T,
        return_pct=True,
        show_breakeven=True,
        title=f"{exercise_style.capitalize()} {option_type.capitalize()} Option - {model.replace('-', ' ').title()}"
    )

except ValueError as e:
    st.error(f"❌ {e}")
