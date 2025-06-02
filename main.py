# main.py

import streamlit as st
from pricing.vanilla_option import price_vanilla_option
from pricing.european_options import plot_option_pnl_curve

def apply_css():
    try:
        with open("plan.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def main():
    apply_css()

    st.title("📊 Vanilla Option Pricing App")

    st.sidebar.header("🧮 Parameters")

    # Basic inputs
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    exercise_style = st.sidebar.selectbox("Exercise Style", ["european", "american"])
    model = st.sidebar.selectbox("Pricing Model", ["black-scholes", "binomial", "monte-carlo"])

    S = st.sidebar.number_input("Spot Price (S)", value=100.0)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0)
    T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
    r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
    q = st.sidebar.number_input("Dividend Yield (q)", value=0.0)

    kwargs = {
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "q": q
    }

    # Model-specific options
    if model == "binomial":
        kwargs["N"] = st.sidebar.slider("Binomial Steps (N)", 10, 500, 200)
    elif model == "monte-carlo":
        kwargs["n_simulations"] = st.sidebar.number_input("Simulations", value=100000)
        if exercise_style == "american":
            kwargs["n_steps"] = st.sidebar.slider("Time Steps", 10, 200, 50)
            kwargs["poly_degree"] = st.sidebar.slider("Polynomial Degree", 1, 5, 2)

    # Calculate price
    st.subheader("💡 Option Price")
    try:
        price = price_vanilla_option(
            option_type=option_type,
            exercise_style=exercise_style,
            model=model,
            **kwargs
        )
        st.success(f"The option price is: **{price:.4f} €**")

        # Plot payoff diagram
        st.subheader("📈 Payoff & Breakeven Plot")
        plot_option_pnl_curve(
            option_type=option_type,
            S=S,
            K=K,
            price=price,
            r=r,
            T=T,
            return_pct=True,
            show_breakeven=True,
            title=f"{exercise_style.capitalize()} {option_type.capitalize()} Option ({model.replace('-', ' ').title()})"
        )

    except ValueError as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
