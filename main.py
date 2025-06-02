# main.py

import streamlit as st
from pricing.vanilla_option import price_vanilla_option
from pricing.european_options import plot_option_pnl_curve
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


# Optional CSS styling
def apply_css():
    try:
        with open("plan.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("No CSS file found — using default Streamlit theme.")

# Main layout
def main():
    apply_css()

    st.set_page_config(page_title="Vanilla Option Pricer", layout="centered")
    st.title("📊 Vanilla Option Pricer")
    st.markdown("Price European and American Call/Put options using different models.")

    with st.form("pricing_form"):
        col1, col2 = st.columns(2)

        with col1:
            option_type = st.selectbox("Option Type", ["call", "put"])
            exercise_style = st.selectbox("Exercise Style", ["european", "american"])
            model = st.selectbox("Pricing Model", ["black-scholes", "binomial", "monte-carlo"])

        with col2:
            S = st.number_input("Spot Price (S)", value=100.0)
            K = st.number_input("Strike Price (K)", value=100.0)
            T = st.number_input("Time to Maturity (T, in years)", value=1.0)
            r = st.number_input("Risk-Free Rate (r)", value=0.05)
            sigma = st.number_input("Volatility (σ)", value=0.2)
            q = st.number_input("Dividend Yield (q)", value=0.0)

        # Advanced model options
        kwargs = {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "q": q
        }

        if model == "binomial":
            kwargs["N"] = st.slider("Steps (N)", 10, 500, 200)

        elif model == "monte-carlo":
            kwargs["n_simulations"] = st.number_input("Simulations", value=100000)
            if exercise_style == "american":
                kwargs["n_steps"] = st.slider("Time Steps", 10, 200, 50)
                kwargs["poly_degree"] = st.slider("Polynomial Degree", 1, 5, 2)

        submitted = st.form_submit_button("🔍 Price Option")

    if submitted:
        try:
            price = price_vanilla_option(
                option_type=option_type,
                exercise_style=exercise_style,
                model=model,
                **kwargs
            )
            st.success(f"💰 Option Price: **{price:.4f} €**")

            st.subheader("📈 Payoff Diagram")
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

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
