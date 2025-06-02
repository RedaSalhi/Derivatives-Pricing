# main.py

import sys
import os

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import numpy as np

from pricing.vanilla_options import price_vanilla_option
from pricing.forward import (
    price_forward_contract,
    plot_forward_mark_to_market,
    plot_forward_payout_and_value
)
from pricing.option_strategies import (
    price_option_strategy,
    compute_strategy_payoff,
    get_predefined_strategy
)


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="📈 Derivatives Pricing App", layout="centered")
st.title("📊 Derivatives Pricing App")
st.caption("Built with ❤️ for students, quants, and finance enthusiasts")


# -----------------------------
# Tabs Layout
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🧮 Vanilla Options", "📉 Forward Contracts", "🧩 Option Strategies"])


# -----------------------------
# Tab 1 – Vanilla Options
# -----------------------------
with tab1:
    st.header("Vanilla Option Pricing")

    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ["call", "put"])
        exercise_style = st.selectbox("Exercise Style", ["european", "american"])
        model = st.selectbox("Pricing Model", ["black-scholes", "binomial", "monte-carlo"])

    with col2:
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
        T = st.number_input("Time to Maturity (T, in years)", value=1.0)
        sigma = st.number_input("Volatility (σ)", value=0.2)
        r = st.number_input("Risk-Free Rate (r)", value=0.05)
        q = st.number_input("Dividend Yield (q)", value=0.0)

    if model == "binomial":
        N = st.slider("Binomial Tree Steps (N)", min_value=10, max_value=10000, value=100)
    elif model == "monte-carlo":
        n_sim = st.slider("Monte Carlo Simulations", min_value=1_000, max_value=100_000, step=5_000, value=10_000)

    if st.button("💰 Compute Option Price"):
        kwargs = {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "q": q
        }

        if model == "binomial":
            kwargs["N"] = N
        elif model == "monte-carlo":
            kwargs["n_simulations"] = n_sim

        try:
            price = price_vanilla_option(option_type, exercise_style, model, **kwargs)
            st.success(f"The {exercise_style} {option_type} option is worth: **{price:.4f}**")
        except Exception as e:
            st.error(f"Error: {e}")


# -----------------------------
# Tab 2 – Forward Contracts
# -----------------------------
with tab2:
    st.header("Forward Contract Pricing")

    col1, col2 = st.columns(2)
    with col1:
        S_fwd = st.number_input("Spot Price", value=100.0, key="fwd_spot")
        K_fwd = st.number_input("Strike Price", value=100.0, key="fwd_strike")
        T_fwd = st.number_input("Time to Maturity (T)", value=1.0, key="fwd_T")
        r_fwd = st.number_input("Risk-Free Rate (r)", value=0.05, key="fwd_r")
    with col2:
        storage_cost = st.number_input("Storage Cost (c)", value=0.0, key="fwd_storage")
        dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, key="fwd_q")
        position = st.radio("Position", ["long", "short"])

    if st.button("📈 Price Forward Contract"):
        F = price_forward_contract(
            spot_price=S_fwd,
            interest_rate=r_fwd,
            time_to_maturity=T_fwd,
            storage_cost=storage_cost,
            dividend_yield=dividend_yield
        )
        st.success(f"Theoretical Forward Price: **{F:.4f}**")

        st.subheader("📉 Forward Payout at Maturity")
        plot_forward_payout_and_value(K_fwd, position)

        st.subheader("🔄 Mark-to-Market Value (Before Maturity)")
        plot_forward_mark_to_market(
            strike_price=K_fwd,
            time_to_maturity=T_fwd,
            interest_rate=r_fwd,
            storage_cost=storage_cost,
            dividend_yield=dividend_yield,
            position=position
        )


# -----------------------------
# Tab 3 – Option Strategies
# -----------------------------
with tab3:
    st.header("Multi-leg Option Strategy")

    strategy = st.selectbox(
        "Choose a Predefined Strategy",
        ["straddle", "bull_call_spread", "bear_put_spread", "butterfly", "iron_condor"],
        key="strat_type"
    )
    model_strat = st.selectbox("Pricing Model", ["black-scholes", "binomial", "monte-carlo"], key="strat_model")
    style_strat = st.selectbox("Exercise Style", ["european", "american"], key="strat_style")

    S_strat = st.number_input("Spot Price (S)", value=100.0, key="strat_S")
    T_strat = st.number_input("Time to Maturity (T)", value=1.0, key="strat_T")
    sigma_strat = st.number_input("Volatility (σ)", value=0.2, key="strat_sigma")
    r_strat = st.number_input("Risk-Free Rate (r)", value=0.05, key="strat_r")
    q_strat = st.number_input("Dividend Yield (q)", value=0.0, key="strat_q")

    st.subheader("Enter Strike Prices")
    strike1 = st.number_input("Strike 1", value=95.0, key="strat_k1")

    strike2 = None
    strike3 = None
    strike4 = None

    if strategy in ["bull_call_spread", "bear_put_spread", "butterfly", "iron_condor"]:
        strike2 = st.number_input("Strike 2", value=100.0, key="strat_k2")
    if strategy in ["butterfly", "iron_condor"]:
        strike3 = st.number_input("Strike 3", value=105.0, key="strat_k3")
    if strategy == "iron_condor":
        strike4 = st.number_input("Strike 4", value=110.0, key="strat_k4")

    if st.button("📊 Price Strategy & Show Payoff"):
        # Pass all strikes (only used if strategy requires them)
        legs = get_predefined_strategy(strategy, strike1, strike2, strike3, strike4=strike4)
        if isinstance(legs, str):
            st.error(legs)
        else:
            kwargs = {
                "S": S_strat,
                "T": T_strat,
                "sigma": sigma_strat,
                "r": r_strat,
                "q": q_strat
            }
            try:
                result = price_option_strategy(legs, style_strat, model_strat, **kwargs)
                st.success(f"Total Strategy Price: **{result['strategy_price']:.4f}**")

                spot_range = np.linspace(0.5 * S_strat, 1.5 * S_strat, 500)
                payoff = compute_strategy_payoff(legs, spot_range)

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(spot_range, payoff, label="Strategy Payoff", color="green")
                ax.axhline(0, color="black", linestyle="--")
                ax.set_xlabel("Spot Price at Maturity (S)")
                ax.set_ylabel("Net Payoff")
                ax.set_title(f"Payoff Diagram: {strategy.title().replace('_', ' ')}")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")
