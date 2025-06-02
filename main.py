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
st.set_page_config(page_title="Derivatives Pricing App", layout="centered")
st.title("Derivatives Pricing App")
st.caption("Built for students, quants, and finance enthusiasts")


# -----------------------------
# Tabs Layout
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Vanilla Options", 
    "Forward Contracts", 
    "Option Strategies", 
    "Exotic Options" 
])



# -----------------------------
# Tab 1 – Vanilla Options
# -----------------------------
with tab1:
    st.header("Vanilla Option Pricing")

    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ["call", "put"])
        exercise_style = st.selectbox("Exercise Style", ["european", "american"])
        model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial", "Monte-Carlo"])

    with col2:
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
        T = st.number_input("Time to Maturity (T, in years)", value=1.0)
        sigma = st.number_input("Volatility (σ)", value=0.2)
        r = st.number_input("Risk-Free Rate (r)", value=0.05)
        q = st.number_input("Dividend Yield (q)", value=0.0)

    if model == "Binomial":
        N = st.slider("Binomial Tree Steps (N)", min_value=10, max_value=10000, value=100)
    elif model == "Monte-Carlo":
        n_sim = st.slider("Monte Carlo Simulations", min_value=1_000, max_value=100_000, step=5_000, value=10_000)

    if st.button("Compute Option Price"):
        kwargs = {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "sigma": sigma,
            "q": q
        }

        if model == "Binomial":
            kwargs["N"] = N
        elif model == "Monte-Carlo":
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

    if st.button("Price Forward Contract"):
        F = price_forward_contract(
            spot_price=S_fwd,
            interest_rate=r_fwd,
            time_to_maturity=T_fwd,
            storage_cost=storage_cost,
            dividend_yield=dividend_yield
        )
        st.success(f"Theoretical Forward Price: **{F:.4f}**")

        st.subheader("Forward Payout at Maturity")
        plot_forward_payout_and_value(K_fwd, position)

        st.subheader("Mark-to-Market Value (Before Maturity)")
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
    st.header("Option Strategies")

    use_manual = st.checkbox("Build Strategy Manually")

    model_strat = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial", "Monte-Carlo"], key="strat_model")
    style_strat = st.selectbox("Exercise Style", ["european", "american"], key="strat_style")

    S_strat = st.number_input("Spot Price (S)", value=100.0, key="strat_S")
    T_strat = st.number_input("Time to Maturity (T)", value=1.0, key="strat_T")
    sigma_strat = st.number_input("Volatility (σ)", value=0.2, key="strat_sigma")
    r_strat = st.number_input("Risk-Free Rate (r)", value=0.05, key="strat_r")
    q_strat = st.number_input("Dividend Yield (q)", value=0.0, key="strat_q")

    kwargs = {
        "S": S_strat,
        "T": T_strat,
        "sigma": sigma_strat,
        "r": r_strat,
        "q": q_strat
    }

    if use_manual:
        st.subheader("➕ Add Legs to Strategy")

        if "custom_legs" not in st.session_state:
            st.session_state.custom_legs = []

        with st.form("add_leg_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                opt_type = st.selectbox("Type", ["call", "put"], key="leg_type")
            with col2:
                strike = st.number_input("Strike", value=100.0, key="leg_strike")
            with col3:
                qty = st.number_input("Quantity", step=1, value=1, key="leg_qty")

            submitted = st.form_submit_button("Add Leg")
            if submitted:
                st.session_state.custom_legs.append({"type": opt_type, "strike": strike, "qty": qty})

        if st.session_state.custom_legs:
            st.markdown("### Strategy Legs")
            for i, leg in enumerate(st.session_state.custom_legs):
                st.write(f"Leg {i+1}: {leg['qty']} × {leg['type'].upper()} @ Strike {leg['strike']}")

            if st.button("Price Custom Strategy"):
                try:
                    result = price_option_strategy(st.session_state.custom_legs, style_strat, model_strat, **kwargs)
                    st.success(f"Total Strategy Price: **{result['strategy_price']:.4f}**")

                    spot_range = np.linspace(0.5 * S_strat, 1.5 * S_strat, 500)
                    payoff = compute_strategy_payoff(st.session_state.custom_legs, spot_range)

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(spot_range, payoff, label="Strategy Payoff", color="blue")
                    ax.axhline(0, color="black", linestyle="--")
                    ax.set_xlabel("Spot Price at Maturity (S)")
                    ax.set_ylabel("Net Payoff")
                    ax.set_title("Payoff Diagram: Custom Strategy")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error: {e}")

        if st.button("Clear Strategy Legs"):
            st.session_state.custom_legs = []

    else:
        st.subheader("Predefined Strategy")

        strategy = st.selectbox(
            "Choose a Strategy",
            ["Straddle", "Bull call Spread", "Bear put Spread", "Butterfly", "Iron Condor"],
            key="strat_type"
        )

        strike1 = st.number_input("Strike 1", value=95.0, key="strat_k1")
        strike2 = strike3 = strike4 = None

        if strategy in ["Bull call Spread", "Bear put Spread", "Butterfly", "Iron Condor"]:
            strike2 = st.number_input("Strike 2", value=100.0, key="strat_k2")
        if strategy in ["Butterfly", "Iron Condor"]:
            strike3 = st.number_input("Strike 3", value=105.0, key="strat_k3")
        if strategy == "Iron Condor":
            strike4 = st.number_input("Strike 4", value=110.0, key="strat_k4")

        if st.button("Price Predefined Strategy"):
            legs = get_predefined_strategy(strategy, strike1, strike2, strike3, strike4=strike4)
            if isinstance(legs, str):
                st.error(legs)
            else:
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


# -----------------------------
# Tab 4 – Exotic Options
# -----------------------------
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff

with tab4:
    st.header("Exotic Option Pricing")

    exotic_type = st.selectbox("Select Exotic Option Type", ["Digital", "Barrier"])

    if exotic_type == "Digital":
        st.subheader("Digital Option")

        col1, col2 = st.columns(2)
        with col1:
            option_type_dig = st.selectbox("Option Type", ["call", "put"], key="dig_type")
            style_dig = st.selectbox("Digital Style", ["cash", "asset"], key="dig_style")
            model_dig = st.selectbox("Model", ["black_scholes"], key="dig_model")
        with col2:
            S_dig = st.number_input("Spot Price (S)", value=100.0, key="dig_S")
            K_dig = st.number_input("Strike Price (K)", value=100.0, key="dig_K")
            T_dig = st.number_input("Time to Maturity (T)", value=1.0, key="dig_T")
            sigma_dig = st.number_input("Volatility (σ)", value=0.2, key="dig_sigma")
            r_dig = st.number_input("Risk-Free Rate (r)", value=0.05, key="dig_r")
            Q_dig = st.number_input("Payout (Q)", value=10.0, key="dig_Q")

        if st.button("Compute Digital Option Price"):
            try:
                price_dig = price_digital_option(
                    model=model_dig,
                    option_type=option_type_dig,
                    style=style_dig,
                    S=S_dig,
                    K=K_dig,
                    T=T_dig,
                    r=r_dig,
                    sigma=sigma_dig,
                    Q=Q_dig
                )
                st.success(f"The {style_dig} digital {option_type_dig} is worth: **{price_dig:.4f}**")
                st.subheader("Payoff at Maturity")
                plot_digital_payoff(K=K_dig, option_type=option_type_dig, style=style_dig, Q=Q_dig)
            except Exception as e:
                st.error(f"Error: {e}")

    elif exotic_type == "Barrier":
        st.subheader("Barrier Option")

        col1, col2 = st.columns(2)
        with col1:
            option_type_bar = st.selectbox("Option Type", ["call", "put"], key="bar_type")
            barrier_type = st.selectbox("Barrier Type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"], key="bar_style")
            model_bar = st.selectbox("Model", ["black_scholes"], key="bar_model")
        with col2:
            S_bar = st.number_input("Spot Price (S)", value=100.0, key="bar_S")
            K_bar = st.number_input("Strike Price (K)", value=100.0, key="bar_K")
            H_bar = st.number_input("Barrier Level (H)", value=120.0, key="bar_H")
            T_bar = st.number_input("Time to Maturity (T)", value=1.0, key="bar_T")
            sigma_bar = st.number_input("Volatility (σ)", value=0.2, key="bar_sigma")
            r_bar = st.number_input("Risk-Free Rate (r)", value=0.05, key="bar_r")

        if st.button("Compute Barrier Option Price"):
            try:
                price_bar = price_barrier_option(
                    model=model_bar,
                    option_type=option_type_bar,
                    barrier_type=barrier_type,
                    S=S_bar,
                    K=K_bar,
                    H=H_bar,
                    T=T_bar,
                    r=r_bar,
                    sigma=sigma_bar
                )
                st.success(f"The {barrier_type} {option_type_bar} option is worth: **{price_bar:.4f}**")
                st.subheader("Payoff at Maturity")
                plot_barrier_payoff(K=K_bar, H=H_bar, option_type=option_type_bar, barrier_type=barrier_type)
            except Exception as e:
                st.error(f"Error: {e}")

