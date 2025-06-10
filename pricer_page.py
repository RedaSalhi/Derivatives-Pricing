# main.py

import sys
import os
import matplotlib.pyplot as plt
# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import numpy as np

from pricing.vanilla_options import price_vanilla_option, plot_option_price_vs_param
from pricing.forward import (
    price_forward_contract,
    plot_forward_mark_to_market,
    plot_forward_payout_and_value
)
from pricing.option_strategies import (
    get_predefined_strategy,
    price_option_strategy,
    compute_strategy_payoff,
    plot_strategy_price_vs_param  
)

st.header("Derivatives Pricer")
# your full app logic for vanilla, forward, etc. goes here

# -----------------------------
# Tabs Layout
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vanilla Options", 
    "Forward Contracts", 
    "Option Strategies", 
    "Exotic Options",
    "Swaps",
    "Interest Rate Instruments"
])


from pricing.utils.greeks_vanilla.plot_single_greek import plot_single_greek_vs_spot

# -----------------------------
# Tab 1 – Vanilla Options
# -----------------------------
with tab1:
    st.header("Vanilla Option Pricing")

    # Init session state
    if "show_plot_controls" not in st.session_state:
        st.session_state["show_plot_controls"] = False
    if "option_kwargs" not in st.session_state:
        st.session_state["option_kwargs"] = {}

    # Input form
    col1, col2 = st.columns(2)
    with col1:
        option_type = st.selectbox("Option Type", ["Call", "Put"]).lower()
        exercise_style = st.selectbox("Exercise Style", ["European", "American"]).lower()
        model = st.selectbox("Pricing Model", ["Black Scholes", "Binomial", "Monte Carlo"])
        model_lower = {
            "Black Scholes": "black-scholes",
            "Monte Carlo": "monte-carlo"
        }.get(model, "binomial")

    with col2:
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
        T = st.number_input("Time to Maturity (T, in years)", value=1.0)
        sigma = st.number_input("Volatility (σ)", value=0.2)
        r = st.number_input("Risk-Free Rate (r)", value=0.05)
        q = st.number_input("Dividend Yield (q)", value=0.0)

    if model_lower == "binomial":
        N = st.slider("Binomial Tree Steps (N)", min_value=1, max_value=10000, step=2, value=100)
    elif model_lower == "monte-carlo":
        n_sim = st.slider("Monte Carlo Simulations", min_value=10, max_value=10000, step=100, value=1000)

    if st.button("Compute Option Price"):
        kwargs = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "q": q}
        if model_lower == "binomial":
            kwargs["N"] = N
        elif model_lower == "monte-carlo":
            kwargs["n_simulations"] = n_sim

        try:
            price = price_vanilla_option(
                option_type=option_type,
                exercise_style=exercise_style,
                model=model_lower,
                **kwargs
            )
            st.success(f"The {exercise_style} {option_type} option is worth: **{price:.4f}**")

            st.session_state["show_plot_controls"] = True
            st.session_state["option_kwargs"] = kwargs.copy()
            st.session_state["model_lower"] = model_lower
            st.session_state["option_type"] = option_type
            st.session_state["exercise_style"] = exercise_style

        except Exception as e:
            st.error(f"Error computing option price: {e}")
            st.session_state["show_plot_controls"] = False

    # -----------------------------
    # Visualization Section
    # -----------------------------
    if st.session_state["show_plot_controls"]:
        st.subheader("Visualize Option Price vs Parameter")
        st.markdown("<small>Compute the option price before generating the plot !</small>", unsafe_allow_html=True)

        param_to_vary = st.selectbox(
            "Select Parameter to Vary",
            ["S", "K", "T", "r", "sigma", "q"],
            key="vary_param"
        )

        fixed_kwargs = st.session_state["option_kwargs"]
        default_val = float(fixed_kwargs.get(param_to_vary, 1.0))
        if param_to_vary in ["r", "q", "sigma"]:
            min_val = st.number_input(f"Minimum value of {param_to_vary}", value=0.0, key="min_val")
            max_val = st.number_input(f"Maximum value of {param_to_vary}", value=1.0, key="max_val")
        elif param_to_vary == "T":
            min_val = st.number_input(f"Minimum value of {param_to_vary}", value=default_val * 0.01, key="min_val")
            max_val = st.number_input(f"Maximum value of {param_to_vary}", value=default_val * 100, key="max_val")
        else:
            min_val = st.number_input(f"Minimum value of {param_to_vary}", value=0.0, key="min_val")
            max_val = st.number_input(f"Maximum value of {param_to_vary}", value=default_val * 1.5, key="max_val")
        n_points = st.slider("Resolution", min_value=100, max_value=1000, value=500, key="n_points_slider")

        if st.button("Generate Plot"):
            try:
                fig = plot_option_price_vs_param(
                    option_type=st.session_state["option_type"],
                    exercise_style=st.session_state["exercise_style"],
                    model=st.session_state["model_lower"],
                    param_name=param_to_vary,
                    param_range=(min_val, max_val),
                    fixed_params=fixed_kwargs,
                    n_points=n_points
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Plotting failed: {e}")

        # -----------------------------
        # Greek Visualizations vs Spot
        # -----------------------------
        st.subheader("Greeks vs Spot Price")
        st.markdown("<small>Delta, Gamma, Vega, Theta, Rho</small>", unsafe_allow_html=True)

        greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"])
        if greek == "Delta":
            greek = "delta"
        elif greek == "Gamma":
            greek = "gamma"
        elif greek == "Vega":
            greek = "vega"
        elif greek == "Theta":
            greek = "theta"
        elif greek == "Rho":
            greek = "rho"
        S_range_min = st.number_input("Min Spot", value=0.5 * S)
        S_range_max = st.number_input("Max Spot", value=1.5 * S)
        n_points_greek = st.slider("Resolution", min_value=50, max_value=1000, value=300)

        if st.button("Plot Greek vs Spot"):
            try:
                fig_greek = plot_single_greek_vs_spot(
                    greek_name=greek,
                    model=st.session_state["model_lower"],
                    option_type=st.session_state["option_type"],
                    S0=S,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    q=q,
                    S_range=np.linspace(S_range_min, S_range_max, n_points_greek)
                )
                st.pyplot(fig_greek)
            except Exception as e:
                st.error(f"Greek plot failed: {e}")




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
        position = st.radio("Position", ["Long", "Short"])
        if position == "Long":
            position = "long"
        elif position == "Short":
            position = "short"

    if st.button("Price Forward Contract"):
        F = price_forward_contract(
            spot_price=S_fwd,
            interest_rate=r_fwd,
            time_to_maturity=T_fwd,
            storage_cost=storage_cost,
            dividend_yield=dividend_yield
        )
        st.success(f"Theoretical Forward Price: **{F:.4f}**")

        with st.expander("Forward Payout at Maturity"):
            plot_forward_payout_and_value(K_fwd, position)

        with st.expander("Mark-to-Market Value (Before Maturity)"):
            plot_forward_mark_to_market(
                strike_price=K_fwd,
                time_to_maturity=T_fwd,
                interest_rate=r_fwd,
                storage_cost=storage_cost,
                dividend_yield=dividend_yield,
                position=position
            )


from pricing.option_strategies import price_option_strategy, compute_strategy_payoff, get_predefined_strategy, plot_strategy_price_vs_param
from pricing.utils.option_strategies_greeks import plot_strategy_greek_vs_spot

# -----------------------------
# Tab 3 – Option Strategies
# -----------------------------
with tab3:
    st.header("Option Strategies")

    use_manual = st.checkbox("Build Strategy Manually")

    model_strat = st.selectbox("Pricing Model", ["Black Scholes", "Binomial", "Monte Carlo"], key="strat_model")
    model_strat = {"Black Scholes": "black-scholes", "Monte Carlo": "monte-carlo"}.get(model_strat, "binomial")

    style_strat = st.selectbox("Exercise Style", ["European", "American"], key="strat_style").lower()

    S_strat = st.number_input("Spot Price (S)", value=100.0, key="strat_S")
    T_strat = st.number_input("Time to Maturity (T)", value=1.0, key="strat_T")
    sigma_strat = st.number_input("Volatility (σ)", value=0.2, key="strat_sigma")
    r_strat = st.number_input("Risk-Free Rate (r)", value=0.05, key="strat_r")
    q_strat = st.number_input("Dividend Yield (q)", value=0.0, key="strat_q")

    kwargs = {"S": S_strat, "T": T_strat, "sigma": sigma_strat, "r": r_strat, "q": q_strat}

    # -----------------------------
    # Manual Strategy
    # -----------------------------
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
            if st.form_submit_button("Add Leg"):
                st.session_state.custom_legs.append({"type": opt_type, "strike": strike, "qty": qty})

        if st.session_state.custom_legs:
            st.markdown("### Strategy Legs")
            for i, leg in enumerate(st.session_state.custom_legs):
                st.write(f"Leg {i+1}: {leg['qty']} × {leg['type'].upper()} @ Strike {leg['strike']}")

            if st.button("Price Custom Strategy"):
                try:
                    legs = st.session_state.custom_legs
                    st.session_state["manual_legs"] = legs
                    result = price_option_strategy(legs, style_strat, model_strat, **kwargs)
                    st.success(f"Total Strategy Price: **{result['strategy_price']:.4f}**")

                    spot_range = np.linspace(0.5 * S_strat, 1.5 * S_strat, 500)
                    payoff = compute_strategy_payoff(legs, spot_range)

                    with st.expander("Payoff Plot"):
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

            if "manual_legs" in st.session_state:
                st.subheader("Visualize Strategy Price vs Parameter")

                param = st.selectbox("Parameter to Vary", ["S", "T", "r", "sigma", "q"], key="manual_vary")
                default_val = float(kwargs.get(param, 1.0))

                min_val = st.number_input(f"Min {param}", value=0.0 if param in ["r", "q", "sigma"] else 0.01, key="manual_min")
                max_val = st.number_input(f"Max {param}", value=default_val * (1.5 if param == "S" else 100), key="manual_max")
                n_points = st.slider("Resolution", 50, 500, 100, key="manual_n")

                if st.button("Generate Plot for Custom Strategy"):
                    try:
                        with st.expander("Strategy Premium vs Parameter"):
                            fig = plot_strategy_price_vs_param(
                                legs=st.session_state["manual_legs"],
                                exercise_style=style_strat,
                                model=model_strat,
                                param_name=param,
                                param_range=(min_val, max_val),
                                fixed_params=kwargs,
                                n_points=n_points
                            )
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Plotting failed: {e}")

                st.subheader("Visualize Strategy Greek vs Spot Price")

                greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"], key="manual_greek")
                if greek == "Delta":
                    greek = "delta"
                elif greek == "Gamma":
                    greek = "gamma"
                elif greek == "Vega":
                    greek = "vega"
                elif greek == "Theta":
                    greek = "theta"
                elif greek == "Rho":
                    greek = "rho"
                S_min = st.number_input("Min Spot (S)", value=0.5 * S_strat, key="manual_greek_smin")
                S_max = st.number_input("Max Spot (S)", value=1.5 * S_strat, key="manual_greek_smax")
                greek_res = st.slider("Greek Plot Resolution", 50, 1000, 300, key="manual_greek_n")

                if st.button("Plot Strategy Greek (Manual)"):
                    try:
                        with st.expander("Strategy Greek"):
                            fig = plot_strategy_greek_vs_spot(
                                greek_name=greek,
                                legs=st.session_state["manual_legs"],
                                model=model_strat,
                                S0=S_strat,
                                T=T_strat,
                                r=r_strat,
                                sigma=sigma_strat,
                                q=q_strat,
                                S_range=np.linspace(S_min, S_max, greek_res)
                            )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Greek plot failed: {e}")

        if st.button("Clear Strategy Legs"):
            st.session_state.custom_legs = []
            st.session_state.pop("manual_legs", None)

    # -----------------------------
    # Predefined Strategy
    # -----------------------------
    else:
        st.subheader("Predefined Strategy")
        strategy = st.selectbox("Choose a Strategy", ["Straddle", "Bull call Spread", "Bear put Spread", "Butterfly", "Iron Condor"], key="strat_type")

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
                    st.session_state["predefined_legs"] = legs
                    result = price_option_strategy(legs, style_strat, model_strat, **kwargs)
                    st.success(f"Total Strategy Price: **{result['strategy_price']:.4f}**")

                    spot_range = np.linspace(0.5 * S_strat, 1.5 * S_strat, 500)
                    payoff = compute_strategy_payoff(legs, spot_range)
                    with st.expander("Payoff Plot"):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(spot_range, payoff, label="Strategy Payoff", color="green")
                        ax.axhline(0, color="black", linestyle="--")
                        ax.set_xlabel("Spot Price at Maturity (S)")
                        ax.set_ylabel("Net Payoff")
                        ax.set_title(f"Payoff Diagram: {strategy}")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")

        if "predefined_legs" in st.session_state:
            st.subheader("Visualize Strategy Price vs Parameter")

            param = st.selectbox("Parameter to Vary", ["S", "T", "r", "sigma", "q"], key="predef_vary")
            default_val = float(kwargs.get(param, 1.0))

            min_val = st.number_input(f"Min {param}", value=0.0 if param in ["r", "q", "sigma"] else 0.01, key="predef_min")
            max_val = st.number_input(f"Max {param}", value=default_val * (1.5 if param == "S" else 100), key="predef_max")
            n_points = st.slider("Resolution", 50, 500, 100, key="predef_n")

            if st.button("Generate Plot for Predefined Strategy"):
                try:
                    with st.expander("Strategy Premium vs Parameter"):
                        fig = plot_strategy_price_vs_param(
                            legs=st.session_state["predefined_legs"],
                            exercise_style=style_strat,
                            model=model_strat,
                            param_name=param,
                            param_range=(min_val, max_val),
                            fixed_params=kwargs,
                            n_points=n_points
                        )
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Plotting failed: {e}")

            st.subheader("Visualize Strategy Greek vs Spot Price")

            greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"], key="manual_greek")
            if greek == "Delta":
                greek = "delta"
            elif greek == "Gamma":
                greek = "gamma"
            elif greek == "Vega":
                greek = "vega"
            elif greek == "Theta":
                greek = "theta"
            elif greek == "Rho":
                greek = "rho"
            S_min = st.number_input("Min Spot (S)", value=0.5 * S_strat, key="predef_greek_smin")
            S_max = st.number_input("Max Spot (S)", value=1.5 * S_strat, key="predef_greek_smax")
            greek_res = st.slider("Greek Plot Resolution", 50, 1000, 300, key="predef_greek_n")

            if st.button("Plot Strategy Greek (Predefined)"):
                try:
                    with st.expander("Strategy Greek"):
                        fig = plot_strategy_greek_vs_spot(
                            greek_name=greek,
                            legs=st.session_state["predefined_legs"],
                            model=model_strat,
                            S0=S_strat,
                            T=T_strat,
                            r=r_strat,
                            sigma=sigma_strat,
                            q=q_strat,
                            S_range=np.linspace(S_min, S_max, greek_res)
                        )
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Greek plot failed: {e}")





# -----------------------------
# Tab 4 – Exotic Options
# -----------------------------
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff, plot_sample_paths_barrier
from pricing.asian_option import price_asian_option, plot_asian_option_payoff, plot_monte_carlo_paths
from pricing.models.asian_monte_carlo import simulate_asian_paths
from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution

with tab4:
    st.header("Exotic Option Pricing")

    exotic_type = st.selectbox("Select Exotic Option Type", ["Digital", "Barrier", "Asian", "Lookback"])

    # ===========================
    # Digital Option Interface
    # ===========================
    if exotic_type == "Digital":
        st.subheader("Digital Option")

        col1, col2 = st.columns(2)
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="dig_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            style = st.selectbox("Digital Style", ["cash", "asset"], key="dig_style")
            model = st.selectbox("Model", ["Black Scholes"], key="dig_model")
            if model == "Black Scholes":
                model = "black_scholes"
        with col2:
            S = st.number_input("Spot Price (S)", value=100.0, key="dig_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="dig_K")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="dig_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="dig_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="dig_r")
            Q = st.number_input("Payout (Q)", value=10.0, key="dig_Q")

        if st.button("Compute Digital Option Price"):
            try:
                price = price_digital_option(
                    model=model,
                    option_type=option_type,
                    style=style,
                    S=S,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    Q=Q
                )
                st.success(f"The {style} digital {option_type} is worth: **{price:.4f}**")
                with st.expander("Payoff at Maturity"):
                    plot_digital_payoff(K=K, option_type=option_type, style=style, Q=Q)

            except Exception as e:
                st.error(f"Error: {e}")

    # ===========================
    # Barrier Option Interface
    # ===========================
    elif exotic_type == "Barrier":
        st.subheader("Barrier Option")

        col1, col2 = st.columns(2)
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="bar_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            barrier_type = st.selectbox("Barrier Type", ["Up and Out", "Up and In", "Down and Out", "Down and In"], key="bar_style")
            if barrier_type == "Up and Out":
                barrier_type = "up-and-out"
            elif barrier_type == "Up and In":
                barrier_type = "up-and-in"
            elif barrier_type == "Down and Out":
                barrier_type = "down-and-out"
            elif barrier_type == "Down and In":
                barrier_type = "down-and-in"
            model = st.selectbox("Model", ["Monte Carlo"], key="bar_model")
            if model == "Monte Carlo":
                model = "monte_carlo"
        with col2:
            S = st.number_input("Spot Price (S)", value=100.0, key="bar_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="bar_K")
            H = st.number_input("Barrier Level (H)", value=120.0, key="bar_H")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="bar_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="bar_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="bar_r")

        # Sliders appear only if Monte Carlo is selected
        if model == "monte_carlo":
            n_sim = st.slider("Number of Simulations", min_value=10, max_value=10000, step=10, value=1000)
            n_steps = st.slider("Steps per Path", min_value=10, max_value=300, step=2, value=252)
        else:
            n_sim = None
            n_steps = None

        if st.button("Compute Barrier Option Price"):
            st.markdown("<small>Wait a few seconds for the plots !</small>", unsafe_allow_html=True)
            try:
                kwargs = dict(
                    model=model,
                    option_type=option_type,
                    barrier_type=barrier_type,
                    S=S,
                    K=K,
                    H=H,
                    T=T,
                    r=r,
                    sigma=sigma
                )
        
                if model == "monte_carlo":
                    kwargs["n_simulations"] = n_sim
                    kwargs["n_steps"] = n_steps
                    price, paths = price_barrier_option(**kwargs)
        
                st.success(f"The {barrier_type} {option_type} option is worth: **{price:.4f}**")
        
                if model == "monte_carlo":
                    with st.expander("Payoff at Maturity"):
                        plot_barrier_payoff(K=K, H=H, option_type=option_type, barrier_type=barrier_type)
                    with st.expander("Monte Carlo Sample Paths"):
                        plot_sample_paths_barrier(paths, K=K, H=H, option_type=option_type, barrier_type=barrier_type)
                    
        
            except Exception as e:
                st.error(f"Error: {e}")
    # ===========================
    # Asian Option Interface
    # ===========================
    elif exotic_type == "Asian":
        st.subheader("Asian Option")

        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Pricing Method", ["Monte Carlo"], key="asian_method")
            if method == "Monte Carlo":
                method = "monte_carlo"
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="asian_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            asian_type = st.selectbox("Asian Type", ["Average Price", "Average Strike"], key="asian_style")
            if asian_type == "Average Price":
                asian_type = "average_price"
            elif asian_type == "Average Strike":
                asian_type = "average_strike"
        with col2:
            S = st.number_input("Spot Price (S)", value=100.0, key="asian_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="asian_K")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="asian_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="asian_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="asian_r")

        if method == "monte_carlo":
            n_paths = st.slider("Monte Carlo Simulations", 10, 10000, step=10, value=10000)
            n_steps = st.slider("Steps per Path", 10, 300, step=2, value=252)

        if st.button("Compute Asian Option Price"):
            st.markdown("<small>Wait a few seconds for the plots !</small>", unsafe_allow_html=True)
            try:
                price = price_asian_option(
                    S0=S, K=K, T=T, r=r, sigma=sigma,
                    n_steps=n_steps,
                    n_paths=n_paths or 10000,
                    method=method,
                    option_type=option_type,
                    asian_type=asian_type
                )

                st.success(f"The {asian_type.replace('_', ' ')} Asian {option_type} is worth: **{price:.4f}**")

                with st.expander("Payoff at Maturity"):    
                    plot_asian_option_payoff(K=K, option_type=option_type, asian_type=asian_type)

                if method == "monte_carlo":
                    paths = simulate_asian_paths(S0=S, T=T, r=r, sigma=sigma,
                                                 n_steps=n_steps, n_paths=n_paths,
                                                 option_type=option_type, asian_type=asian_type)
                    with st.expander("Monte Carlo Sample Paths"):
                        plot_monte_carlo_paths(paths)

            except Exception as e:
                st.error(f"Error: {e}")
    # ===========================
    # Lookback Option Interface
    # ===========================
    elif exotic_type == "Lookback":
        st.subheader("Lookback Option")

        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Pricing Model", ["Monte Carlo"], key="lookback_model")
            if model == "Monte Carlo":
                model = "monte_carlo"
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="lookback_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            floating_strike = st.checkbox("Floating Strike", value=True, key="lookback_floating")
        with col2:
            S0 = st.number_input("Spot Price (S₀)", value=100.0, key="lookback_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="lookback_K")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="lookback_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="lookback_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="lookback_r")
    
        if model == "monte_carlo":
            n_paths = st.slider("Monte Carlo Simulations", 10, 10000, step=10, value=1000, key="lookback_paths")
            n_steps = st.slider("Steps per Path", 10, 300, step=2, value=252, key="lookback_steps")
    
        if st.button("Compute Lookback Option Price"):
            st.markdown("<small>Wait a few seconds for the plots !</small>", unsafe_allow_html=True)
            from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution
    
            try:
                price, stderr = price_lookback_option(
                    S0=S0, K=K if not floating_strike else None, r=r, sigma=sigma, T=T,
                    model=model, option_type=option_type,
                    floating_strike=floating_strike,
                    n_paths=n_paths if model == "monte_carlo" else None,
                    n_steps=n_steps if model == "monte_carlo" else None
                )
    
                if stderr is not None:
                    st.success(f"Monte Carlo Price: **{price:.4f}**")
    
                with st.expander("Payoff Function"):
                    st.pyplot(plot_payoff(S0, option_type, K, floating_strike))
    
                with st.expander("Simulated Asset Paths"):
                    st.pyplot(plot_paths(S0, r, sigma, T, n_paths, n_steps))
    
                if model == "monte_carlo":
                    with st.expander("Distribution of Discounted Payoffs"):
                        st.pyplot(plot_price_distribution(S0, r, sigma, T, option_type, floating_strike, n_paths, n_steps))
    
            except Exception as e:
                st.error(f"Error: {e}")





# -----------------------------
# Tab 5 – Swaps
# -----------------------------

from pricing.swaps import price_swap
from pricing.models.swaps.ois_fx import (
    build_flat_discount_curve,
    build_flat_fx_forward_curve
)

with tab5:
    st.subheader("Swap Pricer (Not accurate)")

    swap_type = st.selectbox("Swap Type", ["IRS", "Currency", "Equity"])
    model = st.selectbox("Model", {
        "IRS": ["DCF", "LMM"],
        "Currency": ["DCF"],
        "Equity": ["DCF"]
    }[swap_type])

    swap_type_lower = swap_type.lower()
    model_lower = model.lower()

    st.markdown("### Parameters")
    fixed_params = {}
    payment_times = []

    if swap_type_lower == "irs":
        notional = st.number_input("Notional", value=100)
        fixed_rate = st.slider("Fixed Rate", 0.01, 0.10, 0.03)
        payment_times = [0.5 * i for i in range(1, 11)]

        if model_lower == "dcf":
            floating_rate = st.slider("Floating Rate", 0.01, 0.10, 0.025)
            discount_rate = st.slider("Discount Rate", 0.00, 0.10, 0.02)
            discount_curve = build_flat_discount_curve(discount_rate)

            fixed_params = {
                "notional": notional,
                "fixed_rate": fixed_rate,
                "floating_rates": [floating_rate] * len(payment_times),
                "payment_times": payment_times,
                "discount_curve": discount_curve
            }

        elif model_lower == "lmm":
            L0 = st.slider("Initial Forward Rate (L0)", 0.01, 0.10, 0.025)
            vol = st.slider("Volatility", 0.01, 0.50, 0.15)
            discount_rate = st.slider("Discount Rate", 0.00, 0.10, 0.02)
            discount_curve = build_flat_discount_curve(discount_rate)

            fixed_params = {
                "notional": notional,
                "fixed_rate": fixed_rate,
                "L0": L0,
                "vol": vol,
                "payment_times": payment_times,
                "discount_curve": discount_curve,
                "n_paths": 5000
            }

    elif swap_type_lower == "currency":
        notional = st.number_input("Notional (Domestic)", value=100)
        r_dom = st.slider("Domestic Rate", 0.00, 0.10, 0.03)
        r_for = st.slider("Foreign Rate", 0.00, 0.10, 0.015)
        fx_spot = st.number_input("Spot FX Rate", value=1.10)
        rate_domestic = [r_dom] * 10
        rate_foreign = [r_for] * 10
        payment_times = [0.5 * i for i in range(1, 11)]

        discount_dom = build_flat_discount_curve(r_dom)
        discount_for = build_flat_discount_curve(r_for)
        fx_curve = build_flat_fx_forward_curve(fx_spot, r_dom, r_for)

        fixed_params = {
            "notional_domestic": notional,
            "rate_domestic": rate_domestic,
            "rate_foreign": rate_foreign,
            "payment_times": payment_times,
            "discount_domestic": discount_dom,
            "discount_foreign": discount_for,
            "fx_forward_curve": fx_curve
        }

    elif swap_type_lower == "equity":
        notional = st.number_input("Notional", value=100)
        S0 = st.number_input("Equity Start Price", value=100.0)
        ST = st.number_input("Equity End Price", value=110.0)
        K = st.number_input("Fixed Strike (K)", value=105.0)
        r = st.slider("Risk-Free Rate", 0.00, 0.10, 0.03)
        q = st.slider("Dividend Yield", 0.00, 0.10, 0.01)
        fixed_rate = st.slider("Fixed Leg Rate", 0.01, 0.10, 0.03)
        T = 5
        payment_times = [i for i in range(1, T + 1)]
        discount_curve = build_flat_discount_curve(r)

        if model_lower == "dcf":
            fixed_params = {
                "notional": notional,
                "equity_start": S0,
                "equity_end": ST,
                "fixed_rate": fixed_rate,
                "payment_times": payment_times,
                "discount_curve": discount_curve
            }

    # ---- Run Pricing ----
    if st.button("Calculate Swap Price"):
        try:
            result = price_swap(swap_type=swap_type_lower, model=model_lower, **fixed_params)
            st.success(f"Swap Price: {result:.2f}")
        except Exception as e:
            st.error(f"Error during pricing: {e}")



# -----------------------------
# Tab 6 – IR Instruments
# -----------------------------

with tab6:
    st.subheader("Interest Rate Instruments Pricer (Not accurate)")
    st.sidebar.selectbox("Choose Interest Rate Model", ["Hull-White (1F)"])
    st.sidebar.slider("Mean Reversion (alpha)", 0.01, 1.0, 0.1)
    st.sidebar.slider("Volatility (sigma)", 0.001, 1.0, 0.01)
    st.sidebar.number_input("Initial Rate r(0)", value=0.03)
    
    instrument = st.selectbox("Select Instrument", ["Zero-Coupon Bond", "Cap", "Floor", "Swaption"])
    if instrument == "Cap":
        # show extra cap inputs
    
    st.button("Run Pricing")
    
