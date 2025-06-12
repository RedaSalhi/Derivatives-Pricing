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
    st.header("Exotic Option Pricing (In Progress)")

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
    st.header("Swap Pricer (In Progress)")

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
"""
with tab6:
    import streamlit as st
    import numpy as np
    import pandas as pd
    from pricing.vanilla_vasicek import price_zero_coupon, price_coupon_bond, price_bond_option
    from pricing.models.interest_rates.monte_carlo_vasicek import simulate_vasicek_paths, plot_vasicek_paths, plot_yield_distribution, vasicek_bond_option_price_mc
    from pricing.models.interest_rates.analytical_vasicek import run_ou_estimation, simulate_vasicek_path, plot_yield_curves, generate_yield_curves, vasicek_bond_option_price

    st.header("Interest Rate Instruments Pricer")

    # --- Model & method ---
    model = st.selectbox("Choose Interest Rate Model", ["Vasicek", "Hull-White (Planned)"], index=0)
    if model == "Vasicek":
        #method = st.radio("Pricing Method", ["Analytical", "Monte Carlo"])
        
        st.markdown("### 🔧 Parameter Setup")
        param_mode = st.radio("How to set parameters?", ["Manual input", "Calibrate from market data (FRED/Yahoo)"])
    
        if param_mode == "Manual input":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                a = st.slider("Mean Reversion Speed (a)", 0.01, 2.0, 0.5 )
            with col2:
                sigma = st.slider("Volatility (σ)", 0.001, 1.0, 0.01)
            with col3:
                lam = st.number_input("Long-Term Mean Level (λ)", 0.04)
            with col4:
                dt = st.number_input("Data frequency (dt)", 0.08)
            r0 = st.number_input("Initial Short Rate r(0)", 0.05)
    
        else:
            ticker = st.text_input("Enter FRED/Yahoo Ticker", value="DGS3MO")
            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    
            if st.button("Calibrate"):
                a, lam, sigma, dt, r0 = run_ou_estimation(ticker, start=start_date.strftime('%Y-%m-%d'))
                st.success(f"✔️ Calibrated: a={a:.4f}, λ={lam:.4f}, σ={sigma:.4f}, r₀={r0:.4f}, dt={dt:.4f}")
    
        st.markdown("### 📈 Select Instrument to Price")
        instrument = st.selectbox("Instrument", [
            "Zero-Coupon Bond",
            "Coupon Bond",
            "Bond Option",
            "Cap (Planned)",
            "Floor (Planned)",
            "Swaption (Planned)"
        ])
    
        if instrument == "Zero-Coupon Bond":
            maturity = st.slider("Maturity (years)", 0.5, 30.0, 5.0, step=0.5)
            t = st.slider("Temps écoulé t (années)", min_value=0.0, max_value=maturity, value=0.0, step=0.25)
            face_value = st.number_input("Face Value", value=1.0)
    
        elif instrument == "Coupon Bond":
            maturity = st.slider("Maturity (years)", 0.5, 30.0, 5.0, step=0.5)
            coupon = st.slider("Coupon Rate", 0.0, 1.0, 0.05, step=0.01)
            face_value = st.number_input("Face Value", value=1.0)
    
        elif instrument == "Bond Option (European)":
            T1 = st.slider("Option Expiry T1 (years)", 0.5, 10.0, 3.0, step=0.5)
            T2 = st.slider("Bond Maturity T2 (years)", T1 + 0.5, 30.0, 5.0, step=0.5)
            if T2 <= T1:
                st.error("T2 must be greater than T1")
            K = st.number_input("Strike Price (P(T1, T2))", value=0.85)
            option_type = st.radio("Option Type", ["Call", "Put"]).lower()
            face = st.number_input("Face Value", value=1.0)
            n_paths = st.number_input("Monte Carlo Paths", value=1000, step=1000)
            
    
        st.markdown("---")

        # --- Yield Curve Plot Button ---
        yield_curve = st.checkbox("Simulate Yield Curve?")
        plot_paths = st.checkbox("Plot Monte Carlo Paths and Rate Distribution at T?")
        greeks = st.checkbox("Visualize Greeks/Option Price? (Check for bond options!)")

        if yield_curve:
            T = st.slider("Simulation horizon (years)", min_value=1, max_value=30, value=10, step=1)
            time, r_path = simulate_vasicek_path(r0, a, lam, sigma, T=T, dt=dt)
            maturities = np.linspace(0.5, T, 60)
            possible_snapshots = list(np.arange(0, T + sim_dt, 0.5))  # 0.0, 0.5, 1.0, ..., 10.0
            snapshot_times = st.multiselect(
                "Select snapshot times (in years):",
                options=possible_snapshots,
                default=[0, 1, 3, 5]
            )
            if not snapshot_times:
                st.warning("Please select at least one snapshot time.")

        if plot_paths:
            if not instrument == "Bond Option (European)":
                n_paths = st.number_input("Monte Carlo Paths", value=1000, step=1000)
                
            T_vec, r_paths = simulate_vasicek_paths(a, lam, sigma, r0, T, dt, n_paths)

        if greeks:
            col1, col2 = st.columns(2)
            with col1:
                greek = st.selectbox("Select Greek or Price:", [ "Price", "Delta", "Vega", "Rho"])
                greek_lower = greek.lower()
            with col2:
                model = st.selectbox("Select Model", [ "Analytical", "Monte Carlo"])
                if model == "Monte Carlo":
                    st.warning("Monte Carlo is not working properly at the moment.")
 
            
    
        # --- Pricing Button ---
        if st.button("Run Pricing"):
            if instrument == "Zero-Coupon Bond":
                price = price_zero_coupon(r0, t, maturity, a, lam, sigma, face_value)
                st.success(f"Zero-Coupon Bond Price: {price:.4f}")
    
            elif instrument == "Coupon Bond":
                price = price_coupon_bond(r0, t, a, lam, sigma, maturity, coupon=coupon, face= face_value, dt=dt)
                st.success(f"Coupon Bond Price: {price:.4f}")
    
            elif instrument == "Bond Option (European)":
                if T2 <= T1:
                    st.error("Invalid maturity: T2 must be > T1")
                
                method = st.radio("Choose The Pricing Method", ["Analytical", "Monte Carlo"])
                if method == "Analytical":
                    price = vasicek_bond_option_price(r0, 0, T1, T2, K, a, lam, sigma, face=face_value, option_type=option_type)
                else:
                    price, std = vasicek_bond_option_price_mc(r0, a, lam, sigma, T1, T2, K, dt, n_paths, face=face_value, option_type=option_type)
                    st.info(f"Monte Carlo Std Error: {std:.6f}")
                st.success(f"Bond Option Price ({option_type}): {price:.6f}")
    
            else:
                st.warning("This instrument is not yet implemented.")

            
            if yield_curve: 
                with st.expander("Yield Curve:"):
                    yield_curves = generate_yield_curves(r_path, snapshots_times, maturities, a, lam, sigma, dt)
                    plot_yield_curves(yield_curves, maturities)

            if plot_paths:
                with st.expander("Paths and Yield Distribution:"):
                    c1, c2 = st.columns(2)
                    with c1:
                        plot_vasicek_paths(T_vec, r_paths, lam)
                    with c2:
                        plot_yield_distribution(r_paths)

            if greeks:
                with st.expander(f"{greek} Visualization"):
                    compute_greek_vs_spot(greek_lower, 0, T1, T2, K, a, lam, sigma, face, option_type=option_type, n_paths=n_paths, model=model)

    
    else:
        st.warning("This instrument is not yet implemented.")
        """


with tab6:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, date

    from pricing.models.interest_rates.analytical_vasicek import *
    from pricing.models.interest_rates.monte_carlo_vasicek import *
    from pricing.utils.greeks_vasicek import *
    
    st.title("📈 Modèle de Vasicek - Pricing d'Obligations et Taux d'intérêt")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔧 Estimation des Paramètres", "📊 Simulation et Courbes", "💰 Pricing d'Obligations", "📈 Options sur Obligations", "🔍 Analyse des Grecques"]
    )
    
    # Session state pour stocker les paramètres estimés
    if 'vasicek_params' not in st.session_state:
        st.session_state.vasicek_params = None
    
    # =============================================
    # TAB 1: ESTIMATION DES PARAMÈTRES
    # =============================================
    with tab1:
        st.header("🔧 Estimation des Paramètres du Modèle de Vasicek")
    
        col1, col2 = st.columns([1, 1])
    
        with col1:
            st.subheader("Configuration des Données")
    
            # Ticker libre
            ticker = st.text_input(
                "Entrer un ticker FRED ou Yahoo (ex: DGS10, DFF, ^IRX)",
                value="DGS10",
                help="Exemples : DGS10 (US 10Y), DGS2 (2Y), DFF (Fed Funds), ^IRX (T-Bill 13W Yahoo)"
            )
    
            # Dates récentes (par défaut : 5 dernières années)
            today = date.today()
            default_start = today.replace(year=today.year - 5)
    
            start_date = st.date_input("Date de début", default_start)
            end_date = st.date_input("Date de fin", today)
    
            # Fréquence de resampling
            freq = st.selectbox("Fréquence", ["ME", "QE", "YE"], index=0)
    
            # Lancement estimation
            if st.button("📊 Estimer les Paramètres", type="primary"):
                if start_date >= end_date:
                    st.error("❌ La date de début doit être antérieure à la date de fin.")
                else:
                    with st.spinner("Chargement des données et estimation..."):
                        try:
                            a, lam, sigma, dt, r0 = run_ou_estimation(ticker, str(start_date), str(end_date), freq)
    
                            st.session_state.vasicek_params = {
                                'a': a, 'lambda': lam, 'sigma': sigma, 'dt': dt, 'r0': r0, 'ticker': ticker
                            }
                            st.success("✅ Paramètres estimés avec succès!")
    
                        except Exception as e:
                            import traceback
                            st.error(f"❌ Erreur lors de l'estimation :\n\n```\n{traceback.format_exc()}\n```")
    
        with col2:
            st.subheader("Paramètres Estimés")
            if st.session_state.vasicek_params:
                params = st.session_state.vasicek_params
    
                col_a, col_lam, col_sig = st.columns(3)
                with col_a:
                    st.metric("Vitesse de retour à la moyenne (a)", f"{params['a']:.4f}")
                with col_lam:
                    st.metric("Niveau moyen long terme (λ)", f"{params['lambda']:.4f}")
                with col_sig:
                    st.metric("Volatilité (σ)", f"{params['sigma']:.4f}")
    
                st.metric("Taux initial (r₀)", f"{params['r0']:.4f}")
                st.info(f"📊 Ticker utilisé : **{params['ticker']}** | Δt: {params['dt']:.4f}")
            else:
                st.info("👆 Cliquez sur 'Estimer les Paramètres' pour commencer")

    
    # =============================================
    # TAB 2: SIMULATION ET COURBES
    # =============================================
    with tab2:
        st.header("📊 Simulation de Trajectoires et Courbes de Taux (Vasicek)")
    
        if not st.session_state.vasicek_params:
            st.warning("⚠️ Veuillez d'abord estimer les paramètres dans l'onglet précédent.")
            st.stop()
    
        params = st.session_state.vasicek_params
    
        col1, col2 = st.columns([1, 2])
    
        with col1:
            st.subheader("⚙️ Paramètres de Simulation")
    
            T = st.slider("Horizon temporel (années)", min_value=1, max_value=30, value=10)
            dt = st.slider("Pas de temps (dt)", min_value=0.01, max_value=1.0, value=float(params["dt"]), step=0.01)
            n_paths = st.slider("Nombre de trajectoires simulées", 100, 10000, 1000, step=100)
    
            st.subheader("📐 Configuration des Courbes de Taux")
    
            available_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
            default_maturities = [m for m in [1, 2, 5, 10] if m <= T]
            maturities = st.multiselect("Maturités (années)", options=available_maturities, default=default_maturities)
    
            # Générer les snapshots lisibles et valides
            max_snapshots = int(T / dt)
            raw_snapshots = [round(i * dt, 2) for i in range(max_snapshots + 1)]
            
            # Formater pour affichage
            labelled_snapshots = {f"{s:.2f} ans": s for s in raw_snapshots}
            
            # Choix par l'utilisateur (affichage propre, valeurs float)
            default_keys = [k for k in labelled_snapshots if float(k.split()[0]) in [0.0, 2.0, 5.0, 10.0]]
            selected_keys = st.multiselect("Temps de snapshot (années)", options=list(labelled_snapshots.keys()), default=default_keys)
            
            # Convertir pour usage technique
            snapshot_times = [labelled_snapshots[k] for k in selected_keys]
    
            simulate_btn = st.button("🚀 Lancer la Simulation", type="primary")
    
        with col2:
            if simulate_btn:
                with st.spinner("Simulation en cours..."):
    
                    # Lancement de la simulation
                    time_vec, r_paths = simulate_vasicek_paths(
                        a=params['a'],
                        lam=params['lambda'],
                        sigma=params['sigma'],
                        r0=params['r0'],
                        T=T,
                        dt=dt,
                        n_paths=n_paths
                    )
    
                    # ✅ Courbes de taux : moyenne sur les paths
                    yield_curves = generate_yield_curves(
                        r_path=np.mean(r_paths, axis=1),
                        snapshot_times=snapshot_times,
                        maturities=maturities,
                        a=params['a'],
                        theta=params['lambda'],
                        sigma=params['sigma'],
                        dt=dt
                    )
    
                    # 📈 Affichage avec Matplotlib (plus rapide pour plusieurs snapshots)
                    st.pyplot(plot_yield_curves(yield_curves, maturities))
    
                    # 📉 Distribution du taux final
                    r_final = r_paths[-1, :]
                    fig_hist = px.histogram(
                        r_final,
                        nbins=50,
                        title="Distribution du Taux Court à l'Horizon",
                        labels={'value': 'Taux', 'count': 'Fréquence'}
                    )
                    fig_hist.add_vline(x=np.mean(r_final), line_dash="dash", line_color="red",
                                       annotation_text=f"Moyenne: {np.mean(r_final):.4f}")
                    st.plotly_chart(fig_hist, use_container_width=True)
    
                    # 🧮 Statistiques descriptives
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Moyenne finale", f"{np.mean(r_final):.4f}")
                    with col_stat2:
                        st.metric("Écart-type", f"{np.std(r_final):.4f}")
                    with col_stat3:
                        st.metric("Min / Max", f"{np.min(r_final):.4f} / {np.max(r_final):.4f}")

    
    # =============================================
    # TAB 3: PRICING D'OBLIGATIONS
    # =============================================
    with tab3:
        st.header("💰 Pricing d'Obligations (Zero-Coupon ou à Coupons)")
    
        if not st.session_state.vasicek_params:
            st.warning("⚠️ Veuillez d'abord estimer les paramètres dans l'onglet précédent.")
            st.stop()
    
        params = st.session_state.vasicek_params
    
        col1, col2 = st.columns([1, 2])
    
        with col1:
            st.subheader("📋 Paramètres de l’Obligation")
    
            bond_type = st.radio("Type d'obligation", ["Zero-Coupon", "Avec Coupons"])
    
            r_current = st.number_input("Taux actuel (r)", min_value=0.0, max_value=0.20, value=params['r0'], step=0.001, format="%.4f")
            t_current = st.number_input("Temps actuel (t)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
            maturity = st.number_input("Maturité (T)", min_value=t_current + 0.1, max_value=30.0, value=5.0, step=0.1)
            face_value = st.number_input("Valeur nominale", min_value=100, max_value=10000, value=100, step=10)
    
            if bond_type == "Avec Coupons":
                coupon_rate = st.number_input("Taux de coupon (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
                freq = st.selectbox("Fréquence des paiements", ["Annuel", "Semestriel"])
                dt_coupon = 1.0 if freq == "Annuel" else 0.5
    
            st.subheader("🔍 Analyse de Sensibilité")
            sensitivity_param = st.selectbox("Paramètre à tester", ["Taux actuel (r)", "Maturité (T)", "Volatilité (σ)"])
    
            price_btn = st.button("💰 Calculer le Prix", type="primary")
    
        with col2:
            if price_btn:
                with st.spinner("Calcul en cours..."):
    
                    try:
                        if bond_type == "Zero-Coupon":
                            price = vasicek_zero_coupon_price(
                                r_t=r_current,
                                t=t_current,
                                T=maturity,
                                a=params['a'],
                                lam=params['lambda'],
                                sigma=params['sigma'],
                                face_value=face_value
                            )
                            st.success(f"💰 Prix de l'obligation Zero-Coupon : **{price:.2f}**")
    
                            ytm = -np.log(price / face_value) / (maturity - t_current)
                            st.info(f"📈 Rendement à l’échéance (YTM) : **{ytm:.4f} ({ytm*100:.2f}%)**")
    
                        else:
                            price = price_coupon_bond(
                                r0=r_current,
                                t=t_current,
                                a=params['a'],
                                lam=params['lambda'],
                                sigma=params['sigma'],
                                maturity=maturity,
                                face=face_value,
                                coupon=coupon_rate,
                                dt=dt_coupon
                            )
                            st.success(f"💰 Prix de l'obligation à coupons : **{price:.2f}**")
                            st.info(f"📊 Coupon : {coupon_rate*100:.2f}% ({freq})")
    
                        # -------- Sensibilité --------
                        st.subheader("📈 Analyse de Sensibilité")
    
                        fig = go.Figure()
    
                        if sensitivity_param == "Taux actuel (r)":
                            r_vals = np.linspace(max(0.001, r_current - 0.05), r_current + 0.05, 100)
                            prices = []
    
                            for r in r_vals:
                                if bond_type == "Zero-Coupon":
                                    p = vasicek_zero_coupon_price(r, t_current, maturity, params['a'], params['lambda'], params['sigma'], face_value)
                                else:
                                    p = price_coupon_bond(r, t_current, params['a'], params['lambda'], params['sigma'], maturity, face_value, coupon_rate, dt_coupon)
                                prices.append(p)
    
                            fig.add_trace(go.Scatter(x=r_vals * 100, y=prices, mode="lines", name="Prix"))
                            fig.add_vline(x=r_current * 100, line_dash="dash", line_color="red", annotation_text=f"Taux actuel: {r_current*100:.2f}%")
                            fig.update_layout(title="Sensibilité du Prix au Taux d’Intérêt", xaxis_title="Taux (%)", yaxis_title="Prix")
    
                        elif sensitivity_param == "Maturité (T)":
                            T_vals = np.linspace(t_current + 0.1, 30, 100)
                            prices = []
    
                            for T_val in T_vals:
                                if bond_type == "Zero-Coupon":
                                    p = vasicek_zero_coupon_price(r_current, t_current, T_val, params['a'], params['lambda'], params['sigma'], face_value)
                                else:
                                    p = price_coupon_bond(r_current, t_current, params['a'], params['lambda'], params['sigma'], T_val, face_value, coupon_rate, dt_coupon)
                                prices.append(p)
    
                            fig.add_trace(go.Scatter(x=T_vals, y=prices, mode="lines", name="Prix"))
                            fig.add_vline(x=maturity, line_dash="dash", line_color="red", annotation_text=f"Maturité actuelle: {maturity:.1f} ans")
                            fig.update_layout(title="Sensibilité du Prix à la Maturité", xaxis_title="Maturité (années)", yaxis_title="Prix")
    
                        elif sensitivity_param == "Volatilité (σ)":
                            sigma_vals = np.linspace(0.001, params['sigma'] * 2, 100)
                            prices = []
    
                            for sig in sigma_vals:
                                if bond_type == "Zero-Coupon":
                                    p = vasicek_zero_coupon_price(r_current, t_current, maturity, params['a'], params['lambda'], sig, face_value)
                                else:
                                    p = price_coupon_bond(r_current, t_current, params['a'], params['lambda'], sig, maturity, face_value, coupon_rate, dt_coupon)
                                prices.append(p)
    
                            fig.add_trace(go.Scatter(x=sigma_vals * 100, y=prices, mode="lines", name="Prix"))
                            fig.add_vline(x=params['sigma'] * 100, line_dash="dash", line_color="red", annotation_text=f"σ actuel: {params['sigma']*100:.2f}%")
                            fig.update_layout(title="Sensibilité du Prix à la Volatilité", xaxis_title="Volatilité (%)", yaxis_title="Prix")
    
                        st.plotly_chart(fig, use_container_width=True)
    
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Erreur lors du calcul :\n\n```\n{traceback.format_exc()}\n```")

    
    # =============================================
    # TAB 4: OPTIONS SUR OBLIGATIONS
    # =============================================
    """with tab4:
        st.header("📈 Pricing d'Options sur Obligations")
        
        if not st.session_state.vasicek_params:
            st.warning("⚠️ Veuillez d'abord estimer les paramètres dans la section 'Estimation des Paramètres'")
            st.stop()
        
        params = st.session_state.vasicek_params
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration de l'Option")
            
            option_type = st.radio("Type d'option", ["Call", "Put"])
            model_type = st.radio("Méthode de calcul", ["Analytique", "Monte Carlo"])
            
            # Paramètres de l'option
            r_current = st.number_input("Taux actuel (r)", 0.0, 0.20, params['r0'], step=0.001, format="%.4f", key="option_r_current")
            T1 = st.number_input("Échéance de l'option (T₁)", 0.1, 10.0, 1.0, step=0.1, key="option_T1")
            T2 = st.number_input("Maturité de l'obligation (T₂)", 0.1, 30.0, 5.0, step=0.1, key="option_T2")
            
            if T2 <= T1:
                st.error("⚠️ La maturité de l'obligation (T₂) doit être supérieure à l'échéance de l'option (T₁)")
                st.stop()
            
            K = st.number_input("Prix d'exercice (K)", 0.1, 2.0, 0.8, step=0.01, key="option_K")
            face_value = st.number_input("Valeur nominale de l'obligation", 100, 10000, 1000, step=100, key="option_face_value")
            
            if model_type == "Monte Carlo":
                n_paths = st.number_input("Nombre de simulations", 1000, 100000, 10000, step=1000, key="option_n_paths")
                dt_mc = st.number_input("Pas de temps MC", 0.001, 0.1, 0.01, step=0.001, key="option_dt_mc")
            
            price_option_btn = st.button("💎 Calculer le Prix de l'Option", type="primary")
        
        with col2:
            if price_option_btn:
                with st.spinner("Calcul du prix de l'option..."):
                    try:
                        if model_type == "Analytique":
                            #Utilisation de la formule analytique
                            option_price = vasicek_bond_option_price(r_current, 0, T1, T2, K, 
                                                                   params['a'], params['lambda'], params['sigma'], 
                                                                   face_value, option_type.lower())
                            
                            
                            st.success(f"💎 **Prix de l'option ({option_type}): {option_price:.4f}**")
                            
                        else:  # Monte Carlo
                            option_price, option_std = vasicek_bond_option_price_mc(r_current, params['a'], params['lambda'], 
                                                                                  params['sigma'], T1, T2, K, dt_mc, 
                                                                                  int(n_paths), face_value, option_type.lower())
                            
                            
                            st.success(f"💎 **Prix de l'option ({option_type}): {option_price:.4f} ± {option_std:.4f}**")
                            st.info(f"📊 Intervalle de confiance 95%: [{option_price - 1.96*option_std:.4f}, {option_price + 1.96*option_std:.4f}]")
                        
                        # Affichage des paramètres
                        st.subheader("📋 Résumé des Paramètres")
                        param_df = pd.DataFrame({
                            'Paramètre': ['Type d\'option', 'Méthode', 'Taux actuel (r)', 'Échéance option (T₁)', 'Maturité obligation (T₂)', 'Strike (K)', 'Valeur faciale'],
                            'Valeur': [option_type, model_type, f"{r_current:.4f}", f"{T1:.2f} ans", f"{T2:.2f} ans", f"{K:.2f}", f"{face_value}"]
                        })
                        st.table(param_df)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors du calcul: {str(e)}")"""

    #
    # =============================================
    # TAB 4: OPTIONS SUR OBLIGATIONS (CORRIGÉ AVEC `key`)
    # =============================================
    with tab4:
        st.header("📈 Pricing d'Options sur Obligations")
    
        if not st.session_state.vasicek_params:
            st.warning("⚠️ Veuillez d'abord estimer les paramètres dans l'onglet précédent.")
            st.stop()
    
        params = st.session_state.vasicek_params
    
        from pricing.models.interest_rates.analytical_vasicek import vasicek_bond_option_price as analytical_option_price
        from pricing.models.interest_rates.monte_carlo_vasicek import vasicek_bond_option_price_mc as mc_option_price
    
        col1, col2 = st.columns([1, 2])
    
        with col1:
            st.subheader("📝 Paramètres de l'Option")
    
            option_type = st.radio("Type d'option", ["Call", "Put"], key="opt_type")
            model_type = st.radio("Méthode de calcul", ["Analytique", "Monte Carlo"], key="opt_model")
    
            r_current = st.number_input("Taux actuel (r)", 0.0, 0.20, params['r0'], step=0.001, format="%.4f", key="opt_r")
            T1 = st.number_input("Échéance de l'option (T₁)", 0.1, 10.0, 1.0, step=0.1, key="opt_T1")
            T2 = st.number_input("Maturité de l'obligation (T₂)", T1 + 0.1, 30.0, 5.0, step=0.1, key="opt_T2")
    
            K = st.number_input("Prix d'exercice (K)", 0.1, 2.0, 0.8, step=0.01, key="opt_K")
            face_value = st.number_input("Valeur nominale", 100, 10000, 1000, step=100, key="opt_face")
    
            if model_type == "Monte Carlo":
                n_paths = st.number_input("Nombre de simulations", 1000, 100000, 10000, step=1000, key="opt_n_paths")
                default_dt = round(params['dt'], 3) if 'dt' in params else 0.01

                dt_mc = st.number_input(
                    "Pas de temps (dt)",
                    min_value=0.001,
                    max_value=0.1,
                    value=default_dt,
                    step=0.001,
                    format="%.3f",
                    key="opt_dt"
                )
    
            price_option_btn = st.button("💎 Calculer le Prix de l'Option", type="primary", key="opt_btn")
    
        with col2:
            if price_option_btn:
                if T2 <= T1:
                    st.error("⚠️ La maturité de l'obligation (T₂) doit être supérieure à l'échéance de l'option (T₁)")
                    st.stop()
    
                with st.spinner("Calcul du prix de l'option..."):
                    try:
                        if model_type == "Analytique":
                            price = analytical_option_price(
                                r_t=r_current,
                                t=0,
                                T1=T1,
                                T2=T2,
                                K=K,
                                a=params['a'],
                                lam=params['lambda'],
                                sigma=params['sigma'],
                                face=face_value,
                                option_type=option_type.lower()
                            )
                            st.success(f"💎 Prix de l'option {option_type} (analytique) : **{price:.4f}**")
    
                        else:
                            price, std = mc_option_price(
                                r0=r_current,
                                a=params['a'],
                                lam=params['lambda'],
                                sigma=params['sigma'],
                                T1=T1,
                                T2=T2,
                                K=K,
                                dt=dt_mc,
                                n_paths=int(n_paths),
                                face=face_value,
                                option_type=option_type.lower()
                            )
                            st.success(f"💎 Prix de l'option {option_type} (MC) : **{price:.4f} ± {std:.4f}**")
                            st.info(f"📊 Intervalle de confiance 95% : [{price - 1.96*std:.4f}, {price + 1.96*std:.4f}]")
    
                        st.subheader("📋 Récapitulatif")
                        df_params = pd.DataFrame({
                            "Paramètre": [
                                "Type d'option", "Méthode", "Taux actuel (r)", "T₁ (échéance)", "T₂ (maturité)",
                                "Prix d'exercice (K)", "Valeur nominale"
                            ],
                            "Valeur": [
                                option_type,
                                model_type,
                                f"{r_current:.4f}",
                                f"{T1:.2f} ans",
                                f"{T2:.2f} ans",
                                f"{K:.2f}",
                                f"{face_value}"
                            ]
                        })
                        st.table(df_params)
    
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Erreur :\n\n```\n{traceback.format_exc()}\n```")

    
    # =============================================
    # TAB 5: ANALYSE DES GRECQUES
    # =============================================
    """with tab5:
        st.header("🔍 Analyse des Grecques pour Options sur Obligations")
        
        if not st.session_state.vasicek_params:
            st.warning("⚠️ Veuillez d'abord estimer les paramètres dans la section 'Estimation des Paramètres'")
            st.stop()
        
        params = st.session_state.vasicek_params
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Configuration")
            
            greek_type = st.selectbox("Grecque à analyser", ["price", "delta", "vega", "rho"])
            option_type = st.radio("Type d'option", ["call", "put"])
            model_type = st.radio("Méthode", ["Analytical", "Monte Carlo"])
            
            # Paramètres
            T1 = st.number_input("Échéance option (T₁)", 0.1, 10.0, 1.0, step=0.1, key="greeks_T1")
            T2 = st.number_input("Maturité obligation (T₂)", 0.1, 30.0, 5.0, step=0.1, key="greeks_T2")
            K = st.number_input("Strike", 0.1, 2.0, 0.8, step=0.01, key="greeks_K")
            face_value = st.number_input("Valeur faciale", 100, 10000, 1000, step=100, key="greeks_face_value")
            
            if model_type == "Monte Carlo":
                n_paths_greeks = st.number_input("Simulations MC", 1000, 50000, 5000, step=1000, key="greeks_n_paths")
            
            compute_greeks_btn = st.button("📊 Calculer les Grecques", type="primary")
        
        with col2:
            if compute_greeks_btn:
                with st.spinner("Calcul des grecques..."):
                    # Simulation temporaire des grecques
                    np.random.seed(42)
                    bond_prices = np.linspace(0.5, 1.5, 100)
                    
                    if greek_type == "price":
                        greek_values = np.maximum(bond_prices - K, 0) if option_type == "call" else np.maximum(K - bond_prices, 0)
                    elif greek_type == "delta":
                        greek_values = np.where(bond_prices > K, 1, 0) if option_type == "call" else np.where(bond_prices < K, -1, 0)
                    elif greek_type == "vega":
                        greek_values = bond_prices * 0.1 * np.exp(-(bond_prices - K)**2 / 0.1)
                    else:  # rho
                        greek_values = (T1 * bond_prices) * np.exp(-(bond_prices - K)**2 / 0.1)
                    
                    # Graphique
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=bond_prices,
                        y=greek_values,
                        mode='lines',
                        name=greek_type.capitalize(),
                        line=dict(width=3)
                    ))
                    
                    fig.add_vline(x=K, line_dash="dash", line_color="red",
                                 annotation_text=f"Strike: {K}")
                    
                    fig.update_layout(
                        title=f"{greek_type.capitalize()} vs Prix de l'Obligation Sous-jacente",
                        xaxis_title="Prix de l'Obligation P(t,T₁)",
                        yaxis_title=greek_type.capitalize(),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    st.subheader("📊 Statistiques")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Valeur max", f"{np.max(greek_values):.4f}")
                    with col_stat2:
                        st.metric("Valeur min", f"{np.min(greek_values):.4f}")
                    with col_stat3:
                        st.metric("Moyenne", f"{np.mean(greek_values):.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown("*Modèle de Vasicek - Interface développée avec Streamlit*")"""
    # =============================================
    # TAB 5: ANALYSE DES GRECQUES
    # =============================================
    with tab5:
        st.header("🔍 Analyse des Grecques pour Options sur Obligations")
    
        if not st.session_state.vasicek_params:
            st.warning("⚠️ Veuillez d'abord estimer les paramètres dans l'onglet précédent.")
            st.stop()
    
        params = st.session_state.vasicek_params
    
        # Import réel des grecques
        from pricing.utils.greeks_vasicek import compute_greek_vs_spot
    
        col1, col2 = st.columns([1, 2])
    
        with col1:
            st.subheader("📋 Configuration")
    
            greek_type = st.selectbox("Grecque à analyser", ["price", "delta", "vega", "rho"], key="greek_type")
            option_type = st.radio("Type d'option", ["call", "put"], key="greek_opt_type")
            model_type = st.radio("Méthode", ["Analytical", "Monte Carlo"], key="greek_model")
    
            T1 = st.number_input("Échéance option (T₁)", 0.1, 10.0, 1.0, step=0.1, key="greek_T1")
            T2 = st.number_input("Maturité obligation (T₂)", T1 + 0.1, 30.0, 5.0, step=0.1, key="greek_T2")
    
            K = st.number_input("Prix d'exercice (K)", 0.1, 2.0, 0.8, step=0.01, key="greek_K")
            face_value = st.number_input("Valeur nominale", 100, 10000, 1000, step=100, key="greek_face")
    
            # Suggérer le dt de Tab 1
            default_dt = round(params['dt'], 3) if 'dt' in params else 0.01
    
            if model_type == "Monte Carlo":
                n_paths = st.number_input("Nombre de simulations MC", 1000, 50000, 5000, step=1000, key="greek_npaths")
                dt = st.number_input("Pas de temps MC (dt)", 0.001, 0.1, default_dt, step=0.001, format="%.3f", key="greek_dt")
            else:
                dt = default_dt
                n_paths = 10000  # valeur par défaut pour analytique, ignorée
    
            compute_btn = st.button("📊 Calculer les Grecques", type="primary", key="greek_btn")
    
        with col2:
            if compute_btn:
                with st.spinner("Calcul des grecques en cours..."):
    
                    try:
                        fig = compute_greek_vs_spot(
                            greek=greek_type,
                            t=0,
                            T1=T1,
                            T2=T2,
                            K=K,
                            a=params['a'],
                            lam=params['lambda'],
                            sigma=params['sigma'],
                            face=face_value,
                            dt=dt
                            option_type=option_type,
                            n_paths=n_paths,
                            model=model_type,
                        )
                        
                        st.pyplot(fig)

    
                    except Exception as e:
                        import traceback
                        st.error(f"❌ Erreur :\n\n```\n{traceback.format_exc()}\n```")

