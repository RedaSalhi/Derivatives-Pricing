# tabs/interest_rate_instruments.py
# Interest Rate Instruments Tab - Tab 6

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date

# Import your pricing functions
from pricing.models.interest_rates.analytical_vasicek import *
from pricing.models.interest_rates.monte_carlo_vasicek import *
from pricing.utils.greeks_vasicek import *


def interest_rate_instruments_tab():
    """Interest Rate Instruments Tab Content"""
    
    st.markdown('<div class="main-header">Interest Rate Model Selector</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Model selection
    model = st.selectbox(
        "Choose a model to explore:",
        ["Vasicek Model", "Hull-White Model (Coming Soon)", "Cox-Ingersoll-Ross (CIR) (Coming Soon)"],
        index=0,
        help="Select the interest rate model you want to explore."
    )

    if model != "Vasicek Model":
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("🚧 This model is not yet available. Stay tuned!")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Vasicek Model Interface
    st.markdown('<div class="sub-header">Vasicek Model – Bond Pricing and Interest Rates</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state for estimated parameters
    if 'vasicek_params' not in st.session_state:
        st.session_state.vasicek_params = None
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Parameter Estimation", "Simulation & Yield Curves", "Bond Pricing", "Bond Options", "Greeks Analysis"]
    )
    
    with tab1:
        _parameter_estimation_tab()
    
    with tab2:
        _simulation_yield_curves_tab()
    
    with tab3:
        _bond_pricing_tab()
    
    with tab4:
        _bond_options_tab()
    
    with tab5:
        _greeks_analysis_tab()


def _parameter_estimation_tab():
    """Parameter Estimation Tab"""
    st.markdown('<div class="sub-header">Vasicek Model Parameter Estimation</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data Configuration")

        # Ticker input
        ticker = st.text_input(
            "Enter a FRED or Yahoo ticker (e.g., DGS10, DFF, ^IRX)",
            value="DGS10",
            help="Examples: DGS10 (US 10Y), DGS2 (2Y), DFF (Fed Funds), ^IRX (Yahoo 13W T-Bill)"
        )

        # Default to the last 5 years
        today = date.today()
        default_start = today.replace(year=today.year - 5)

        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", today)

        # Resampling frequency
        freq = st.selectbox("Resampling frequency", ["ME", "QE", "YE"], index=0)

        # Trigger estimation
        if st.button("Estimate Parameters", type="primary"):
            if start_date >= end_date:
                st.error("❌ Start date must be before end date.")
            else:
                with st.spinner("Loading data and estimating..."):
                    try:
                        a, lam, sigma, dt, r0 = run_ou_estimation(ticker, str(start_date), str(end_date), freq)

                        st.session_state.vasicek_params = {
                            'a': a, 'lambda': lam, 'sigma': sigma, 'dt': dt, 'r0': r0, 'ticker': ticker
                        }
                        st.success("✅ Parameters successfully estimated!")

                    except Exception as e:
                        import traceback
                        st.error(f"❌ Error during estimation:\n\n```\n{traceback.format_exc()}\n```")

    with col2:
        st.subheader("Estimated Parameters")
        if st.session_state.vasicek_params:
            params = st.session_state.vasicek_params

            col_a, col_lam, col_sig = st.columns(3)
            with col_a:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Speed of mean reversion (a)", f"{params['a']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_lam:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Long-term mean level (λ)", f"{params['lambda']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_sig:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Volatility (σ)", f"{params['sigma']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Initial rate (r₀)", f"{params['r0']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"Ticker used: **{params['ticker']}** | Δt: {params['dt']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("Click 'Estimate Parameters' to get started")
            st.markdown('</div>', unsafe_allow_html=True)


def _simulation_yield_curves_tab():
    """Simulation and Yield Curves Tab"""
    st.markdown('<div class="sub-header">Simulation of Rate Paths and Yield Curves (Vasicek)</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please estimate the parameters in the previous tab first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Simulation Parameters")

        T = st.slider("Time horizon (years)", min_value=1, max_value=30, value=10)
        dt = st.slider("Time step (dt)", min_value=0.01, max_value=1.0, value=float(params["dt"]), step=0.01)
        n_paths = st.slider("Number of simulated paths", 100, 10000, 1000, step=100)

        st.subheader("Yield Curve Configuration")

        available_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        default_maturities = [m for m in [1, 2, 5, 10] if m <= T]
        maturities = st.multiselect("Maturities (years)", options=available_maturities, default=default_maturities)

        # Generate readable and valid snapshot times
        max_snapshots = int(T / dt)
        raw_snapshots = [round(i * dt, 2) for i in range(max_snapshots + 1)]

        # Format for display
        labelled_snapshots = {f"{s:.2f} years": s for s in raw_snapshots}

        # User selection (pretty labels, float values)
        default_keys = [k for k in labelled_snapshots if float(k.split()[0]) in [0.0, 2.0, 5.0, 10.0]]
        selected_keys = st.multiselect("Snapshot times (years)", options=list(labelled_snapshots.keys()), default=default_keys)

        # Convert for technical use
        snapshot_times = [labelled_snapshots[k] for k in selected_keys]

        simulate_btn = st.button("Run Simulation", type="primary")

    with col2:
        if simulate_btn:
            with st.spinner("Running simulation..."):

                # Run the simulation
                time_vec, r_paths = simulate_vasicek_paths(
                    a=params['a'],
                    lam=params['lambda'],
                    sigma=params['sigma'],
                    r0=params['r0'],
                    T=T,
                    dt=dt,
                    n_paths=n_paths
                )

                # Yield curves: average over all paths
                yield_curves = generate_yield_curves(
                    r_path=np.mean(r_paths, axis=1),
                    snapshot_times=snapshot_times,
                    maturities=maturities,
                    a=params['a'],
                    theta=params['lambda'],
                    sigma=params['sigma'],
                    dt=dt
                )

                st.pyplot(plot_yield_curves(yield_curves, maturities))

                # Final short rate distribution
                r_final = r_paths[-1, :]
                fig_hist = px.histogram(
                    r_final,
                    nbins=50,
                    title="Final Short Rate Distribution",
                    labels={'value': 'Rate', 'count': 'Frequency'}
                )
                fig_hist.add_vline(x=np.mean(r_final), line_dash="dash", line_color="red",
                                   annotation_text=f"Mean: {np.mean(r_final):.4f}")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Descriptive statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Final Mean", f"{np.mean(r_final):.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_stat2:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Standard Deviation", f"{np.std(r_final):.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_stat3:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Min / Max", f"{np.min(r_final):.4f} / {np.max(r_final):.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)


def _bond_pricing_tab():
    """Bond Pricing Tab"""
    st.markdown('<div class="sub-header">Bond Pricing (Zero-Coupon or Coupon Bonds)</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please estimate the parameters in the previous tab first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Bond Parameters")

        bond_type = st.radio("Bond type", ["Zero-Coupon", "With Coupons"])

        r_current = st.number_input("Current interest rate (r)", min_value=0.0, max_value=0.20, value=params['r0'], step=0.001, format="%.4f")
        t_current = st.number_input("Current time (t)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
        maturity = st.number_input("Maturity (T)", min_value=t_current + 0.1, max_value=30.0, value=5.0, step=0.1)
        face_value = st.number_input("Face value", min_value=100, max_value=10000, value=100, step=10)

        if bond_type == "With Coupons":
            coupon_rate = st.number_input("Coupon rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
            freq = st.selectbox("Payment frequency", ["Annual", "Semi-Annual"])
            dt_coupon = 1.0 if freq == "Annual" else 0.5

        st.subheader("Sensitivity Analysis")
        sensitivity_param = st.selectbox("Parameter to test", ["Current rate (r)", "Maturity (T)", "Volatility (σ)"])

        price_btn = st.button("Compute Price", type="primary")

    with col2:
        if price_btn:
            with st.spinner("Calculating..."):

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
                        
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.success(f"Zero-Coupon Bond Price: **${price:.2f}**")
                        st.markdown('</div>', unsafe_allow_html=True)

                        ytm = -np.log(price / face_value) / (maturity - t_current)
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown(f"Yield to Maturity (YTM): **{ytm:.4f} ({ytm*100:.2f}%)**")
                        st.markdown('</div>', unsafe_allow_html=True)

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
                        
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.success(f"Coupon Bond Price: **${price:.2f}**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown(f"Coupon: {coupon_rate*100:.2f}% ({freq})")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Sensitivity Analysis
                    st.markdown('<div class="sub-header">Sensitivity Analysis</div>', unsafe_allow_html=True)

                    fig = go.Figure()

                    if sensitivity_param == "Current rate (r)":
                        r_vals = np.linspace(max(0.001, r_current - 0.05), r_current + 0.05, 100)
                        prices = []

                        for r in r_vals:
                            if bond_type == "Zero-Coupon":
                                p = vasicek_zero_coupon_price(r, t_current, maturity, params['a'], params['lambda'], params['sigma'], face_value)
                            else:
                                p = price_coupon_bond(r, t_current, params['a'], params['lambda'], params['sigma'], maturity, face_value, coupon_rate, dt_coupon)
                            prices.append(p)

                        fig.add_trace(go.Scatter(x=r_vals * 100, y=prices, mode="lines", name="Price"))
                        fig.add_vline(x=r_current * 100, line_dash="dash", line_color="red", annotation_text=f"Current rate: {r_current*100:.2f}%")
                        fig.update_layout(title="Price Sensitivity to Interest Rate", xaxis_title="Rate (%)", yaxis_title="Price")

                    elif sensitivity_param == "Maturity (T)":
                        T_vals = np.linspace(t_current + 0.1, 30, 100)
                        prices = []

                        for T_val in T_vals:
                            if bond_type == "Zero-Coupon":
                                p = vasicek_zero_coupon_price(r_current, t_current, T_val, params['a'], params['lambda'], params['sigma'], face_value)
                            else:
                                p = price_coupon_bond(r_current, t_current, params['a'], params['lambda'], params['sigma'], T_val, face_value, coupon_rate, dt_coupon)
                            prices.append(p)

                        fig.add_trace(go.Scatter(x=T_vals, y=prices, mode="lines", name="Price"))
                        fig.add_vline(x=maturity, line_dash="dash", line_color="red", annotation_text=f"Current maturity: {maturity:.1f} years")
                        fig.update_layout(title="Price Sensitivity to Maturity", xaxis_title="Maturity (years)", yaxis_title="Price")

                    elif sensitivity_param == "Volatility (σ)":
                        sigma_vals = np.linspace(0.001, params['sigma'] * 2, 100)
                        prices = []

                        for sig in sigma_vals:
                            if bond_type == "Zero-Coupon":
                                p = vasicek_zero_coupon_price(r_current, t_current, maturity, params['a'], params['lambda'], sig, face_value)
                            else:
                                p = price_coupon_bond(r_current, t_current, params['a'], params['lambda'], sig, maturity, face_value, coupon_rate, dt_coupon)
                            prices.append(p)

                        fig.add_trace(go.Scatter(x=sigma_vals * 100, y=prices, mode="lines", name="Price"))
                        fig.add_vline(x=params['sigma'] * 100, line_dash="dash", line_color="red", annotation_text=f"Current σ: {params['sigma']*100:.2f}%")
                        fig.update_layout(title="Price Sensitivity to Volatility", xaxis_title="Volatility (%)", yaxis_title="Price")

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    import traceback
                    st.error(f"❌ Error during calculation:\n\n```\n{traceback.format_exc()}\n```")


def _bond_options_tab():
    """Bond Options Pricing Tab"""
    st.markdown('<div class="sub-header">Bond Option Pricing</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please estimate the parameters in the previous tab first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    from pricing.models.interest_rates.analytical_vasicek import vasicek_bond_option_price as analytical_option_price
    from pricing.models.interest_rates.monte_carlo_vasicek import vasicek_bond_option_price_mc as mc_option_price

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Option Parameters")

        option_type = st.radio("Option type", ["Call", "Put"], key="opt_type")
        model_type = st.radio("Calculation method", ["Analytical", "Monte Carlo"], key="opt_model")

        r_current = st.number_input("Current rate (r)", 0.0, 0.20, params['r0'], step=0.001, format="%.4f", key="opt_r")
        T1 = st.number_input("Option maturity (T₁)", 0.1, 10.0, 1.0, step=0.1, key="opt_T1")
        T2 = st.number_input("Bond maturity (T₂)", T1 + 0.1, 30.0, 5.0, step=0.1, key="opt_T2")

        K = st.number_input("Strike price (K)", 0.1, 2.0, 0.8, step=0.01, key="opt_K")
        face_value = st.number_input("Face value", 100, 10000, 1000, step=100, key="opt_face")

        if model_type == "Monte Carlo":
            n_paths = st.number_input("Number of simulations", 1000, 100000, 10000, step=1000, key="opt_n_paths")
            default_dt = round(params['dt'], 3) if 'dt' in params else 0.01

            dt_mc = st.number_input(
                "Time step (dt)",
                min_value=0.001,
                max_value=0.1,
                value=default_dt,
                step=0.001,
                format="%.3f",
                key="opt_dt"
            )

        price_option_btn = st.button("Compute Option Price", type="primary", key="opt_btn")

    with col2:
        if price_option_btn:
            if T2 <= T1:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("⚠️ The bond maturity (T₂) must be greater than the option maturity (T₁).")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            with st.spinner("Computing option price..."):
                try:
                    if model_type == "Analytical":
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
                        
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.success(f"{option_type} Option Price (Analytical): **{price:.4f}**")
                        st.markdown('</div>', unsafe_allow_html=True)

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
                        
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.success(f"{option_type} Option Price (Monte Carlo): **{price:.4f} ± {std:.4f}**")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown(f"95% Confidence Interval: [{price - 1.96*std:.4f}, {price + 1.96*std:.4f}]")
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="sub-header">Summary</div>', unsafe_allow_html=True)
                    df_params = pd.DataFrame({
                        "Parameter": [
                            "Option Type", "Method", "Current Rate (r)", "T₁ (Option Maturity)", "T₂ (Bond Maturity)",
                            "Strike Price (K)", "Face Value"
                        ],
                        "Value": [
                            option_type,
                            model_type,
                            f"{r_current:.4f}",
                            f"{T1:.2f} years",
                            f"{T2:.2f} years",
                            f"{K:.2f}",
                            f"{face_value}"
                        ]
                    })
                    st.table(df_params)

                except Exception as e:
                    import traceback
                    st.error(f"❌ Error:\n\n```\n{traceback.format_exc()}\n```")


def _greeks_analysis_tab():
    """Greeks Analysis Tab"""
    st.markdown('<div class="sub-header">Greeks Analysis for Bond Options</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please estimate the parameters in the previous tab first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    # Import Greeks computation
    from pricing.utils.greeks_vasicek import compute_greek_vs_spot

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")

        greek_type = st.selectbox("Greek to analyze", ["price", "delta", "vega", "rho"], key="greek_type")
        option_type = st.radio("Option type", ["call", "put"], key="greek_opt_type")
        model_type = st.radio("Calculation method", ["Analytical", "Monte Carlo"], key="greek_model")

        T1 = st.number_input("Option maturity (T₁)", 0.1, 10.0, 1.0, step=0.1, key="greek_T1")
        T2 = st.number_input("Bond maturity (T₂)", T1 + 0.1, 30.0, 5.0, step=0.1, key="greek_T2")

        K = st.number_input("Strike price (K)", 0.1, 2.0, 0.8, step=0.01, key="greek_K")
        face_value = st.number_input("Face value", 100, 10000, 1000, step=100, key="greek_face")

        # Suggest dt from Tab 1
        default_dt = round(params['dt'], 3) if 'dt' in params else 0.01

        if model_type == "Monte Carlo":
            n_paths = st.number_input("Number of Monte Carlo simulations", 1000, 50000, 5000, step=1000, key="greek_npaths")
            dt = st.number_input("Time step (MC dt)", 0.001, 0.1, default_dt, step=0.001, format="%.3f", key="greek_dt")
        else:
            dt = default_dt
            n_paths = 10000  # default value for analytical, ignored

        compute_btn = st.button("Compute Greeks", type="primary", key="greek_btn")

    with col2:
        if compute_btn:
            with st.spinner("Computing Greeks..."):

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
                        dt=dt,
                        option_type=option_type,
                        n_paths=n_paths,
                        model=model_type,
                    )
                    
                    st.pyplot(fig)

                except Exception as e:
                    import traceback
                    st.error(f"❌ Error:\n\n```\n{traceback.format_exc()}\n```")
    
    # Educational content
    st.markdown("---")
    st.markdown('<div class="sub-header">Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("📚 Understanding the Vasicek Model"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### The Vasicek Interest Rate Model
        
        The Vasicek model is a mathematical model describing the evolution of interest rates. It is a type of "one-factor short-rate model" as it describes interest rate movements as driven by only one source of market risk.
        
        **Model Equation:**
        ```
        dr(t) = a(λ - r(t))dt + σ dW(t)
        ```
        
        **Parameters:**
        - **a**: Speed of mean reversion (how quickly rates return to long-term mean)
        - **λ**: Long-term mean level of interest rates
        - **σ**: Volatility of interest rate changes
        - **r(t)**: Short-term interest rate at time t
        - **dW(t)**: Wiener process (random component)
        
        **Key Features:**
        - **Mean Reversion**: Rates tend to drift back toward the long-term mean
        - **Analytical Solutions**: Closed-form formulas for bond prices and options
        - **Negative Rates**: Model allows for negative interest rates
        - **Normal Distribution**: Rate changes are normally distributed
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🏦 Bond Pricing Formulas"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Vasicek Bond Pricing Formulas
        
        **Zero-Coupon Bond Price:**
        ```
        P(t,T) = A(t,T) × exp(-B(t,T) × r(t))
        ```
        
        Where:
        ```
        B(t,T) = (1 - exp(-a(T-t))) / a
        A(t,T) = exp((λ - σ²/2a²)(B(t,T) - T + t) - σ²B(t,T)²/4a)
        ```
        
        **Coupon Bond Price:**
        Sum of discounted coupon payments plus principal:
        ```
        P = Σ C × P(t,Ti) + Face × P(t,T)
        ```
        
        **Bond Option Price:**
        Uses the Black-Scholes formula adapted for bonds:
        ```
        Call = P(t,T₁) × N(d₁) - K × P(t,T₂) × N(d₂)
        Put = K × P(t,T₂) × N(-d₂) - P(t,T₁) × N(-d₁)
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Greeks for Bond Options"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Bond Option Greeks
        
        **Delta (Δ)**: Sensitivity to changes in the underlying bond price
        - Measures how much the option price changes for a $1 change in bond price
        - Range: 0 to 1 for calls, -1 to 0 for puts
        
        **Rho (ρ)**: Sensitivity to changes in interest rates
        - Measures how much the option price changes for a 1% change in rates
        - More important for bond options than equity options
        - **Negative** for calls (higher rates → lower bond prices → lower call values)
        
        **Vega (ν)**: Sensitivity to changes in interest rate volatility
        - Measures how much the option price changes for a 1% change in volatility
        - **Positive** for both calls and puts (higher volatility → higher option values)
        
        **Theta (Θ)**: Time decay
        - Measures how much the option price changes as time passes
        - Usually **negative** (options lose value as expiration approaches)
        
        **Gamma (Γ)**: Rate of change of delta
        - Measures the convexity of the option price
        - Highest for **at-the-money** options near expiration
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("⚠️ Model Limitations & Considerations"):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Vasicek Model Limitations
        
        **Theoretical Limitations:**
        - **Negative Rates**: Model allows unrealistic negative rates (though less problematic now)
        - **Constant Parameters**: Assumes constant mean reversion speed and volatility
        - **Normal Distribution**: Real rate changes may have fat tails
        - **Single Factor**: Ignores multiple sources of interest rate risk
        
        **Practical Considerations:**
        - **Parameter Estimation**: Historical data may not reflect future behavior
        - **Calibration**: Model may not fit current market prices perfectly
        - **Volatility Clustering**: Real rates show periods of high/low volatility
        - **Regime Changes**: Central bank policy changes can break model assumptions
        
        **Risk Management:**
        - Use multiple models for comparison
        - Regular recalibration with fresh data
        - Stress testing with extreme scenarios
        - Consider model uncertainty in risk measures
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p><strong>Vasicek Interest Rate Model</strong> | Built with Streamlit & Python</p>
        <p>⚠️ For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)
