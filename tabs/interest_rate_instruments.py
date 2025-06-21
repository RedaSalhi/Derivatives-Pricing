# tabs/interest_rate_instruments.py
# Interest Rate Instruments Tab - Tab 6

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from styles.app_styles import load_theme

# Import your pricing functions
from pricing.models.interest_rates.analytical_vasicek import *
from pricing.models.interest_rates.monte_carlo_vasicek import *
from pricing.utils.greeks_vasicek import *


def interest_rate_instruments_tab():
    """Interest Rate Instruments Tab Content"""

    load_theme()
    
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
        st.markdown("""
        <div class="warning-box">
            <h4>🚧 Coming Soon</h4>
            <p>This model is not yet available. Stay tuned for future updates!</p>
        </div>
        """, unsafe_allow_html=True)
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
        st.markdown("""
        <div class="info-box">
            <h4>Data Configuration</h4>
        </div>
        """, unsafe_allow_html=True)

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
                st.markdown("""
                <div class="warning-box">
                    <h4>❌ Invalid Date Range</h4>
                    <p>Start date must be before end date.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Loading data and estimating..."):
                    try:
                        a, lam, sigma, dt, r0 = run_ou_estimation(ticker, str(start_date), str(end_date), freq)

                        st.session_state.vasicek_params = {
                            'a': a, 'lambda': lam, 'sigma': sigma, 'dt': dt, 'r0': r0, 'ticker': ticker
                        }
                        st.markdown("""
                        <div class="info-box">
                            <h4>✅ Success!</h4>
                            <p>Parameters successfully estimated!</p>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        import traceback
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>❌ Estimation Error</h4>
                            <p>Error during estimation:</p>
                            <pre>{traceback.format_exc()}</pre>
                        </div>
                        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Estimated Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.vasicek_params:
            params = st.session_state.vasicek_params

            st.markdown(f"""
            <div class="metric-container">
                <h4>Vasicek Model Parameters</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                        <td style="padding: 10px; font-weight: bold;">Parameter</td>
                        <td style="padding: 10px; font-weight: bold;">Symbol</td>
                        <td style="padding: 10px; font-weight: bold;">Value</td>
                        <td style="padding: 10px; font-weight: bold;">Description</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px;">Speed of Reversion</td>
                        <td style="padding: 10px; font-weight: bold; color: #1f77b4;">a</td>
                        <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{params['a']:.4f}</td>
                        <td style="padding: 10px; font-size: 0.9em; font-style: italic;">How quickly rates return to mean</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px;">Long-term Mean</td>
                        <td style="padding: 10px; font-weight: bold; color: #1f77b4;">λ</td>
                        <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{params['lambda']:.4f}</td>
                        <td style="padding: 10px; font-size: 0.9em; font-style: italic;">Target interest rate level</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px;">Volatility</td>
                        <td style="padding: 10px; font-weight: bold; color: #1f77b4;">σ</td>
                        <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{params['sigma']:.4f}</td>
                        <td style="padding: 10px; font-size: 0.9em; font-style: italic;">Standard deviation of changes</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px;">Initial Rate</td>
                        <td style="padding: 10px; font-weight: bold; color: #1f77b4;">r₀</td>
                        <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{params['r0']:.4f}</td>
                        <td style="padding: 10px; font-size: 0.9em; font-style: italic;">Starting interest rate</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px;">Time Step</td>
                        <td style="padding: 10px; font-weight: bold; color: #1f77b4;">Δt</td>
                        <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{params['dt']:.4f}</td>
                        <td style="padding: 10px; font-size: 0.9em; font-style: italic;">Observation frequency</td>
                    </tr>
                </table>
                <br>
                <p style="text-align: center; font-weight: bold; color: #1f77b4; font-size: 1.1em;">
                    Data Source: {params['ticker']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>👆 Get Started</h4>
                <p>Click <strong>'Estimate Parameters'</strong> to begin analyzing interest rate data.</p>
                <br>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #1f77b4;">
                    <p style="margin: 0;"><em>The Vasicek model follows the stochastic differential equation:</em></p>
                    <p style="text-align: center; font-size: 1.2em; font-weight: bold; color: #1f77b4; margin: 10px 0;">
                        dr(t) = a(λ - r(t))dt + σ dW(t)
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)


def _simulation_yield_curves_tab():
    """Simulation and Yield Curves Tab"""
    st.markdown('<div class="sub-header">Simulation of Rate Paths and Yield Curves (Vasicek)</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Parameters Required</h4>
            <p>Please estimate the parameters in the <strong>Parameter Estimation</strong> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Simulation Parameters</h4>
        </div>
        """, unsafe_allow_html=True)

        T = st.slider("Time horizon (years)", min_value=1, max_value=30, value=10)
        dt = st.slider("Time step (Δt)", min_value=0.01, max_value=1.0, value=float(params["dt"]), step=0.01)
        n_paths = st.slider("Number of simulated paths", 100, 10000, 1000, step=100)

        st.markdown("""
        <div class="info-box">
            <h4>Yield Curve Configuration</h4>
        </div>
        """, unsafe_allow_html=True)

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
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Final Rate Statistics</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                            <td style="padding: 12px; font-weight: bold;">Statistic</td>
                            <td style="padding: 12px; font-weight: bold;">Value</td>
                            <td style="padding: 12px; font-weight: bold;">Formula</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 10px; font-weight: bold;">Mean (r̄)</td>
                            <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{np.mean(r_final):.4f}</td>
                            <td style="padding: 10px; font-style: italic;">(1/n) × Σ rᵢ</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 10px; font-weight: bold;">Std Dev (σᵣ)</td>
                            <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{np.std(r_final):.4f}</td>
                            <td style="padding: 10px; font-style: italic;">√[(1/n-1) × Σ(rᵢ - r̄)²]</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 10px; font-weight: bold;">Minimum</td>
                            <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #FF6347;">{np.min(r_final):.4f}</td>
                            <td style="padding: 10px; font-style: italic;">min(r₁, r₂, ..., rₙ)</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; font-weight: bold;">Maximum</td>
                            <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #FF6347;">{np.max(r_final):.4f}</td>
                            <td style="padding: 10px; font-style: italic;">max(r₁, r₂, ..., rₙ)</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)


def _bond_pricing_tab():
    """Bond Pricing Tab"""
    st.markdown('<div class="sub-header">Bond Pricing (Zero-Coupon or Coupon Bonds)</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Parameters Required</h4>
            <p>Please estimate the parameters in the <strong>Parameter Estimation</strong> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Bond Parameters</h4>
        </div>
        """, unsafe_allow_html=True)

        bond_type = st.radio("Bond type", ["Zero-Coupon", "With Coupons"])

        r_current = st.number_input("Current interest rate (r)", min_value=0.0, max_value=0.20, value=params['r0'], step=0.001, format="%.4f")
        t_current = st.number_input("Current time (t)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
        maturity = st.number_input("Maturity (T)", min_value=t_current + 0.1, max_value=30.0, value=5.0, step=0.1)
        face_value = st.number_input("Face value", min_value=100, max_value=10000, value=100, step=10)

        if bond_type == "With Coupons":
            coupon_rate = st.number_input("Coupon rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
            freq = st.selectbox("Payment frequency", ["Annual", "Semi-Annual"])
            dt_coupon = 1.0 if freq == "Annual" else 0.5

        st.markdown("""
        <div class="info-box">
            <h4>Sensitivity Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        
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
                        
                        ytm = -np.log(price / face_value) / (maturity - t_current)
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>Zero-Coupon Bond Pricing Results</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                                    <td style="padding: 12px; font-weight: bold;">Value</td>
                                    <td style="padding: 12px; font-weight: bold;">Formula</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Bond Price</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold; font-size: 1.2em;">${price:.2f}</td>
                                    <td style="padding: 10px; font-style: italic;">P(t,T) = A(t,T) × exp(-B(t,T) × r(t))</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; font-weight: bold;">Yield to Maturity</td>
                                    <td style="padding: 10px; font-family: monospace; color: #FF6347; font-weight: bold;">{ytm:.4f} ({ytm*100:.2f}%)</td>
                                    <td style="padding: 10px; font-style: italic;">YTM = -ln(P/F) / (T-t)</td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

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
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>Coupon Bond Pricing Results</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                                    <td style="padding: 12px; font-weight: bold;">Value</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Bond Price</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold; font-size: 1.2em;">${price:.2f}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Coupon Rate</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">{coupon_rate*100:.2f}%</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; font-weight: bold;">Payment Frequency</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">{freq}</td>
                                </tr>
                            </table>
                            <br>
                            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center;">
                                <p style="margin: 0; font-style: italic; color: #1f77b4;">
                                    <strong>Formula:</strong> P = Σ C × P(t,Tᵢ) + F × P(t,T)
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

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

                        fig.add_trace(go.Scatter(x=r_vals * 100, y=prices, mode="lines", name="Price", line=dict(width=3, color="#1f77b4")))
                        fig.add_vline(x=r_current * 100, line_dash="dash", line_color="red", annotation_text=f"Current rate: {r_current*100:.2f}%")
                        fig.update_layout(title="Price Sensitivity to Interest Rate", xaxis_title="Rate (%)", yaxis_title="Price ($)")

                    elif sensitivity_param == "Maturity (T)":
                        T_vals = np.linspace(t_current + 0.1, 30, 100)
                        prices = []

                        for T_val in T_vals:
                            if bond_type == "Zero-Coupon":
                                p = vasicek_zero_coupon_price(r_current, t_current, T_val, params['a'], params['lambda'], params['sigma'], face_value)
                            else:
                                p = price_coupon_bond(r_current, t_current, params['a'], params['lambda'], params['sigma'], T_val, face_value, coupon_rate, dt_coupon)
                            prices.append(p)

                        fig.add_trace(go.Scatter(x=T_vals, y=prices, mode="lines", name="Price", line=dict(width=3, color="#1f77b4")))
                        fig.add_vline(x=maturity, line_dash="dash", line_color="red", annotation_text=f"Current maturity: {maturity:.1f} years")
                        fig.update_layout(title="Price Sensitivity to Maturity", xaxis_title="Maturity (years)", yaxis_title="Price ($)")

                    elif sensitivity_param == "Volatility (σ)":
                        sigma_vals = np.linspace(0.001, params['sigma'] * 2, 100)
                        prices = []

                        for sig in sigma_vals:
                            if bond_type == "Zero-Coupon":
                                p = vasicek_zero_coupon_price(r_current, t_current, maturity, params['a'], params['lambda'], sig, face_value)
                            else:
                                p = price_coupon_bond(r_current, t_current, params['a'], params['lambda'], sig, maturity, face_value, coupon_rate, dt_coupon)
                            prices.append(p)

                        fig.add_trace(go.Scatter(x=sigma_vals * 100, y=prices, mode="lines", name="Price", line=dict(width=3, color="#1f77b4")))
                        fig.add_vline(x=params['sigma'] * 100, line_dash="dash", line_color="red", annotation_text=f"Current σ: {params['sigma']*100:.2f}%")
                        fig.update_layout(title="Price Sensitivity to Volatility", xaxis_title="Volatility (%)", yaxis_title="Price ($)")

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    import traceback
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Calculation Error</h4>
                        <pre>{traceback.format_exc()}</pre>
                    </div>
                    """, unsafe_allow_html=True)


def _bond_options_tab():
    """Bond Options Pricing Tab"""
    st.markdown('<div class="sub-header">Bond Option Pricing</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Parameters Required</h4>
            <p>Please estimate the parameters in the <strong>Parameter Estimation</strong> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    from pricing.models.interest_rates.analytical_vasicek import vasicek_bond_option_price as analytical_option_price
    from pricing.models.interest_rates.monte_carlo_vasicek import vasicek_bond_option_price_mc as mc_option_price

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Option Parameters</h4>
        </div>
        """, unsafe_allow_html=True)

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
                "Time step (Δt)",
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
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Invalid Maturity Structure</h4>
                    <p>The bond maturity (T₂) must be greater than the option maturity (T₁).</p>
                </div>
                """, unsafe_allow_html=True)
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
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{option_type} Option Pricing Results (Analytical)</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                                    <td style="padding: 12px; font-weight: bold;">Value</td>
                                    <td style="padding: 12px; font-weight: bold;">Formula</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; font-weight: bold;">Option Price</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold; font-size: 1.2em;">{price:.4f}</td>
                                    <td style="padding: 10px; font-style: italic;">P(t,T₁) × N(d₁) - K × P(t,T₂) × N(d₂)</td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

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
                        
                        ci_lower = price - 1.96*std
                        ci_upper = price + 1.96*std
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{option_type} Option Pricing Results (Monte Carlo)</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                                    <td style="padding: 12px; font-weight: bold;">Value</td>
                                    <td style="padding: 12px; font-weight: bold;">Formula</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Option Price</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold; font-size: 1.2em;">{price:.4f}</td>
                                    <td style="padding: 10px; font-style: italic;">(1/N) × Σ max(Sᵢ - K, 0)</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Standard Error</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">± {std:.4f}</td>
                                    <td style="padding: 10px; font-style: italic;">σ / √N</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; font-weight: bold;">95% Confidence Interval</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">[{ci_lower:.4f}, {ci_upper:.4f}]</td>
                                    <td style="padding: 10px; font-style: italic;">X̄ ± 1.96 × SE</td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                    # Summary parameters table
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Option Configuration Summary</h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                <td style="padding: 10px; font-weight: bold;">Parameter</td>
                                <td style="padding: 10px; font-weight: bold;">Symbol</td>
                                <td style="padding: 10px; font-weight: bold;">Value</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Option Type</td>
                                <td style="padding: 8px;">-</td>
                                <td style="padding: 8px; font-weight: bold; color: #1f77b4;">{option_type}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Calculation Method</td>
                                <td style="padding: 8px;">-</td>
                                <td style="padding: 8px; font-weight: bold; color: #1f77b4;">{model_type}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Current Rate</td>
                                <td style="padding: 8px; font-weight: bold;">r</td>
                                <td style="padding: 8px; font-family: monospace;">{r_current:.4f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Option Maturity</td>
                                <td style="padding: 8px; font-weight: bold;">T₁</td>
                                <td style="padding: 8px; font-family: monospace;">{T1:.2f} years</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Bond Maturity</td>
                                <td style="padding: 8px; font-weight: bold;">T₂</td>
                                <td style="padding: 8px; font-family: monospace;">{T2:.2f} years</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Strike Price</td>
                                <td style="padding: 8px; font-weight: bold;">K</td>
                                <td style="padding: 8px; font-family: monospace;">{K:.2f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px;">Face Value</td>
                                <td style="padding: 8px; font-weight: bold;">F</td>
                                <td style="padding: 8px; font-family: monospace;">{face_value}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    import traceback
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Calculation Error</h4>
                        <pre>{traceback.format_exc()}</pre>
                    </div>
                    """, unsafe_allow_html=True)


def _greeks_analysis_tab():
    """Greeks Analysis Tab"""
    st.markdown('<div class="sub-header">Greeks Analysis for Bond Options</div>', unsafe_allow_html=True)

    if not st.session_state.vasicek_params:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Parameters Required</h4>
            <p>Please estimate the parameters in the <strong>Parameter Estimation</strong> tab first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    params = st.session_state.vasicek_params

    # Import Greeks computation
    from pricing.utils.greeks_vasicek import compute_greek_vs_spot

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Greeks Configuration</h4>
        </div>
        """, unsafe_allow_html=True)

        greek_type = st.selectbox("Greek to analyze", ["price", "delta", "vega", "rho"], key="ir_greek_type")
        option_type = st.radio("Option type", ["call", "put"], key="greek_opt_type")
        model_type = st.radio("Calculation method", ["Analytical", "Monte Carlo"], key="greek_model")

        T1 = st.number_input("Option maturity (T₁)", 0.1, 10.0, 1.0, step=0.1, key="greek_T1")
        T2 = st.number_input("Bond maturity (T₂)", T1 + 0.1, 30.0, 5.0, step=0.1, key="greek_T2")

        K = st.number_input("Strike price (K)", 0.1, 2.0, 0.8, step=0.01, key="ir_greek_K")
        face_value = st.number_input("Face value", 100, 10000, 1000, step=100, key="greek_face")

        # Suggest dt from Tab 1
        default_dt = round(params['dt'], 3) if 'dt' in params else 0.01

        if model_type == "Monte Carlo":
            n_paths = st.number_input("Number of Monte Carlo simulations", 1000, 50000, 5000, step=1000, key="greek_npaths")
            dt = st.number_input("Time step (Δt)", 0.001, 0.1, default_dt, step=0.001, format="%.3f", key="greek_dt")
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
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Calculation Error</h4>
                        <pre>{traceback.format_exc()}</pre>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Educational content
    st.markdown("---")
    st.markdown('<div class="sub-header">Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("Understanding the Vasicek Model"):
        st.markdown("#### The Vasicek Interest Rate Model")
        st.info("The Vasicek model is a mathematical model describing the evolution of interest rates. It is a type of 'one-factor short-rate model' as it describes interest rate movements as driven by only one source of market risk.")
        
        st.markdown("#### Model Equation:")
        st.markdown(
            '<div style="text-align: center; font-size: 1.3em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">dr(t) = a(λ - r(t))dt + σ dW(t)</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("#### Parameters:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("• **a**: Speed of mean reversion")
            st.markdown("• **λ**: Long-term mean level")
            st.markdown("• **σ**: Volatility of rate changes")
        with col2:
            st.markdown("• **r(t)**: Interest rate at time t")
            st.markdown("• **dW(t)**: Wiener process")
        
        st.markdown("#### Key Features:")
        st.markdown("""
        • **Mean Reversion**: Rates tend to drift back toward the long-term mean  
        • **Analytical Solutions**: Closed-form formulas for bond prices and options  
        • **Negative Rates**: Model allows for negative interest rates  
        • **Normal Distribution**: Rate changes are normally distributed
        """)
    
    with st.expander("Bond Pricing Formulas"):
        st.markdown("#### Vasicek Bond Pricing Formulas")
        
        st.markdown("##### Zero-Coupon Bond Price:")
        st.markdown(
            '<div style="text-align: center; font-size: 1.2em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">P(t,T) = A(t,T) × exp(-B(t,T) × r(t))</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("**Where:**")
        st.latex(r"""
B(t,T) = \frac{1 - e^{-a(T - t)}}{a} \\
A(t,T) = \exp\left( \left( \lambda - \frac{\sigma^2}{2a^2} \right)(B(t,T) - (T - t)) - \frac{\sigma^2 B(t,T)^2}{4a} \right)
""")

        
        st.markdown("##### Coupon Bond Price:")
        st.markdown("Sum of discounted coupon payments plus principal:")
        st.markdown(
            '<div style="text-align: center; font-size: 1.2em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">P = Σ C × P(t,Tᵢ) + F × P(t,T)</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("##### Bond Option Price:")
        st.markdown("Uses the Black-Scholes formula adapted for bonds:")
        st.latex(r"""
\text{Call: } P(t,T_1) \cdot N(d_1) - K \cdot P(t,T_2) \cdot N(d_2) \\
\text{Put: } K \cdot P(t,T_2) \cdot N(-d_2) - P(t,T_1) \cdot N(-d_1)
""")
    
    with st.expander("Greeks for Bond Options"):
        st.markdown("#### Bond Option Greeks")
        
        # Delta
        st.markdown(
            '<div style="background-color: #e8f4f8; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1f77b4;"><strong>Delta (Δ): Sensitivity to Bond Price Changes</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • Measures how much the option price changes for a $1 change in bond price  
        • **Range:** 0 to 1 for calls, -1 to 0 for puts  
        • **Formula:** Δ = ∂V/∂S
        """)
        
        # Rho
        st.markdown(
            '<div style="background-color: #fff3cd; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffc107;"><strong>Rho (ρ): Sensitivity to Interest Rate Changes</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • Measures how much the option price changes for a 1% change in rates  
        • More important for bond options than equity options  
        • **Negative** for calls (higher rates → lower bond prices → lower call values)  
        • **Formula:** ρ = ∂V/∂r
        """)
        
        # Vega
        st.markdown(
            '<div style="background-color: #d1ecf1; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #17a2b8;"><strong>Vega (ν): Sensitivity to Volatility Changes</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • Measures how much the option price changes for a 1% change in volatility  
        • **Positive** for both calls and puts (higher volatility → higher option values)  
        • **Formula:** ν = ∂V/∂σ
        """)
        
        # Theta
        st.markdown(
            '<div style="background-color: #f8d7da; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #dc3545;"><strong>Theta (Θ): Time Decay</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • Measures how much the option price changes as time passes  
        • Usually **negative** (options lose value as expiration approaches)  
        • **Formula:** Θ = ∂V/∂t
        """)
        
        # Gamma
        st.markdown(
            '<div style="background-color: #d4edda; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745;"><strong>Gamma (Γ): Rate of Change of Delta</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • Measures the convexity of the option price  
        • Highest for **at-the-money** options near expiration  
        • **Formula:** Γ = ∂²V/∂S²
        """)
    
    with st.expander("⚠️ Model Limitations & Considerations"):
        st.warning("#### Vasicek Model Limitations")
        
        st.markdown("##### Theoretical Limitations:")
        st.markdown("""
        • **Negative Rates:** Model allows unrealistic negative rates (though less problematic now)  
        • **Constant Parameters:** Assumes constant mean reversion speed and volatility  
        • **Normal Distribution:** Real rate changes may have fat tails  
        • **Single Factor:** Ignores multiple sources of interest rate risk
        """)
        
        st.markdown("##### Practical Considerations:")
        st.markdown("""
        • **Parameter Estimation:** Historical data may not reflect future behavior  
        • **Calibration:** Model may not fit current market prices perfectly  
        • **Volatility Clustering:** Real rates show periods of high/low volatility  
        • **Regime Changes:** Central bank policy changes can break model assumptions
        """)
        
        st.markdown("##### Risk Management Best Practices:")
        st.markdown("""
        • Use multiple models for comparison and validation  
        • Regular recalibration with fresh market data  
        • Stress testing with extreme market scenarios  
        • Consider model uncertainty in risk measures  
        • Monitor model performance against market prices
        """)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 1px solid #dee2e6;'>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2rem;"></span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">Vasicek Interest Rate Model</p>
        <p style="margin: 8px 0; color: #6c757d;">Built with Streamlit & Python</p>
        <p style="margin: 0; color: #dc3545; font-weight: bold;">⚠️ For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)
