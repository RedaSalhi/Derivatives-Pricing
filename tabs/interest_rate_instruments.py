# tabs/interest_rate_instruments.py
# Interest Rate Instruments Tab - Tab 6 - CORRECTED VERSION

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
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Parameter Estimation", "Simulation & Yield Curves", "Bond Pricing", "Bond Options"]
    )
    
    with tab1:
        _parameter_estimation_tab()
    
    with tab2:
        _simulation_yield_curves_tab()
    
    with tab3:
        _bond_pricing_tab()
    
    with tab4:
        _bond_options_tab()

    _display_educational_content()


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
                        <div class="success-box">
                            <h4>✅ Success!</h4>
                            <p>Parameters successfully estimated!</p>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>❌ Estimation Error</h4>
                            <p>Error during estimation: {str(e)}</p>
                            <p>Please try with different data or parameters.</p>
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
        n_paths = st.slider("Number of simulated paths", 100, 5000, 1000, step=100)

        st.markdown("""
        <div class="info-box">
            <h4>Yield Curve Configuration</h4>
        </div>
        """, unsafe_allow_html=True)

        available_maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        default_maturities = [m for m in [1, 2, 5, 10] if m <= T]
        maturities = st.multiselect("Maturities (years)", options=available_maturities, default=default_maturities)

        # Simplified snapshot times
        max_snapshots = min(10, int(T))
        snapshot_options = [f"{i} years" for i in range(0, max_snapshots + 1, max(1, max_snapshots // 5))]
        selected_snapshots = st.multiselect("Snapshot times", options=snapshot_options, default=snapshot_options[:4])
        
        # Convert back to numeric values
        snapshot_times = [float(s.split()[0]) for s in selected_snapshots]

        simulate_btn = st.button("Run Simulation", type="primary")

    with col2:
        if simulate_btn and maturities and snapshot_times:
            with st.spinner("Running simulation..."):
                try:
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

                    # Calculate yield curves for average path
                    avg_path = np.mean(r_paths, axis=1)
                    yield_curves = {}
                    
                    for t_snap in snapshot_times:
                        if t_snap < len(avg_path) * dt:
                            idx = int(t_snap / dt)
                            r_t = avg_path[idx] if idx < len(avg_path) else avg_path[-1]
                            yields = []
                            
                            for m in maturities:
                                T_target = t_snap + m
                                try:
                                    P = vasicek_zero_coupon_price(r_t, t_snap, T_target, params['a'], params['lambda'], params['sigma'])
                                    y = -np.log(P) / m if m > 0 else r_t
                                    yields.append(y)
                                except:
                                    yields.append(r_t)
                            
                            yield_curves[t_snap] = yields

                    # Plot yield curves
                    if yield_curves:
                        fig = go.Figure()
                        
                        colors = ['blue', 'red', 'green', 'orange', 'purple']
                        for i, (t_snap, yields) in enumerate(yield_curves.items()):
                            fig.add_trace(go.Scatter(
                                x=maturities,
                                y=yields,
                                mode='lines+markers',
                                name=f'Time {t_snap:.1f}y',
                                line=dict(color=colors[i % len(colors)], width=2)
                            ))
                        
                        fig.update_layout(
                            title='Simulated Yield Curves under Vasicek Model',
                            xaxis_title='Maturity (years)',
                            yaxis_title='Yield (continuously compounded)',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

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

                    # Statistics
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>Final Rate Statistics</h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                <td style="padding: 12px; font-weight: bold;">Statistic</td>
                                <td style="padding: 12px; font-weight: bold;">Value</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 10px; font-weight: bold;">Mean</td>
                                <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{np.mean(r_final):.4f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 10px; font-weight: bold;">Std Dev</td>
                                <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #2E8B57;">{np.std(r_final):.4f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 10px; font-weight: bold;">Minimum</td>
                                <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #FF6347;">{np.min(r_final):.4f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 10px; font-weight: bold;">Maximum</td>
                                <td style="padding: 10px; font-family: monospace; font-weight: bold; color: #FF6347;">{np.max(r_final):.4f}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Simulation Error</h4>
                        <p>Error during simulation: {str(e)}</p>
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

        r_current = st.number_input(
            "Current interest rate (r)", 
            min_value=0.0, 
            max_value=0.20, 
            value=max(0.001, params['r0']), 
            step=0.001, 
            format="%.4f"
        )
        t_current = st.number_input("Current time (t)", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
        maturity = st.number_input("Maturity (T)", min_value=t_current + 0.1, max_value=30.0, value=5.0, step=0.1)
        face_value = st.number_input("Face value", min_value=10, max_value=10000, value=100, step=10)

        if bond_type == "With Coupons":
            coupon_rate = st.number_input("Coupon rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100
            freq = st.selectbox("Payment frequency", ["Annual", "Semi-Annual"])
            dt_coupon = 1.0 if freq == "Annual" else 0.5

        price_btn = st.button("Compute Price", type="primary")

    with col2:
        if price_btn:
            with st.spinner("Calculating..."):
                try:
                    # Validate parameters
                    if r_current <= 0:
                        st.error("Interest rate must be positive")
                        return
                    
                    if maturity <= t_current:
                        st.error("Maturity must be greater than current time")
                        return

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
                        
                        # Validate results
                        if price <= 0 or price > face_value:
                            st.error(f"Invalid bond price: ${price:.2f}. Check parameters.")
                            return
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>Zero-Coupon Bond Pricing Results</h4>
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
                                    <td style="padding: 10px; font-weight: bold;">Yield to Maturity</td>
                                    <td style="padding: 10px; font-family: monospace; color: #FF6347; font-weight: bold;">{ytm:.4f} ({ytm*100:.2f}%)</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; font-weight: bold;">Discount Factor</td>
                                    <td style="padding: 10px; font-family: monospace; color: #1f77b4; font-weight: bold;">{price/face_value:.4f}</td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                    else:  # Coupon bond
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
                        
                        # Validate results
                        if price <= 0:
                            st.error(f"Invalid bond price: ${price:.2f}. Check parameters.")
                            return
                        
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
                        </div>
                        """, unsafe_allow_html=True)

                    # Sensitivity Analysis
                    st.markdown('<div class="sub-header">Price Sensitivity Analysis</div>', unsafe_allow_html=True)

                    # Interest rate sensitivity
                    r_vals = np.linspace(max(0.001, r_current - 0.03), r_current + 0.03, 50)
                    prices = []

                    for r in r_vals:
                        try:
                            if bond_type == "Zero-Coupon":
                                p = vasicek_zero_coupon_price(r, t_current, maturity, params['a'], params['lambda'], params['sigma'], face_value)
                            else:
                                p = price_coupon_bond(r, t_current, params['a'], params['lambda'], params['sigma'], maturity, face_value, coupon_rate, dt_coupon)
                            prices.append(max(0, p))  # Ensure non-negative prices
                        except:
                            prices.append(0)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=r_vals * 100, 
                        y=prices, 
                        mode="lines", 
                        name="Price", 
                        line=dict(width=3, color="#1f77b4")
                    ))
                    fig.add_vline(
                        x=r_current * 100, 
                        line_dash="dash", 
                        line_color="red", 
                        annotation_text=f"Current rate: {r_current*100:.2f}%"
                    )
                    fig.update_layout(
                        title="Bond Price Sensitivity to Interest Rate", 
                        xaxis_title="Interest Rate (%)", 
                        yaxis_title="Bond Price ($)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Duration calculation
                    try:
                        dr = 0.0001  # 1 basis point
                        if bond_type == "Zero-Coupon":
                            price_up = vasicek_zero_coupon_price(r_current + dr, t_current, maturity, params['a'], params['lambda'], params['sigma'], face_value)
                            price_down = vasicek_zero_coupon_price(r_current - dr, t_current, maturity, params['a'], params['lambda'], params['sigma'], face_value)
                        else:
                            price_up = price_coupon_bond(r_current + dr, t_current, params['a'], params['lambda'], params['sigma'], maturity, face_value, coupon_rate, dt_coupon)
                            price_down = price_coupon_bond(r_current - dr, t_current, params['a'], params['lambda'], params['sigma'], maturity, face_value, coupon_rate, dt_coupon)
                        
                        duration = -(price_up - price_down) / (2 * dr * price)
                        convexity = (price_up + price_down - 2 * price) / (dr**2 * price)
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>Risk Metrics</h4>
                            <p><strong>Modified Duration:</strong> {duration:.2f} years</p>
                            <p><strong>Convexity:</strong> {convexity:.2f}</p>
                            <p><strong>Price Change per 100bp:</strong> ${-duration * price * 0.01:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except:
                        st.warning("Could not calculate duration metrics")

                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Calculation Error</h4>
                        <p>Error: {str(e)}</p>
                        <p>Please check your parameters and try again.</p>
                    </div>
                    """, unsafe_allow_html=True)


def _bond_options_tab():
    """Bond Options Pricing Tab - FIXED VERSION"""
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

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Option Parameters</h4>
        </div>
        """, unsafe_allow_html=True)

        option_type = st.radio("Option type", ["Call", "Put"], key="opt_type")
        model_type = st.radio("Calculation method", ["Analytical", "Monte Carlo"], key="opt_model")

        r_current = st.number_input(
            "Current rate (r)", 
            0.001, 
            1.0, 
            max(0.001, params['r0']), 
            step=0.001, 
            format="%.4f", 
            key="opt_r"
        )
        T1 = st.number_input("Option maturity (T₁)", 0.1, 10.0, 1.0, step=0.1, key="opt_T1")
        T2 = st.number_input("Bond maturity (T₂)", T1 + 0.1, 30.0, 5.0, step=0.1, key="opt_T2")

        # Calculate reasonable strike range based on current bond price (normalized to face=1)
        try:
            current_bond_price = vasicek_zero_coupon_price(r_current, 0, T2, params['a'], params['lambda'], params['sigma'], 1.0)
            default_strike = round(current_bond_price * 0.95, 4)  # Slightly OTM
            st.info(f"**Current bond price:** "
                f"${current_bond_price:,.4f}\n\n"
                "Pick a strike reasonably close to that price.")
        except:
            default_strike = 0.8
            current_bond_price = 0.8

        # Strike should be reasonable relative to bond price
        K = st.number_input(
            "Strike price (per $1 face value)", 
            0.1, 
            1.5, 
            min(1.0, max(0.1, default_strike)), 
            step=0.01, 
            key="opt_K",
            help=(
                "Pick a strike reasonably close to the bond price."
                f"${current_bond_price:,.4f}\n\n"
            ),
        )
            
        
        # Face value for final scaling
        face_value = st.number_input("Face value ($)", 100, 10000, 1000, step=100, key="opt_face")

        if model_type == "Monte Carlo":
            n_paths = st.number_input("Number of simulations", 1000, 50000, 10000, step=1000, key="opt_n_paths")
            dt_mc = st.number_input(
                "Time step (Δt)",
                min_value=0.001,
                max_value=0.1,
                value=min(0.01, T1/50),  # Adaptive default
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
                    # Validate inputs
                    if r_current <= 0:
                        st.error("Interest rate must be positive")
                        return
                    
                    if K <= 0:
                        st.error("Strike price must be positive")
                        return

                    # Calculate underlying bond price for reference (normalized)
                    underlying_bond_price = vasicek_zero_coupon_price(
                        r_current, 0, T2, params['a'], params['lambda'], params['sigma'], 1.0
                    )
                    
                    # Show current market info
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>📊 Market Information</h4>
                        <p><strong>Bond price (per $1 face):</strong> ${underlying_bond_price:.4f}</p>
                        <p><strong>Strike price (per $1 face):</strong> ${K:.4f}</p>
                        <p><strong>Moneyness (S/K):</strong> {underlying_bond_price/K:.4f}</p>
                        <p><strong>Status:</strong> {"ITM" if (option_type=="Call" and underlying_bond_price > K) or (option_type=="Put" and underlying_bond_price < K) else "OTM"}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if model_type == "Analytical":
                        from pricing.models.interest_rates.analytical_vasicek import vasicek_bond_option_price
                        
                        # Calculate option price per $1 face value first
                        price_per_dollar = vasicek_bond_option_price(
                            r_t=r_current,
                            t=0,
                            T1=T1,
                            T2=T2,
                            K=K,
                            a=params['a'],
                            lam=params['lambda'],
                            sigma=params['sigma'],
                            face=1.0,  # Normalized to $1
                            option_type=option_type.lower()
                        )
                        
                        # Scale by actual face value
                        total_option_price = price_per_dollar * face_value
                        
                        # Validate result
                        if price_per_dollar < 0:
                            st.error("Negative option price calculated. Check parameters.")
                            return
                        
                        # Calculate additional metrics
                        intrinsic_per_dollar = 0
                        if option_type.lower() == "call":
                            intrinsic_per_dollar = max(0, underlying_bond_price - K)
                        else:
                            intrinsic_per_dollar = max(0, K - underlying_bond_price)
                        
                        total_intrinsic = intrinsic_per_dollar * face_value
                        time_value_per_dollar = max(0, price_per_dollar - intrinsic_per_dollar)
                        total_time_value = time_value_per_dollar * face_value
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{option_type} Option Pricing Results (Analytical)</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                                    <td style="padding: 12px; font-weight: bold;">Per $1 Face</td>
                                    <td style="padding: 12px; font-weight: bold;">Total (${face_value:,})</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Option Price</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">${price_per_dollar:.6f}</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold; font-size: 1.2em;">${total_option_price:.2f}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Intrinsic Value</td>
                                    <td style="padding: 10px; font-family: monospace; color: #1f77b4; font-weight: bold;">${intrinsic_per_dollar:.6f}</td>
                                    <td style="padding: 10px; font-family: monospace; color: #1f77b4; font-weight: bold;">${total_intrinsic:.2f}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Time Value</td>
                                    <td style="padding: 10px; font-family: monospace; color: #FF6347; font-weight: bold;">${time_value_per_dollar:.6f}</td>
                                    <td style="padding: 10px; font-family: monospace; color: #FF6347; font-weight: bold;">${total_time_value:.2f}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; font-weight: bold;">Option Delta</td>
                                    <td style="padding: 10px; font-family: monospace; color: #6c757d; font-weight: bold;" colspan="2">{"N/A" if abs(time_value_per_dollar) < 1e-8 else f"{(price_per_dollar/underlying_bond_price):.4f}"}</td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                    else:  # Monte Carlo
                        from pricing.models.interest_rates.monte_carlo_vasicek import vasicek_bond_option_price_mc
                        
                        # Calculate option price per $1 face value first
                        price_per_dollar, std_per_dollar = vasicek_bond_option_price_mc(
                            r0=r_current,
                            a=params['a'],
                            lam=params['lambda'],
                            sigma=params['sigma'],
                            T1=T1,
                            T2=T2,
                            K=K,
                            dt=dt_mc,
                            n_paths=int(n_paths),
                            face=1.0,  # Normalized to $1
                            option_type=option_type.lower()
                        )
                        
                        # Scale by actual face value
                        total_option_price = price_per_dollar * face_value
                        total_std = std_per_dollar * face_value
                        
                        # Validate result
                        if price_per_dollar < 0:
                            st.error("Negative option price calculated. Check parameters.")
                            return
                        
                        ci_lower_per_dollar = max(0, price_per_dollar - 1.96*std_per_dollar)
                        ci_upper_per_dollar = price_per_dollar + 1.96*std_per_dollar
                        ci_lower_total = ci_lower_per_dollar * face_value
                        ci_upper_total = ci_upper_per_dollar * face_value
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{option_type} Option Pricing Results (Monte Carlo)</h4>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                                    <td style="padding: 12px; font-weight: bold;">Per $1 Face</td>
                                    <td style="padding: 12px; font-weight: bold;">Total (${face_value:,})</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Option Price</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">${price_per_dollar:.6f}</td>
                                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold; font-size: 1.2em;">${total_option_price:.2f}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">Standard Error</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">± ${std_per_dollar:.6f}</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">± ${total_std:.2f}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid #eee;">
                                    <td style="padding: 10px; font-weight: bold;">95% Confidence Interval</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">[${ci_lower_per_dollar:.6f}, ${ci_upper_per_dollar:.6f}]</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">[${ci_lower_total:.2f}, ${ci_upper_total:.2f}]</td>
                                </tr>
                                <tr>
                                    <td style="padding: 10px; font-weight: bold;">Simulation Quality</td>
                                    <td style="padding: 10px; font-family: monospace; font-weight: bold;" colspan="2">{"Excellent" if std_per_dollar/price_per_dollar < 0.02 else "Good" if std_per_dollar/price_per_dollar < 0.05 else "Fair" if std_per_dollar/price_per_dollar < 0.1 else "Poor"}</td>
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
                                <td style="padding: 10px; font-weight: bold;">Value</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Option Type</td>
                                <td style="padding: 8px; font-weight: bold; color: #1f77b4;">{option_type}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Current Rate</td>
                                <td style="padding: 8px; font-family: monospace;">{r_current:.4f} ({r_current*100:.2f}%)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Option Maturity</td>
                                <td style="padding: 8px; font-family: monospace;">{T1:.2f} years</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Bond Maturity</td>
                                <td style="padding: 8px; font-family: monospace;">{T2:.2f} years</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 8px;">Strike Price</td>
                                <td style="padding: 8px; font-family: monospace;">${K:.4f} per $1 face</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px;">Face Value</td>
                                <td style="padding: 8px; font-family: monospace;">${face_value:,}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                    # Strike sensitivity analysis
                    st.markdown('<div class="sub-header">Strike Sensitivity Analysis</div>', unsafe_allow_html=True)
                    
                    try:
                        strike_range = np.linspace(max(0.1, underlying_bond_price * 0.7), underlying_bond_price * 1.3, 20)
                        option_prices = []
                        
                        for strike in strike_range:
                            try:
                                if model_type == "Analytical":
                                    opt_price = vasicek_bond_option_price(
                                        r_current, 0, T1, T2, strike, params['a'], params['lambda'], 
                                        params['sigma'], 1.0, option_type.lower()  # Use face=1.0
                                    )
                                else:
                                    opt_price, _ = vasicek_bond_option_price_mc(
                                        r_current, params['a'], params['lambda'], params['sigma'], 
                                        T1, T2, strike, dt_mc, min(5000, n_paths), 1.0, option_type.lower()  # Use face=1.0
                                    )
                                option_prices.append(max(0, opt_price))
                            except:
                                option_prices.append(0)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=strike_range,
                            y=option_prices,
                            mode='lines+markers',
                            name=f'{option_type} Option Price (per $1)',
                            line=dict(color='blue', width=3),
                            marker=dict(size=6)
                        ))
                        
                        fig.add_vline(
                            x=K, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"Current Strike: ${K:.4f}"
                        )
                        
                        fig.add_vline(
                            x=underlying_bond_price,
                            line_dash="dot",
                            line_color="green", 
                            annotation_text=f"Bond Price: ${underlying_bond_price:.4f}"
                        )
                        
                        fig.update_layout(
                            title=f'{option_type} Option Price vs Strike ({model_type}) - Per $1 Face Value',
                            xaxis_title='Strike Price ($)',
                            yaxis_title='Option Price ($)',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"Could not generate sensitivity analysis: {str(e)}")

                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Calculation Error</h4>
                        <p>Error: {str(e)}</p>
                        <p>Suggestions:</p>
                        <ul>
                            <li>Check that T₂ > T₁</li>
                            <li>Ensure strike price is reasonable (typically 0.5 to 1.2 for bonds)</li>
                            <li>Verify interest rate is positive</li>
                            <li>For Monte Carlo: try fewer simulations or larger time step</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

# Educational content and footer remain the same...
def _display_educational_content():
    """Display educational content about Vasicek model"""
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
            st.markdown("• **dW(t)**: Wiener process (random walk)")
        
        st.markdown("#### Key Features:")
        st.markdown("""
        • **Mean Reversion**: Rates tend to drift back toward the long-term mean  
        • **Analytical Solutions**: Closed-form formulas for bond prices and options  
        • **Normal Distribution**: Rate changes are normally distributed  
        • **Tractable Mathematics**: Allows for exact pricing formulas
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

        st.markdown("##### Bond Option Price:")
        st.markdown("Uses the Black-Scholes formula adapted for bonds:")
        st.latex(r"""
\text{Call: } P(t,T_1) \cdot N(d_1) - K \cdot P(t,T_2) \cdot N(d_2) \\
\text{Put: } K \cdot P(t,T_2) \cdot N(-d_2) - P(t,T_1) \cdot N(-d_1)
""")
    
    with st.expander("⚠️ Model Limitations & Best Practices"):
        st.warning("#### Vasicek Model Limitations")
        
        st.markdown("##### Theoretical Limitations:")
        st.markdown("""
        • **Negative Rates**: Model allows negative rates (which can be unrealistic)  
        • **Constant Parameters**: Assumes constant mean reversion speed and volatility  
        • **Normal Distribution**: Real rate changes may have fat tails  
        • **Single Factor**: Ignores multiple sources of interest rate risk
        """)
        
        st.markdown("##### Practical Considerations:")
        st.markdown("""
        • **Parameter Estimation**: Historical data may not reflect future behavior  
        • **Calibration**: Model may not fit current market prices perfectly  
        • **Volatility**: Interest rate volatility changes over time  
        • **Regime Changes**: Central bank policy changes can break model assumptions
        """)
        
        st.markdown("##### Best Practices:")
        st.markdown("""
        • Use multiple models for comparison and validation  
        • Regular recalibration with fresh market data  
        • Stress testing with extreme scenarios  
        • Consider model uncertainty in risk measures  
        • Validate results against market prices when available
        """)

    # Footer
    st.markdown("---")
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

