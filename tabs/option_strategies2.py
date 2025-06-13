# tabs/exotic_options.py
# Exotic Options Tab - Tab 4

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import math

# Import your pricing functions
from pricing.exotic_options import *
from pricing.utils.monte_carlo import *
from pricing.utils.numerical_methods import *


def exotic_options_tab():
    """Exotic Options Tab Content"""
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #9c27b0;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ff6f00;
            margin: 1rem 0;
        }
        .metric-container {
            background-color: #f3e5f5;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #9c27b0;
        }
        .info-box {
            background-color: #f3e5f5;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #9c27b0;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff8e1;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff8f00;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #e8f5e8;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #4caf50;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #ffebee;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #f44336;
            margin: 1rem 0;
        }
        .exotic-type-card {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #9c27b0;
            margin: 10px 0;
        }
        .parameter-grid {
            background-color: #fafafa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            margin: 10px 0;
        }
        .pricing-result {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #4caf50;
        }
        .complexity-indicator {
            background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ff8f00;
            text-align: center;
        }
        .monte-carlo-progress {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #2196f3;
        }
        .barrier-visualization {
            background: linear-gradient(135deg, #f1f8e9 0%, #dcedc8 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #8bc34a;
        }
        .path-dependent-info {
            background: linear-gradient(135deg, #fce4ec 0%, #f8bbd9 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e91e63;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<div class="main-header">🌟 Exotic Options Pricing Laboratory</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'exotic_setup_completed' not in st.session_state:
        st.session_state.exotic_setup_completed = False
    
    if 'exotic_params' not in st.session_state:
        st.session_state.exotic_params = {
            'spot_price': 100.0,
            'risk_free_rate': 0.05,
            'dividend_yield': 0.0,
            'volatility': 0.25,
            'time_to_expiry': 1.0,
            'n_simulations': 100000,
            'n_time_steps': 252
        }

    # Tab structure
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Exotic Setup",
        "🎯 Barrier Options", 
        "🔗 Path-Dependent Options", 
        "🏺 Asian Options", 
        "🎲 Monte Carlo Analysis",
        "📊 Exotic Strategies"
    ])
    
    with tab0:
        _exotic_setup_tab()
    
    # Extract parameters for other tabs
    if st.session_state.exotic_setup_completed:
        params = st.session_state.exotic_params
        
        with tab1:
            _barrier_options_tab(**params)
        
        with tab2:
            _path_dependent_tab(**params)
        
        with tab3:
            _asian_options_tab(**params)
        
        with tab4:
            _monte_carlo_tab(**params)
        
        with tab5:
            _exotic_strategies_tab(**params)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); border-radius: 15px; border: 1px solid #9c27b0;'>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2rem;">🌟</span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #9c27b0;">Exotic Options Pricing Laboratory</p>
        <p style="margin: 8px 0; color: #6c757d;">Advanced derivatives pricing with Monte Carlo simulation</p>
        <p style="margin: 0; color: #f44336; font-weight: bold;">⚠️ For educational and research purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


def _exotic_setup_tab():
    """Exotic Options Setup Tab"""
    st.markdown('<div class="sub-header">🚀 Exotic Options Configuration</div>', unsafe_allow_html=True)
    
    # Welcome message
    if not st.session_state.exotic_setup_completed:
        st.markdown("""
        <div class="warning-box">
            <h2 style="color: #e65100; margin-top: 0;">🌟 Welcome to Exotic Options!</h2>
            <p style="font-size: 1.1em;">
                Configure your market parameters to unlock advanced exotic derivatives pricing and analysis tools.
            </p>
            <p><strong>Featured:</strong> Barrier options, Asian options, lookback options, and Monte Carlo analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <h3 style="color: #2e7d32; margin-top: 0;">✅ Exotic Options Ready!</h3>
            <p>All advanced pricing tools are available. Modify parameters anytime to see instant updates.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Exotic options overview
    st.markdown("""
    <div class="info-box">
        <h3>🌟 Exotic Options Laboratory</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
            <div>
                <h4 style="color: #9c27b0; margin-bottom: 10px;">🎯 Barrier Options</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Knock-in and knock-out options</li>
                    <li>Up-and-in/out, down-and-in/out variants</li>
                    <li>Continuous barrier monitoring</li>
                    <li>Rebate payments upon barrier breach</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #9c27b0; margin-bottom: 10px;">🏺 Path-Dependent Options</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Asian options (average price/strike)</li>
                    <li>Lookback options (floating/fixed)</li>
                    <li>Cliquet options with performance caps</li>
                    <li>Monte Carlo simulation required</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Market Parameters
    st.markdown('<div class="sub-header">📊 Market Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #9c27b0; margin-top: 0;">💰 Core Market Data</h4>
        </div>
        """, unsafe_allow_html=True)
        
        spot_price = st.number_input(
            "Current Spot Price (S₀)", 
            value=st.session_state.exotic_params['spot_price'], 
            min_value=0.1, 
            step=0.1, 
            key="exotic_spot",
            help="Current price of the underlying asset"
        )
        
        risk_free_rate = st.number_input(
            "Risk-free Rate (r)", 
            value=st.session_state.exotic_params['risk_free_rate'], 
            min_value=0.0, 
            max_value=1.0, 
            step=0.001, 
            format="%.3f", 
            key="exotic_rate",
            help="Continuously compounded risk-free rate"
        )
        
        dividend_yield = st.number_input(
            "Dividend Yield (q)", 
            value=st.session_state.exotic_params['dividend_yield'], 
            min_value=0.0, 
            max_value=1.0, 
            step=0.001, 
            format="%.3f", 
            key="exotic_dividend",
            help="Continuous dividend yield"
        )
    
    with col2:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #9c27b0; margin-top: 0;">📈 Volatility & Time</h4>
        </div>
        """, unsafe_allow_html=True)
        
        volatility = st.number_input(
            "Volatility (σ)", 
            value=st.session_state.exotic_params['volatility'], 
            min_value=0.01, 
            max_value=3.0, 
            step=0.01, 
            format="%.3f", 
            key="exotic_vol",
            help="Annual volatility for geometric Brownian motion"
        )
        
        time_to_expiry = st.number_input(
            "Time to Expiry (T)", 
            value=st.session_state.exotic_params['time_to_expiry'], 
            min_value=0.01, 
            step=0.01, 
            format="%.3f", 
            key="exotic_time",
            help="Time to expiration in years"
        )
    
    st.markdown("---")
    
    # Monte Carlo Configuration
    st.markdown('<div class="sub-header">🎲 Monte Carlo Configuration</div>', unsafe_allow_html=True)
    
    mc_col1, mc_col2 = st.columns(2)
    
    with mc_col1:
        n_simulations = st.number_input(
            "Number of Simulations", 
            value=st.session_state.exotic_params['n_simulations'], 
            min_value=1000, 
            max_value=1000000, 
            step=1000, 
            key="exotic_n_sims",
            help="More simulations = higher accuracy but slower computation"
        )
        
        # Performance indicator
        if n_simulations <= 10000:
            perf_indicator = "🟢 Fast (< 5 seconds)"
            perf_color = "#4caf50"
        elif n_simulations <= 100000:
            perf_indicator = "🟡 Moderate (5-30 seconds)"
            perf_color = "#ff9800"
        else:
            perf_indicator = "🔴 Slow (30+ seconds)"
            perf_color = "#f44336"
        
        st.markdown(f"""
        <div class="complexity-indicator">
            <p style="margin: 0; color: {perf_color}; font-weight: bold;">{perf_indicator}</p>
            <p style="margin: 5px 0 0 0; font-size: 0.9em;">Standard Error: ±{1.96/np.sqrt(n_simulations)*100:.3f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with mc_col2:
        n_time_steps = st.number_input(
            "Time Steps per Year", 
            value=st.session_state.exotic_params['n_time_steps'], 
            min_value=50, 
            max_value=1000, 
            step=1, 
            key="exotic_steps",
            help="Daily steps = 252, weekly = 52, monthly = 12"
        )
        
        # Time step interpretation
        if n_time_steps >= 250:
            step_desc = "📅 Daily monitoring"
        elif n_time_steps >= 50:
            step_desc = "📊 Weekly monitoring"
        else:
            step_desc = "📈 Monthly monitoring"
        
        st.markdown(f"""
        <div class="parameter-grid">
            <h5>⏱️ Monitoring Frequency</h5>
            <p><strong>{step_desc}</strong></p>
            <p>Δt = {time_to_expiry/n_time_steps:.4f} years per step</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Parameter Summary
    st.markdown('<div class="sub-header">📋 Configuration Summary</div>', unsafe_allow_html=True)
    
    # Market regime analysis
    vol_regime = "High Volatility" if volatility > 0.4 else "Moderate Volatility" if volatility > 0.2 else "Low Volatility"
    rate_regime = "High Rate" if risk_free_rate > 0.05 else "Low Rate"
    
    st.markdown(f"""
    <div class="parameter-grid">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 2px solid #9c27b0; background-color: #f3e5f5;">
                <td style="padding: 12px; font-weight: bold;">Parameter</td>
                <td style="padding: 12px; font-weight: bold;">Value</td>
                <td style="padding: 12px; font-weight: bold;">Market Regime</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Spot Price</td>
                <td style="padding: 10px; font-family: monospace; color: #7b1fa2;">${spot_price:.2f}</td>
                <td style="padding: 10px; font-style: italic;">Reference level</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Volatility</td>
                <td style="padding: 10px; font-family: monospace; color: #7b1fa2;">{volatility*100:.1f}%</td>
                <td style="padding: 10px; font-style: italic;">{vol_regime}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Interest Rate</td>
                <td style="padding: 10px; font-family: monospace; color: #7b1fa2;">{risk_free_rate*100:.2f}%</td>
                <td style="padding: 10px; font-style: italic;">{rate_regime} Environment</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Time Horizon</td>
                <td style="padding: 10px; font-family: monospace; color: #7b1fa2;">{time_to_expiry:.2f} years</td>
                <td style="padding: 10px; font-style: italic;">{int(time_to_expiry*365)} days to expiry</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Simulation Quality</td>
                <td style="padding: 10px; font-family: monospace; color: #7b1fa2;">{n_simulations:,} paths</td>
                <td style="padding: 10px; font-style: italic;">{perf_indicator.split()[1]} computation</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup completion
    st.markdown("---")
    
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    
    with col_button2:
        setup_button_text = "🔄 Update Exotic Configuration" if st.session_state.exotic_setup_completed else "🚀 Initialize Exotic Options Lab"
        
        if st.button(setup_button_text, type="primary", use_container_width=True):
            # Update session state
            st.session_state.exotic_params.update({
                'spot_price': spot_price,
                'risk_free_rate': risk_free_rate,
                'dividend_yield': dividend_yield,
                'volatility': volatility,
                'time_to_expiry': time_to_expiry,
                'n_simulations': n_simulations,
                'n_time_steps': n_time_steps
            })
            
            st.session_state.exotic_setup_completed = True
            st.success("✅ Exotic options laboratory is now ready! All advanced pricing tools are available.")
            if not st.session_state.get('exotic_setup_completed_before', False):
                st.balloons()
                st.session_state.exotic_setup_completed_before = True
    
    if not st.session_state.exotic_setup_completed:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Setup Required</h4>
            <p>Please complete the setup above to unlock advanced exotic options analysis!</p>
            <p><strong>What you'll get:</strong> Barrier options, Asian options, lookback options, and Monte Carlo simulation tools.</p>
        </div>
        """, unsafe_allow_html=True)


def _barrier_options_tab():
    """Barrier Options Tab"""
    st.markdown('<div class="sub-header">Barrier Options Pricing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Parameters")
        
        barrier_S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.1, key="barrier_s")
        barrier_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="barrier_k")
        barrier_H = st.number_input("Barrier Level (H)", value=120.0, min_value=0.1, key="barrier_h")
        barrier_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="barrier_t")
        barrier_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="barrier_r")
        barrier_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="barrier_sigma")
        barrier_n_sim = st.number_input("Number of Simulations", value=10000, min_value=1000, max_value=100000, key="barrier_sim")
        barrier_n_steps = st.number_input("Steps per Path", value=100, min_value=10, max_value=500, key="barrier_steps")
        
        barrier_option_type = st.selectbox("Option Type", ["call", "put"], key="barrier_option_type")
        barrier_type = st.selectbox("Barrier Type", 
                                  ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], 
                                  key="barrier_type")
        
        calculate_barrier = st.button("🔢 Calculate Barrier Option", key="calc_barrier")
        show_paths_barrier = st.checkbox("📈 Show Sample Paths", key="show_paths_barrier")
    
    with col2:
        if calculate_barrier:
            with st.spinner("Calculating barrier option price..."):
                try:
                    # Validate barrier level
                    if barrier_type.startswith("up") and barrier_H <= max(barrier_S, barrier_K):
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("⚠️ For up barriers, H should typically be above current spot and strike prices")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif barrier_type.startswith("down") and barrier_H >= min(barrier_S, barrier_K):
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("⚠️ For down barriers, H should typically be below current spot and strike prices")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate option price
                    barrier_price, paths = price_barrier_option(
                        S=barrier_S, K=barrier_K, H=barrier_H, T=barrier_T,
                        r=barrier_r, sigma=barrier_sigma,
                        option_type=barrier_option_type, barrier_type=barrier_type,
                        n_simulations=barrier_n_sim, n_steps=barrier_n_steps
                    )
                    
                    # Display results
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.success(f"**Barrier Option Price: ${barrier_price:.4f}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show payoff diagram
                    st.markdown('<div class="sub-header">📈 Payoff Diagram</div>', unsafe_allow_html=True)
                    plot_barrier_payoff(
                        barrier_K, barrier_H, barrier_option_type, barrier_type,
                        S_min=barrier_S*0.5, S_max=barrier_S*1.5
                    )
                    
                    # Show sample paths if requested
                    if show_paths_barrier and paths is not None:
                        st.markdown('<div class="sub-header">📊 Sample Monte Carlo Paths</div>', unsafe_allow_html=True)
                        plot_sample_paths_barrier(
                            paths[:20], barrier_K, barrier_H, 
                            barrier_option_type, barrier_type
                        )
                    
                    # Market insights
                    st.markdown('<div class="sub-header">💡 Market Insights</div>', unsafe_allow_html=True)
                    if "out" in barrier_type:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**Knock-out options** are cheaper than vanilla options as they can expire worthless if the barrier is breached.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**Knock-in options** are cheaper than vanilla options as they only become active if the barrier is breached.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error calculating barrier option: {str(e)}")


def _digital_options_tab():
    """Digital Options Tab"""
    st.markdown('<div class="sub-header">Digital Options Pricing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Parameters")
        
        digital_S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.1, key="digital_s")
        digital_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="digital_k")
        digital_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="digital_t")
        digital_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="digital_r")
        digital_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="digital_sigma")
        
        digital_option_type = st.selectbox("Option Type", ["call", "put"], key="digital_option_type")
        digital_style = st.selectbox("Digital Style", ["cash", "asset"], key="digital_style")
        
        if digital_style == "cash":
            digital_Q = st.number_input("Cash Payout (Q)", value=1.0, min_value=0.01, key="digital_q")
        else:
            digital_Q = 1.0
        
        calculate_digital = st.button("🔢 Calculate Digital Option", key="calc_digital")
        show_greeks_digital = st.checkbox("📈 Show Greeks", key="show_greeks_digital")
    
    with col2:
        if calculate_digital:
            with st.spinner("Calculating digital option price..."):
                try:
                    # Calculate option price
                    digital_price = price_digital_option(
                        model="black_scholes", option_type=digital_option_type,
                        style=digital_style, S=digital_S, K=digital_K,
                        T=digital_T, r=digital_r, sigma=digital_sigma, Q=digital_Q
                    )
                    
                    # Display results
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.success(f"**Digital Option Price: ${digital_price:.4f}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate and display Greeks if requested
                    if show_greeks_digital:
                        greeks = calculate_greeks_digital(
                            digital_S, digital_K, digital_T, digital_r, digital_sigma,
                            digital_option_type, digital_style, digital_Q
                        )
                        
                        col_g1, col_g2, col_g3 = st.columns(3)
                        with col_g1:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Delta (Δ)", f"{greeks['Delta']:.4f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Gamma (Γ)", f"{greeks['Gamma']:.6f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col_g2:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Theta (Θ)", f"{greeks['Theta']:.4f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Vega (ν)", f"{greeks['Vega']:.4f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col_g3:
                            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                            st.metric("Rho (ρ)", f"{greeks['Rho']:.4f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show payoff diagram
                    st.markdown('<div class="sub-header">📈 Payoff Diagram</div>', unsafe_allow_html=True)
                    plot_digital_payoff(
                        digital_K, digital_option_type, digital_style, digital_Q,
                        S_min=digital_S*0.5, S_max=digital_S*1.5
                    )
                    
                    # Educational content
                    st.markdown('<div class="sub-header">📚 Digital Options Explained</div>', unsafe_allow_html=True)
                    if digital_style == "cash":
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown(f"**Cash-or-Nothing**: Pays ${digital_Q:.2f} if the option finishes in-the-money, nothing otherwise.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**Asset-or-Nothing**: Pays the asset price if the option finishes in-the-money, nothing otherwise.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error calculating digital option: {str(e)}")


def _lookback_options_tab():
    """Lookback Options Tab"""
    st.markdown('<div class="sub-header">Lookback Options Pricing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 Parameters")
        
        lookback_S0 = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=0.1, key="lookback_s0")
        
        lookback_floating = st.checkbox("Floating Strike", value=True, key="lookback_floating")
        
        if not lookback_floating:
            lookback_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="lookback_k")
        else:
            lookback_K = None
        
        lookback_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="lookback_t")
        lookback_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="lookback_r")
        lookback_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="lookback_sigma")
        lookback_n_paths = st.number_input("Number of Paths", value=100000, min_value=10000, max_value=1000000, key="lookback_paths")
        lookback_n_steps = st.number_input("Steps per Path", value=252, min_value=50, max_value=1000, key="lookback_steps")
        
        lookback_option_type = st.selectbox("Option Type", ["call", "put"], key="lookback_option_type")
        
        calculate_lookback = st.button("🔢 Calculate Lookback Option", key="calc_lookback")
        show_paths_lookback = st.checkbox("📈 Show Sample Paths", key="show_paths_lookback")
        show_distribution = st.checkbox("📊 Show Payoff Distribution", key="show_dist_lookback")
    
    with col2:
        if calculate_lookback:
            with st.spinner("Calculating lookback option price..."):
                try:
                    # Calculate option price
                    if lookback_floating:
                        lookback_price, lookback_stderr = price_lookback_option(
                            S0=lookback_S0, r=lookback_r, sigma=lookback_sigma, T=lookback_T,
                            option_type=lookback_option_type, floating_strike=True,
                            n_paths=lookback_n_paths, n_steps=lookback_n_steps
                        )
                    else:
                        lookback_price, lookback_stderr = price_lookback_option(
                            S0=lookback_S0, K=lookback_K, r=lookback_r, sigma=lookback_sigma, T=lookback_T,
                            option_type=lookback_option_type, floating_strike=False,
                            n_paths=lookback_n_paths, n_steps=lookback_n_steps
                        )
                    
                    # Display results
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.success(f"**Lookback Option Price: ${lookback_price:.4f} ± {lookback_stderr:.4f}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence interval
                    ci_lower = lookback_price - 1.96 * lookback_stderr
                    ci_upper = lookback_price + 1.96 * lookback_stderr
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown(f"**95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show payoff diagram
                    st.markdown('<div class="sub-header">📈 Payoff Function</div>', unsafe_allow_html=True)
                    fig_payoff = plot_payoff(lookback_S0, lookback_option_type, lookback_K, lookback_floating)
                    st.pyplot(fig_payoff)
                    
                    # Show sample paths if requested
                    if show_paths_lookback:
                        st.markdown('<div class="sub-header">📊 Sample Price Paths</div>', unsafe_allow_html=True)
                        fig_paths = plot_paths(lookback_S0, lookback_r, lookback_sigma, lookback_T, 
                                             min(10, lookback_n_paths), lookback_n_steps)
                        st.pyplot(fig_paths)
                    
                    # Show payoff distribution if requested
                    if show_distribution:
                        st.markdown('<div class="sub-header">📈 Payoff Distribution</div>', unsafe_allow_html=True)
                        fig_dist = plot_price_distribution(
                            lookback_S0, lookback_r, lookback_sigma, lookback_T,
                            lookback_option_type, lookback_floating,
                            min(10000, lookback_n_paths), lookback_n_steps
                        )
                        st.pyplot(fig_dist)
                    
                    # Educational content
                    st.markdown('<div class="sub-header">💡 Lookback Options Explained</div>', unsafe_allow_html=True)
                    if lookback_floating:
                        if lookback_option_type == "call":
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown("**Floating Strike Call**: Pays S_T - min(S_t), where min(S_t) is the minimum price during the option's life.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown("**Floating Strike Put**: Pays max(S_t) - S_T, where max(S_t) is the maximum price during the option's life.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        if lookback_option_type == "call":
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown(f"**Fixed Strike Call**: Pays max(0, max(S_t) - ${lookback_K}), based on the maximum price reached.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="info-box">', unsafe_allow_html=True)
                            st.markdown(f"**Fixed Strike Put**: Pays max(0, ${lookback_K} - min(S_t)), based on the minimum price reached.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error calculating lookback option: {str(e)}")


def _portfolio_analysis_tab():
    """Portfolio Analysis Tab"""
    st.markdown('<div class="sub-header">Portfolio Analysis & Comparison</div>', unsafe_allow_html=True)
    
    st.subheader("📊 Multi-Option Comparison")
    
    # Portfolio builder
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🏗️ Build Portfolio")
        
        # Common parameters
        port_S0 = st.number_input("Current Stock Price", value=100.0, key="port_s0")
        port_K = st.number_input("Strike Price", value=100.0, key="port_k")
        port_T = st.number_input("Time to Maturity", value=1.0, key="port_t")
        port_r = st.number_input("Risk-free Rate", value=0.05, format="%.4f", key="port_r")
        port_sigma = st.number_input("Volatility", value=0.2, format="%.4f", key="port_sigma")
        
        # Option selections
        include_vanilla = st.checkbox("Include Vanilla Option", value=True, key="include_vanilla")
        include_asian = st.checkbox("Include Asian Option", value=True, key="include_asian")
        include_barrier = st.checkbox("Include Barrier Option", value=True, key="include_barrier")
        include_digital = st.checkbox("Include Digital Option", value=True, key="include_digital")
        include_lookback = st.checkbox("Include Lookback Option", value=True, key="include_lookback")
        
        analyze_portfolio = st.button("📈 Analyze Portfolio", key="analyze_port")
    
    with col2:
        if analyze_portfolio:
            with st.spinner("Analyzing portfolio..."):
                try:
                    results = []
                    
                    # Calculate prices for selected options
                    if include_vanilla:
                        # Simple Black-Scholes for vanilla
                        from scipy.stats import norm
                        d1 = (np.log(port_S0/port_K) + (port_r + 0.5*port_sigma**2)*port_T) / (port_sigma*np.sqrt(port_T))
                        d2 = d1 - port_sigma*np.sqrt(port_T)
                        vanilla_price = port_S0*norm.cdf(d1) - port_K*np.exp(-port_r*port_T)*norm.cdf(d2)
                        results.append({"Option Type": "Vanilla Call", "Price": vanilla_price, "Complexity": "Low"})
                    
                    if include_asian:
                        asian_price = price_asian_option(
                            port_S0, port_K, port_T, port_r, port_sigma, 252, 10000, 
                            "call", "average_price"
                        )
                        results.append({"Option Type": "Asian Call", "Price": asian_price, "Complexity": "Medium"})
                    
                    if include_barrier:
                        barrier_price, _ = price_barrier_option(
                            port_S0, port_K, port_S0*1.2, port_T, port_r, port_sigma,
                            "call", "up-and-out", 10000, 100
                        )
                        results.append({"Option Type": "Barrier Call", "Price": barrier_price, "Complexity": "Medium"})
                    
                    if include_digital:
                        digital_price = price_digital_option(
                            "black_scholes", "call", "cash", port_S0, port_K, port_T, port_r, port_sigma
                        )
                        results.append({"Option Type": "Digital Call", "Price": digital_price, "Complexity": "Low"})
                    
                    if include_lookback:
                        lookback_price, _ = price_lookback_option(
                            port_S0, None, port_r, port_sigma, port_T, "call", True, 10000, 252
                        )
                        results.append({"Option Type": "Lookback Call", "Price": lookback_price, "Complexity": "High"})
                    
                    # Display results
                    if results:
                        df_results = pd.DataFrame(results)
                        df_results['Price'] = df_results['Price'].round(4)
                        
                        st.markdown('<div class="sub-header">💰 Portfolio Summary</div>', unsafe_allow_html=True)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Total portfolio value
                        total_value = df_results['Price'].sum()
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("**Total Portfolio Value**", f"${total_value:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Price comparison chart
                        fig = px.bar(df_results, x='Option Type', y='Price', 
                                   color='Complexity', title="Option Prices Comparison")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk analysis
                        st.markdown('<div class="sub-header">⚠️ Risk Analysis</div>', unsafe_allow_html=True)
                        
                        risk_metrics = []
                        for _, row in df_results.iterrows():
                            if "Barrier" in row['Option Type']:
                                risk = "High - Path dependent with knock-out risk"
                            elif "Lookback" in row['Option Type']:
                                risk = "Medium - Path dependent but no knock-out"
                            elif "Asian" in row['Option Type']:
                                risk = "Medium - Averaging reduces volatility impact"
                            elif "Digital" in row['Option Type']:
                                risk = "High - Binary payoff creates gamma risk"
                            else:
                                risk = "Low - Standard European option"
                            
                            risk_metrics.append({"Option": row['Option Type'], "Risk Level": risk})
                        
                        df_risk = pd.DataFrame(risk_metrics)
                        st.dataframe(df_risk, use_container_width=True)
                        
                        # Hedging suggestions
                        st.markdown('<div class="sub-header">🛡️ Hedging Suggestions</div>', unsafe_allow_html=True)
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("""
                        **Portfolio Hedging Strategies:**
                        - **Delta Hedging**: Regularly adjust underlying position to maintain delta neutrality
                        - **Gamma Hedging**: Use options to hedge gamma exposure, especially for digital options
                        - **Vega Hedging**: Consider volatility swaps for high vega exposure
                        - **Barrier Monitoring**: Set up real-time alerts for barrier levels
                        - **Diversification**: Spread risk across different option types and underlyings
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Monte Carlo analysis for portfolio
                        st.markdown('<div class="sub-header">🎲 Portfolio Monte Carlo Analysis</div>', unsafe_allow_html=True)
                        
                        mc_runs = st.slider("Monte Carlo Runs", 1000, 10000, 5000, key="mc_runs")
                        
                        if st.button("Run Monte Carlo Analysis", key="run_mc"):
                            with st.spinner("Running Monte Carlo simulation..."):
                                # Simulate price paths
                                dt = port_T / 252
                                n_sims = mc_runs
                                
                                # Generate correlated price paths
                                Z = np.random.normal(0, 1, (n_sims, 252))
                                price_paths = np.zeros((n_sims, 253))
                                price_paths[:, 0] = port_S0
                                
                                for t in range(1, 253):
                                    price_paths[:, t] = price_paths[:, t-1] * np.exp(
                                        (port_r - 0.5 * port_sigma**2) * dt + port_sigma * np.sqrt(dt) * Z[:, t-1]
                                    )
                                
                                final_prices = price_paths[:, -1]
                                
                                # Calculate portfolio P&L distribution
                                portfolio_pnl = []
                                
                                for final_price in final_prices:
                                    pnl = 0
                                    
                                    if include_vanilla:
                                        pnl += max(final_price - port_K, 0) - vanilla_price
                                    
                                    # For simplicity, approximate other options' P&L
                                    # In practice, you'd re-price each option at the final price
                                    if include_asian:
                                        approx_asian_pnl = max(final_price - port_K, 0) * 0.8 - asian_price
                                        pnl += approx_asian_pnl
                                    
                                    if include_digital:
                                        digital_pnl = (1.0 if final_price > port_K else 0.0) - digital_price
                                        pnl += digital_pnl
                                    
                                    portfolio_pnl.append(pnl)
                                
                                portfolio_pnl = np.array(portfolio_pnl)
                                
                                # Display Monte Carlo results
                                col_mc1, col_mc2, col_mc3 = st.columns(3)
                                
                                with col_mc1:
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.metric("Expected P&L", f"${np.mean(portfolio_pnl):.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.metric("P&L Std Dev", f"${np.std(portfolio_pnl):.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col_mc2:
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.metric("VaR (95%)", f"${np.percentile(portfolio_pnl, 5):.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.metric("CVaR (95%)", f"${np.mean(portfolio_pnl[portfolio_pnl <= np.percentile(portfolio_pnl, 5)]):.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col_mc3:
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.metric("Max Loss", f"${np.min(portfolio_pnl):.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                                    st.metric("Max Gain", f"${np.max(portfolio_pnl):.2f}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # P&L distribution plot
                                fig_dist = go.Figure()
                                fig_dist.add_trace(go.Histogram(
                                    x=portfolio_pnl, 
                                    nbinsx=50, 
                                    name="P&L Distribution",
                                    opacity=0.7
                                ))
                                
                                # Add VaR line
                                var_95 = np.percentile(portfolio_pnl, 5)
                                fig_dist.add_vline(
                                    x=var_95, 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text=f"VaR 95%: ${var_95:.2f}"
                                )
                                
                                fig_dist.update_layout(
                                    title="Portfolio P&L Distribution",
                                    xaxis_title="P&L ($)",
                                    yaxis_title="Frequency",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_dist, use_container_width=True)
                    
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown("Please select at least one option type to analyze.")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error in portfolio analysis: {str(e)}")
    
    # Educational Resources
    st.markdown("---")
    st.markdown('<div class="sub-header">📚 Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("🎓 Exotic Options Overview"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Understanding Exotic Options
        
        **Asian Options (Average Options)**
        - Payoff depends on the average price of the underlying over a specific period
        - Less volatile than vanilla options due to averaging effect
        - Popular in commodity markets and FX
        
        **Barrier Options**
        - Payoff depends on whether the underlying crosses a barrier level
        - Knock-out: Option extinguished if barrier is crossed
        - Knock-in: Option activated only if barrier is crossed
        - Cheaper than vanilla options due to additional risk
        
        **Digital Options (Binary Options)**
        - All-or-nothing payoff structure
        - Either pays a fixed amount or nothing at all
        - High gamma risk near expiration and strike
        
        **Lookback Options**
        - Payoff based on the maximum or minimum price during the option's life
        - Floating strike: Strike is set to the optimal level at expiration
        - Fixed strike: Payoff based on extrema vs. fixed strike
        - Expensive due to path-dependent nature
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("⚠️ Risk Management Guidelines"):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Key Risk Considerations
        
        **Model Risk**
        - Monte Carlo simulations have sampling error
        - Model assumptions may not hold in practice
        - Calibration to market data is crucial
        
        **Market Risk**
        - Exotic options often have complex Greeks
        - Path-dependent options require sophisticated hedging
        - Barrier options have discontinuous payoffs
        
        **Operational Risk**
        - Real-time monitoring of barrier levels
        - Accurate averaging calculations for Asian options
        - Proper settlement procedures for digital options
        
        **Liquidity Risk**
        - Exotic options may be harder to trade
        - Wider bid-ask spreads
        - Limited market makers
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Market Data & Volatility Analysis"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Interactive Implied Volatility Surface
        
        **Surface Parameters that affect exotic options:**
        
        **Base Volatility**: The fundamental volatility level for at-the-money options
        - Affects all exotic options but impact varies by type
        - Asian options less sensitive due to averaging
        - Lookback options more sensitive due to path dependence
        
        **Smile Intensity**: Controls the "volatility smile" or "smirk"
        - Higher values create more pronounced curves
        - Barrier options particularly sensitive to volatility skew
        - Digital options affected by local volatility near strike
        
        **Term Structure**: How volatility changes with time
        - Longer-term exotic options more affected
        - Path-dependent options sensitive to volatility term structure
        
        **Market Phenomena affecting exotics:**
        - **Volatility Smile**: Affects barrier and digital option pricing
        - **Volatility Clustering**: Important for path-dependent options
        - **Jump Risk**: Can trigger barriers unexpectedly
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Exotic Options Pricing Tool</strong></p>
        <p>Built with Streamlit • Educational and Research Purposes Only</p>
        <p>⚠️ Not for actual trading decisions</p>
    </div>
    """, unsafe_allow_html=True)


# Helper functions for Greeks calculations (these would need to be implemented in your utils)
def calculate_greeks_asian(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type):
    """Calculate Greeks for Asian options using finite differences"""
    # This is a placeholder - you'd implement finite difference calculations
    # or use your existing Greeks functions
    return {
        'Delta': 0.5,  # Placeholder values
        'Gamma': 0.05,
        'Theta': -0.01,
        'Vega': 0.15,
        'Rho': 0.08
    }

def calculate_greeks_digital(S, K, T, r, sigma, option_type, style, Q):
    """Calculate Greeks for Digital options"""
    # This is a placeholder - you'd implement actual digital Greeks calculations
    return {
        'Delta': 0.3,  # Placeholder values  
        'Gamma': 0.8,
        'Theta': -0.05,
        'Vega': 0.1,
        'Rho': 0.05
    }

def plot_sensitivity_analysis(option_type, base_params, param_name, param_range, option_family, **kwargs):
    """Create sensitivity analysis plots"""
    # This would create plotly charts showing how option prices vary with parameters
    # Placeholder implementation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_range, y=param_range*0.5, mode='lines', name=f'{option_family} {option_type}'))
    fig.update_layout(title=f"Sensitivity to {param_name}", xaxis_title=param_name, yaxis_title="Option Price")
    return fig
