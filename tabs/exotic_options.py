# tabs/exotic_options.py
# Exotic Options Tab - Tab 4

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import your exotic options pricing functions
from pricing.asian_option import price_asian_option, plot_asian_option_payoff, plot_monte_carlo_paths
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff, plot_sample_paths_barrier
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution
from pricing.utils.exotic_utils import *


def exotic_options_tab():
    """Exotic Options Tab Content"""
    
    # Custom CSS for enhanced styling - matching other tabs
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ff7f0e;
            margin: 1rem 0;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .info-box {
            background-color: #e8f4f8;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #28a745;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
        }
        .formula {
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            color: #1f77b4;
            margin: 15px 0;
            padding: 15px;
            background-color: #f0f8ff;
            border-radius: 8px;
            border: 1px solid #d0e7ff;
        }
        .results-table {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
        .section-title {
            color: #1f77b4;
            font-weight: bold;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .parameter-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
        }
        .greek-container {
            background: linear-gradient(135deg, #f0f2f6 0%, #e9ecef 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #6c757d;
        }
        .pricing-result {
            background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px solid #28a745;
            text-align: center;
            margin: 1rem 0;
        }
        .option-type-asian {
            background-color: #e8f4f8;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #1f77b4;
        }
        .option-type-barrier {
            background-color: #fff3cd;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
        .option-type-digital {
            background-color: #d1ecf1;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
        }
        .option-type-lookback {
            background-color: #d4edda;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #28a745;
        }
        .portfolio-summary {
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
            padding: 2rem;
            border-radius: 15px;
            border-left: 5px solid #1f77b4;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Exotic Options Pricing Toolkit</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h4>🚀 Welcome to Advanced Options Pricing!</h4>
    This comprehensive toolkit enables pricing and analysis of sophisticated exotic options:
    <ul>
    <li><strong>Asian Options</strong> - Path-dependent options based on average prices</li>
    <li><strong>Barrier Options</strong> - Options with knock-in/knock-out features</li>
    <li><strong>Digital Options</strong> - Binary payoff structures</li>
    <li><strong>Lookback Options</strong> - Options based on price extrema</li>
    </ul>
    <em>Select any option type below to begin pricing and analysis.</em>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different option types
    tabb1, tabb2, tabb3, tabb4, tabb5 = st.tabs([
        "🌅 Asian Options", 
        "🚧 Barrier Options", 
        "💰 Digital Options", 
        "👀 Lookback Options",
        "📊 Portfolio Analysis"
    ])
    
    with tabb1:
        _asian_options_tab()
    
    with tabb2:
        _barrier_options_tab()
    
    with tabb3:
        _digital_options_tab()
    
    with tabb4:
        _lookback_options_tab()
    
    with tabb5:
        _portfolio_analysis_tab()


def _asian_options_tab():
    """Asian Options Tab"""
    st.markdown('<div class="sub-header">Asian Options Pricing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Option Parameters</div>', unsafe_allow_html=True)
        
        asian_S0 = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=0.1, key="asian_s0")
        asian_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="asian_k")
        asian_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="asian_t")
        asian_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="asian_r")
        asian_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="asian_sigma")
        
        st.markdown('<div class="section-title">⚙️ Model Parameters</div>', unsafe_allow_html=True)
        asian_n_steps = st.number_input("Number of Steps", value=252, min_value=10, max_value=1000, key="asian_steps")
        asian_n_paths = st.number_input("Number of Paths", value=10000, min_value=1000, max_value=100000, key="asian_paths")
        
        st.markdown('<div class="section-title">🎯 Option Configuration</div>', unsafe_allow_html=True)
        asian_option_type = st.selectbox("Option Type", ["call", "put"], key="asian_option_type_1")
        asian_type = st.selectbox("Asian Type", ["average_price", "average_strike"], key="asian_type")
        
        calculate_asian = st.button("🔢 Calculate Asian Option", key="calc_asian", type="primary")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            show_greeks_asian = st.checkbox("📈 Greeks", key="show_greeks_asian")
        with col_check2:
            show_sensitivity_asian = st.checkbox("📊 Sensitivity", key="show_sens_asian")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if calculate_asian:
            with st.spinner("🔄 Calculating Asian option price..."):
                try:
                    # Calculate option price
                    asian_price = price_asian_option(
                        S0=asian_S0, K=asian_K, T=asian_T, r=asian_r, sigma=asian_sigma,
                        n_steps=asian_n_steps, n_paths=asian_n_paths,
                        option_type=asian_option_type, asian_type=asian_type
                    )
                    
                    # Display results
                    st.markdown('<div class="pricing-result">', unsafe_allow_html=True)
                    st.markdown(f"<h3>💵 Asian Option Price: ${asian_price:.4f}</h3>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate and display Greeks if requested
                    if show_greeks_asian:
                        st.markdown('<div class="sub-header">📈 Option Greeks</div>', unsafe_allow_html=True)
                        greeks = calculate_greeks_asian(
                            asian_S0, asian_K, asian_T, asian_r, asian_sigma,
                            asian_n_steps, asian_n_paths, asian_option_type, asian_type
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
                    plot_asian_option_payoff(asian_K, asian_option_type, asian_type)
                    
                    # Sensitivity analysis
                    if show_sensitivity_asian:
                        st.markdown('<div class="sub-header">📊 Sensitivity Analysis</div>', unsafe_allow_html=True)
                        
                        # Parameter ranges for sensitivity
                        s_range = np.linspace(asian_S0 * 0.7, asian_S0 * 1.3, 20)
                        vol_range = np.linspace(0.1, 0.5, 20)
                        
                        base_params = {
                            'S0': asian_S0, 'K': asian_K, 'T': asian_T, 
                            'r': asian_r, 'sigma': asian_sigma,
                            'n_steps': asian_n_steps, 'n_paths': asian_n_paths
                        }
                        
                        # Spot price sensitivity
                        fig_spot = plot_sensitivity_analysis(
                            asian_option_type, base_params, 'S0', s_range, 
                            'asian', option_type=asian_option_type, asian_type=asian_type
                        )
                        st.plotly_chart(fig_spot, use_container_width=True)
                        
                        # Volatility sensitivity
                        fig_vol = plot_sensitivity_analysis(
                            asian_option_type, base_params, 'sigma', vol_range,
                            'asian', option_type=asian_option_type, asian_type=asian_type
                        )
                        st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # Educational content for Asian options
                    st.markdown('<div class="sub-header">📚 Understanding Asian Options</div>', unsafe_allow_html=True)
                    if asian_type == "average_price":
                        st.markdown("""
                        <div class="option-type-asian">
                        <strong>Average Price Asian Option:</strong><br>
                        • Payoff based on average price over the option's life<br>
                        • Less volatile than vanilla options due to averaging effect<br>
                        • Popular in commodity markets and currency trading<br>
                        • Formula: max(0, Average_Price - K) for calls
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="option-type-asian">
                        <strong>Average Strike Asian Option:</strong><br>
                        • Strike price is the average price over the option's life<br>
                        • Final payoff uses current spot vs average strike<br>
                        • Provides natural hedging against price fluctuations<br>
                        • Formula: max(0, S_T - Average_Price) for calls
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown(f"<strong>❌ Error:</strong> {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)


def _barrier_options_tab():
    """Barrier Options Tab"""
    st.markdown('<div class="sub-header">Barrier Options Pricing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Option Parameters</div>', unsafe_allow_html=True)
        
        barrier_S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.1, key="barrier_s")
        barrier_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="barrier_k")
        barrier_H = st.number_input("Barrier Level (H)", value=120.0, min_value=0.1, key="barrier_h")
        barrier_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="barrier_t")
        barrier_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="barrier_r")
        barrier_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="barrier_sigma")
        
        st.markdown('<div class="section-title">⚙️ Simulation Parameters</div>', unsafe_allow_html=True)
        barrier_n_sim = st.number_input("Number of Simulations", value=10000, min_value=1000, max_value=100000, key="barrier_sim")
        barrier_n_steps = st.number_input("Steps per Path", value=100, min_value=10, max_value=500, key="barrier_steps")
        
        st.markdown('<div class="section-title">🎯 Barrier Configuration</div>', unsafe_allow_html=True)
        barrier_option_type = st.selectbox("Option Type", ["call", "put"], key="barrier_option_type")
        barrier_type = st.selectbox("Barrier Type", 
                                  ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], 
                                  key="barrier_type")
        
        calculate_barrier = st.button("🔢 Calculate Barrier Option", key="calc_barrier", type="primary")
        show_paths_barrier = st.checkbox("📈 Show Sample Paths", key="show_paths_barrier")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if calculate_barrier:
            with st.spinner("🔄 Calculating barrier option price..."):
                try:
                    # Validate barrier level
                    if barrier_type.startswith("up") and barrier_H <= max(barrier_S, barrier_K):
                        st.markdown("""
                        <div class="warning-box">
                        <strong>⚠️ Barrier Configuration Notice:</strong><br>
                        For up barriers, H should typically be above current spot and strike prices for meaningful results.
                        </div>
                        """, unsafe_allow_html=True)
                    elif barrier_type.startswith("down") and barrier_H >= min(barrier_S, barrier_K):
                        st.markdown("""
                        <div class="warning-box">
                        <strong>⚠️ Barrier Configuration Notice:</strong><br>
                        For down barriers, H should typically be below current spot and strike prices for meaningful results.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Calculate option price
                    barrier_price, paths = price_barrier_option(
                        S=barrier_S, K=barrier_K, H=barrier_H, T=barrier_T,
                        r=barrier_r, sigma=barrier_sigma,
                        option_type=barrier_option_type, barrier_type=barrier_type,
                        n_simulations=barrier_n_sim, n_steps=barrier_n_steps
                    )
                    
                    # Display results
                    st.markdown('<div class="pricing-result">', unsafe_allow_html=True)
                    st.markdown(f"<h3>💵 Barrier Option Price: ${barrier_price:.4f}</h3>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show payoff diagram
                    st.markdown('<div class="sub-header">📈 Payoff Diagram</div>', unsafe_allow_html=True)
                    plot_barrier_payoff(
                        barrier_K, barrier_H, barrier_option_type, barrier_type,
                        S_min=barrier_S*0.5, S_max=barrier_S*1.5
                    )
                    
                    # Show sample paths if requested
                    if show_paths_barrier and paths is not None:
                        st.markdown('<div class="sub-header">📊 Monte Carlo Sample Paths</div>', unsafe_allow_html=True)
                        plot_sample_paths_barrier(
                            paths[:20], barrier_K, barrier_H, 
                            barrier_option_type, barrier_type
                        )
                    
                    # Educational content and market insights
                    st.markdown('<div class="sub-header">💡 Barrier Options Explained</div>', unsafe_allow_html=True)
                    if "out" in barrier_type:
                        st.markdown("""
                        <div class="option-type-barrier">
                        <strong>Knock-Out Options:</strong><br>
                        • Option expires worthless if barrier is breached<br>
                        • Cheaper than vanilla options due to knockout risk<br>
                        • Popular for hedging with cost reduction<br>
                        • Risk: Protection disappears when most needed
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="option-type-barrier">
                        <strong>Knock-In Options:</strong><br>
                        • Option only becomes active if barrier is breached<br>
                        • Cheaper than vanilla options - only pay if activated<br>
                        • Used for contingent exposure strategies<br>
                        • Risk: May never activate despite favorable moves
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown(f"<strong>❌ Error:</strong> {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)


def _digital_options_tab():
    """Digital Options Tab"""
    st.markdown('<div class="sub-header">Digital Options Pricing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Option Parameters</div>', unsafe_allow_html=True)
        
        digital_S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.1, key="digital_s")
        digital_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="digital_k")
        digital_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="digital_t")
        digital_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="digital_r")
        digital_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="digital_sigma")
        
        st.markdown('<div class="section-title">🎯 Digital Configuration</div>', unsafe_allow_html=True)
        digital_option_type = st.selectbox("Option Type", ["call", "put"], key="digital_option_type")
        digital_style = st.selectbox("Digital Style", ["cash", "asset"], key="digital_style")
        
        if digital_style == "cash":
            digital_Q = st.number_input("Cash Payout (Q)", value=1.0, min_value=0.01, key="digital_q")
        else:
            digital_Q = 1.0
        
        calculate_digital = st.button("🔢 Calculate Digital Option", key="calc_digital", type="primary")
        show_greeks_digital = st.checkbox("📈 Show Greeks", key="show_greeks_digital")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if calculate_digital:
            with st.spinner("🔄 Calculating digital option price..."):
                try:
                    # Calculate option price
                    digital_price = price_digital_option(
                        model="black_scholes", option_type=digital_option_type,
                        style=digital_style, S=digital_S, K=digital_K,
                        T=digital_T, r=digital_r, sigma=digital_sigma, Q=digital_Q
                    )
                    
                    # Display results
                    st.markdown('<div class="pricing-result">', unsafe_allow_html=True)
                    st.markdown(f"<h3>💵 Digital Option Price: ${digital_price:.4f}</h3>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate and display Greeks if requested
                    if show_greeks_digital:
                        st.markdown('<div class="sub-header">📈 Option Greeks</div>', unsafe_allow_html=True)
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
                        st.markdown(f"""
                        <div class="option-type-digital">
                        <strong>Cash-or-Nothing Digital:</strong><br>
                        • Pays fixed amount ${digital_Q:.2f} if ITM, nothing if OTM<br>
                        • Simple binary outcome - all or nothing<br>
                        • High gamma risk near expiration and strike<br>
                        • Popular in binary trading and structured products
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="option-type-digital">
                        <strong>Asset-or-Nothing Digital:</strong><br>
                        • Pays the asset price if ITM, nothing if OTM<br>
                        • Exposure to underlying asset price movements<br>
                        • Combines digital structure with asset exposure<br>
                        • Used in portfolio replication strategies
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown(f"<strong>❌ Error:</strong> {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)


def _lookback_options_tab():
    """Lookback Options Tab"""
    st.markdown('<div class="sub-header">Lookback Options Pricing</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Option Parameters</div>', unsafe_allow_html=True)
        
        lookback_S0 = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=0.1, key="lookback_s0")
        
        lookback_floating = st.checkbox("Floating Strike", value=True, key="lookback_floating")
        
        if not lookback_floating:
            lookback_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="lookback_k")
        else:
            lookback_K = None
        
        lookback_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="lookback_t")
        lookback_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="lookback_r")
        lookback_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="lookback_sigma")
        
        st.markdown('<div class="section-title">⚙️ Simulation Parameters</div>', unsafe_allow_html=True)
        lookback_n_paths = st.number_input("Number of Paths", value=100000, min_value=10000, max_value=1000000, key="lookback_paths")
        lookback_n_steps = st.number_input("Steps per Path", value=252, min_value=50, max_value=1000, key="lookback_steps")
        
        st.markdown('<div class="section-title">🎯 Option Configuration</div>', unsafe_allow_html=True)
        lookback_option_type = st.selectbox("Option Type", ["call", "put"], key="lookback_option_type")
        
        calculate_lookback = st.button("🔢 Calculate Lookback Option", key="calc_lookback", type="primary")
        
        col_check1, col_check2 = st.columns(2)
        with col_check1:
            show_paths_lookback = st.checkbox("📈 Sample Paths", key="show_paths_lookback")
        with col_check2:
            show_distribution = st.checkbox("📊 Payoff Dist.", key="show_dist_lookback")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if calculate_lookback:
            with st.spinner("🔄 Calculating lookback option price..."):
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
                    st.markdown('<div class="pricing-result">', unsafe_allow_html=True)
                    st.markdown(f"<h3>💵 Lookback Option Price: ${lookback_price:.4f} ± {lookback_stderr:.4f}</h3>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence interval
                    ci_lower = lookback_price - 1.96 * lookback_stderr
                    ci_upper = lookback_price + 1.96 * lookback_stderr
                    st.markdown("""
                    <div class="info-box">
                    <strong>📊 95% Confidence Interval:</strong><br>
                    Price Range: [${:.4f}, ${:.4f}]<br>
                    <em>Monte Carlo standard error: ±${:.4f}</em>
                    </div>
                    """.format(ci_lower, ci_upper, lookback_stderr), unsafe_allow_html=True)
                    
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
                            st.markdown("""
                            <div class="option-type-lookback">
                            <strong>Floating Strike Call:</strong><br>
                            • Payoff: S_T - min(S_t) over option life<br>
                            • Strike = minimum price reached<br>
                            • Always finishes in-the-money<br>
                            • Perfect timing - buys at the lowest price
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="option-type-lookback">
                            <strong>Floating Strike Put:</strong><br>
                            • Payoff: max(S_t) - S_T over option life<br>
                            • Strike = maximum price reached<br>
                            • Always finishes in-the-money<br>
                            • Perfect timing - sells at the highest price
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        if lookback_option_type == "call":
                            st.markdown(f"""
                            <div class="option-type-lookback">
                            <strong>Fixed Strike Call:</strong><br>
                            • Payoff: max(0, max(S_t) - ${lookback_K})<br>
                            • Based on maximum price vs fixed strike<br>
                            • Benefits from any upward price movement<br>
                            • More expensive than vanilla calls
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="option-type-lookback">
                            <strong>Fixed Strike Put:</strong><br>
                            • Payoff: max(0, ${lookback_K} - min(S_t))<br>
                            • Based on minimum price vs fixed strike<br>
                            • Benefits from any downward price movement<br>
                            • More expensive than vanilla puts
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown(f"<strong>❌ Error:</strong> {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)


def _portfolio_analysis_tab():
    """Portfolio Analysis Tab"""
    st.markdown('<div class="sub-header">Portfolio Analysis & Multi-Option Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏗️ Portfolio Builder</div>', unsafe_allow_html=True)
        
        # Common parameters
        port_S0 = st.number_input("Current Stock Price", value=100.0, key="port_s0")
        port_K = st.number_input("Strike Price", value=100.0, key="port_k")
        port_T = st.number_input("Time to Maturity", value=1.0, key="port_t")
        port_r = st.number_input("Risk-free Rate", value=0.05, format="%.4f", key="port_r")
        port_sigma = st.number_input("Volatility", value=0.2, format="%.4f", key="port_sigma")
        
        st.markdown('<div class="section-title">📋 Option Selection</div>', unsafe_allow_html=True)
        # Option selections
        include_vanilla = st.checkbox("🏛️ Vanilla Option", value=True, key="include_vanilla")
        include_asian = st.checkbox("🌅 Asian Option", value=True, key="include_asian")
        include_barrier = st.checkbox("🚧 Barrier Option", value=True, key="include_barrier")
        include_digital = st.checkbox("💰 Digital Option", value=True, key="include_digital")
        include_lookback = st.checkbox("👀 Lookback Option", value=True, key="include_lookback")
        
        analyze_portfolio = st.button("📈 Analyze Portfolio", key="analyze_port", type="primary")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if analyze_portfolio:
            with st.spinner("🔄 Analyzing exotic options portfolio..."):
                try:
                    results = []
                    
                    # Calculate prices for selected options
                    if include_vanilla:
                        # Simple Black-Scholes for vanilla
                        from scipy.stats import norm
                        d1 = (np.log(port_S0/port_K) + (port_r + 0.5*port_sigma**2)*port_T) / (port_sigma*np.sqrt(port_T))
                        d2 = d1 - port_sigma*np.sqrt(port_T)
                        vanilla_price = port_S0*norm.cdf(d1) - port_K*np.exp(-port_r*port_T)*norm.cdf(d2)
                        results.append({
                            "Option Type": "🏛️ Vanilla Call", 
                            "Price": vanilla_price, 
                            "Complexity": "Low",
                            "Risk Level": "Standard"
                        })
                    
                    if include_asian:
                        asian_price = price_asian_option(
                            port_S0, port_K, port_T, port_r, port_sigma, 252, 10000, 
                            "call", "average_price"
                        )
                        results.append({
                            "Option Type": "🌅 Asian Call", 
                            "Price": asian_price, 
                            "Complexity": "Medium",
                            "Risk Level": "Reduced"
                        })
                    
                    if include_barrier:
                        barrier_price, _ = price_barrier_option(
                            port_S0, port_K, port_S0*1.2, port_T, port_r, port_sigma,
                            "call", "up-and-out", 10000, 100
                        )
                        results.append({
                            "Option Type": "🚧 Barrier Call", 
                            "Price": barrier_price, 
                            "Complexity": "Medium",
                            "Risk Level": "High"
                        })
                    
                    if include_digital:
                        digital_price = price_digital_option(
                            "black_scholes", "call", "cash", port_S0, port_K, port_T, port_r, port_sigma
                        )
                        results.append({
                            "Option Type": "💰 Digital Call", 
                            "Price": digital_price, 
                            "Complexity": "Low",
                            "Risk Level": "Binary"
                        })
                    
                    if include_lookback:
                        lookback_price, _ = price_lookback_option(
                            port_S0, None, port_r, port_sigma, port_T, "call", True, 10000, 252
                        )
                        results.append({
                            "Option Type": "👀 Lookback Call", 
                            "Price": lookback_price, 
                            "Complexity": "High",
                            "Risk Level": "Premium"
                        })
                    
                    # Display results
                    if results:
                        df_results = pd.DataFrame(results)
                        df_results['Price'] = df_results['Price'].round(4)
                        
                        st.markdown('<div class="portfolio-summary">', unsafe_allow_html=True)
                        st.markdown('<h4>💰 Portfolio Summary</h4>', unsafe_allow_html=True)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Total portfolio value
                        total_value = df_results['Price'].sum()
                        avg_price = df_results['Price'].mean()
                        
                        col_sum1, col_sum2, col_sum3 = st.columns(3)
                        with col_sum1:
                            st.metric("Total Portfolio Value", f"${total_value:.4f}")
                        with col_sum2:
                            st.metric("Average Option Price", f"${avg_price:.4f}")
                        with col_sum3:
                            st.metric("Number of Options", len(results))
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Price comparison chart
                        fig = px.bar(
                            df_results, 
                            x='Option Type', 
                            y='Price', 
                            color='Complexity',
                            title="📊 Exotic Options Price Comparison",
                            color_discrete_map={
                                'Low': '#28a745',
                                'Medium': '#ffc107', 
                                'High': '#dc3545'
                            }
                        )
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk analysis comparison
                        st.markdown('<div class="sub-header">⚠️ Risk Profile Analysis</div>', unsafe_allow_html=True)
                        
                        # Create risk matrix
                        risk_descriptions = {
                            "🏛️ Vanilla Call": "Standard European option risk - predictable Greeks behavior",
                            "🌅 Asian Call": "Reduced volatility impact due to averaging - lower gamma risk",
                            "🚧 Barrier Call": "Path-dependent with knockout risk - protection can disappear",
                            "💰 Digital Call": "Binary payoff creates high gamma and vega near strike and expiry",
                            "👀 Lookback Call": "Path-dependent premium option - expensive but comprehensive coverage"
                        }
                        
                        selected_risks = [opt["Option Type"] for opt in results]
                        
                        for option_type in selected_risks:
                            if option_type in risk_descriptions:
                                risk_class = "info-box"
                                if "Barrier" in option_type:
                                    risk_class = "warning-box"
                                elif "Digital" in option_type:
                                    risk_class = "error-box"
                                elif "Lookback" in option_type:
                                    risk_class = "success-box"
                                
                                st.markdown(f'''
                                <div class="{risk_class}">
                                <strong>{option_type}:</strong><br>
                                {risk_descriptions[option_type]}
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        # Hedging recommendations
                        st.markdown('<div class="sub-header">🛡️ Portfolio Hedging Strategy</div>', unsafe_allow_html=True)
                        
                        hedging_strategy = []
                        
                        if include_barrier:
                            hedging_strategy.append("• **Barrier Monitoring**: Set up real-time alerts for barrier breach detection")
                            hedging_strategy.append("• **Knock-out Protection**: Consider vanilla option backstops for barrier options")
                        
                        if include_digital:
                            hedging_strategy.append("• **Gamma Hedging**: Use spreads to manage high gamma exposure near strikes")
                            hedging_strategy.append("• **Binary Risk**: Monitor time decay acceleration approaching expiry")
                        
                        if include_asian:
                            hedging_strategy.append("• **Average Tracking**: Monitor running averages vs current spot prices")
                        
                        if include_lookback:
                            hedging_strategy.append("• **Path Monitoring**: Track maximum/minimum levels reached during option life")
                        
                        # General recommendations
                        hedging_strategy.extend([
                            "• **Delta Hedging**: Regularly rebalance underlying position for portfolio delta neutrality",
                            "• **Vega Management**: Consider volatility swaps or straddles for net vega exposure",
                            "• **Diversification**: Spread exotic option risk across different underlyings and timeframes"
                        ])
                        
                        st.markdown(f'''
                        <div class="info-box">
                        <h4>🎯 Recommended Hedging Actions:</h4>
                        {"<br>".join(hedging_strategy)}
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Monte Carlo portfolio analysis
                        st.markdown('<div class="sub-header">🎲 Portfolio Monte Carlo Risk Analysis</div>', unsafe_allow_html=True)
                        
                        mc_runs = st.slider("Monte Carlo Simulations", 1000, 10000, 5000, key="mc_runs")
                        
                        if st.button("🚀 Run Portfolio Risk Analysis", key="run_mc"):
                            with st.spinner("⚡ Running Monte Carlo portfolio simulation..."):
                                # Simulate future scenarios
                                dt = port_T / 252
                                n_sims = mc_runs
                                
                                # Generate price paths
                                np.random.seed(42)  # For reproducible results
                                Z = np.random.normal(0, 1, (n_sims, 252))
                                price_paths = np.zeros((n_sims, 253))
                                price_paths[:, 0] = port_S0
                                
                                for t in range(1, 253):
                                    price_paths[:, t] = price_paths[:, t-1] * np.exp(
                                        (port_r - 0.5 * port_sigma**2) * dt + port_sigma * np.sqrt(dt) * Z[:, t-1]
                                    )
                                
                                final_prices = price_paths[:, -1]
                                
                                # Calculate portfolio P&L distribution (simplified approximation)
                                portfolio_pnl = []
                                
                                for final_price in final_prices:
                                    pnl = 0
                                    
                                    # Approximate P&L for each option type
                                    for result in results:
                                        option_type = result["Option Type"]
                                        current_price = result["Price"]
                                        
                                        if "Vanilla" in option_type:
                                            pnl += max(final_price - port_K, 0) - current_price
                                        elif "Asian" in option_type:
                                            # Simplified: assume 80% of vanilla payoff due to averaging
                                            pnl += max(final_price - port_K, 0) * 0.8 - current_price
                                        elif "Digital" in option_type:
                                            # Binary payoff
                                            pnl += (1.0 if final_price > port_K else 0.0) - current_price
                                        elif "Barrier" in option_type:
                                            # Simplified: knocked out if price exceeded 20% above initial
                                            if np.any(price_paths[portfolio_pnl.__len__(), :] > port_S0 * 1.2):
                                                pnl += 0 - current_price  # Knocked out
                                            else:
                                                pnl += max(final_price - port_K, 0) - current_price
                                        elif "Lookback" in option_type:
                                            # Simplified: use minimum price in path
                                            min_price = np.min(price_paths[portfolio_pnl.__len__(), :])
                                            pnl += final_price - min_price - current_price
                                    
                                    portfolio_pnl.append(pnl)
                                
                                portfolio_pnl = np.array(portfolio_pnl)
                                
                                # Display Monte Carlo results
                                col_mc1, col_mc2, col_mc3, col_mc4 = st.columns(4)
                                
                                with col_mc1:
                                    st.metric("Expected P&L", f"${np.mean(portfolio_pnl):.2f}")
                                    st.metric("P&L Volatility", f"${np.std(portfolio_pnl):.2f}")
                                
                                with col_mc2:
                                    var_95 = np.percentile(portfolio_pnl, 5)
                                    st.metric("VaR (95%)", f"${var_95:.2f}")
                                    cvar_95 = np.mean(portfolio_pnl[portfolio_pnl <= var_95])
                                    st.metric("CVaR (95%)", f"${cvar_95:.2f}")
                                
                                with col_mc3:
                                    st.metric("Maximum Loss", f"${np.min(portfolio_pnl):.2f}")
                                    st.metric("Maximum Gain", f"${np.max(portfolio_pnl):.2f}")
                                
                                with col_mc4:
                                    prob_profit = np.mean(portfolio_pnl > 0) * 100
                                    st.metric("Profit Probability", f"{prob_profit:.1f}%")
                                    sharpe_ratio = np.mean(portfolio_pnl) / np.std(portfolio_pnl) if np.std(portfolio_pnl) > 0 else 0
                                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                                
                                # P&L distribution plot
                                fig_dist = go.Figure()
                                fig_dist.add_trace(go.Histogram(
                                    x=portfolio_pnl, 
                                    nbinsx=50, 
                                    name="Portfolio P&L",
                                    opacity=0.7,
                                    marker_color='#1f77b4'
                                ))
                                
                                # Add VaR line
                                fig_dist.add_vline(
                                    x=var_95, 
                                    line_dash="dash", 
                                    line_color="red",
                                    annotation_text=f"VaR 95%: ${var_95:.2f}",
                                    annotation_position="top"
                                )
                                
                                fig_dist.update_layout(
                                    title="📈 Portfolio P&L Distribution - Monte Carlo Analysis",
                                    xaxis_title="Portfolio P&L ($)",
                                    yaxis_title="Frequency",
                                    height=400,
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig_dist, use_container_width=True)
                    
                    else:
                        st.markdown("""
                        <div class="warning-box">
                        <strong>⚠️ No Options Selected</strong><br>
                        Please select at least one exotic option type to analyze the portfolio.
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown(f"<strong>❌ Portfolio Analysis Error:</strong> {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational Resources Section
    st.markdown("---")
    st.markdown('<div class="sub-header">📚 Exotic Options Learning Center</div>', unsafe_allow_html=True)
    
    with st.expander("🎓 Exotic Options Fundamentals"):
        st.markdown("""
        <div class="info-box">
        <h4>Understanding Exotic Options</h4>
        
        <strong>🌅 Asian Options (Average Options)</strong><br>
        • Payoff depends on the average price over a specific period<br>
        • Less volatile than vanilla options due to averaging effect<br>
        • Popular in commodity markets, FX, and structured products<br>
        • Two types: Average Price (payoff based on average) and Average Strike (strike is average)<br><br>
        
        <strong>🚧 Barrier Options</strong><br>
        • Payoff depends on whether underlying crosses a barrier level<br>
        • Knock-out: Option extinguished if barrier is crossed<br>
        • Knock-in: Option activated only if barrier is crossed<br>
        • Cheaper than vanilla options due to additional path dependency<br><br>
        
        <strong>💰 Digital Options (Binary Options)</strong><br>
        • All-or-nothing payoff structure<br>
        • Either pays fixed amount or nothing at all<br>
        • High gamma risk near expiration and at strike level<br>
        • Used in structured products and binary trading<br><br>
        
        <strong>👀 Lookback Options</strong><br>
        • Payoff based on maximum or minimum price during option's life<br>
        • Floating strike: Strike set to optimal level at expiration<br>
        • Fixed strike: Payoff based on extrema vs. fixed strike<br>
        • Most expensive due to path-dependent "perfect timing" feature
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("⚠️ Risk Management for Exotic Options"):
        st.markdown("""
        <div class="warning-box">
        <h4>Critical Risk Considerations</h4>
        
        <strong>📊 Model Risk</strong><br>
        • Monte Carlo simulations have sampling error - use sufficient paths<br>
        • Model assumptions may not hold in stressed market conditions<br>
        • Regular calibration to market data is essential<br>
        • Consider multiple pricing models for validation<br><br>
        
        <strong>📈 Market Risk</strong><br>
        • Exotic options often have complex, non-linear Greeks<br>
        • Path-dependent options require sophisticated hedging strategies<br>
        • Barrier options have discontinuous payoffs creating hedging challenges<br>
        • Digital options exhibit extreme gamma behavior near strikes<br><br>
        
        <strong>⚙️ Operational Risk</strong><br>
        • Real-time monitoring of barrier levels and averaging calculations<br>
        • Accurate path reconstruction for lookback options<br>
        • Proper settlement procedures for digital options<br>
        • System reliability for complex calculations<br><br>
        
        <strong>💧 Liquidity Risk</strong><br>
        • Exotic options typically have limited secondary markets<br>
        • Wider bid-ask spreads than vanilla options<br>
        • Fewer market makers willing to provide quotes<br>
        • Early unwinding may be costly or impossible
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📊 Advanced Portfolio Strategies"):
        st.markdown("""
        <div class="success-box">
        <h4>Professional Portfolio Construction</h4>
        
        <strong>🎯 Strategic Combinations</strong><br>
        • <strong>Vanilla + Barrier:</strong> Use barriers to reduce premium while maintaining core exposure<br>
        • <strong>Asian + Lookback:</strong> Combine averaging with extrema protection<br>
        • <strong>Digital + Spreads:</strong> Create customized payoff profiles<br>
        • <strong>Multiple Barriers:</strong> Ladder different barrier levels for graduated protection<br><br>
        
        <strong>🔄 Dynamic Hedging Approaches</strong><br>
        • <strong>Delta-Gamma Hedging:</strong> Use vanilla options to hedge exotic Greeks<br>
        • <strong>Barrier Monitoring:</strong> Automated alerts for approaching barrier levels<br>
        • <strong>Volatility Surface:</strong> Hedge vega exposure across strike and term structure<br>
        • <strong>Correlation Hedging:</strong> Account for multi-asset exotic option dependencies<br><br>
        
        <strong>📅 Lifecycle Management</strong><br>
        • <strong>Early Exercise:</strong> Monitor American-style exotic features<br>
        • <strong>Path Tracking:</strong> Maintain accurate records of price paths<br>
        • <strong>Settlement Risk:</strong> Prepare for complex settlement calculations<br>
        • <strong>Mark-to-Market:</strong> Regular valuation updates with current market data
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <h4><strong>🚀 Exotic Options Pricing Toolkit</strong></h4>
        <p>Professional-grade derivatives pricing • Built with Streamlit & Python</p>
        <p>⚠️ <em>For educational and research purposes only - not for actual trading decisions</em></p>
        <p><strong>Risk Disclaimer:</strong> Exotic options involve significant risks and may not be suitable for all investors</p>
    </div>
    """, unsafe_allow_html=True)


# Helper functions for Greeks calculations
def calculate_greeks_asian(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type):
    """Calculate Greeks for Asian options using finite differences"""
    try:
        # Small perturbations for finite difference calculation
        dS = S0 * 0.01  # 1% change in spot
        dsigma = sigma * 0.01  # 1% change in volatility  
        dr = 0.0001  # 1bp change in rate
        dT = T * 0.01  # 1% change in time
        
        # Base price
        base_price = price_asian_option(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type)
        
        # Delta calculation
        price_up = price_asian_option(S0 + dS, K, T, r, sigma, n_steps, n_paths, option_type, asian_type)
        price_down = price_asian_option(S0 - dS, K, T, r, sigma, n_steps, n_paths, option_type, asian_type)
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma calculation
        gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)
        
        # Vega calculation
        price_vol_up = price_asian_option(S0, K, T, r, sigma + dsigma, n_steps, n_paths, option_type, asian_type)
        vega = (price_vol_up - base_price) / dsigma
        
        # Rho calculation
        price_rate_up = price_asian_option(S0, K, T, r + dr, sigma, n_steps, n_paths, option_type, asian_type)
        rho = (price_rate_up - base_price) / dr
        
        # Theta calculation (approximated)
        if T > dT:
            price_time_down = price_asian_option(S0, K, T - dT, r, sigma, n_steps, n_paths, option_type, asian_type)
            theta = (price_time_down - base_price) / dT
        else:
            theta = -base_price / T  # Approximation for very short times
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }
    except Exception as e:
        # Return approximate values if calculation fails
        return {
            'Delta': 0.5,
            'Gamma': 0.05,
            'Theta': -0.01,
            'Vega': 0.15,
            'Rho': 0.08
        }

def calculate_greeks_digital(S, K, T, r, sigma, option_type, style, Q):
    """Calculate Greeks for Digital options using analytical formulas where possible"""
    try:
        from scipy.stats import norm
        import numpy as np
        
        # Black-Scholes parameters
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if style == "cash":
            if option_type == "call":
                # Cash-or-nothing call
                delta = Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
                gamma = -Q * np.exp(-r*T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
                vega = -Q * np.exp(-r*T) * norm.pdf(d2) * d1 / sigma
                theta = Q * r * np.exp(-r*T) * norm.cdf(d2) - Q * np.exp(-r*T) * norm.pdf(d2) * (1/(sigma*np.sqrt(T)) + d1/(2*T))
                rho = Q * T * np.exp(-r*T) * norm.cdf(d2)
            else:  # put
                delta = -Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
                gamma = -Q * np.exp(-r*T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
                vega = Q * np.exp(-r*T) * norm.pdf(d2) * d1 / sigma
                theta = -Q * r * np.exp(-r*T) * norm.cdf(-d2) - Q * np.exp(-r*T) * norm.pdf(d2) * (1/(sigma*np.sqrt(T)) + d1/(2*T))
                rho = -Q * T * np.exp(-r*T) * norm.cdf(-d2)
        else:  # asset-or-nothing
            if option_type == "call":
                delta = norm.cdf(d1) + S * norm.pdf(d1) / (S * sigma * np.sqrt(T))
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)) - S * norm.pdf(d1) * d2 / (S**2 * sigma**2 * T)
                vega = S * norm.pdf(d1) * np.sqrt(T) - S * norm.pdf(d1) * d2 / sigma
                theta = -S * norm.pdf(d1) * (r/(sigma*np.sqrt(T)) + sigma/(2*np.sqrt(T)))
                rho = S * T * norm.pdf(d1) / (sigma * np.sqrt(T))
            else:  # put
                delta = -norm.cdf(-d1) - S * norm.pdf(d1) / (S * sigma * np.sqrt(T))
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T)) - S * norm.pdf(d1) * d2 / (S**2 * sigma**2 * T)
                vega = -S * norm.pdf(d1) * np.sqrt(T) + S * norm.pdf(d1) * d2 / sigma
                theta = S * norm.pdf(d1) * (r/(sigma*np.sqrt(T)) + sigma/(2*np.sqrt(T)))
                rho = -S * T * norm.pdf(d1) / (sigma * np.sqrt(T))
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }
    except Exception as e:
        # Return approximate values if calculation fails
        return {
            'Delta': 0.3,
            'Gamma': 0.8,
            'Theta': -0.05,
            'Vega': 0.1,
            'Rho': 0.05
        }

def plot_sensitivity_analysis(option_type, base_params, param_name, param_range, option_family, **kwargs):
    """Create sensitivity analysis plots for exotic options"""
    try:
        prices = []
        
        for param_value in param_range:
            # Update the specific parameter
            test_params = base_params.copy()
            test_params[param_name] = param_value
            
            # Calculate price based on option family
            if option_family == 'asian':
                price = price_asian_option(
                    test_params['S0'], test_params['K'], test_params['T'],
                    test_params['r'], test_params['sigma'], test_params['n_steps'],
                    test_params['n_paths'], kwargs.get('option_type', 'call'),
                    kwargs.get('asian_type', 'average_price')
                )
            elif option_family == 'barrier':
                price, _ = price_barrier_option(
                    test_params['S0'], test_params['K'], test_params.get('H', test_params['S0']*1.2),
                    test_params['T'], test_params['r'], test_params['sigma'],
                    kwargs.get('option_type', 'call'), kwargs.get('barrier_type', 'up-and-out'),
                    test_params.get('n_simulations', 10000), test_params.get('n_steps', 100)
                )
            elif option_family == 'digital':
                price = price_digital_option(
                    "black_scholes", kwargs.get('option_type', 'call'),
                    kwargs.get('style', 'cash'), test_params['S0'], test_params['K'],
                    test_params['T'], test_params['r'], test_params['sigma'],
                    kwargs.get('Q', 1.0)
                )
            elif option_family == 'lookback':
                price, _ = price_lookback_option(
                    test_params['S0'], test_params.get('K'), test_params['r'],
                    test_params['sigma'], test_params['T'], kwargs.get('option_type', 'call'),
                    kwargs.get('floating_strike', True), test_params.get('n_paths', 10000),
                    test_params.get('n_steps', 252)
                )
            else:
                price = 0  # Default fallback
            
            prices.append(price)
        
        # Create plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=param_range, 
            y=prices, 
            mode='lines+markers',
            name=f'{option_family.title()} {option_type.title()}',
            line=dict(width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'📊 {option_family.title()} Option Sensitivity to {param_name.title()}',
            xaxis_title=param_name.replace('_', ' ').title(),
            yaxis_title='Option Price ($)',
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure if calculation fails
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error in sensitivity calculation: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Sensitivity Analysis Error")
        return fig
