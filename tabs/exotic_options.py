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
    
    st.markdown('<div class="main-header">Exotic Options Pricing Toolkit</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Introduction
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    <strong>Welcome to the Exotic Options Pricing Toolkit!</strong><br>
    This comprehensive application allows you to price and analyze various exotic options including:
    <ul>
    <li><strong>Asian Options</strong> - Options based on average prices</li>
    <li><strong>Barrier Options</strong> - Options with knock-in/knock-out features</li>
    <li><strong>Digital Options</strong> - Binary payoff options</li>
    <li><strong>Lookback Options</strong> - Options based on extrema</li>
    </ul>
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different option types
    tabb1, tabb2, tabb3, tabb4, tabb5 = st.tabs([
        "Asian Options", 
        "Barrier Options", 
        "Digital Options", 
        "Lookback Options",
        "Portfolio Analysis"
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
        st.subheader("📋 Parameters")
        
        asian_S0 = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=0.1, key="asian_s0")
        asian_K = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, key="asian_k")
        asian_T = st.number_input("Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="asian_t")
        asian_r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="asian_r")
        asian_sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="asian_sigma")
        asian_n_steps = st.number_input("Number of Steps", value=252, min_value=10, max_value=1000, key="asian_steps")
        asian_n_paths = st.number_input("Number of Paths", value=10000, min_value=1000, max_value=100000, key="asian_paths")
        
        asian_option_type = st.selectbox("Option Type", ["call", "put"], key="asian_option_type_1")
        asian_type = st.selectbox("Asian Type", ["average_price", "average_strike"], key="asian_type")
        
        calculate_asian = st.button("🔢 Calculate Asian Option", key="calc_asian")
        show_greeks_asian = st.checkbox("📈 Show Greeks", key="show_greeks_asian")
        show_sensitivity_asian = st.checkbox("📊 Sensitivity Analysis", key="show_sens_asian")
    
    with col2:
        if calculate_asian:
            with st.spinner("Calculating Asian option price..."):
                try:
                    # Calculate option price
                    asian_price = price_asian_option(
                        S0=asian_S0, K=asian_K, T=asian_T, r=asian_r, sigma=asian_sigma,
                        n_steps=asian_n_steps, n_paths=asian_n_paths,
                        option_type=asian_option_type, asian_type=asian_type
                    )
                    
                    # Display results
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.success(f"**Asian Option Price: ${asian_price:.4f}**")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate and display Greeks if requested
                    if show_greeks_asian:
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
                    
                except Exception as e:
                    st.error(f"Error calculating Asian option: {str(e)}")


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
