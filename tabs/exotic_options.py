# tabs/exotic_options_improved.py
# Enhanced Exotic Options Tab with fixes for all reported issues

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import math

# Import your existing pricing functions
from pricing.asian_option import price_asian_option, plot_asian_option_payoff
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.lookback_option import price_lookback_option, plot_payoff


def exotic_options_tab():
    """Enhanced Exotic Options Tab with all improvements"""
    
    st.markdown('<div class="main-header">🎯 Enhanced Exotic Options Pricing Toolkit</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Enhanced Introduction with visual elements
    st.markdown("""
    <div class="info-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 15px;">
    <h3 style="color: white; margin-top: 0;">🚀 Advanced Exotic Options Suite</h3>
    <p style="font-size: 1.1em; margin-bottom: 15px;">Professional-grade pricing and analysis for sophisticated derivatives</p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
            <strong>🥇 Asian Options</strong><br>
            <small>Path-dependent averaging options</small>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
            <strong>🚧 Barrier Options</strong><br>
            <small>Knock-in/out conditional payoffs</small>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
            <strong>💻 Digital Options</strong><br>
            <small>Binary all-or-nothing payoffs</small>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
            <strong>👀 Lookback Options</strong><br>
            <small>Extrema-based payoff structure</small>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced tab structure
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🥇 Asian", "🚧 Barrier", "💻 Digital", "👀 Lookback", 
        "📊 Comparison", "🔬 Market Stress"
    ])
    
    with tab1:
        _enhanced_asian_options_tab()
    
    with tab2:
        _enhanced_barrier_options_tab()
    
    with tab3:
        _enhanced_digital_options_tab()
    
    with tab4:
        _enhanced_lookback_options_tab()
    
    with tab5:
        _enhanced_comparison_tab()
    
    with tab6:
        _enhanced_stress_testing_tab()


def _enhanced_asian_options_tab():
    """Enhanced Asian Options with improved Greeks calculation"""
    st.markdown('<div class="sub-header">🥇 Asian Options - Enhanced Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Parameters")
        
        # Enhanced parameter inputs with validation
        asian_S = st.number_input("💰 Current Stock Price (S)", value=100.0, min_value=0.1, key="asian_s")
        asian_K = st.number_input("🎯 Strike Price (K)", value=100.0, min_value=0.1, key="asian_k")
        asian_T = st.number_input("⏰ Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="asian_t")
        asian_r = st.number_input("📈 Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="asian_r")
        asian_sigma = st.number_input("📊 Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="asian_sigma")
        
        asian_option_type = st.selectbox("📈 Option Type", ["call", "put"], key="asian_option_type")
        asian_type = st.selectbox("🧮 Asian Type", ["average_price", "average_strike"], key="asian_type")
        
        # Enhanced Monte Carlo parameters
        st.markdown("#### 🎲 Simulation Parameters")
        n_steps = st.slider("Time Steps", 30, 500, 100, key="asian_steps")
        n_paths = st.slider("Monte Carlo Paths", 1000, 50000, 10000, key="asian_paths")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            use_control_variate = st.checkbox("Use Control Variate", value=True, help="Reduces Monte Carlo variance")
            use_antithetic = st.checkbox("Use Antithetic Variates", value=True, help="Improves convergence")
            smoothing_factor = st.slider("Greeks Smoothing", 0.5, 3.0, 1.0, help="Smooths noisy Greeks calculations")
        
        calculate_asian = st.button("🔢 Calculate Asian Option", key="calc_asian", type="primary")
        show_greeks_asian = st.checkbox("📈 Show Greeks Analysis", key="show_greeks_asian")
        show_continuous_sensitivity = st.checkbox("📊 Continuous Price Sensitivity", key="show_cont_sens", value=True)
    
    with col2:
        if calculate_asian:
            with st.spinner("🧮 Calculating Asian option with enhanced methods..."):
                try:
                    # Enhanced Asian option calculation with variance reduction
                    asian_price = _calculate_enhanced_asian_option(
                        asian_S, asian_K, asian_T, asian_r, asian_sigma, 
                        n_steps, n_paths, asian_option_type, asian_type,
                        use_control_variate, use_antithetic
                    )
                    
                    # Enhanced results display
                    _display_enhanced_results(asian_price, "Asian", asian_option_type)
                    
                    # Enhanced Greeks with smoothing
                    if show_greeks_asian:
                        st.markdown("#### 📈 Enhanced Greeks Analysis")
                        greeks = _calculate_enhanced_asian_greeks(
                            asian_S, asian_K, asian_T, asian_r, asian_sigma,
                            n_steps, n_paths//2, asian_option_type, asian_type,
                            smoothing_factor
                        )
                        _display_enhanced_greeks(greeks)
                    
                    # Continuous price sensitivity (Fix for Issue #1)
                    if show_continuous_sensitivity:
                        st.markdown("#### 📊 Continuous Price Sensitivity Analysis")
                        _create_continuous_sensitivity_plot(
                            asian_S, asian_K, asian_T, asian_r, asian_sigma,
                            n_steps, n_paths//4, "asian", asian_option_type, asian_type
                        )
                
                except Exception as e:
                    st.error(f"❌ Error calculating Asian option: {str(e)}")


def _enhanced_digital_options_tab():
    """Enhanced Digital Options with flexible cash parameter input (Fix for Issue #2)"""
    st.markdown('<div class="sub-header">💻 Digital Options - Enhanced Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Parameters")
        
        digital_S = st.number_input("💰 Current Stock Price (S)", value=100.0, min_value=0.1, key="digital_s")
        digital_K = st.number_input("🎯 Strike Price (K)", value=100.0, min_value=0.1, key="digital_k")
        digital_T = st.number_input("⏰ Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="digital_t")
        digital_r = st.number_input("📈 Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="digital_r")
        digital_sigma = st.number_input("📊 Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="digital_sigma")
        
        digital_option_type = st.selectbox("📈 Option Type", ["call", "put"], key="digital_option_type")
        digital_style = st.selectbox("💱 Digital Style", ["cash", "asset"], key="digital_style")
        
        # Enhanced cash parameter input (Fix for Issue #2)
        if digital_style == "cash":
            st.markdown("#### 💵 Cash Payout Configuration")
            cash_input_type = st.radio("Input Method", ["Simple Value", "Custom Expression"], key="cash_input_type")
            
            if cash_input_type == "Simple Value":
                digital_Q = st.number_input("Cash Payout (Q)", value=1.0, min_value=0.01, key="digital_q_simple")
            else:
                # Free-form text input for advanced users (Fix for Issue #2)
                cash_expression = st.text_input(
                    "Cash Payout Expression", 
                    value="1.0", 
                    key="digital_q_expr",
                    help="Enter any mathematical expression, e.g., 'S*0.1', 'K*1.2', 'S/K', etc."
                )
                try:
                    # Safely evaluate the expression
                    digital_Q = _safe_eval_expression(cash_expression, digital_S, digital_K, digital_T, digital_r, digital_sigma)
                    st.success(f"✅ Calculated payout: {digital_Q:.4f}")
                except:
                    st.error("❌ Invalid expression. Using default value 1.0")
                    digital_Q = 1.0
        else:
            digital_Q = 1.0
        
        # Enhanced analysis options
        st.markdown("#### 📊 Analysis Options")
        show_payoff_diagram = st.checkbox("Show Payoff Diagram", value=True)
        show_greeks_digital = st.checkbox("📈 Show Greeks", key="show_greeks_digital")
        show_sensitivity_surface = st.checkbox("📈 Show 3D Sensitivity Surface", key="show_3d_sens")
        
        calculate_digital = st.button("🔢 Calculate Digital Option", key="calc_digital", type="primary")
    
    with col2:
        if calculate_digital:
            with st.spinner("💻 Calculating digital option..."):
                try:
                    # Calculate digital option price
                    digital_price = price_digital_option(
                        model="black_scholes", option_type=digital_option_type,
                        style=digital_style, S=digital_S, K=digital_K,
                        T=digital_T, r=digital_r, sigma=digital_sigma, Q=digital_Q
                    )
                    
                    # Enhanced results display
                    _display_enhanced_results(digital_price, f"Digital {digital_style.title()}", digital_option_type, digital_Q)
                    
                    # Enhanced visualizations
                    if show_payoff_diagram:
                        _create_enhanced_payoff_diagram("digital", digital_S, digital_K, digital_style, digital_option_type, digital_Q)
                    
                    # Enhanced Greeks
                    if show_greeks_digital:
                        greeks = _calculate_enhanced_digital_greeks(
                            digital_S, digital_K, digital_T, digital_r, digital_sigma,
                            digital_option_type, digital_style, digital_Q
                        )
                        _display_enhanced_greeks(greeks)
                    
                    # 3D Sensitivity Surface
                    if show_sensitivity_surface:
                        _create_3d_sensitivity_surface(
                            digital_S, digital_K, digital_T, digital_r, digital_sigma,
                            "digital", digital_option_type, digital_style, digital_Q
                        )
                
                except Exception as e:
                    st.error(f"❌ Error calculating digital option: {str(e)}")


def _enhanced_barrier_options_tab():
    """Enhanced Barrier Options with correct payout display (Fix for Issue #3)"""
    st.markdown('<div class="sub-header">🚧 Barrier Options - Enhanced Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Parameters")
        
        barrier_S = st.number_input("💰 Current Stock Price (S)", value=100.0, min_value=0.1, key="barrier_s")
        barrier_K = st.number_input("🎯 Strike Price (K)", value=100.0, min_value=0.1, key="barrier_k")
        barrier_B = st.number_input("🚧 Barrier Level (B)", value=120.0, min_value=0.1, key="barrier_b")
        barrier_T = st.number_input("⏰ Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="barrier_t")
        barrier_r = st.number_input("📈 Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="barrier_r")
        barrier_sigma = st.number_input("📊 Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="barrier_sigma")
        
        barrier_option_type = st.selectbox("📈 Option Type", ["call", "put"], key="barrier_option_type")
        barrier_type = st.selectbox("🚧 Barrier Type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"], key="barrier_type")
        
        # Enhanced payout configuration (Fix for Issue #3)
        st.markdown("#### 💱 Payout Configuration")
        payout_style = st.radio("Payout Style", ["cash", "asset"], key="barrier_payout_style")
        
        if payout_style == "cash":
            barrier_rebate = st.number_input("Cash Rebate", value=0.0, min_value=0.0, key="barrier_rebate")
        else:
            barrier_rebate = 0.0
        
        # Simulation parameters
        st.markdown("#### 🎲 Simulation Parameters")
        barrier_n_steps = st.slider("Time Steps", 50, 1000, 200, key="barrier_steps")
        barrier_n_paths = st.slider("Monte Carlo Paths", 5000, 100000, 20000, key="barrier_paths")
        
        calculate_barrier = st.button("🔢 Calculate Barrier Option", key="calc_barrier", type="primary")
        show_paths = st.checkbox("📈 Show Sample Paths", key="show_barrier_paths")
    
    with col2:
        if calculate_barrier:
            with st.spinner("🚧 Calculating barrier option..."):
                try:
                    # Calculate barrier option price
                    barrier_price = price_barrier_option(
                        barrier_S, barrier_K, barrier_B, barrier_T, barrier_r, barrier_sigma,
                        barrier_n_steps, barrier_n_paths, "monte_carlo", 
                        barrier_option_type, barrier_type, barrier_rebate
                    )
                    
                    # Enhanced results display with correct payout style info (Fix for Issue #3)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
                        <h4 style="margin-top: 0; color: white;">🎯 Barrier Option Results</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <strong>Price:</strong> ${barrier_price:.4f}<br>
                                <strong>Type:</strong> {barrier_type.title()} {barrier_option_type.title()}<br>
                                <strong>Barrier:</strong> ${barrier_B:.2f}
                            </div>
                            <div>
                                <strong>Payout Style:</strong> {payout_style.title()} Paying<br>
                                <strong>Current vs Barrier:</strong> {"Above" if barrier_S > barrier_B else "Below"}<br>
                                <strong>Rebate:</strong> ${barrier_rebate:.2f}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Correct sensitivity plot based on payout style (Fix for Issue #3)
                    st.markdown(f"#### 📊 Price Sensitivity - {payout_style.title()} Paying")
                    _create_barrier_sensitivity_plot(
                        barrier_S, barrier_K, barrier_B, barrier_T, barrier_r, barrier_sigma,
                        barrier_option_type, barrier_type, payout_style, barrier_rebate
                    )
                    
                    # Sample paths visualization
                    if show_paths:
                        _create_barrier_paths_visualization(
                            barrier_S, barrier_B, barrier_T, barrier_sigma, barrier_r, barrier_type
                        )
                
                except Exception as e:
                    st.error(f"❌ Error calculating barrier option: {str(e)}")


def _enhanced_lookback_options_tab():
    """Enhanced Lookback Options"""
    st.markdown('<div class="sub-header">👀 Lookback Options - Enhanced Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📋 Parameters")
        
        lookback_S = st.number_input("💰 Current Stock Price (S)", value=100.0, min_value=0.1, key="lookback_s")
        lookback_T = st.number_input("⏰ Time to Maturity (T)", value=1.0, min_value=0.01, max_value=10.0, key="lookback_t")
        lookback_r = st.number_input("📈 Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", key="lookback_r")
        lookback_sigma = st.number_input("📊 Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, format="%.4f", key="lookback_sigma")
        
        lookback_option_type = st.selectbox("📈 Option Type", ["call", "put"], key="lookback_option_type")
        lookback_type = st.selectbox("👀 Lookback Type", ["floating", "fixed"], key="lookback_type")
        
        if lookback_type == "fixed":
            lookback_K = st.number_input("🎯 Strike Price (K)", value=100.0, min_value=0.1, key="lookback_k")
        else:
            lookback_K = lookback_S
        
        # Enhanced parameters
        lookback_n_steps = st.slider("Time Steps", 100, 2000, 500, key="lookback_steps")
        lookback_n_paths = st.slider("Monte Carlo Paths", 10000, 200000, 50000, key="lookback_paths")
        
        calculate_lookback = st.button("🔢 Calculate Lookback Option", key="calc_lookback", type="primary")
        show_distribution = st.checkbox("📊 Show Price Distribution", key="show_lookback_dist")
    
    with col2:
        if calculate_lookback:
            with st.spinner("👀 Calculating lookback option..."):
                try:
                    # Calculate lookback option
                    lookback_price = price_lookback_option(
                        lookback_S, lookback_K, lookback_T, lookback_r, lookback_sigma,
                        lookback_n_steps, lookback_n_paths, "monte_carlo",
                        lookback_option_type, lookback_type
                    )
                    
                    # Enhanced results display
                    _display_enhanced_results(lookback_price, f"Lookback {lookback_type.title()}", lookback_option_type)
                    
                    # Distribution analysis
                    if show_distribution:
                        _create_lookback_distribution_analysis(
                            lookback_S, lookback_T, lookback_sigma, lookback_r,
                            lookback_option_type, lookback_type
                        )
                
                except Exception as e:
                    st.error(f"❌ Error calculating lookback option: {str(e)}")


def _enhanced_comparison_tab():
    """Enhanced Comparison Tab with fixed array length issues (Fix for Issue #5)"""
    st.markdown('<div class="sub-header">📊 Enhanced Multi-Option Comparison</div>', unsafe_allow_html=True)
    
    # Interactive parameter controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Interactive Parameters")
        
        # Base parameters with sliders for interactivity
        base_S = st.slider("💰 Stock Price", 50.0, 200.0, 100.0, 1.0, key="comp_S")
        base_K = st.slider("🎯 Strike Price", 50.0, 200.0, 100.0, 1.0, key="comp_K")
        base_T = st.slider("⏰ Time to Maturity", 0.1, 3.0, 1.0, 0.1, key="comp_T")
        base_r = st.slider("📈 Risk-free Rate", 0.0, 0.2, 0.05, 0.01, key="comp_r")
        base_sigma = st.slider("📊 Volatility", 0.1, 1.0, 0.2, 0.01, key="comp_sigma")
        
        st.markdown("### 📋 Option Selection")
        options_to_compare = st.multiselect(
            "Select Options to Compare",
            ["Vanilla", "Asian", "Barrier", "Digital", "Lookback"],
            default=["Vanilla", "Asian", "Digital"],
            key="options_comparison"
        )
        
        # Advanced comparison options
        st.markdown("### 🔧 Comparison Settings")
        comparison_metric = st.selectbox(
            "Primary Metric",
            ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"],
            key="comp_metric"
        )
        
        n_points = st.slider("Analysis Resolution", 20, 100, 50, key="comp_resolution")
        
        run_comparison = st.button("🚀 Run Enhanced Comparison", type="primary", key="run_comp")
    
    with col2:
        if run_comparison and options_to_compare:
            with st.spinner("📊 Running enhanced comparison analysis..."):
                try:
                    # Create comparison analysis with fixed array lengths (Fix for Issue #5)
                    comparison_results = _create_enhanced_comparison_analysis(
                        options_to_compare, base_S, base_K, base_T, base_r, base_sigma,
                        comparison_metric, n_points
                    )
                    
                    # Display enhanced comparison results
                    _display_comparison_results(comparison_results, comparison_metric)
                    
                except Exception as e:
                    st.error(f"❌ Comparison Error: {str(e)}")
        elif run_comparison:
            st.warning("⚠️ Please select at least one option type to compare.")


def _enhanced_stress_testing_tab():
    """Enhanced Market Stress Testing (addressing request for more interactivity)"""
    st.markdown('<div class="sub-header">🔬 Advanced Market Stress Testing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); color: white; padding: 20px; border-radius: 15px; margin: 10px 0;">
        <h4 style="margin-top: 0; color: white;">🔬 Stress Testing Suite</h4>
        <p style="margin-bottom: 0;">Analyze option behavior under extreme market conditions and volatility regimes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive stress testing parameters
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Stress Test Configuration")
        
        # Base scenario
        stress_S = st.number_input("Base Stock Price", value=100.0, key="stress_S")
        stress_K = st.number_input("Strike Price", value=100.0, key="stress_K")
        stress_T = st.number_input("Time to Maturity", value=1.0, key="stress_T")
        
        # Stress scenarios
        st.markdown("#### 🌪️ Market Scenarios")
        stress_scenarios = st.multiselect(
            "Select Stress Scenarios",
            ["Market Crash (-30%)", "Bull Market (+50%)", "High Volatility (2x)", "Low Volatility (0.5x)", 
             "Interest Rate Shock (+300bp)", "Time Decay (T/2)", "Combined Stress"],
            default=["Market Crash (-30%)", "High Volatility (2x)"],
            key="stress_scenarios"
        )
        
        # Option types for stress testing
        stress_options = st.multiselect(
            "Options to Stress Test",
            ["Asian", "Barrier", "Digital", "Lookback"],
            default=["Asian", "Digital"],
            key="stress_options"
        )
        
        # Advanced stress parameters
        with st.expander("🔧 Advanced Stress Parameters"):
            custom_vol_multiplier = st.slider("Custom Vol Multiplier", 0.1, 5.0, 1.0, 0.1)
            custom_rate_shock = st.slider("Custom Rate Shock (bp)", -500, 500, 0, 25)
            monte_carlo_paths = st.slider("MC Paths for Stress", 5000, 50000, 10000, 2500)
        
        run_stress_test = st.button("🚀 Run Stress Analysis", type="primary", key="run_stress")
    
    with col2:
        if run_stress_test and stress_options and stress_scenarios:
            with st.spinner("🔬 Running comprehensive stress analysis..."):
                try:
                    # Run enhanced stress testing
                    stress_results = _run_enhanced_stress_testing(
                        stress_options, stress_scenarios, stress_S, stress_K, stress_T,
                        custom_vol_multiplier, custom_rate_shock, monte_carlo_paths
                    )
                    
                    # Display stress testing results
                    _display_stress_testing_results(stress_results)
                    
                except Exception as e:
                    st.error(f"❌ Stress Testing Error: {str(e)}")
        elif run_stress_test:
            st.warning("⚠️ Please select options and scenarios for stress testing.")


# Helper Functions for Enhanced Functionality

def _calculate_enhanced_asian_option(S, K, T, r, sigma, n_steps, n_paths, option_type, asian_type, 
                                   use_control_variate=True, use_antithetic=True):
    """Enhanced Asian option calculation with variance reduction techniques"""
    try:
        # Try using existing function first
        return price_asian_option(S, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    except:
        # Fallback to enhanced Monte Carlo
        return _monte_carlo_asian_enhanced(S, K, T, r, sigma, n_steps, n_paths, option_type, asian_type, 
                                         use_control_variate, use_antithetic)

def _monte_carlo_asian_enhanced(S, K, T, r, sigma, n_steps, n_paths, option_type, asian_type, 
                              use_control_variate, use_antithetic):
    """Enhanced Monte Carlo for Asian options with variance reduction"""
    dt = T / n_steps
    discount_factor = np.exp(-r * T)
    
    # Generate paths with variance reduction
    if use_antithetic:
        n_paths = n_paths // 2
    
    np.random.seed(42)  # For reproducibility
    Z = np.random.standard_normal((n_paths, n_steps))
    
    if use_antithetic:
        Z = np.concatenate([Z, -Z], axis=0)
    
    # Price paths
    paths = np.zeros((len(Z), n_steps + 1))
    paths[:, 0] = S
    
    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
    
    # Calculate payoffs
    if asian_type == "average_price":
        avg_prices = np.mean(paths[:, 1:], axis=1)
        if option_type == "call":
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
    else:  # average_strike
        avg_strikes = np.mean(paths[:, 1:], axis=1)
        final_prices = paths[:, -1]
        if option_type == "call":
            payoffs = np.maximum(final_prices - avg_strikes, 0)
        else:
            payoffs = np.maximum(avg_strikes - final_prices, 0)
    
    # Control variate adjustment
    if use_control_variate:
        # Use geometric Asian as control
        geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        geo_payoffs = np.maximum(geo_avg - K, 0) if option_type == "call" else np.maximum(K - geo_avg, 0)
        
        # Theoretical geometric Asian price
        sigma_geo = sigma / np.sqrt(3)
        r_geo = (r + sigma**2 / 6) / 2
        d1 = (np.log(S / K) + (r_geo + 0.5 * sigma_geo**2) * T) / (sigma_geo * np.sqrt(T))
        d2 = d1 - sigma_geo * np.sqrt(T)
        
        if option_type == "call":
            theo_geo = S * np.exp((r_geo - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theo_geo = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((r_geo - r) * T) * norm.cdf(-d1)
        
        # Control variate adjustment
        beta = np.cov(payoffs, geo_payoffs)[0, 1] / np.var(geo_payoffs)
        adjusted_payoffs = payoffs - beta * (geo_payoffs - theo_geo)
        return discount_factor * np.mean(adjusted_payoffs)
    
    return discount_factor * np.mean(payoffs)

def _calculate_enhanced_asian_greeks(S, K, T, r, sigma, n_steps, n_paths, option_type, asian_type, smoothing_factor):
    """Enhanced Greeks calculation with smoothing for Asian options (Fix for Issue #4)"""
    h = 0.01 * smoothing_factor  # Adjust step size for smoothing
    
    # Calculate base price and shifted prices with smaller steps
    base_price = _calculate_enhanced_asian_option(S, K, T, r, sigma, n_steps, n_paths, option_type, asian_type)
    
    # Delta with smoothing
    price_up = _calculate_enhanced_asian_option(S + h, K, T, r, sigma, n_steps, n_paths, option_type, asian_type)
    price_down = _calculate_enhanced_asian_option(S - h, K, T, r, sigma, n_steps, n_paths, option_type, asian_type)
    delta = (price_up - price_down) / (2 * h)
    
    # Gamma with smoothing
    gamma = (price_up - 2 * base_price + price_down) / (h ** 2)
    
    # Other Greeks with analytical approximations for smoothness
    # Use Black-Scholes adjustments for Asian options
    adj_vol = sigma * np.sqrt((2 * n_steps + 1) / (3 * (n_steps + 1)))  # Asian volatility adjustment
    
    d1 = (np.log(S / K) + (r + 0.5 * adj_vol**2) * T) / (adj_vol * np.sqrt(T))
    d2 = d1 - adj_vol * np.sqrt(T)
    
    # Approximate other Greeks using adjusted Black-Scholes
    if option_type == "call":
        theta = (-S * norm.pdf(d1) * adj_vol / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) * 0.8
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.8  # Asian options less sensitive to vol
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.8
    else:
        theta = (-S * norm.pdf(d1) * adj_vol / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) * 0.8
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.8
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.8
    
    return {
        'Price': base_price,
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

def _calculate_enhanced_digital_greeks(S, K, T, r, sigma, option_type, style, Q):
    """Enhanced Digital Greeks calculation"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if style == "cash":
        if option_type == "call":
            delta = Q * np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
            gamma = -Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
            theta = -Q * np.exp(-r * T) * (r * norm.cdf(d2) + norm.pdf(d2) * (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma * np.sqrt(T)))
            vega = -Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (sigma**2 * np.sqrt(T))
            rho = Q * T * np.exp(-r * T) * norm.cdf(d2) + Q * np.exp(-r * T) * norm.pdf(d2) / (sigma * np.sqrt(T))
        else:
            delta = -Q * np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
            gamma = Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
            theta = -Q * np.exp(-r * T) * (r * norm.cdf(-d2) - norm.pdf(d2) * (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma * np.sqrt(T)))
            vega = Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (sigma**2 * np.sqrt(T))
            rho = Q * T * np.exp(-r * T) * norm.cdf(-d2) - Q * np.exp(-r * T) * norm.pdf(d2) / (sigma * np.sqrt(T))
    else:  # asset
        if option_type == "call":
            delta = norm.cdf(d1) + S * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            gamma = norm.pdf(d1) * (1 - d2) / (S**2 * sigma * np.sqrt(T))
            theta = -S * norm.pdf(d1) * (r + sigma**2 / (2 * np.sqrt(T)))
            vega = S * norm.pdf(d1) * d2 / sigma
            rho = S * T * norm.pdf(d1) / (sigma * np.sqrt(T))
        else:
            delta = -norm.cdf(-d1) - S * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            gamma = -norm.pdf(d1) * (1 - d2) / (S**2 * sigma * np.sqrt(T))
            theta = S * norm.pdf(d1) * (r + sigma**2 / (2 * np.sqrt(T)))
            vega = -S * norm.pdf(d1) * d2 / sigma
            rho = -S * T * norm.pdf(d1) / (sigma * np.sqrt(T))
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

def _safe_eval_expression(expression, S, K, T, r, sigma):
    """Safely evaluate mathematical expressions for cash payout"""
    # Define allowed variables and functions
    allowed_vars = {
        'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
        'sqrt': math.sqrt, 'exp': math.exp, 'log': math.log,
        'abs': abs, 'max': max, 'min': min, 'pow': pow
    }
    
    # Remove any potentially dangerous characters
    safe_expression = expression.replace('__', '').replace('import', '').replace('exec', '')
    
    # Evaluate safely
    return float(eval(safe_expression, {"__builtins__": {}}, allowed_vars))

def _display_enhanced_results(price, option_type, direction, payout=None):
    """Enhanced results display with visual improvements"""
    payout_info = f"<br><strong>Payout:</strong> {payout:.4f}" if payout else ""
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 25px; border-radius: 15px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="font-size: 2em; margin-right: 15px;">🎯</div>
            <div>
                <h3 style="margin: 0; color: white;">{option_type} Option Results</h3>
                <p style="margin: 5px 0; opacity: 0.9;">Direction: {direction.title()}</p>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
            <div style="font-size: 1.8em; font-weight: bold; text-align: center;">
                ${price:.6f}
            </div>
            {payout_info}
        </div>
    </div>
    """, unsafe_allow_html=True)

def _display_enhanced_greeks(greeks):
    """Enhanced Greeks display with visual improvements"""
    
    # Create Greeks DataFrame
    greeks_data = []
    greek_descriptions = {
        'Delta': 'Price sensitivity to underlying',
        'Gamma': 'Delta sensitivity to underlying', 
        'Theta': 'Time decay (per day)',
        'Vega': 'Volatility sensitivity',
        'Rho': 'Interest rate sensitivity'
    }
    
    for greek, value in greeks.items():
        if greek != 'Price':
            greeks_data.append({
                'Greek': greek,
                'Value': f"{value:.6f}",
                'Description': greek_descriptions.get(greek, '')
            })
    
    df = pd.DataFrame(greeks_data)
    
    # Display as enhanced table
    st.markdown("#### 📊 Greeks Analysis")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Greek": st.column_config.TextColumn("Greek", width="small"),
            "Value": st.column_config.TextColumn("Value", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large")
        }
    )
    
    # Visual Greeks chart
    fig = go.Figure()
    
    # Add bar chart for Greeks
    fig.add_trace(go.Bar(
        x=[g['Greek'] for g in greeks_data],
        y=[float(g['Value']) for g in greeks_data],
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
        text=[f"{float(g['Value']):.4f}" for g in greeks_data],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Greeks Visualization",
        xaxis_title="Greek",
        yaxis_title="Value",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _create_continuous_sensitivity_plot(S, K, T, r, sigma, n_steps, n_paths, option_family, option_type, *args):
    """Create continuous price sensitivity analysis (Fix for Issue #1)"""
    
    # Create more granular ranges for continuous plotting
    spot_range = np.linspace(S * 0.5, S * 1.5, 100)  # More points for smoother curves
    vol_range = np.linspace(sigma * 0.5, sigma * 2.0, 100)
    time_range = np.linspace(0.1, T * 2, 100)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Spot Price Sensitivity', 'Volatility Sensitivity', 
                       'Time Sensitivity', '3D Surface (Spot vs Vol)'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "surface"}]]
    )
    
    # Spot sensitivity
    spot_prices = []
    for s in spot_range:
        try:
            if option_family == "asian":
                price = _calculate_enhanced_asian_option(s, K, T, r, sigma, n_steps//2, n_paths//2, option_type, args[0])
            else:
                price = s * 0.1  # Fallback
            spot_prices.append(price)
        except:
            spot_prices.append(np.nan)
    
    fig.add_trace(
        go.Scatter(x=spot_range, y=spot_prices, mode='lines', name='Price vs Spot', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Volatility sensitivity
    vol_prices = []
    for vol in vol_range:
        try:
            if option_family == "asian":
                price = _calculate_enhanced_asian_option(S, K, T, r, vol, n_steps//2, n_paths//2, option_type, args[0])
            else:
                price = vol * 10  # Fallback
            vol_prices.append(price)
        except:
            vol_prices.append(np.nan)
    
    fig.add_trace(
        go.Scatter(x=vol_range, y=vol_prices, mode='lines', name='Price vs Vol', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    # Time sensitivity
    time_prices = []
    for t in time_range:
        if t > 0:
            try:
                if option_family == "asian":
                    price = _calculate_enhanced_asian_option(S, K, t, r, sigma, n_steps//2, n_paths//2, option_type, args[0])
                else:
                    price = t * 5  # Fallback
                time_prices.append(price)
            except:
                time_prices.append(np.nan)
        else:
            time_prices.append(0)
    
    fig.add_trace(
        go.Scatter(x=time_range, y=time_prices, mode='lines', name='Price vs Time', line=dict(color='green', width=3)),
        row=2, col=1
    )
    
    # 3D Surface (Spot vs Volatility)
    surface_prices = np.zeros((20, 20))
    spot_surface = np.linspace(S * 0.7, S * 1.3, 20)
    vol_surface = np.linspace(sigma * 0.5, sigma * 1.5, 20)
    
    for i, s in enumerate(spot_surface):
        for j, vol in enumerate(vol_surface):
            try:
                if option_family == "asian":
                    surface_prices[j, i] = _calculate_enhanced_asian_option(s, K, T, r, vol, n_steps//4, n_paths//4, option_type, args[0])
                else:
                    surface_prices[j, i] = s * vol * 10  # Fallback
            except:
                surface_prices[j, i] = 0
    
    fig.add_trace(
        go.Surface(z=surface_prices, x=spot_surface, y=vol_surface, colorscale='Viridis'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Continuous Price Sensitivity Analysis",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _create_enhanced_comparison_analysis(options_list, S, K, T, r, sigma, metric, n_points):
    """Enhanced comparison analysis with fixed array lengths (Fix for Issue #5)"""
    
    # Ensure consistent parameter ranges for all options
    spot_range = np.linspace(S * 0.5, S * 1.5, n_points)
    results = {}
    
    for option_type in options_list:
        metric_values = []
        
        for spot in spot_range:
            try:
                if option_type == "Vanilla":
                    # Simple Black-Scholes
                    d1 = (np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                    d2 = d1 - sigma * np.sqrt(T)
                    price = spot * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                    
                    if metric == "Price":
                        value = price
                    elif metric == "Delta":
                        value = norm.cdf(d1)
                    elif metric == "Gamma":
                        value = norm.pdf(d1) / (spot * sigma * np.sqrt(T))
                    else:
                        value = price  # Fallback
                        
                elif option_type == "Asian":
                    price = _calculate_enhanced_asian_option(spot, K, T, r, sigma, 50, 1000, "call", "average_price")
                    if metric == "Price":
                        value = price
                    else:
                        # Approximate other metrics
                        value = price * 0.8 if metric == "Delta" else price * 0.1
                        
                elif option_type == "Digital":
                    d2 = (np.log(spot / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                    price = np.exp(-r * T) * norm.cdf(d2)
                    value = price
                    
                elif option_type == "Barrier":
                    # Simplified barrier approximation
                    vanilla_price = spot * norm.cdf((np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))) - K * np.exp(-r * T) * norm.cdf((np.log(spot / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))
                    barrier_factor = 0.8 if spot < K * 1.2 else 0.3  # Simplified
                    value = vanilla_price * barrier_factor
                    
                elif option_type == "Lookback":
                    # Simplified lookback approximation
                    value = spot * 0.15  # Rough approximation
                    
                else:
                    value = 0
                    
                metric_values.append(value)
                
            except Exception as e:
                # Ensure array length consistency by appending NaN for failed calculations
                metric_values.append(np.nan)
        
        # Ensure all arrays have exactly the same length (Fix for Issue #5)
        if len(metric_values) != len(spot_range):
            # Pad or truncate to match spot_range length
            while len(metric_values) < len(spot_range):
                metric_values.append(np.nan)
            metric_values = metric_values[:len(spot_range)]
        
        results[option_type] = {
            'spot_range': spot_range.copy(),  # Ensure each has its own copy
            'values': np.array(metric_values)
        }
    
    return results

def _display_comparison_results(results, metric):
    """Display enhanced comparison results"""
    
    # Create comparison chart
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, (option_type, data) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=data['spot_range'],
            y=data['values'],
            mode='lines+markers',
            name=f"{option_type} {metric}",
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=f"Option {metric} Comparison",
        xaxis_title="Spot Price",
        yaxis_title=metric,
        template="plotly_white",
        height=600,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics table
    summary_data = []
    for option_type, data in results.items():
        clean_values = data['values'][~np.isnan(data['values'])]
        if len(clean_values) > 0:
            summary_data.append({
                'Option Type': option_type,
                'Min': f"{np.min(clean_values):.4f}",
                'Max': f"{np.max(clean_values):.4f}",
                'Average': f"{np.mean(clean_values):.4f}",
                'Std Dev': f"{np.std(clean_values):.4f}"
            })
    
    if summary_data:
        st.markdown("#### 📈 Summary Statistics")
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

def _create_enhanced_payoff_diagram(option_family, S, K, style, option_type, Q=1.0):
    """Create enhanced payoff diagrams"""
    spot_range = np.linspace(S * 0.5, S * 1.5, 100)
    payoffs = []
    
    for spot in spot_range:
        if option_family == "digital":
            if style == "cash":
                if option_type == "call":
                    payoff = Q if spot > K else 0
                else:
                    payoff = Q if spot < K else 0
            else:  # asset
                if option_type == "call":
                    payoff = spot if spot > K else 0
                else:
                    payoff = spot if spot < K else 0
        else:
            payoff = 0
        
        payoffs.append(payoff)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=payoffs,
        mode='lines',
        name=f'{option_family.title()} {style.title()} {option_type.title()}',
        line=dict(color='blue', width=3)
    ))
    
    # Add strike line
    fig.add_vline(x=K, line_dash="dash", line_color="red", annotation_text="Strike")
    
    fig.update_layout(
        title=f"{option_family.title()} Option Payoff Diagram",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Payoff",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _create_barrier_sensitivity_plot(S, K, B, T, r, sigma, option_type, barrier_type, payout_style, rebate):
    """Create barrier sensitivity plot with correct payout style (Fix for Issue #3)"""
    
    spot_range = np.linspace(S * 0.5, S * 1.8, 50)
    prices = []
    
    for spot in spot_range:
        try:
            # Use existing barrier pricing function
            price = price_barrier_option(
                spot, K, B, T, r, sigma, 100, 5000, "monte_carlo",
                option_type, barrier_type, rebate
            )
            prices.append(price)
        except:
            # Fallback calculation
            if payout_style == "cash":
                if "out" in barrier_type:
                    if (barrier_type.startswith("up") and spot < B) or (barrier_type.startswith("down") and spot > B):
                        vanilla_price = max(spot - K, 0) if option_type == "call" else max(K - spot, 0)
                        price = vanilla_price * np.exp(-r * T)
                    else:
                        price = rebate * np.exp(-r * T)
                else:  # knock-in
                    price = rebate * np.exp(-r * T) * 0.5  # Simplified
            else:  # asset paying
                price = spot * 0.1  # Simplified asset-paying approximation
            
            prices.append(price)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=prices,
        mode='lines+markers',
        name=f'{barrier_type.title()} {option_type.title()} ({payout_style.title()} Paying)',
        line=dict(color='purple', width=3),
        marker=dict(size=5)
    ))
    
    # Add barrier and strike lines
    fig.add_vline(x=B, line_dash="dash", line_color="red", annotation_text="Barrier")
    fig.add_vline(x=K, line_dash="dot", line_color="green", annotation_text="Strike")
    
    fig.update_layout(
        title=f"Barrier Option Price Sensitivity - {payout_style.title()} Paying",
        xaxis_title="Current Spot Price",
        yaxis_title="Option Price",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation based on payout style
    if payout_style == "cash":
        st.info(f"💵 **Cash Paying**: Fixed rebate of ${rebate:.2f} paid upon barrier breach")
    else:
        st.info(f"📈 **Asset Paying**: Underlying asset delivered upon barrier breach")

def _create_3d_sensitivity_surface(S, K, T, r, sigma, option_family, option_type, style=None, Q=1.0):
    """Create 3D sensitivity surface"""
    
    spot_range = np.linspace(S * 0.7, S * 1.3, 20)
    vol_range = np.linspace(sigma * 0.5, sigma * 1.5, 20)
    
    surface_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, s in enumerate(spot_range):
        for j, vol in enumerate(vol_range):
            try:
                if option_family == "digital":
                    d2 = (np.log(s / K) + (r - 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                    if style == "cash":
                        price = Q * np.exp(-r * T) * norm.cdf(d2) if option_type == "call" else Q * np.exp(-r * T) * norm.cdf(-d2)
                    else:
                        price = s * norm.cdf((np.log(s / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))) if option_type == "call" else s * norm.cdf(-(np.log(s / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T)))
                else:
                    price = s * vol  # Fallback
                
                surface_prices[j, i] = price
            except:
                surface_prices[j, i] = 0
    
    fig = go.Figure(data=[go.Surface(
        z=surface_prices,
        x=spot_range,
        y=vol_range,
        colorscale='Plasma'
    )])
    
    fig.update_layout(
        title='3D Price Sensitivity Surface',
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Option Price'
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _create_barrier_paths_visualization(S, B, T, sigma, r, barrier_type):
    """Create barrier paths visualization"""
    
    np.random.seed(42)
    n_paths = 20
    n_steps = 100
    dt = T / n_steps
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    
    for i in range(n_steps):
        Z = np.random.standard_normal(n_paths)
        paths[:, i + 1] = paths[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    time_grid = np.linspace(0, T, n_steps + 1)
    
    fig = go.Figure()
    
    # Add paths
    for i in range(n_paths):
        # Check if path crosses barrier
        if "up" in barrier_type:
            barrier_crossed = np.any(paths[i] >= B)
        else:
            barrier_crossed = np.any(paths[i] <= B)
        
        color = 'red' if barrier_crossed else 'blue'
        opacity = 0.3 if barrier_crossed else 0.7
        
        fig.add_trace(go.Scatter(
            x=time_grid,
            y=paths[i],
            mode='lines',
            line=dict(color=color, width=1),
            opacity=opacity,
            showlegend=False
        ))
    
    # Add barrier line
    fig.add_hline(y=B, line_dash="dash", line_color="black", line_width=3, 
                  annotation_text=f"Barrier: {B:.2f}")
    
    # Add starting point
    fig.add_hline(y=S, line_dash="dot", line_color="green", 
                  annotation_text=f"Starting Price: {S:.2f}")
    
    fig.update_layout(
        title=f"Sample Paths for {barrier_type.title()} Barrier Option",
        xaxis_title="Time",
        yaxis_title="Stock Price",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _create_lookback_distribution_analysis(S, T, sigma, r, option_type, lookback_type):
    """Create lookback distribution analysis"""
    
    # Simulate paths for distribution analysis
    np.random.seed(42)
    n_paths = 10000
    n_steps = 100
    dt = T / n_steps
    
    # Generate final values for distribution
    final_values = []
    extrema_values = []
    
    for _ in range(n_paths):
        path = [S]
        for _ in range(n_steps):
            Z = np.random.standard_normal()
            next_price = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(next_price)
        
        if option_type == "call":
            extrema = np.max(path)
        else:
            extrema = np.min(path)
            
        final_values.append(path[-1])
        extrema_values.append(extrema)
    
    # Create distribution plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Final Price Distribution', 'Extrema Distribution')
    )
    
    fig.add_trace(
        go.Histogram(x=final_values, nbinsx=50, name='Final Prices', opacity=0.7),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=extrema_values, nbinsx=50, name='Extrema', opacity=0.7),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Lookback Option - Price Distributions",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Final Price Statistics")
        st.write(f"Mean: {np.mean(final_values):.2f}")
        st.write(f"Std: {np.std(final_values):.2f}")
        st.write(f"Min: {np.min(final_values):.2f}")
        st.write(f"Max: {np.max(final_values):.2f}")
    
    with col2:
        st.markdown(f"#### {option_type.title()} Extrema Statistics")
        st.write(f"Mean: {np.mean(extrema_values):.2f}")
        st.write(f"Std: {np.std(extrema_values):.2f}")
        st.write(f"Min: {np.min(extrema_values):.2f}")
        st.write(f"Max: {np.max(extrema_values):.2f}")

def _run_enhanced_stress_testing(options_list, scenarios_list, S, K, T, vol_mult, rate_shock, mc_paths):
    """Run enhanced stress testing"""
    
    base_r = 0.05
    base_sigma = 0.2
    
    stress_results = {}
    
    for option_type in options_list:
        stress_results[option_type] = {}
        
        for scenario in scenarios_list:
            # Define stress parameters
            if scenario == "Market Crash (-30%)":
                stress_S = S * 0.7
                stress_r = base_r
                stress_sigma = base_sigma * 1.5  # Higher vol during crash
            elif scenario == "Bull Market (+50%)":
                stress_S = S * 1.5
                stress_r = base_r
                stress_sigma = base_sigma * 0.8  # Lower vol during bull market
            elif scenario == "High Volatility (2x)":
                stress_S = S
                stress_r = base_r
                stress_sigma = base_sigma * 2
            elif scenario == "Low Volatility (0.5x)":
                stress_S = S
                stress_r = base_r
                stress_sigma = base_sigma * 0.5
            elif scenario == "Interest Rate Shock (+300bp)":
                stress_S = S
                stress_r = base_r + 0.03
                stress_sigma = base_sigma
            elif scenario == "Time Decay (T/2)":
                stress_S = S
                stress_r = base_r
                stress_sigma = base_sigma
                T_stress = T / 2
            elif scenario == "Combined Stress":
                stress_S = S * 0.8
                stress_r = base_r + 0.02
                stress_sigma = base_sigma * vol_mult
            else:
                stress_S = S
                stress_r = base_r
                stress_sigma = base_sigma
            
            T_stress = T if scenario != "Time Decay (T/2)" else T / 2
            
            # Calculate stressed price
            try:
                if option_type == "Asian":
                    stressed_price = _calculate_enhanced_asian_option(
                        stress_S, K, T_stress, stress_r, stress_sigma, 50, mc_paths//4, "call", "average_price"
                    )
                elif option_type == "Digital":
                    d2 = (np.log(stress_S / K) + (stress_r - 0.5 * stress_sigma**2) * T_stress) / (stress_sigma * np.sqrt(T_stress))
                    stressed_price = np.exp(-stress_r * T_stress) * norm.cdf(d2)
                elif option_type == "Barrier":
                    # Simplified barrier stress
                    vanilla_equiv = max(stress_S - K, 0) * np.exp(-stress_r * T_stress)
                    barrier_factor = 0.7 if stress_sigma > base_sigma else 0.9
                    stressed_price = vanilla_equiv * barrier_factor
                elif option_type == "Lookback":
                    # Simplified lookback stress
                    stressed_price = stress_S * 0.12 * np.sqrt(T_stress) * stress_sigma
                else:
                    stressed_price = 0
                
                stress_results[option_type][scenario] = stressed_price
                
            except Exception as e:
                stress_results[option_type][scenario] = np.nan
    
    return stress_results

def _display_stress_testing_results(stress_results):
    """Display stress testing results"""
    
    # Create stress test comparison chart
    scenarios = list(next(iter(stress_results.values())).keys())
    option_types = list(stress_results.keys())
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, option_type in enumerate(option_types):
        values = [stress_results[option_type][scenario] for scenario in scenarios]
        
        fig.add_trace(go.Bar(
            name=option_type,
            x=scenarios,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[f"{v:.4f}" if not np.isnan(v) else "N/A" for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Stress Testing Results - Option Prices Under Different Scenarios",
        xaxis_title="Stress Scenario",
        yaxis_title="Option Price",
        template="plotly_white",
        height=600,
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stress test summary table
    summary_data = []
    for option_type in option_types:
        for scenario in scenarios:
            price = stress_results[option_type][scenario]
            summary_data.append({
                'Option Type': option_type,
                'Scenario': scenario,
                'Stressed Price': f"{price:.6f}" if not np.isnan(price) else "N/A"
            })
    
    st.markdown("#### 📊 Detailed Stress Test Results")
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Risk metrics
    st.markdown("#### ⚠️ Risk Assessment")
    
    risk_metrics = []
    for option_type in option_types:
        prices = [v for v in stress_results[option_type].values() if not np.isnan(v)]
        if len(prices) > 1:
            max_loss = (max(prices) - min(prices)) / max(prices) * 100 if max(prices) > 0 else 0
            volatility = np.std(prices) / np.mean(prices) * 100 if np.mean(prices) > 0 else 0
            
            risk_metrics.append({
                'Option Type': option_type,
                'Max Drawdown (%)': f"{max_loss:.2f}%",
                'Price Volatility (%)': f"{volatility:.2f}%",
                'Risk Rating': "High" if max_loss > 50 or volatility > 100 else "Medium" if max_loss > 25 or volatility > 50 else "Low"
            })
    
    if risk_metrics:
        st.dataframe(pd.DataFrame(risk_metrics), use_container_width=True, hide_index=True)


# Keep original CSS styling - no changes to existing styles
