# tabs/vanilla_options.py
# Vanilla Options Tab - Tab 1

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your pricing functions
from pricing.vanilla_options import price_vanilla_option, plot_option_price_vs_param
from pricing.utils.greeks_vanilla.plot_single_greek import plot_single_greek_vs_spot
from pricing.utils.greeks_vanilla.greeks_interface import *


def vanilla_options_tab():
    """Vanilla Options Tab Content"""
    
    # Custom CSS for enhanced styling
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
        .greek-delta {
            background-color: #e8f4f8;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #1f77b4;
        }
        .greek-gamma {
            background-color: #d4edda;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #28a745;
        }
        .greek-theta {
            background-color: #f8d7da;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #dc3545;
        }
        .greek-vega {
            background-color: #d1ecf1;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
        }
        .greek-rho {
            background-color: #fff3cd;
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Vanilla Option Pricing Tool</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Input Parameters Section
    st.markdown('<div class="sub-header">📊 Option Parameters</div>', unsafe_allow_html=True)
    
    # Create columns for input parameters
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        st.markdown("""
        <div class="info-box">
            <h4>📋 Option Details</h4>
        </div>
        """, unsafe_allow_html=True)
        
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        exercise_style = st.selectbox("Exercise Style", ["European", "American"])
        
        st.markdown("""
        <div class="info-box">
            <h4>💰 Market Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.01, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0)
    
    with param_col2:
        st.markdown("""
        <div class="info-box">
            <h4>⏰ Time & Risk</h4>
        </div>
        """, unsafe_allow_html=True)
        
        T = st.number_input("Time to Maturity (T) in years", value=1.0, min_value=0.01, max_value=10.0, step=0.01)
        r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
        sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.001, max_value=2.0, step=0.01, format="%.3f")
        q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    
    with param_col3:
        st.markdown("""
        <div class="info-box">
            <h4>🔧 Model Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        N = st.number_input("Binomial Steps (N)", value=100, min_value=10, max_value=1000, step=10)
        n_simulations = st.number_input("Monte Carlo Simulations", value=100000, min_value=1000, max_value=1000000, step=1000)
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Model Selection</h4>
        </div>
        """, unsafe_allow_html=True)
        
        models = st.multiselect(
            "Select Pricing Models", 
            ["Black-Scholes", "Binomial", "Monte-Carlo"],
            default=["Black-Scholes", "Binomial"]
        )
    
    st.divider()
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="sub-header">💵 Pricing Results</div>', unsafe_allow_html=True)
        
        if models:
            results = {}
            
            for model in models:
                try:
                    # Prepare parameters
                    params = {
                        'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'q': q
                    }
                    
                    if model == "Binomial":
                        params['N'] = N
                    elif model == "Monte-Carlo":
                        params['n_simulations'] = n_simulations
                    
                    # Calculate price
                    price = price_vanilla_option(
                        option_type=option_type,
                        exercise_style=exercise_style,
                        model=model,
                        **params
                    )
                    
                    results[model] = price
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Error calculating {model} price</h4>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    results[model] = None
            
            # Display results
            if results:
                results_df = pd.DataFrame([results]).T
                results_df.columns = ['Option Price']
                results_df = results_df.round(4)
                
                st.markdown("#### 📊 Pricing Comparison")
                st.markdown('<div class="results-table">', unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualization
                if len([v for v in results.values() if v is not None]) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    valid_results = {k: v for k, v in results.items() if v is not None}
                    
                    bars = ax.bar(valid_results.keys(), valid_results.values(), 
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(valid_results)])
                    ax.set_ylabel('Option Price ($)')
                    ax.set_title(f'{exercise_style} {option_type} Option Pricing Comparison')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, valid_results.values()):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'${value:.4f}', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
        else:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ No Models Selected</h4>
                <p>Please select at least one pricing model to see results.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">📈 Analysis & Information</div>', unsafe_allow_html=True)
        
        # Option Information
        st.markdown("#### 📊 Option Metrics")
        moneyness = S / K
        intrinsic_value = max(S - K, 0) if option_type.lower() == "call" else max(K - S, 0)
        time_value = max(results.get("Black-Scholes", 0) - intrinsic_value, 0) if "Black-Scholes" in results and results.get("Black-Scholes") else 0
        
        # Create metrics in a clean layout
        st.markdown(f"""
        <div class="metric-container">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                    <td style="padding: 10px; font-weight: bold;">Metric</td>
                    <td style="padding: 10px; font-weight: bold;">Value</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; font-weight: bold;">Moneyness (S/K)</td>
                    <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{moneyness:.4f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; font-weight: bold;">Days to Expiry</td>
                    <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{T * 365:.0f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; font-weight: bold;">Intrinsic Value</td>
                    <td style="padding: 8px; font-family: monospace; color: #2E8B57;">${intrinsic_value:.4f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">Time Value</td>
                    <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{"${:.4f}".format(time_value) if "Black-Scholes" in results and results.get("Black-Scholes") else "N/A"}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Greeks Section
        st.markdown("#### 📊 Greeks Analysis")
        greek_model_option = st.selectbox("Model for Greeks:", ["Black-Scholes", "Binomial", "Monte-Carlo"], key="greek_model_option")
        
        if st.button("Calculate Greeks", type="primary"):
            try:
                # Calculate Greeks for current spot price
                greeks_list = ["delta", "gamma", "theta", "vega", "rho"]
                greeks_values = {}
                
                for greek in greeks_list:
                    try:
                        greek_val = compute_greek(
                            greek_name=greek,
                            model=greek_model_option,
                            option_type=option_type,
                            S_values=[S],  # Single value for current calculation
                            K=K, T=T, r=r, sigma=sigma, q=q
                        )[0]  # Get first (and only) value
                        greeks_values[greek.capitalize()] = f"{greek_val:.6f}"
                    except Exception as e:
                        greeks_values[greek.capitalize()] = f"Error: {str(e)}"
                
                # Display Greeks in a nice format
                st.markdown(f"""
                <div class="metric-container">
                    <h4>Greeks Values ({greek_model_option})</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                            <td style="padding: 10px; font-weight: bold;">Greek</td>
                            <td style="padding: 10px; font-weight: bold;">Value</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 8px; font-weight: bold;">Delta (Δ)</td>
                            <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{greeks_values.get('Delta', 'N/A')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 8px; font-weight: bold;">Gamma (Γ)</td>
                            <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{greeks_values.get('Gamma', 'N/A')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 8px; font-weight: bold;">Theta (Θ)</td>
                            <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{greeks_values.get('Theta', 'N/A')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 8px; font-weight: bold;">Vega (ν)</td>
                            <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{greeks_values.get('Vega', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Rho (ρ)</td>
                            <td style="padding: 8px; font-family: monospace; color: #2E8B57;">{greeks_values.get('Rho', 'N/A')}</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>❌ Error calculating Greeks</h4>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Sensitivity Analysis Section
    st.divider()
    st.markdown('<div class="sub-header">🔬 Advanced Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different analysis types
    t1, t2 = st.tabs(["Parameter Sensitivity", "Greeks Analysis"])
    
    with t1:
        st.markdown("#### 📊 Parameter Sensitivity Analysis")
        sens_col1, sens_col2, sens_col3 = st.columns([1, 1, 2])
        
        with sens_col1:
            param_to_analyze = st.selectbox(
                "Parameter to vary:",
                ["S", "K", "T", "r", "sigma"]
            )
        
        with sens_col2:
            model_for_analysis = st.selectbox(
                "Model for analysis:",
                ["Black-Scholes", "Binomial", "Monte-Carlo"]
            )
        
        with sens_col3:
            run_analysis = st.button("Run Parameter Sensitivity", use_container_width=True, type="primary")
        
        if run_analysis:
            with st.spinner("Running sensitivity analysis..."):
                # Define parameter ranges
                param_ranges = {
                    "S": (S * 0.5, S * 1.5),
                    "K": (K * 0.5, K * 1.5),
                    "T": (0.1, min(2.0, T * 2)),
                    "r": (0.01, min(0.2, r * 3)),
                    "sigma": (sigma * 0.5, min(1.0, sigma * 2))
                }
                
                param_range = param_ranges[param_to_analyze]
                param_values = np.linspace(param_range[0], param_range[1], 20)
                prices = []
                
                base_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'q': q}
                
                for val in param_values:
                    temp_params = base_params.copy()
                    temp_params[param_to_analyze] = val
                    
                    if model_for_analysis == "Binomial":
                        temp_params['N'] = N
                    elif model_for_analysis == "Monte-Carlo":
                        temp_params['n_simulations'] = n_simulations
                    
                    try:
                        price = price_vanilla_option(
                            option_type=option_type,
                            exercise_style=exercise_style,
                            model=model_for_analysis,
                            **temp_params
                        )
                        prices.append(price)
                    except:
                        prices.append(np.nan)
                
                # Plot sensitivity
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(param_values, prices, 'b-', linewidth=3, label='Option Price')
                ax.set_xlabel(f'{param_to_analyze}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Option Price ($)', fontsize=12, fontweight='bold')
                ax.set_title(f'Option Price Sensitivity to {param_to_analyze} ({model_for_analysis})', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Mark current value
                current_val = base_params[param_to_analyze]
                current_price = results.get(model_for_analysis, 0) if 'results' in locals() else 0
                if current_price:
                    ax.axvline(x=current_val, color='red', linestyle='--', alpha=0.7, label='Current Value', linewidth=2)
                    ax.scatter([current_val], [current_price], color='red', s=100, zorder=5, label='Current Price')
                    ax.legend()
                
                st.pyplot(fig)
                
                # Show sensitivity metrics
                if len(prices) > 1:
                    price_change = (max(prices) - min(prices)) / min(prices) * 100
                    param_change = (param_range[1] - param_range[0]) / param_range[0] * 100
                    sensitivity = price_change / param_change
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>📊 Sensitivity Metrics</h4>
                        <p><strong>Price Range:</strong> ${min(prices):.4f} - ${max(prices):.4f}</p>
                        <p><strong>Price Sensitivity:</strong> {sensitivity:.4f}% per 1% change in {param_to_analyze}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with t2:
        st.markdown("#### 📈 Greeks Sensitivity Analysis")
        
        greek_sens_col1, greek_sens_col2, greek_sens_col3 = st.columns([1, 1, 2])
        
        with greek_sens_col1:
            greek_to_analyze = st.selectbox(
                "Greek to analyze:",
                ["Delta", "Gamma", "Theta", "Vega", "Rho"]
            )
        
        with greek_sens_col2:
            greek_model_analysis = st.selectbox(
                "Model for Greeks:",
                ["Black-Scholes", "Binomial", "Monte-Carlo"],
                key="greek_model_analysis"
            )
        
        with greek_sens_col3:
            run_greek_analysis = st.button("Run Greeks Analysis", use_container_width=True, type="primary")
        
        if run_greek_analysis:
            with st.spinner("Computing Greeks sensitivity..."):
                # Analyze Greek vs Spot Price
                spot_range = np.linspace(S * 0.7, S * 1.3, 30)
                
                try:
                    greek_values = compute_greek(
                        greek_name=greek_to_analyze.lower(),
                        model=greek_model_analysis,
                        option_type=option_type,
                        S_values=spot_range,
                        K=K, T=T, r=r, sigma=sigma, q=q
                    )
                    
                    # Plot Greek sensitivity
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(spot_range, greek_values, 'g-', linewidth=3, label=f'{greek_to_analyze}')
                    ax.set_xlabel('Spot Price ($)', fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'{greek_to_analyze}', fontsize=12, fontweight='bold')
                    ax.set_title(f'{greek_to_analyze} vs Spot Price ({greek_model_analysis})', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Mark current spot price
                    ax.axvline(x=S, color='red', linestyle='--', alpha=0.7, label='Current S', linewidth=2)
                    current_greek = compute_greek(
                        greek_name=greek_to_analyze.lower(),
                        model=greek_model_analysis,
                        option_type=option_type,
                        S_values=[S],
                        K=K, T=T, r=r, sigma=sigma, q=q
                    )[0]
                    ax.scatter([S], [current_greek], color='red', s=100, zorder=5, label='Current Value')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Show current Greek value
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Current {greek_to_analyze} Analysis</h4>
                        <p><strong>Current {greek_to_analyze} value:</strong> {current_greek:.6f}</p>
                        <p><strong>Spot Price:</strong> ${S:.2f}</p>
                        <p><strong>Model Used:</strong> {greek_model_analysis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Error calculating {greek_to_analyze}</h4>
                        <p>{str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer with educational content
    st.markdown("---")
    st.markdown('<div class="sub-header">📚 Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Understanding Option Pricing Models"):
        st.markdown("#### Option Pricing Models Comparison")
        
        st.markdown(
            '<div style="text-align: center; font-size: 1.2em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">Black-Scholes Formula: C = S₀N(d₁) - Ke^(-rT)N(d₂)</div>', 
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                '<div style="background-color: #e8f4f8; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1f77b4;"><strong>📈 Black-Scholes Model</strong></div>', 
                unsafe_allow_html=True
            )
            st.markdown("""
            • **Analytical solution** for European options  
            • Assumes constant volatility and interest rates  
            • **Fastest computation** but limited to European exercise  
            • **Best for:** Quick pricing, theoretical analysis
            """)
        
        with col2:
            st.markdown(
                '<div style="background-color: #fff3cd; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffc107;"><strong>🌳 Binomial Tree Model</strong></div>', 
                unsafe_allow_html=True
            )
            st.markdown("""
            • **Discrete time model** that can handle American options  
            • More flexible than Black-Scholes  
            • **Convergence:** More steps → more accurate (but slower)  
            • **Best for:** American options, dividend modeling
            """)
        
        with col3:
            st.markdown(
                '<div style="background-color: #d4edda; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745;"><strong>🎲 Monte Carlo Simulation</strong></div>', 
                unsafe_allow_html=True
            )
            st.markdown("""
            • **Simulation-based** approach using random paths  
            • Can handle complex payoffs and exotic features  
            • **Statistical accuracy:** More simulations → lower standard error  
            • **Best for:** Path-dependent options, complex models
            """)
    
    with st.expander("📈 Understanding the Greeks"):
        st.markdown("#### The Greeks - Risk Sensitivities")
        
        # Delta
        st.markdown(
            '<div style="background-color: #e8f4f8; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1f77b4;"><strong>📈 Delta (Δ): Price Sensitivity</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • **Definition:** Rate of change of option price with respect to underlying price  
        • **Range:** 0 to 1 (calls), -1 to 0 (puts)  
        • **Interpretation:** A delta of 0.5 means $0.50 price change per $1 stock move  
        • **Use:** Hedge ratio for delta-neutral portfolios
        """)
        
        # Gamma
        st.markdown(
            '<div style="background-color: #d4edda; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745;"><strong>🔄 Gamma (Γ): Delta Acceleration</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • **Definition:** Rate of change of delta with respect to underlying price  
        • **Highest near ATM** options with medium time to expiry  
        • **Interpretation:** Measures **convexity** of option price  
        • **Use:** Risk management for large price moves
        """)
        
        # Theta
        st.markdown(
            '<div style="background-color: #f8d7da; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #dc3545;"><strong>⏰ Theta (Θ): Time Decay</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • **Definition:** Rate of change of option price with respect to time  
        • Usually **negative** (options lose value over time)  
        • **Accelerates** as expiration approaches  
        • **Use:** Time decay strategies, income generation
        """)
        
        # Vega
        st.markdown(
            '<div style="background-color: #d1ecf1; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #17a2b8;"><strong>🌊 Vega (ν): Volatility Sensitivity</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • **Definition:** Rate of change of option price with respect to volatility  
        • **Positive** for both calls and puts  
        • **Highest for ATM** options with medium time to expiry  
        • **Use:** Volatility trading strategies
        """)
        
        # Rho
        st.markdown(
            '<div style="background-color: #fff3cd; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffc107;"><strong>💰 Rho (ρ): Interest Rate Sensitivity</strong></div>', 
            unsafe_allow_html=True
        )
        st.markdown("""
        • **Definition:** Rate of change of option price with respect to interest rate  
        • **Positive** for calls, **negative** for puts  
        • More important for **longer-term** options  
        • **Use:** Interest rate risk management
        """)
    
    with st.expander("⚠️ Risk Management & Trading Tips"):
        st.markdown("#### Key Risk Considerations")
        
        st.markdown("##### 🎯 Model Risk")
        st.markdown("""
        • Different models can give different prices  
        • **Validate** results across multiple models  
        • Consider **model assumptions** vs. real market conditions  
        • Be aware of **computational limitations** (steps, simulations)
        """)
        
        st.markdown("##### 📊 Parameter Risk")
        st.markdown("""
        • **Volatility** is the most critical and hardest to estimate  
        • Use **implied volatility** from market prices when available  
        • **Interest rates** and **dividends** can change over option life  
        • **Time decay** accelerates near expiration
        """)
        
        st.markdown("##### 💼 Execution Risk")
        st.markdown("""
        • **Bid-ask spreads** affect real trading costs  
        • **Liquidity** varies significantly across strikes and expirations  
        • **Pin risk** near expiration for short positions  
        • **Assignment risk** for American-style options
        """)
        
        st.markdown("##### 🛡️ Greeks Management")
        st.markdown("""
        • **Delta hedge** regularly to manage directional risk  
        • Monitor **gamma** for large price moves  
        • Be aware of **theta decay**, especially for short positions  
        • Use **vega** to manage volatility exposure
        """)
    
    with st.expander("📊 Trading Strategies & Applications"):
        st.markdown("#### Common Option Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📈 Bullish Strategies")
            st.markdown("""
            • **Long Call:** Limited risk, unlimited profit potential  
            • **Bull Call Spread:** Lower cost, limited profit  
            • **Cash-Secured Put:** Income generation, potential ownership  
            • **Covered Call:** Income on existing stock positions
            """)
            
            st.markdown("##### 📉 Bearish Strategies")
            st.markdown("""
            • **Long Put:** Limited risk, high profit potential  
            • **Bear Put Spread:** Lower cost, limited profit  
            • **Short Call:** Income generation, unlimited risk  
            • **Protective Put:** Portfolio insurance
            """)
        
        with col2:
            st.markdown("##### 🔄 Neutral Strategies")
            st.markdown("""
            • **Short Straddle:** Profit from low volatility  
            • **Iron Condor:** Profit from range-bound movement  
            • **Calendar Spread:** Profit from time decay  
            • **Butterfly Spread:** Profit from minimal movement
            """)
            
            st.markdown("##### 🌊 Volatility Strategies")
            st.markdown("""
            • **Long Straddle:** Profit from high volatility  
            • **Long Strangle:** Lower cost volatility play  
            • **Volatility Trading:** Buy low IV, sell high IV  
            • **Gamma Scalping:** Dynamic hedging for profits
            """)
    
    with st.expander("🔢 Key Formulas & Calculations"):
        st.markdown("#### Mathematical Foundations")
        
        st.markdown("##### Black-Scholes Formula")
        st.markdown(
            '<div style="text-align: center; font-size: 1.1em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">Call: C = S₀N(d₁) - Ke^(-rT)N(d₂)<br>Put: P = Ke^(-rT)N(-d₂) - S₀N(-d₁)</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("##### d₁ and d₂ Parameters")
        st.markdown(
            '<div style="text-align: center; font-size: 1.0em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)<br>d₂ = d₁ - σ√T</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("##### Greeks Formulas")
        st.code("""
Delta (Δ):
  Call: N(d₁)
  Put: N(d₁) - 1

Gamma (Γ):
  φ(d₁) / (S₀σ√T)

Theta (Θ):
  Call: -[S₀φ(d₁)σ/(2√T) + rKe^(-rT)N(d₂)]
  Put: -[S₀φ(d₁)σ/(2√T) - rKe^(-rT)N(-d₂)]

Vega (ν):
  S₀φ(d₁)√T

Rho (ρ):
  Call: KTe^(-rT)N(d₂)
  Put: -KTe^(-rT)N(-d₂)

Where φ(x) is the standard normal PDF
        """)
        
        st.markdown("##### Put-Call Parity")
        st.markdown(
            '<div style="text-align: center; font-size: 1.2em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">C - P = S₀ - Ke^(-rT)</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("This relationship must hold for European options to prevent arbitrage opportunities.")
    
    # Final footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 1px solid #dee2e6;'>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2rem;">📊</span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">Vanilla Options Pricing Tool</p>
        <p style="margin: 8px 0; color: #6c757d;">Built with Streamlit & Python</p>
        <p style="margin: 0; color: #dc3545; font-weight: bold;">⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
