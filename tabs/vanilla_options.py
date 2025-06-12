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
    
    st.markdown('<div class="main-header">Vanilla Option Pricing Tool</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Input Parameters Section
    st.markdown('<div class="sub-header">Option Parameters</div>', unsafe_allow_html=True)
    
    # Create columns for input parameters
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        st.subheader("Option Details")
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        exercise_style = st.selectbox("Exercise Style", ["European", "American"])
        
        st.subheader("Market Parameters")
        S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.01, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0)
    
    with param_col2:
        st.subheader("Time & Risk")
        T = st.number_input("Time to Maturity (T) in years", value=1.0, min_value=0.01, max_value=10.0, step=0.01)
        r = st.number_input("Risk-free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
        sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.001, max_value=2.0, step=0.01, format="%.3f")
        q = st.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
    
    with param_col3:
        st.subheader("Model Parameters")
        N = st.number_input("Binomial Steps (N)", value=100, min_value=10, max_value=1000, step=10)
        n_simulations = st.number_input("Monte Carlo Simulations", value=100000, min_value=1000, max_value=1000000, step=1000)
        
        st.subheader("Model Selection")
        models = st.multiselect(
            "Select Pricing Models", 
            ["Black-Scholes", "Binomial", "Monte-Carlo"],
            default=["Black-Scholes", "Binomial"]
        )
    
    st.divider()
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="sub-header">Pricing Results</div>', unsafe_allow_html=True)
        
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
                    st.error(f"Error calculating {model} price: {str(e)}")
                    results[model] = None
            
            # Display results
            if results:
                results_df = pd.DataFrame([results]).T
                results_df.columns = ['Option Price']
                results_df = results_df.round(4)
                
                st.subheader("Pricing Comparison")
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                if len([v for v in results.values() if v is not None]) > 1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    valid_results = {k: v for k, v in results.items() if v is not None}
                    
                    bars = ax.bar(valid_results.keys(), valid_results.values(), 
                                 color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(valid_results)])
                    ax.set_ylabel('Option Price')
                    ax.set_title(f'{exercise_style} {option_type} Option Pricing Comparison')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, valid_results.values()):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.4f}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
    
    with col2:
        st.markdown('<div class="sub-header">Analysis & Information</div>', unsafe_allow_html=True)
        
        # Option Information
        st.subheader("Option Metrics")
        moneyness = S / K
        time_value = max(results.get("Black-Scholes", 0) - max(S - K, 0) if option_type.lower() == "call" 
                        else results.get("Black-Scholes", 0) - max(K - S, 0), 0) if "Black-Scholes" in results and results.get("Black-Scholes") else 0
        
        # Create metrics in a clean layout
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Moneyness (S/K)", f"{moneyness:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Days to Expiry", f"{T * 365:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Intrinsic Value", f"{max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0):.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Time Value", f"{time_value:.4f}" if "Black-Scholes" in results and results.get("Black-Scholes") else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Greeks Section
        st.subheader("Greeks")
        greek_model_option = st.selectbox("Model for Greeks:", ["Black-Scholes", "Binomial", "Monte-Carlo"], key="greek_model_option")
        
        if st.button("Calculate Greeks"):
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
                        greeks_values[greek.capitalize()] = greek_val
                    except Exception as e:
                        greeks_values[greek.capitalize()] = f"Error: {str(e)}"
                
                # Display Greeks in a nice format
                greeks_df = pd.DataFrame([greeks_values]).T
                greeks_df.columns = ['Value']
                st.dataframe(greeks_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating Greeks: {str(e)}")
    
    # Sensitivity Analysis Section
    st.divider()
    st.markdown('<div class="sub-header">Advanced Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different analysis types
    t1, t2 = st.tabs(["Parameter Sensitivity", "Greeks Analysis"])
    
    with t1:
        st.subheader("Parameter Sensitivity Analysis")
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
            run_analysis = st.button("Run Parameter Sensitivity", use_container_width=True)
        
        if run_analysis:
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
            ax.plot(param_values, prices, 'b-', linewidth=2)
            ax.set_xlabel(param_to_analyze)
            ax.set_ylabel('Option Price')
            ax.set_title(f'Option Price Sensitivity to {param_to_analyze}')
            ax.grid(True, alpha=0.3)
            
            # Mark current value
            current_val = base_params[param_to_analyze]
            current_price = results.get(model_for_analysis, 0) if 'results' in locals() else 0
            if current_price:
                ax.axvline(x=current_val, color='red', linestyle='--', alpha=0.7, label='Current')
                ax.scatter([current_val], [current_price], color='red', s=100, zorder=5)
                ax.legend()
            
            st.pyplot(fig)
    
    with t2:
        st.subheader("Greeks Sensitivity Analysis")
        
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
            run_greek_analysis = st.button("Run Greeks Analysis", use_container_width=True)
        
        if run_greek_analysis:
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
                ax.plot(spot_range, greek_values, 'g-', linewidth=2)
                ax.set_xlabel('Spot Price (S)')
                ax.set_ylabel(f'{greek_to_analyze}')
                ax.set_title(f'{greek_to_analyze} vs Spot Price ({greek_model_analysis})')
                ax.grid(True, alpha=0.3)
                
                # Mark current spot price
                ax.axvline(x=S, color='red', linestyle='--', alpha=0.7, label='Current S')
                current_greek = compute_greek(
                    greek_name=greek_to_analyze.lower(),
                    model=greek_model_analysis,
                    option_type=option_type,
                    S_values=[S],
                    K=K, T=T, r=r, sigma=sigma, q=q
                )[0]
                ax.scatter([S], [current_greek], color='red', s=100, zorder=5)
                ax.legend()
                
                st.pyplot(fig)
                
                # Show current Greek value
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write(f"Current {greek_to_analyze} value: {current_greek:.6f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error calculating {greek_to_analyze}: {str(e)}")
    
    # Footer with educational content
    st.markdown("---")
    st.markdown('<div class="sub-header">Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("📚 Understanding Option Pricing Models"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Black-Scholes Model
        - **Analytical solution** for European options
        - Assumes constant volatility and interest rates
        - **Fastest computation** but limited to European exercise
        
        ### Binomial Tree Model
        - **Discrete time model** that can handle American options
        - More flexible than Black-Scholes
        - **Convergence**: More steps → more accurate (but slower)
        
        ### Monte Carlo Simulation
        - **Simulation-based** approach using random paths
        - Can handle complex payoffs and exotic features
        - **Statistical accuracy**: More simulations → lower standard error
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("📈 Understanding the Greeks"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### The Greeks - Risk Sensitivities
        
        **Delta (Δ)**: Price sensitivity to underlying asset price
        - Range: 0 to 1 (calls), -1 to 0 (puts)
        - **Higher delta** = more sensitive to stock price changes
        
        **Gamma (Γ)**: Rate of change of delta
        - **Highest near ATM** options
        - Measures **convexity** of option price
        
        **Theta (Θ)**: Time decay
        - Usually **negative** (options lose value over time)
        - **Accelerates** as expiration approaches
        
        **Vega (ν)**: Volatility sensitivity
        - **Positive** for both calls and puts
        - **Highest for ATM** options with medium time to expiry
        
        **Rho (ρ)**: Interest rate sensitivity
        - **Positive** for calls, **negative** for puts
        - More important for **longer-term** options
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("⚠️ Risk Management Tips"):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Key Risk Considerations
        
        **Model Risk**
        - Different models can give different prices
        - **Validate** results across multiple models
        - Consider **model assumptions** vs. real market conditions
        
        **Parameter Risk**
        - **Volatility** is the most critical and hardest to estimate
        - Use **implied volatility** from market prices when available
        - **Interest rates** and **dividends** can change
        
        **Execution Risk**
        - **Bid-ask spreads** affect real trading costs
        - **Liquidity** varies significantly across strikes and expirations
        - **Pin risk** near expiration for short positions
        
        **Greeks Management**
        - **Delta hedge** regularly to manage directional risk
        - Monitor **gamma** for large price moves
        - Be aware of **theta decay**, especially for short positions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Final footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p><strong>Vanilla Options Pricing Tool</strong> | Built with Streamlit & Python</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
