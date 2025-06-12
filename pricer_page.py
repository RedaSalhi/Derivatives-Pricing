# main.py
# Licensed under the MIT License. See LICENSE file in the project root for full license text.


import sys
import os
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import plotly.express as px
from io import BytesIO
import seaborn as sns
from scipy.stats import norm
import time



from pricing.vanilla_options import price_vanilla_option, plot_option_price_vs_param
from pricing.forward import *
from pricing.option_strategies import *
from pricing.utils.greeks_vanilla.plot_single_greek import plot_single_greek_vs_spot
from pricing.utils.greeks_vanilla.greeks_interface import *
from pricing.asian_option import price_asian_option, plot_asian_option_payoff, plot_monte_carlo_paths
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff, plot_sample_paths_barrier
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution
from pricing.utils.exotic_utils import *
from pricing.models.interest_rates.analytical_vasicek import *
from pricing.models.interest_rates.monte_carlo_vasicek import *
from pricing.utils.greeks_vasicek import *

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

    

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
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)




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




# -----------------------------
# Tab 1 – Vanilla Options
# -----------------------------
with tab1:
    st.markdown('<h1 class="main-header">Vanilla Option Pricing Tool</h1>', 
                unsafe_allow_html=True)

    st.markdown("---")
    # Input Parameters Section
    st.header("Option Parameters")
    
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
        st.header("Pricing Results")
        
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
        st.header("Analysis & Information")
        
        # Option Information
        st.subheader("Option Metrics")
        moneyness = S / K
        time_value = max(results.get("Black-Scholes", 0) - max(S - K, 0) if option_type.lower() == "call" 
                        else results.get("Black-Scholes", 0) - max(K - S, 0), 0) if "Black-Scholes" in results and results.get("Black-Scholes") else 0
        
        # Create metrics in a clean layout
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Moneyness (S/K)", f"{moneyness:.4f}")
            st.metric("Days to Expiry", f"{T * 365:.0f}")
        with metric_col2:
            st.metric("Intrinsic Value", f"{max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0):.4f}")
            st.metric("Time Value", f"{time_value:.4f}" if "Black-Scholes" in results and results.get("Black-Scholes") else "N/A")
        
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
    st.header("Advanced Analysis")
    
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
                st.info(f"Current {greek_to_analyze} value: {current_greek:.6f}")
                
            except Exception as e:
                st.error(f"Error calculating {greek_to_analyze}: {str(e)}")




# -----------------------------
# Tab 2 – Forward Contracts
# -----------------------------

with tab2:
    st.markdown('<h1 class="main-header">Forward Contract Pricing & Analysis</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Parameter input section
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("## Contract Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spot_price = st.number_input(
            "Spot Price ($)", 
            min_value=0.01, 
            value=100.0, 
            step=1.0,
            help="Current price of the underlying asset"
        )
        
        strike_price = st.number_input(
            "Strike Price ($)", 
            min_value=0.01, 
            value=105.0, 
            step=1.0,
            help="Agreed delivery price at maturity"
        )
        
    
    with col2:
        interest_rate = st.number_input(
            "Risk-free Rate (%)", 
            min_value=0.0, 
            value=5.0, 
            step=0.1
        ) / 100
        
        time_input_method = st.selectbox(
            "Time Input Method",
            ["Years", "Days", "Calendar Date"]
        )
    
    with col3:
        storage_cost = st.number_input(
            "Storage Cost Rate (%)", 
            min_value=0.0, 
            value=0.0, 
            step=0.1
        ) / 100
        
        dividend_yield = st.number_input(
            "Dividend Yield (%)", 
            min_value=0.0, 
            value=0.0, 
            step=0.1
        ) / 100
    
    # Time to maturity calculation
    if time_input_method == "Years":
        time_to_maturity = st.number_input(
            "Time to Maturity (years)", 
            min_value=0.01, 
            value=1.0, 
            step=0.1
        )
    elif time_input_method == "Days":
        days_to_maturity = st.number_input(
            "Days to Maturity", 
            min_value=1, 
            value=365, 
            step=1
        )
        time_to_maturity = days_to_maturity / 365.25
    else:  # Calendar Date
        maturity_date = st.date_input(
            "Maturity Date",
            value=datetime.now().date() + timedelta(days=365)
        )
        days_to_maturity = (maturity_date - datetime.now().date()).days
        time_to_maturity = max(days_to_maturity / 365.25, 0.01)
    
    position = st.selectbox("Position", ["Long", "Short"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate forward price
    forward_price = price_forward_contract(
        spot_price, interest_rate, time_to_maturity, storage_cost, dividend_yield
    )
    
    # Main tabs
    ta1, ta2, ta3, ta4 = st.tabs(["Pricing Results", "Mark-to-Market", "Payout Analysis", "Sensitivity"])
    
    with ta1:
        st.markdown("## Forward Contract Pricing Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Forward Price", f"${forward_price:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            basis = forward_price - spot_price
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Basis", f"${basis:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            carry_cost = (interest_rate + storage_cost - dividend_yield) * time_to_maturity
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Carry Cost", f"{carry_cost*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            total_return = forward_price / spot_price - 1
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Return", f"{total_return*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Pricing formula
        st.markdown("### Cost of Carry Model")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.latex(r"F = S_0 \cdot e^{(r + c - q) \cdot T}")
        st.markdown("""
        **Where:**
        - **F** = Forward Price = ${:.2f}
        - **S₀** = Current Spot Price = ${:.2f}
        - **r** = Risk-free Interest Rate = {:.2f}%
        - **c** = Storage Cost Rate = {:.2f}%
        - **q** = Dividend Yield = {:.2f}%
        - **T** = Time to Maturity = {:.1f} years
        """.format(forward_price, spot_price, interest_rate*100, storage_cost*100, dividend_yield*100, time_to_maturity))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed breakdown
        with st.expander("Calculation Breakdown"):
            net_carry = interest_rate + storage_cost - dividend_yield
            exp_factor = np.exp(net_carry * time_to_maturity)
            forward_price = spot_price * exp_factor
            #st.write(f"**Net Carry Rate:** {interest_rate*100:.2f}% + {storage_cost*100:.2f}% - {dividend_yield*100:.2f}% = {net_carry*100:.2f}%")
            #st.write(f"**Exponential Factor:** e^({net_carry*100:.2f}% × {time_to_maturity:.4f}) = {np.exp(net_carry * time_to_maturity):.6f}")
            #st.write(f"**Forward Price:** ${spot_price:.2f} × {np.exp(net_carry * time_to_maturity):.6f} = **${forward_price:.2f}**")
            #st.write(f"**Forward Price:** ${spot_price:.2f} × {np.exp(net_carry * time_to_maturity):.6f} = **${forward_price:.2f}**")
            st.write(f"**Net Carry Rate:** {interest_rate*100:.2f}% + {storage_cost*100:.2f}% - {dividend_yield*100:.2f}% = {net_carry*100:.2f}%")
            st.latex(r"\text{Exponential Factor: } e^{(r + c - q) \cdot T}")
            st.latex(fr"e^{{({net_carry*100:.2f}\% \cdot {time_to_maturity:.4f})}} = {exp_factor:.6f}")
            st.latex(r"\text{Forward Price: } F = S \cdot e^{(r + c - q) \cdot T}")
            st.latex(fr"F = {spot_price:.2f} \cdot {exp_factor:.6f} = {forward_price:.2f}")
    
    with ta2:
        st.markdown("## Mark-to-Market Analysis")
        st.markdown("*Contract value before maturity (t < T)*")
        
        # Interactive Plotly chart
        fig_mtm = create_plotly_mtm_chart(
            strike_price, time_to_maturity, interest_rate, 
            storage_cost, dividend_yield, position.lower()
        )
        st.plotly_chart(fig_mtm, use_container_width=True)
        
        # Current contract value
        current_value = spot_price * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity) - strike_price * np.exp(-interest_rate * time_to_maturity)
        if position.lower() == "short":
            current_value = -current_value
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Contract Value", f"${current_value:.2f}")
        with col2:
            breakeven = strike_price * np.exp(-interest_rate * time_to_maturity) / np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity)
            st.metric("Breakeven Spot Price", f"${breakeven:.2f}")
        with col3:
            st.metric("Time to Maturity", f"{time_to_maturity:.3f} years")
        
        # Original matplotlib chart
        st.markdown("### Original Chart (Matplotlib)")
        plot_forward_mark_to_market(
            strike_price, time_to_maturity, interest_rate, 
            storage_cost, dividend_yield, position.lower()
        )
    
    with ta3:
        st.markdown("## Payout Analysis at Maturity")
        st.markdown("*Profit/Loss when contract expires (t = T)*")
        
        # Interactive Plotly payout chart
        fig_payout = create_plotly_payout_chart(strike_price, position.lower())
        st.plotly_chart(fig_payout, use_container_width=True)
        
        # Scenario analysis
        st.markdown("### Payout Scenarios")
        spot_scenarios = [strike_price * mult for mult in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
        
        scenarios_data = []
        for spot in spot_scenarios:
            long_payout = spot - strike_price
            short_payout = strike_price - spot
            scenarios_data.append({
                'Spot Price at Maturity': f"${spot:.2f}",
                'Long Position Payout': f"${long_payout:.2f}",
                'Short Position Payout': f"${short_payout:.2f}",
                'Current Position Payout': f"${long_payout if position.lower() == 'long' else short_payout:.2f}"
            })
        
        scenarios_df = pd.DataFrame(scenarios_data)
        st.dataframe(scenarios_df, use_container_width=True)
        
        # Original matplotlib chart
        st.markdown("### Original Chart (Matplotlib)")
        plot_forward_payout_and_value(strike_price, position.lower())
    
    with ta4:
        st.markdown("## Sensitivity Analysis")
        st.markdown("*How forward prices respond to parameter changes*")
        
        # Sensitivity analysis
        base_params = {
            'spot_price': spot_price,
            'interest_rate': interest_rate,
            'time_to_maturity': time_to_maturity,
            'storage_cost': storage_cost,
            'dividend_yield': dividend_yield,
            'base_forward': forward_price
        }
        
        fig_sensitivity = create_sensitivity_analysis(base_params)
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Risk metrics
        st.markdown("### Risk Sensitivities")
        
        # Calculate numerical derivatives
        delta_s = 0.01 * spot_price
        delta_r = 0.0001
        delta_t = 0.01
        
        # Spot sensitivity (Delta equivalent)
        spot_up = price_forward_contract(spot_price + delta_s, interest_rate, time_to_maturity, storage_cost, dividend_yield)
        spot_sensitivity = (spot_up - forward_price) / delta_s
        
        # Rate sensitivity (Rho equivalent)
        rate_up = price_forward_contract(spot_price, interest_rate + delta_r, time_to_maturity, storage_cost, dividend_yield)
        rate_sensitivity = (rate_up - forward_price) / delta_r
        
        # Time sensitivity (Theta equivalent)
        time_up = price_forward_contract(spot_price, interest_rate, time_to_maturity + delta_t, storage_cost, dividend_yield)
        time_sensitivity = (time_up - forward_price) / delta_t
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spot Sensitivity", f"{spot_sensitivity:.4f}", 
                     help="Forward price change per $1 change in spot price")
        with col2:
            st.metric("Rate Sensitivity", f"{rate_sensitivity:.2f}", 
                     help="Forward price change per 1bp change in interest rate")
        with col3:
            st.metric("Time Sensitivity", f"{time_sensitivity:.4f}", 
                     help="Forward price change per 1% change in time to maturity")
    
    # Footer
    st.markdown("---")
    st.markdown("*Forward Contract Pricing & Analysis • Built with Streamlit*")







from pricing.option_strategies import price_option_strategy, compute_strategy_payoff, get_predefined_strategy, plot_strategy_price_vs_param
from pricing.utils.option_strategies_greeks import plot_strategy_greek_vs_spot

# -----------------------------
# Tab 3 – Option Strategies
# -----------------------------
with tab3:
    # Main title
    st.markdown('<h1 class="main-header">Options Pricing Suite</h1>', unsafe_allow_html=True)

    st.markdown("---")
    # Initialize session state for setup completion
    if 'setup_completed' not in st.session_state:
        st.session_state.setup_completed = False
    
    # Initialize session state for parameters
    if 'global_params' not in st.session_state:
        st.session_state.global_params = {
            'spot_price': 100.0,
            'risk_free_rate': 0.05,
            'dividend_yield': 0.0,
            'volatility': 0.2,
            'time_to_expiry': 1.0,
            'model': 'black-scholes',
            'n_steps': 100,
            'n_simulations': 10000
        }

    # Tab structure
    taa0, taa1, taa2, taa3, taa4, taa5 = st.tabs([
        "Setup & Parameters",
        "Single Option Pricing", 
        "Strategy Builder", 
        "Payoff Analysis", 
        "Greeks Analysis",
        "Sensitivity Analysis"
    ])
    
    with taa0:
        st.markdown('<h2 class="sub-header">Welcome to the Options Pricing Suite!</h2>', unsafe_allow_html=True)
        
        # Tips and Instructions
        st.markdown("""
        ### **Quick Start Guide**
        
        Welcome to your comprehensive options pricing toolkit! This application provides advanced analytics for:
        - **Single Option Pricing** with multiple models (Black-Scholes, Binomial Trees, Monte Carlo)
        - **Multi-leg Strategy Construction** with predefined and custom strategies
        - **Interactive Payoff Diagrams** with breakeven analysis
        - **Greeks Visualization** across different market conditions
        - **Sensitivity Analysis** with multi-parameter heatmaps
        
        ### ⚠️ **Important Notes:**
        - All parameters below are **required** before accessing other tabs
        - Your settings will be saved throughout your session
        - Use realistic market parameters for accurate results
        - For educational purposes only - not financial advice
        """)
        
        st.markdown("---")
        
        # Global Parameters Section
        st.markdown('<h3 class="sub-header">Global Market Parameters</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Asset & Market Parameters**")
            spot_price = st.number_input(
                "Spot Price (S)", 
                value=st.session_state.global_params['spot_price'], 
                min_value=0.1, 
                step=0.1, 
                key="setup_spot",
                help="Current price of the underlying asset"
            )
            
            risk_free_rate = st.number_input(
                "Risk-free Rate (r)", 
                value=st.session_state.global_params['risk_free_rate'], 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                format="%.3f", 
                key="setup_rate",
                help="Annual risk-free interest rate (e.g., 0.05 = 5%)"
            )
            
            dividend_yield = st.number_input(
                "Dividend Yield (q)", 
                value=st.session_state.global_params['dividend_yield'], 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                format="%.3f", 
                key="setup_dividend",
                help="Annual dividend yield (e.g., 0.02 = 2%)"
            )
        
        with col2:
            st.markdown("**Option Parameters**")
            volatility = st.number_input(
                "Volatility (σ)", 
                value=st.session_state.global_params['volatility'], 
                min_value=0.01, 
                max_value=2.0, 
                step=0.01, 
                format="%.3f", 
                key="setup_vol",
                help="Annual volatility (e.g., 0.20 = 20%)"
            )
            
            time_to_expiry = st.number_input(
                "Time to Expiry (T)", 
                value=st.session_state.global_params['time_to_expiry'], 
                min_value=0.001, 
                step=0.01, 
                format="%.3f", 
                key="setup_time",
                help="Time to expiration in years (e.g., 0.25 = 3 months)"
            )
        
        st.markdown("---")
        
        # Model Selection
        st.markdown('<h3 class="sub-header">Pricing Model Configuration</h3>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            model = st.selectbox(
                "Select Pricing Model", 
                ["black-scholes", "binomial", "monte-carlo"],
                index=["black-scholes", "binomial", "monte-carlo"].index(st.session_state.global_params['model']),
                key="setup_model",
                help="Choose the mathematical model for option pricing"
            )
            
            # Model descriptions
            model_descriptions = {
                "black-scholes": "**Black-Scholes**: We compute the analytical solution for European options. Fast and precise.",
                "binomial": "**Binomial Tree**: Discrete model supporting American options. Flexible but slower.",
                "monte-carlo": "**Monte Carlo**: Simulation-based approach. Handles complex payoffs."
            }
            
            st.info(model_descriptions[model])
        
        with col4:
            # Additional parameters for specific models
            if model == "binomial":
                n_steps = st.number_input(
                    "Number of Steps (N)", 
                    value=st.session_state.global_params['n_steps'], 
                    min_value=1, 
                    max_value=1000, 
                    step=1, 
                    key="setup_n_steps",
                    help="More steps = higher accuracy but slower computation"
                )
            elif model == "monte-carlo":
                n_simulations = st.number_input(
                    "Number of Simulations", 
                    value=st.session_state.global_params['n_simulations'], 
                    min_value=100, 
                    max_value=100000, 
                    step=100, 
                    key="setup_n_sims",
                    help="More simulations = higher accuracy but slower computation"
                )
        
        st.markdown("---")
        
        # Parameter Summary
        st.markdown('<h3 class="sub-header">Parameter Summary</h3>', unsafe_allow_html=True)
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Spot Price", f"${spot_price:.2f}")
            st.metric("Strike Price Range", f"${spot_price*0.8:.0f} - ${spot_price*1.2:.0f}")
        
        with summary_col2:
            st.metric("Volatility", f"{volatility*100:.1f}%")
            st.metric("Risk-free Rate", f"{risk_free_rate*100:.2f}%")
        
        with summary_col3:
            st.metric("Time to Expiry", f"{time_to_expiry:.3f} years")
            st.metric("Model", model.title())
        
        # Setup completion button
        st.markdown("---")
        
        col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
        
        with col_button2:
            if st.button("**Complete Setup & Start Analysis**", type="primary", use_container_width=True):
                # Update session state with all parameters
                st.session_state.global_params.update({
                    'spot_price': spot_price,
                    'risk_free_rate': risk_free_rate,
                    'dividend_yield': dividend_yield,
                    'volatility': volatility,
                    'time_to_expiry': time_to_expiry,
                    'model': model,
                    'n_steps': n_steps if model == "binomial" else st.session_state.global_params['n_steps'],
                    'n_simulations': n_simulations if model == "monte-carlo" else st.session_state.global_params['n_simulations']
                })
                
                st.session_state.setup_completed = True
                st.success("✅ Setup completed successfully! You can now access all analysis tabs.")
                st.balloons()
        
        if not st.session_state.setup_completed:
            st.warning("⚠️ Please complete the setup above to unlock all analysis features!")
    
    # Extract parameters for other tabs
    if st.session_state.setup_completed:
        spot_price = st.session_state.global_params['spot_price']
        risk_free_rate = st.session_state.global_params['risk_free_rate']
        dividend_yield = st.session_state.global_params['dividend_yield']
        volatility = st.session_state.global_params['volatility']
        time_to_expiry = st.session_state.global_params['time_to_expiry']
        model = st.session_state.global_params['model']
        n_steps = st.session_state.global_params['n_steps']
        n_simulations = st.session_state.global_params['n_simulations']



    
    # Tab 1: Single Option Pricing
    with taa1:
        if not st.session_state.setup_completed:
            st.warning("⚠️ Please complete the setup in the 'Setup & Parameters' tab first!")
        elif 'legs' in locals() and not isinstance(legs, str):
            st.markdown('<h2 class="sub-header">Single Option Pricing</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                option_type = st.selectbox("Option Type", ["call", "put"])
                exercise_style = st.selectbox("Exercise Style", ["european", "american"])
                strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=0.1, key="single_strike")
                
                # Price calculation
                try:
                    kwargs = {
                        "S": spot_price, "K": strike_price, "T": time_to_expiry,
                        "r": risk_free_rate, "sigma": volatility, "q": dividend_yield
                    }
                    
                    if model == "binomial":
                        kwargs["N"] = n_steps
                    elif model == "monte-carlo":
                        kwargs["n_simulations"] = n_simulations
                        
                    option_price = price_vanilla_option(
                        option_type=option_type,
                        exercise_style=exercise_style,
                        model=model,
                        **kwargs
                    )
                    
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric(
                        label=f"{exercise_style.title()} {option_type.title()} Option Price",
                        value=f"${option_price:.4f}",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error calculating option price: {str(e)}")
            
            with col2:
                # Interactive parameter sensitivity
                st.subheader("Parameter Sensitivity")
                param_to_vary = st.selectbox("Parameter to Vary", ["S", "K", "T", "r", "sigma", "q"])
                
                # Get current parameter value and create range
                current_val = {
                    "S": spot_price, "K": strike_price, "T": time_to_expiry,
                    "r": risk_free_rate, "sigma": volatility, "q": dividend_yield
                }[param_to_vary]
                
                param_min = st.number_input(f"Min {param_to_vary}", value=current_val * 0.5, step=0.01, key="single_param_min")
                param_max = st.number_input(f"Max {param_to_vary}", value=current_val * 1.5, step=0.01, key="single_param_max")
                
                if st.button("Generate Sensitivity Plot"):
                    try:
                        fixed_params = {
                            "S": spot_price, "K": strike_price, "T": time_to_expiry,
                            "r": risk_free_rate, "sigma": volatility, "q": dividend_yield
                        }
                        if model == "binomial":
                            fixed_params["N"] = n_steps
                        elif model == "monte-carlo":
                            fixed_params["n_simulations"] = n_simulations
                        
                        fig = plot_option_price_vs_param(
                            option_type=option_type,
                            exercise_style=exercise_style,
                            model=model,
                            param_name=param_to_vary,
                            param_range=(param_min, param_max),
                            fixed_params=fixed_params
                        )
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating plot: {str(e)}")
    
    #  Tab 2: Strategy Builder
    with taa2:
        if not st.session_state.setup_completed:
            st.warning("⚠️ Please complete the setup in the 'Setup & Parameters' tab first!")
        elif 'legs' in locals() and not isinstance(legs, str):
            st.markdown('<h2 class="sub-header">Option Strategy Builder</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Strategy Selection")
                strategy_method = st.radio("Choose Method", ["Predefined Strategy", "Custom Strategy"])
                
                if strategy_method == "Predefined Strategy":
                    strategy_name = st.selectbox(
                        "Select Strategy",
                        ["straddle", "bull call spread", "bear put spread", "butterfly", "iron condor"]
                    )
                    
                    # Dynamic strike inputs based on strategy
                    if strategy_name == "straddle":
                        strike1 = st.number_input("Strike Price", value=100.0, key="pred_k1")
                        legs = get_predefined_strategy(strategy_name, strike1)
                    elif strategy_name in ["bull call spread", "bear put spread"]:
                        strike1 = st.number_input("Strike 1 (Long)", value=95.0, key="pred_k1")
                        strike2 = st.number_input("Strike 2 (Short)", value=105.0, key="pred_k2")
                        legs = get_predefined_strategy(strategy_name, strike1, strike2)
                    elif strategy_name == "butterfly":
                        strike1 = st.number_input("Lower Strike", value=90.0, key="pred_k1")
                        strike2 = st.number_input("Middle Strike", value=100.0, key="pred_k2")
                        strike3 = st.number_input("Upper Strike", value=110.0, key="pred_k3")
                        legs = get_predefined_strategy(strategy_name, strike1, strike2, strike3)
                    elif strategy_name == "iron condor":
                        strike1 = st.number_input("Put Long Strike", value=85.0, key="pred_k1")
                        strike2 = st.number_input("Put Short Strike", value=95.0, key="pred_k2")
                        strike3 = st.number_input("Call Short Strike", value=105.0, key="pred_k3")
                        strike4 = st.number_input("Call Long Strike", value=115.0, key="pred_k4")
                        legs = get_predefined_strategy(strategy_name, strike1, strike2, strike3, strike4)
                        
                else:  # Custom Strategy
                    st.subheader("Build Custom Strategy")
                    num_legs = st.number_input("Number of Legs", value=2, min_value=1, max_value=10, step=1, key="custom_num_legs")
                    
                    legs = []
                    for i in range(num_legs):
                        st.write(f"**Leg {i+1}**")
                        col_type, col_strike, col_qty = st.columns(3)
                        with col_type:
                            leg_type = st.selectbox(f"Type", ["call", "put"], key=f"leg_type_{i}")
                        with col_strike:
                            leg_strike = st.number_input(f"Strike", value=100.0, key=f"leg_strike_{i}")
                        with col_qty:
                            leg_qty = st.number_input(f"Quantity", value=1.0, step=0.1, key=f"leg_qty_{i}")
                        
                        legs.append({"type": leg_type, "strike": leg_strike, "qty": leg_qty})
                
                # Exercise style for strategy
                strategy_exercise = st.selectbox("Exercise Style", ["european", "american"], key="strategy_exercise")
            
            with col2:
                st.subheader("Strategy Analysis")
                
                if isinstance(legs, str):  # Error message from predefined strategy
                    st.error(legs)
                else:
                    # Display strategy legs
                    strategy_df = pd.DataFrame(legs)
                    st.dataframe(strategy_df, use_container_width=True)
                    
                    # Price the strategy
                    try:
                        kwargs = {
                            "S": spot_price, "T": time_to_expiry,
                            "r": risk_free_rate, "sigma": volatility, "q": dividend_yield
                        }
                        
                        if model == "binomial":
                            kwargs["N"] = n_steps
                        elif model == "monte-carlo":
                            kwargs["n_simulations"] = n_simulations
                        
                        strategy_result = price_option_strategy(
                            legs=legs,
                            exercise_style=strategy_exercise,
                            model=model,
                            **kwargs
                        )
                        
                        col_price1, col_price2 = st.columns(2)
                        with col_price1:
                            st.metric("Total Strategy Price", f"${strategy_result['strategy_price']:.4f}")
                        with col_price2:
                            net_premium = strategy_result['strategy_price']
                            strategy_type = "Credit" if net_premium < 0 else "Debit"
                            st.metric("Strategy Type", strategy_type, f"${abs(net_premium):.4f}")
                        
                        # Individual leg prices
                        st.subheader("Individual Leg Prices")
                        for i, (leg, price) in enumerate(zip(legs, strategy_result['individual_prices'])):
                            position = "Long" if leg['qty'] > 0 else "Short"
                            st.write(f"**Leg {i+1}:** {position} {abs(leg['qty'])} {leg['type'].title()} @ {leg['strike']} = ${price:.4f}")
                        
                    except Exception as e:
                        st.error(f"Error pricing strategy: {str(e)}")
    
    # Tab 3: Payoff Analysis
    with taa3:
        if not st.session_state.setup_completed:
            st.warning("⚠️ Please complete the setup in the 'Setup & Parameters' tab first!")
        elif 'legs' in locals() and not isinstance(legs, str):
            st.markdown('<h2 class="sub-header">Strategy Payoff Analysis</h2>', unsafe_allow_html=True)
            
            if 'legs' in locals() and not isinstance(legs, str):
                # Spot price range for payoff calculation
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.subheader("Payoff Parameters")
                    strikes = [leg['strike'] for leg in legs]
                    min_strike, max_strike = min(strikes), max(strikes)
                    
                    spot_min = st.number_input("Min Spot Price", value=min_strike * 0.7, step=1.0, key="payoff_spot_min")
                    spot_max = st.number_input("Max Spot Price", value=max_strike * 1.3, step=1.0, key="payoff_spot_max")
                    n_points = st.slider("Number of Points", 50, 500, 200, key="payoff_n_points")
                    
                    show_breakeven = st.checkbox("Show Breakeven Points", value=True)
                    show_profit_loss = st.checkbox("Include Premium Cost", value=True)
                
                with col2:
                    # Calculate payoff
                    spot_range = np.linspace(spot_min, spot_max, n_points)
                    payoffs = compute_strategy_payoff(legs, spot_range)
                    
                    # Create interactive plotly chart
                    fig = go.Figure()
                    
                    # Payoff at expiration
                    fig.add_trace(go.Scatter(
                        x=spot_range,
                        y=payoffs,
                        mode='lines',
                        name='Payoff at Expiration',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Add profit/loss line if premium is included
                    if show_profit_loss and 'strategy_result' in locals():
                        pnl = payoffs - strategy_result['strategy_price']
                        fig.add_trace(go.Scatter(
                            x=spot_range,
                            y=pnl,
                            mode='lines',
                            name='Profit/Loss (incl. premium)',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                    
                    # Add strike lines
                    for i, leg in enumerate(legs):
                        fig.add_vline(
                            x=leg['strike'],
                            line_dash="dot",
                            line_color="gray",
                            annotation_text=f"K{i+1}: {leg['strike']}"
                        )
                    
                    # Add current spot line
                    fig.add_vline(
                        x=spot_price,
                        line_dash="dash",
                        line_color="green",
                        annotation_text=f"Current Spot: {spot_price}"
                    )
                    
                    # Breakeven analysis
                    if show_breakeven and show_profit_loss and 'strategy_result' in locals():
                        pnl = payoffs - strategy_result['strategy_price']
                        # Find breakeven points (where PnL crosses zero)
                        zero_crossings = []
                        for i in range(len(pnl)-1):
                            if pnl[i] * pnl[i+1] < 0:  # Sign change
                                # Linear interpolation to find exact crossing
                                x_cross = spot_range[i] + (spot_range[i+1] - spot_range[i]) * (-pnl[i] / (pnl[i+1] - pnl[i]))
                                zero_crossings.append(x_cross)
                        
                        for i, breakeven in enumerate(zero_crossings):
                            fig.add_vline(
                                x=breakeven,
                                line_color="orange",
                                line_width=2,
                                annotation_text=f"Breakeven: {breakeven:.2f}"
                            )
                    
                    fig.update_layout(
                        title="Strategy Payoff Diagram",
                        xaxis_title="Spot Price at Expiration",
                        yaxis_title="Payoff ($)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Payoff Statistics")
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Max Payoff", f"${np.max(payoffs):.2f}")
                    with col_stat2:
                        st.metric("Min Payoff", f"${np.min(payoffs):.2f}")
                    with col_stat3:
                        if show_profit_loss and 'strategy_result' in locals():
                            max_profit = np.max(payoffs - strategy_result['strategy_price'])
                            st.metric("Max Profit", f"${max_profit:.2f}")
                        else:
                            st.metric("Max Profit", "N/A")
                    with col_stat4:
                        if show_profit_loss and 'strategy_result' in locals():
                            max_loss = np.min(payoffs - strategy_result['strategy_price'])
                            st.metric("Max Loss", f"${max_loss:.2f}")
                        else:
                            st.metric("Max Loss", "N/A")
    
    # Tab 4: Greeks Analysis
    with taa4:
        if not st.session_state.setup_completed:
            st.warning("⚠️ Please complete the setup in the 'Setup & Parameters' tab first!")
        elif 'legs' in locals() and not isinstance(legs, str):
            st.markdown('<h2 class="sub-header">Greeks Analysis</h2>', unsafe_allow_html=True)
            
            if 'legs' in locals() and not isinstance(legs, str):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Greeks Parameters")
                    greek_name = st.selectbox("Select Greek", ["delta", "gamma", "vega", "theta", "rho"])
                    
                    # Spot range for Greeks
                    strikes = [leg['strike'] for leg in legs]
                    min_strike, max_strike = min(strikes), max(strikes)
                    
                    greek_spot_min = st.number_input("Min Spot for Greeks", value=min_strike * 0.8, step=1.0, key="greeks_spot_min")
                    greek_spot_max = st.number_input("Max Spot for Greeks", value=max_strike * 1.2, step=1.0, key="greeks_spot_max")
                    greek_points = st.slider("Number of Points for Greeks", 50, 300, 150, key="greeks_n_points")
                
                with col2:
                    st.subheader(f"Strategy {greek_name.title()} vs Spot Price")
                    
                    try:
                        # Generate Greeks plot
                        fig = plot_strategy_greek_vs_spot(
                            greek_name=greek_name,
                            legs=legs,
                            model=model,
                            S0=spot_price,
                            T=time_to_expiry,
                            r=risk_free_rate,
                            sigma=volatility,
                            q=dividend_yield,
                            S_range=np.linspace(greek_spot_min, greek_spot_max, greek_points),
                            n_points=greek_points
                        )
                        st.pyplot(fig)
                        
                        # Greeks explanation
                        greek_explanations = {
                            "delta": "Measures price sensitivity to underlying price changes",
                            "gamma": "Measures the rate of change of delta",
                            "vega": "Measures sensitivity to volatility changes",
                            "theta": "Measures time decay (typically negative)",
                            "rho": "Measures sensitivity to interest rate changes"
                        }
                        
                        st.info(f"**{greek_name.title()}**: {greek_explanations[greek_name]}")
                        
                    except Exception as e:
                        st.error(f"Error calculating Greeks: {str(e)}")
                        st.info("Greeks analysis requires the Greeks computation module to be properly implemented.")
        
    # Tab 5: Sensitivity Analysis
    with taa5:
        if not st.session_state.setup_completed:
            st.warning("⚠️ Please complete the setup in the 'Setup & Parameters' tab first!")
        elif 'legs' in locals() and not isinstance(legs, str):
            st.markdown('<h2 class="sub-header">Advanced Sensitivity Analysis</h2>', unsafe_allow_html=True)
            
            if 'legs' in locals() and not isinstance(legs, str):
                st.subheader("Multi-Parameter Sensitivity")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Parameter 1
                    param1 = st.selectbox("First Parameter", ["S", "T", "r", "sigma", "q"], key="param1")
                    param1_range = st.slider(
                        f"{param1} Range (%)",
                        -50, 100, (-20, 20),
                        step=5,
                        key="param1_range"
                    )
                    
                    # Parameter 2
                    param2 = st.selectbox("Second Parameter", ["S", "T", "r", "sigma", "q"], key="param2")
                    param2_range = st.slider(
                        f"{param2} Range (%)",
                        -50, 100, (-20, 20),
                        step=5,
                        key="param2_range"
                    )
                
                with col2:
                    resolution = st.slider("Grid Resolution", 10, 50, 20, key="sensitivity_resolution")
                    
                    if st.button("Generate Sensitivity Heatmap", type="primary"):
                        try:
                            # Get base values
                            base_values = {
                                "S": spot_price, "T": time_to_expiry, "r": risk_free_rate,
                                "sigma": volatility, "q": dividend_yield
                            }
                            
                            # Create parameter grids
                            param1_vals = np.linspace(
                                base_values[param1] * (1 + param1_range[0]/100),
                                base_values[param1] * (1 + param1_range[1]/100),
                                resolution
                            )
                            param2_vals = np.linspace(
                                base_values[param2] * (1 + param2_range[0]/100),
                                base_values[param2] * (1 + param2_range[1]/100),
                                resolution
                            )
                            
                            # Calculate strategy prices for each combination
                            price_grid = np.zeros((len(param2_vals), len(param1_vals)))
                            
                            progress_bar = st.progress(0)
                            total_iterations = len(param1_vals) * len(param2_vals)
                            iteration = 0
                            
                            for i, p1_val in enumerate(param1_vals):
                                for j, p2_val in enumerate(param2_vals):
                                    kwargs = base_values.copy()
                                    kwargs[param1] = p1_val
                                    kwargs[param2] = p2_val
                                    
                                    if model == "binomial":
                                        kwargs["N"] = n_steps
                                    elif model == "monte-carlo":
                                        kwargs["n_simulations"] = min(n_simulations, 1000)  # Reduce for speed
                                    
                                    try:
                                        result = price_option_strategy(
                                            legs=legs,
                                            exercise_style=strategy_exercise,
                                            model=model,
                                            **kwargs
                                        )
                                        price_grid[j, i] = result["strategy_price"]
                                    except:
                                        price_grid[j, i] = np.nan
                                    
                                    iteration += 1
                                    progress_bar.progress(iteration / total_iterations)
                            
                            # Create heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=price_grid,
                                x=param1_vals,
                                y=param2_vals,
                                colorscale='RdYlBu_r',
                                colorbar=dict(title="Strategy Price")
                            ))
                            
                            fig.update_layout(
                                title=f"Strategy Price Sensitivity: {param1} vs {param2}",
                                xaxis_title=param1,
                                yaxis_title=param2,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating sensitivity analysis: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Advanced Options Pricing Suite | Built with Streamlit & Python</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    # Error handling and warnings
    if model == "black-scholes" and any(leg.get('qty', 1) != int(leg.get('qty', 1)) for leg in locals().get('legs', [])):
        st.sidebar.warning("⚠️ Black-Scholes assumes integer quantities")
    
    if model == "monte-carlo" and 'n_simulations' in locals() and n_simulations < 1000:
        st.sidebar.warning("⚠️ Low simulation count may affect accuracy")









# -----------------------------
# Tab 4 – Exotic Options
# -----------------------------
with tab4:
    # Main title
    st.markdown('<div class="main-header">Exotic Options Pricing Toolkit</div>', unsafe_allow_html=True)
    
    if not MODULES_LOADED:
        st.stop()
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <strong>Welcome to the Exotic Options Pricing Toolkit!</strong><br>
    This comprehensive application allows you to price and analyze various exotic options including:
    <ul>
    <li>🔄 <strong>Asian Options</strong> - Options based on average prices</li>
    <li>🚧 <strong>Barrier Options</strong> - Options with knock-in/knock-out features</li>
    <li>💻 <strong>Digital Options</strong> - Binary payoff options</li>
    <li>👀 <strong>Lookback Options</strong> - Options based on extrema</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different option types
    tabb1, tabb2, tabb3, tabb4, tabb5 = st.tabs([
        "🔄 Asian Options", 
        "🚧 Barrier Options", 
        "💻 Digital Options", 
        "👀 Lookback Options",
        "📊 Portfolio Analysis"
    ])
    
    # Tab 1: Asian Options
    with tabb1:
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
            
            asian_option_type = st.selectbox("Option Type", ["call", "put"], key="asian_option_type")
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
                        st.success(f"**Asian Option Price: ${asian_price:.4f}**")
                        
                        # Calculate and display Greeks if requested
                        if show_greeks_asian:
                            greeks = calculate_greeks_asian(
                                asian_S0, asian_K, asian_T, asian_r, asian_sigma,
                                asian_n_steps, asian_n_paths, asian_option_type, asian_type
                            )
                            
                            col_g1, col_g2, col_g3 = st.columns(3)
                            with col_g1:
                                st.metric("Delta (Δ)", f"{greeks['Delta']:.4f}")
                                st.metric("Gamma (Γ)", f"{greeks['Gamma']:.6f}")
                            with col_g2:
                                st.metric("Theta (Θ)", f"{greeks['Theta']:.4f}")
                                st.metric("Vega (ν)", f"{greeks['Vega']:.4f}")
                            with col_g3:
                                st.metric("Rho (ρ)", f"{greeks['Rho']:.4f}")
                        
                        # Show payoff diagram
                        st.subheader("📈 Payoff Diagram")
                        plot_asian_option_payoff(asian_K, asian_option_type, asian_type)
                        
                        # Sensitivity analysis
                        if show_sensitivity_asian:
                            st.subheader("📊 Sensitivity Analysis")
                            
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
    
    # Tab 2: Barrier Options
    with tabb2:
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
                            st.warning("⚠️ For up barriers, H should typically be above current spot and strike prices")
                        elif barrier_type.startswith("down") and barrier_H >= min(barrier_S, barrier_K):
                            st.warning("⚠️ For down barriers, H should typically be below current spot and strike prices")
                        
                        # Calculate option price
                        barrier_price, paths = price_barrier_option(
                            S=barrier_S, K=barrier_K, H=barrier_H, T=barrier_T,
                            r=barrier_r, sigma=barrier_sigma,
                            option_type=barrier_option_type, barrier_type=barrier_type,
                            n_simulations=barrier_n_sim, n_steps=barrier_n_steps
                        )
                        
                        # Display results
                        st.success(f"**Barrier Option Price: ${barrier_price:.4f}**")
                        
                        # Show payoff diagram
                        st.subheader("📈 Payoff Diagram")
                        plot_barrier_payoff(
                            barrier_K, barrier_H, barrier_option_type, barrier_type,
                            S_min=barrier_S*0.5, S_max=barrier_S*1.5
                        )
                        
                        # Show sample paths if requested
                        if show_paths_barrier and paths is not None:
                            st.subheader("📊 Sample Monte Carlo Paths")
                            plot_sample_paths_barrier(
                                paths[:20], barrier_K, barrier_H, 
                                barrier_option_type, barrier_type
                            )
                        
                        # Market insights
                        st.subheader("💡 Market Insights")
                        if "out" in barrier_type:
                            st.info("**Knock-out options** are cheaper than vanilla options as they can expire worthless if the barrier is breached.")
                        else:
                            st.info("**Knock-in options** are cheaper than vanilla options as they only become active if the barrier is breached.")
                        
                    except Exception as e:
                        st.error(f"Error calculating barrier option: {str(e)}")
    
    # Tab 3: Digital Options
    with tabb3:
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
                        st.success(f"**Digital Option Price: ${digital_price:.4f}**")
                        
                        # Calculate and display Greeks if requested
                        if show_greeks_digital:
                            greeks = calculate_greeks_digital(
                                digital_S, digital_K, digital_T, digital_r, digital_sigma,
                                digital_option_type, digital_style, digital_Q
                            )
                            
                            col_g1, col_g2, col_g3 = st.columns(3)
                            with col_g1:
                                st.metric("Delta (Δ)", f"{greeks['Delta']:.4f}")
                                st.metric("Gamma (Γ)", f"{greeks['Gamma']:.6f}")
                            with col_g2:
                                st.metric("Theta (Θ)", f"{greeks['Theta']:.4f}")
                                st.metric("Vega (ν)", f"{greeks['Vega']:.4f}")
                            with col_g3:
                                st.metric("Rho (ρ)", f"{greeks['Rho']:.4f}")
                        
                        # Show payoff diagram
                        st.subheader("📈 Payoff Diagram")
                        plot_digital_payoff(
                            digital_K, digital_option_type, digital_style, digital_Q,
                            S_min=digital_S*0.5, S_max=digital_S*1.5
                        )
                        
                        # Educational content
                        st.subheader("📚 Digital Options Explained")
                        if digital_style == "cash":
                            st.info(f"**Cash-or-Nothing**: Pays ${digital_Q:.2f} if the option finishes in-the-money, nothing otherwise.")
                        else:
                            st.info("**Asset-or-Nothing**: Pays the asset price if the option finishes in-the-money, nothing otherwise.")
                        
                    except Exception as e:
                        st.error(f"Error calculating digital option: {str(e)}")
    
    # Tab 4: Lookback Options
    with tabb4:
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
                        st.success(f"**Lookback Option Price: ${lookback_price:.4f} ± {lookback_stderr:.4f}**")
                        
                        # Confidence interval
                        ci_lower = lookback_price - 1.96 * lookback_stderr
                        ci_upper = lookback_price + 1.96 * lookback_stderr
                        st.info(f"**95% Confidence Interval: [${ci_lower:.4f}, ${ci_upper:.4f}]**")
                        
                        # Show payoff diagram
                        st.subheader("📈 Payoff Function")
                        fig_payoff = plot_payoff(lookback_S0, lookback_option_type, lookback_K, lookback_floating)
                        st.pyplot(fig_payoff)
                        
                        # Show sample paths if requested
                        if show_paths_lookback:
                            st.subheader("📊 Sample Price Paths")
                            fig_paths = plot_paths(lookback_S0, lookback_r, lookback_sigma, lookback_T, 
                                                 min(10, lookback_n_paths), lookback_n_steps)
                            st.pyplot(fig_paths)
                        
                        # Show payoff distribution if requested
                        if show_distribution:
                            st.subheader("📈 Payoff Distribution")
                            fig_dist = plot_price_distribution(
                                lookback_S0, lookback_r, lookback_sigma, lookback_T,
                                lookback_option_type, lookback_floating,
                                min(10000, lookback_n_paths), lookback_n_steps
                            )
                            st.pyplot(fig_dist)
                        
                        # Educational content
                        st.subheader("💡 Lookback Options Explained")
                        if lookback_floating:
                            if lookback_option_type == "call":
                                st.info("**Floating Strike Call**: Pays S_T - min(S_t), where min(S_t) is the minimum price during the option's life.")
                            else:
                                st.info("**Floating Strike Put**: Pays max(S_t) - S_T, where max(S_t) is the maximum price during the option's life.")
                        else:
                            if lookback_option_type == "call":
                                st.info(f"**Fixed Strike Call**: Pays max(0, max(S_t) - ${lookback_K}), based on the maximum price reached.")
                            else:
                                st.info(f"**Fixed Strike Put**: Pays max(0, ${lookback_K} - min(S_t)), based on the minimum price reached.")
                        
                    except Exception as e:
                        st.error(f"Error calculating lookback option: {str(e)}")
    
    # Tab 5: Portfolio Analysis
    with tabb5:
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
                                "monte_carlo", "call", "average_price"
                            )
                            results.append({"Option Type": "Asian Call", "Price": asian_price, "Complexity": "Medium"})
                        
                        if include_barrier:
                            barrier_price, _ = price_barrier_option(
                                port_S0, port_K, port_S0*1.2, port_T, port_r, port_sigma,
                                "call", "up-and-out", "monte_carlo", 10000, 100
                            )
                            results.append({"Option Type": "Barrier Call", "Price": barrier_price, "Complexity": "Medium"})
                        
                        if include_digital:
                            digital_price = price_digital_option(
                                "black_scholes", "call", "cash", port_S0, port_K, port_T, port_r, port_sigma
                            )
                            results.append({"Option Type": "Digital Call", "Price": digital_price, "Complexity": "Low"})
                        
                        if include_lookback:
                            lookback_price, _ = price_lookback_option(
                                port_S0, None, port_r, port_sigma, port_T, "monte_carlo", "call", True, 10000, 252
                            )
                            results.append({"Option Type": "Lookback Call", "Price": lookback_price, "Complexity": "High"})
                        
                        # Display results
                        if results:
                            df_results = pd.DataFrame(results)
                            df_results['Price'] = df_results['Price'].round(4)
                            
                            st.subheader("💰 Portfolio Summary")
                            st.dataframe(df_results, use_container_width=True)
                            
                            # Total portfolio value
                            total_value = df_results['Price'].sum()
                            st.metric("**Total Portfolio Value**", f"${total_value:.4f}")
                            
                            # Price comparison chart
                            fig = px.bar(df_results, x='Option Type', y='Price', 
                                       color='Complexity', title="Option Prices Comparison")
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Risk analysis
                            st.subheader("⚠️ Risk Analysis")
                            
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
                            st.subheader("🛡️ Hedging Suggestions")
                            st.info("""
                            **Portfolio Hedging Strategies:**
                            - **Delta Hedging**: Regularly adjust underlying position to maintain

    





# -----------------------------
# Tab 5 – Swaps
# -----------------------------

from pricing.swaps import price_swap
from pricing.models.swaps.ois_fx import (
    build_flat_discount_curve,
    build_flat_fx_forward_curve
)

with tab5:
    st.markdown('<h1 class="main-header">Swap Pricer (In Progress)</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
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


with tab6:
    # Add model selection info box
    st.markdown('<h1 class="main-header">Interest Rate Model Selector</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    model = st.selectbox(
        "Choose a model to explore:",
        ["Vasicek Model", "Hull-White Model (Coming Soon)", "Cox-Ingersoll-Ross (CIR) (Coming Soon)"],
        index=0,
        help="Select the interest rate model you want to explore."
    )

    if model != "Vasicek Model":
        st.warning("🚧 This model is not yet available. Stay tuned!")
        st.stop()  # Exit until Vasicek is selected

        # Proceed with Vasicek UI if selected...
    else:
        st.subheader("Vasicek Model – Bond Pricing and Interest Rates")
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Parameter Estimation", "Simulation & Yield Curves", "Bond Pricing", "Bond Options", "Greeks Analysis"]
        )
        
        # Session state to store estimated parameters
        if 'vasicek_params' not in st.session_state:
            st.session_state.vasicek_params = None
        
        # =============================================
        # TAB 1: PARAMETER ESTIMATION
        # =============================================
        with tab1:
            st.header("Vasicek Model Parameter Estimation")
        
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
                        st.metric("Speed of mean reversion (a)", f"{params['a']:.4f}")
                    with col_lam:
                        st.metric("Long-term mean level (λ)", f"{params['lambda']:.4f}")
                    with col_sig:
                        st.metric("Volatility (σ)", f"{params['sigma']:.4f}")
        
                    st.metric("Initial rate (r₀)", f"{params['r0']:.4f}")
                    st.info(f"Ticker used: **{params['ticker']}** | Δt: {params['dt']:.4f}")
                else:
                    st.info("Click 'Estimate Parameters' to get started")

    
        
        # =============================================
        # TAB 2: SIMULATION AND YIELD CURVES
        # =============================================
        with tab2:
            st.header("Simulation of Rate Paths and Yield Curves (Vasicek)")
        
            if not st.session_state.vasicek_params:
                st.warning("⚠️ Please estimate the parameters in the previous tab first.")
                st.stop()
        
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
        
                        # 📉 Final short rate distribution
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
                            st.metric("Final Mean", f"{np.mean(r_final):.4f}")
                        with col_stat2:
                            st.metric("Standard Deviation", f"{np.std(r_final):.4f}")
                        with col_stat3:
                            st.metric("Min / Max", f"{np.min(r_final):.4f} / {np.max(r_final):.4f}")

    
        
        # =============================================
        # TAB 3: BOND PRICING
        # =============================================
        with tab3:
            st.header("Bond Pricing (Zero-Coupon or Coupon Bonds)")
        
            if not st.session_state.vasicek_params:
                st.warning("⚠️ Please estimate the parameters in the previous tab first.")
                st.stop()
        
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
                                st.success(f"Zero-Coupon Bond Price: **{price:.2f}**")
        
                                ytm = -np.log(price / face_value) / (maturity - t_current)
                                st.info(f"Yield to Maturity (YTM): **{ytm:.4f} ({ytm*100:.2f}%)**")
        
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
                                st.success(f"Coupon Bond Price: **{price:.2f}**")
                                st.info(f"Coupon: {coupon_rate*100:.2f}% ({freq})")
        
                            # -------- Sensitivity Analysis --------
                            st.subheader("Sensitivity Analysis")
        
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

    
    
        # =============================================
        # TAB 4: BOND OPTIONS PRICING
        # =============================================
        with tab4:
            st.header("Bond Option Pricing")
        
            if not st.session_state.vasicek_params:
                st.warning("⚠️ Please estimate the parameters in the previous tab first.")
                st.stop()
        
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
                        st.error("⚠️ The bond maturity (T₂) must be greater than the option maturity (T₁).")
                        st.stop()
        
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
                                st.success(f"{option_type} Option Price (Analytical): **{price:.4f}**")
        
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
                                st.success(f"{option_type} Option Price (Monte Carlo): **{price:.4f} ± {std:.4f}**")
                                st.info(f"95% Confidence Interval: [{price - 1.96*std:.4f}, {price + 1.96*std:.4f}]")
        
                            st.subheader("Summary")
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

        
        # =============================================
        # TAB 5: GREEKS ANALYSIS
        # =============================================
        with tab5:
            st.header("Greeks Analysis for Bond Options")
        
            if not st.session_state.vasicek_params:
                st.warning("⚠️ Please estimate the parameters in the previous tab first.")
                st.stop()
        
            params = st.session_state.vasicek_params
        
            # Actual import of Greeks computation
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


