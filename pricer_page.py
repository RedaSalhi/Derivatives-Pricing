# main.py

import sys
import os
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from pricing.vanilla_options import price_vanilla_option, plot_option_price_vs_param
from pricing.forward import *
from pricing.option_strategies import *
from pricing.utils.greeks_vanilla.plot_single_greek import plot_single_greek_vs_spot
from pricing.utils.greeks_vanilla.greeks_interface import *

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .parameter-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
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
    st.title("Advanced Option Pricing Tool")
    st.markdown("Price options using Black-Scholes, Binomial Tree, and Monte Carlo methods")
    
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
        greek_model = st.selectbox("Model for Greeks:", ["Black-Scholes", "Binomial", "Monte-Carlo"], key="greek_model")
        
        if st.button("Calculate Greeks"):
            try:
                # Calculate Greeks for current spot price
                greeks_list = ["delta", "gamma", "theta", "vega", "rho"]
                greeks_values = {}
                
                for greek in greeks_list:
                    try:
                        greek_val = compute_greek(
                            greek_name=greek,
                            model=greek_model,
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
"""
with tab2:
    st.header("Forward Contract Pricing")

    col1, col2 = st.columns(2)
    with col1:
        S_fwd = st.number_input("Spot Price", value=100.0, key="fwd_spot")
        K_fwd = st.number_input("Strike Price", value=100.0, key="fwd_strike")
        T_fwd = st.number_input("Time to Maturity (T)", value=1.0, key="fwd_T")
        r_fwd = st.number_input("Risk-Free Rate (r)", value=0.05, key="fwd_r")
    with col2:
        storage_cost = st.number_input("Storage Cost (c)", value=0.0, key="fwd_storage")
        dividend_yield = st.number_input("Dividend Yield (q)", value=0.0, key="fwd_q")
        position = st.radio("Position", ["Long", "Short"])
        if position == "Long":
            position = "long"
        elif position == "Short":
            position = "short"

    if st.button("Price Forward Contract"):
        F = price_forward_contract(
            spot_price=S_fwd,
            interest_rate=r_fwd,
            time_to_maturity=T_fwd,
            storage_cost=storage_cost,
            dividend_yield=dividend_yield
        )
        st.success(f"Theoretical Forward Price: **{F:.4f}**")

        with st.expander("Forward Payout at Maturity"):
            plot_forward_payout_and_value(K_fwd, position)

        with st.expander("Mark-to-Market Value (Before Maturity)"):
            plot_forward_mark_to_market(
                strike_price=K_fwd,
                time_to_maturity=T_fwd,
                interest_rate=r_fwd,
                storage_cost=storage_cost,
                dividend_yield=dividend_yield,
                position=position
            )
"""


with tab2:
    st.markdown('<h1 class="main-header">📈 Forward Contract Pricing & Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Parameter input section
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("## 🔧 Contract Parameters")
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Pricing Results", "📈 Mark-to-Market", "💰 Payout Analysis", "🔬 Sensitivity"])
    
    with tab1:
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
        st.markdown("### 📝 Cost of Carry Model")
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.latex(r"F = S_0 \cdot e^{(r + c - q) \cdot T}")
        st.markdown("""
        **Where:**
        - **F** = Forward Price = ${:.2f}
        - **S₀** = Current Spot Price = ${:.2f}
        - **r** = Risk-free Interest Rate = {:.2f}%
        - **c** = Storage Cost Rate = {:.2f}%
        - **q** = Dividend Yield = {:.2f}%
        - **T** = Time to Maturity = {:.4f} years
        """.format(forward_price, spot_price, interest_rate*100, storage_cost*100, dividend_yield*100, time_to_maturity))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed breakdown
        with st.expander("🧮 Calculation Breakdown"):
            net_carry = interest_rate + storage_cost - dividend_yield
            st.write(f"**Net Carry Rate:** {interest_rate*100:.2f}% + {storage_cost*100:.2f}% - {dividend_yield*100:.2f}% = {net_carry*100:.2f}%")
            st.write(f"**Exponential Factor:** e^({net_carry*100:.2f}% × {time_to_maturity:.4f}) = {np.exp(net_carry * time_to_maturity):.6f}")
            st.write(f"**Forward Price:** ${spot_price:.2f} × {np.exp(net_carry * time_to_maturity):.6f} = **${forward_price:.2f}**")
    
    with tab2:
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
        st.markdown("### 📊 Original Chart (Matplotlib)")
        plot_forward_mark_to_market_plotly(
            strike_price, time_to_maturity, interest_rate, 
            storage_cost, dividend_yield, position.lower()
        )
    
    with tab3:
        st.markdown("## Payout Analysis at Maturity")
        st.markdown("*Profit/Loss when contract expires (t = T)*")
        
        # Interactive Plotly payout chart
        fig_payout = create_plotly_payout_chart(strike_price, position.lower())
        st.plotly_chart(fig_payout, use_container_width=True)
        
        # Scenario analysis
        st.markdown("### 📋 Payout Scenarios")
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
        st.markdown("### 📊 Original Chart (Matplotlib)")
        plot_forward_payout_and_value(strike_price, position.lower())
    
    with tab4:
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
        st.markdown("### 📈 Risk Sensitivities")
        
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
    st.header("Option Strategies")

    use_manual = st.checkbox("Build Strategy Manually")

    model_strat = st.selectbox("Pricing Model", ["Black Scholes", "Binomial", "Monte Carlo"], key="strat_model")
    model_strat = {"Black Scholes": "black-scholes", "Monte Carlo": "monte-carlo"}.get(model_strat, "binomial")

    style_strat = st.selectbox("Exercise Style", ["European", "American"], key="strat_style").lower()

    S_strat = st.number_input("Spot Price (S)", value=100.0, key="strat_S")
    T_strat = st.number_input("Time to Maturity (T)", value=1.0, key="strat_T")
    sigma_strat = st.number_input("Volatility (σ)", value=0.2, key="strat_sigma")
    r_strat = st.number_input("Risk-Free Rate (r)", value=0.05, key="strat_r")
    q_strat = st.number_input("Dividend Yield (q)", value=0.0, key="strat_q")

    kwargs = {"S": S_strat, "T": T_strat, "sigma": sigma_strat, "r": r_strat, "q": q_strat}

    # -----------------------------
    # Manual Strategy
    # -----------------------------
    if use_manual:
        st.subheader("➕ Add Legs to Strategy")

        if "custom_legs" not in st.session_state:
            st.session_state.custom_legs = []

        with st.form("add_leg_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                opt_type = st.selectbox("Type", ["call", "put"], key="leg_type")
            with col2:
                strike = st.number_input("Strike", value=100.0, key="leg_strike")
            with col3:
                qty = st.number_input("Quantity", step=1, value=1, key="leg_qty")
            if st.form_submit_button("Add Leg"):
                st.session_state.custom_legs.append({"type": opt_type, "strike": strike, "qty": qty})

        if st.session_state.custom_legs:
            st.markdown("### Strategy Legs")
            for i, leg in enumerate(st.session_state.custom_legs):
                st.write(f"Leg {i+1}: {leg['qty']} × {leg['type'].upper()} @ Strike {leg['strike']}")

            if st.button("Price Custom Strategy"):
                try:
                    legs = st.session_state.custom_legs
                    st.session_state["manual_legs"] = legs
                    result = price_option_strategy(legs, style_strat, model_strat, **kwargs)
                    st.success(f"Total Strategy Price: **{result['strategy_price']:.4f}**")

                    spot_range = np.linspace(0.5 * S_strat, 1.5 * S_strat, 500)
                    payoff = compute_strategy_payoff(legs, spot_range)

                    with st.expander("Payoff Plot"):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(spot_range, payoff, label="Strategy Payoff", color="blue")
                        ax.axhline(0, color="black", linestyle="--")
                        ax.set_xlabel("Spot Price at Maturity (S)")
                        ax.set_ylabel("Net Payoff")
                        ax.set_title("Payoff Diagram: Custom Strategy")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")

            if "manual_legs" in st.session_state:
                st.subheader("Visualize Strategy Price vs Parameter")

                param = st.selectbox("Parameter to Vary", ["S", "T", "r", "sigma", "q"], key="manual_vary")
                default_val = float(kwargs.get(param, 1.0))

                min_val = st.number_input(f"Min {param}", value=0.0 if param in ["r", "q", "sigma"] else 0.01, key="manual_min")
                max_val = st.number_input(f"Max {param}", value=default_val * (1.5 if param == "S" else 100), key="manual_max")
                n_points = st.slider("Resolution", 50, 500, 100, key="manual_n")

                if st.button("Generate Plot for Custom Strategy"):
                    try:
                        with st.expander("Strategy Premium vs Parameter"):
                            fig = plot_strategy_price_vs_param(
                                legs=st.session_state["manual_legs"],
                                exercise_style=style_strat,
                                model=model_strat,
                                param_name=param,
                                param_range=(min_val, max_val),
                                fixed_params=kwargs,
                                n_points=n_points
                            )
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Plotting failed: {e}")

                st.subheader("Visualize Strategy Greek vs Spot Price")

                greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"], key="manual_greek")
                if greek == "Delta":
                    greek = "delta"
                elif greek == "Gamma":
                    greek = "gamma"
                elif greek == "Vega":
                    greek = "vega"
                elif greek == "Theta":
                    greek = "theta"
                elif greek == "Rho":
                    greek = "rho"
                S_min = st.number_input("Min Spot (S)", value=0.5 * S_strat, key="manual_greek_smin")
                S_max = st.number_input("Max Spot (S)", value=1.5 * S_strat, key="manual_greek_smax")
                greek_res = st.slider("Greek Plot Resolution", 50, 1000, 300, key="manual_greek_n")

                if st.button("Plot Strategy Greek (Manual)"):
                    try:
                        with st.expander("Strategy Greek"):
                            fig = plot_strategy_greek_vs_spot(
                                greek_name=greek,
                                legs=st.session_state["manual_legs"],
                                model=model_strat,
                                S0=S_strat,
                                T=T_strat,
                                r=r_strat,
                                sigma=sigma_strat,
                                q=q_strat,
                                S_range=np.linspace(S_min, S_max, greek_res)
                            )
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Greek plot failed: {e}")

        if st.button("Clear Strategy Legs"):
            st.session_state.custom_legs = []
            st.session_state.pop("manual_legs", None)

    # -----------------------------
    # Predefined Strategy
    # -----------------------------
    else:
        st.subheader("Predefined Strategy")
        strategy = st.selectbox("Choose a Strategy", ["Straddle", "Bull call Spread", "Bear put Spread", "Butterfly", "Iron Condor"], key="strat_type")

        strike1 = st.number_input("Strike 1", value=95.0, key="strat_k1")
        strike2 = strike3 = strike4 = None
        if strategy in ["Bull call Spread", "Bear put Spread", "Butterfly", "Iron Condor"]:
            strike2 = st.number_input("Strike 2", value=100.0, key="strat_k2")
        if strategy in ["Butterfly", "Iron Condor"]:
            strike3 = st.number_input("Strike 3", value=105.0, key="strat_k3")
        if strategy == "Iron Condor":
            strike4 = st.number_input("Strike 4", value=110.0, key="strat_k4")

        if st.button("Price Predefined Strategy"):
            legs = get_predefined_strategy(strategy, strike1, strike2, strike3, strike4=strike4)
            if isinstance(legs, str):
                st.error(legs)
            else:
                try:
                    st.session_state["predefined_legs"] = legs
                    result = price_option_strategy(legs, style_strat, model_strat, **kwargs)
                    st.success(f"Total Strategy Price: **{result['strategy_price']:.4f}**")

                    spot_range = np.linspace(0.5 * S_strat, 1.5 * S_strat, 500)
                    payoff = compute_strategy_payoff(legs, spot_range)
                    with st.expander("Payoff Plot"):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(spot_range, payoff, label="Strategy Payoff", color="green")
                        ax.axhline(0, color="black", linestyle="--")
                        ax.set_xlabel("Spot Price at Maturity (S)")
                        ax.set_ylabel("Net Payoff")
                        ax.set_title(f"Payoff Diagram: {strategy}")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")

        if "predefined_legs" in st.session_state:
            st.subheader("Visualize Strategy Price vs Parameter")

            param = st.selectbox("Parameter to Vary", ["S", "T", "r", "sigma", "q"], key="predef_vary")
            default_val = float(kwargs.get(param, 1.0))

            min_val = st.number_input(f"Min {param}", value=0.0 if param in ["r", "q", "sigma"] else 0.01, key="predef_min")
            max_val = st.number_input(f"Max {param}", value=default_val * (1.5 if param == "S" else 100), key="predef_max")
            n_points = st.slider("Resolution", 50, 500, 100, key="predef_n")

            if st.button("Generate Plot for Predefined Strategy"):
                try:
                    with st.expander("Strategy Premium vs Parameter"):
                        fig = plot_strategy_price_vs_param(
                            legs=st.session_state["predefined_legs"],
                            exercise_style=style_strat,
                            model=model_strat,
                            param_name=param,
                            param_range=(min_val, max_val),
                            fixed_params=kwargs,
                            n_points=n_points
                        )
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Plotting failed: {e}")

            st.subheader("Visualize Strategy Greek vs Spot Price")

            greek = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"], key="manual_greek")
            if greek == "Delta":
                greek = "delta"
            elif greek == "Gamma":
                greek = "gamma"
            elif greek == "Vega":
                greek = "vega"
            elif greek == "Theta":
                greek = "theta"
            elif greek == "Rho":
                greek = "rho"
            S_min = st.number_input("Min Spot (S)", value=0.5 * S_strat, key="predef_greek_smin")
            S_max = st.number_input("Max Spot (S)", value=1.5 * S_strat, key="predef_greek_smax")
            greek_res = st.slider("Greek Plot Resolution", 50, 1000, 300, key="predef_greek_n")

            if st.button("Plot Strategy Greek (Predefined)"):
                try:
                    with st.expander("Strategy Greek"):
                        fig = plot_strategy_greek_vs_spot(
                            greek_name=greek,
                            legs=st.session_state["predefined_legs"],
                            model=model_strat,
                            S0=S_strat,
                            T=T_strat,
                            r=r_strat,
                            sigma=sigma_strat,
                            q=q_strat,
                            S_range=np.linspace(S_min, S_max, greek_res)
                        )
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Greek plot failed: {e}")





# -----------------------------
# Tab 4 – Exotic Options
# -----------------------------
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff, plot_sample_paths_barrier
from pricing.asian_option import price_asian_option, plot_asian_option_payoff, plot_monte_carlo_paths
from pricing.models.asian_monte_carlo import simulate_asian_paths
from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution

with tab4:
    st.header("Exotic Option Pricing (In Progress)")

    exotic_type = st.selectbox("Select Exotic Option Type", ["Digital", "Barrier", "Asian", "Lookback"])

    # ===========================
    # Digital Option Interface
    # ===========================
    if exotic_type == "Digital":
        st.subheader("Digital Option")

        col1, col2 = st.columns(2)
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="dig_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            style = st.selectbox("Digital Style", ["cash", "asset"], key="dig_style")
            model = st.selectbox("Model", ["Black Scholes"], key="dig_model")
            if model == "Black Scholes":
                model = "black_scholes"
        with col2:
            S = st.number_input("Spot Price (S)", value=100.0, key="dig_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="dig_K")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="dig_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="dig_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="dig_r")
            Q = st.number_input("Payout (Q)", value=10.0, key="dig_Q")

        if st.button("Compute Digital Option Price"):
            try:
                price = price_digital_option(
                    model=model,
                    option_type=option_type,
                    style=style,
                    S=S,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    Q=Q
                )
                st.success(f"The {style} digital {option_type} is worth: **{price:.4f}**")
                with st.expander("Payoff at Maturity"):
                    plot_digital_payoff(K=K, option_type=option_type, style=style, Q=Q)

            except Exception as e:
                st.error(f"Error: {e}")

    # ===========================
    # Barrier Option Interface
    # ===========================
    elif exotic_type == "Barrier":
        st.subheader("Barrier Option")

        col1, col2 = st.columns(2)
        with col1:
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="bar_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            barrier_type = st.selectbox("Barrier Type", ["Up and Out", "Up and In", "Down and Out", "Down and In"], key="bar_style")
            if barrier_type == "Up and Out":
                barrier_type = "up-and-out"
            elif barrier_type == "Up and In":
                barrier_type = "up-and-in"
            elif barrier_type == "Down and Out":
                barrier_type = "down-and-out"
            elif barrier_type == "Down and In":
                barrier_type = "down-and-in"
            model = st.selectbox("Model", ["Monte Carlo"], key="bar_model")
            if model == "Monte Carlo":
                model = "monte_carlo"
        with col2:
            S = st.number_input("Spot Price (S)", value=100.0, key="bar_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="bar_K")
            H = st.number_input("Barrier Level (H)", value=120.0, key="bar_H")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="bar_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="bar_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="bar_r")

        # Sliders appear only if Monte Carlo is selected
        if model == "monte_carlo":
            n_sim = st.slider("Number of Simulations", min_value=10, max_value=10000, step=10, value=1000)
            n_steps = st.slider("Steps per Path", min_value=10, max_value=300, step=2, value=252)
        else:
            n_sim = None
            n_steps = None

        if st.button("Compute Barrier Option Price"):
            st.markdown("<small>Wait a few seconds for the plots !</small>", unsafe_allow_html=True)
            try:
                kwargs = dict(
                    model=model,
                    option_type=option_type,
                    barrier_type=barrier_type,
                    S=S,
                    K=K,
                    H=H,
                    T=T,
                    r=r,
                    sigma=sigma
                )
        
                if model == "monte_carlo":
                    kwargs["n_simulations"] = n_sim
                    kwargs["n_steps"] = n_steps
                    price, paths = price_barrier_option(**kwargs)
        
                st.success(f"The {barrier_type} {option_type} option is worth: **{price:.4f}**")
        
                if model == "monte_carlo":
                    with st.expander("Payoff at Maturity"):
                        plot_barrier_payoff(K=K, H=H, option_type=option_type, barrier_type=barrier_type)
                    with st.expander("Monte Carlo Sample Paths"):
                        plot_sample_paths_barrier(paths, K=K, H=H, option_type=option_type, barrier_type=barrier_type)
                    
        
            except Exception as e:
                st.error(f"Error: {e}")
    # ===========================
    # Asian Option Interface
    # ===========================
    elif exotic_type == "Asian":
        st.subheader("Asian Option")

        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Pricing Method", ["Monte Carlo"], key="asian_method")
            if method == "Monte Carlo":
                method = "monte_carlo"
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="asian_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            asian_type = st.selectbox("Asian Type", ["Average Price", "Average Strike"], key="asian_style")
            if asian_type == "Average Price":
                asian_type = "average_price"
            elif asian_type == "Average Strike":
                asian_type = "average_strike"
        with col2:
            S = st.number_input("Spot Price (S)", value=100.0, key="asian_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="asian_K")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="asian_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="asian_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="asian_r")

        if method == "monte_carlo":
            n_paths = st.slider("Monte Carlo Simulations", 10, 10000, step=10, value=10000)
            n_steps = st.slider("Steps per Path", 10, 300, step=2, value=252)

        if st.button("Compute Asian Option Price"):
            st.markdown("<small>Wait a few seconds for the plots !</small>", unsafe_allow_html=True)
            try:
                price = price_asian_option(
                    S0=S, K=K, T=T, r=r, sigma=sigma,
                    n_steps=n_steps,
                    n_paths=n_paths or 10000,
                    method=method,
                    option_type=option_type,
                    asian_type=asian_type
                )

                st.success(f"The {asian_type.replace('_', ' ')} Asian {option_type} is worth: **{price:.4f}**")

                with st.expander("Payoff at Maturity"):    
                    plot_asian_option_payoff(K=K, option_type=option_type, asian_type=asian_type)

                if method == "monte_carlo":
                    paths = simulate_asian_paths(S0=S, T=T, r=r, sigma=sigma,
                                                 n_steps=n_steps, n_paths=n_paths,
                                                 option_type=option_type, asian_type=asian_type)
                    with st.expander("Monte Carlo Sample Paths"):
                        plot_monte_carlo_paths(paths)

            except Exception as e:
                st.error(f"Error: {e}")
    # ===========================
    # Lookback Option Interface
    # ===========================
    elif exotic_type == "Lookback":
        st.subheader("Lookback Option")

        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Pricing Model", ["Monte Carlo"], key="lookback_model")
            if model == "Monte Carlo":
                model = "monte_carlo"
            option_type = st.selectbox("Option Type", ["Call", "Put"], key="lookback_type")
            if option_type == "Call":
                option_type = "call"
            elif option_type == "Put":
                option_type = "put"
            floating_strike = st.checkbox("Floating Strike", value=True, key="lookback_floating")
        with col2:
            S0 = st.number_input("Spot Price (S₀)", value=100.0, key="lookback_S")
            K = st.number_input("Strike Price (K)", value=100.0, key="lookback_K")
            T = st.number_input("Time to Maturity (T)", value=1.0, key="lookback_T")
            sigma = st.number_input("Volatility (σ)", value=0.2, key="lookback_sigma")
            r = st.number_input("Risk-Free Rate (r)", value=0.05, key="lookback_r")
    
        if model == "monte_carlo":
            n_paths = st.slider("Monte Carlo Simulations", 10, 10000, step=10, value=1000, key="lookback_paths")
            n_steps = st.slider("Steps per Path", 10, 300, step=2, value=252, key="lookback_steps")
    
        if st.button("Compute Lookback Option Price"):
            st.markdown("<small>Wait a few seconds for the plots !</small>", unsafe_allow_html=True)
            from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution
    
            try:
                price, stderr = price_lookback_option(
                    S0=S0, K=K if not floating_strike else None, r=r, sigma=sigma, T=T,
                    model=model, option_type=option_type,
                    floating_strike=floating_strike,
                    n_paths=n_paths if model == "monte_carlo" else None,
                    n_steps=n_steps if model == "monte_carlo" else None
                )
    
                if stderr is not None:
                    st.success(f"Monte Carlo Price: **{price:.4f}**")
    
                with st.expander("Payoff Function"):
                    st.pyplot(plot_payoff(S0, option_type, K, floating_strike))
    
                with st.expander("Simulated Asset Paths"):
                    st.pyplot(plot_paths(S0, r, sigma, T, n_paths, n_steps))
    
                if model == "monte_carlo":
                    with st.expander("Distribution of Discounted Payoffs"):
                        st.pyplot(plot_price_distribution(S0, r, sigma, T, option_type, floating_strike, n_paths, n_steps))
    
            except Exception as e:
                st.error(f"Error: {e}")





# -----------------------------
# Tab 5 – Swaps
# -----------------------------

from pricing.swaps import price_swap
from pricing.models.swaps.ois_fx import (
    build_flat_discount_curve,
    build_flat_fx_forward_curve
)

with tab5:
    st.header("Swap Pricer (In Progress)")

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
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime, date

    from pricing.models.interest_rates.analytical_vasicek import *
    from pricing.models.interest_rates.monte_carlo_vasicek import *
    from pricing.utils.greeks_vasicek import *

    # Add model selection info box
    st.markdown("### Interest Rate Model Selector")
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


