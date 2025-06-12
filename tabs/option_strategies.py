# tabs/option_strategies.py
# Option Strategies Tab - Tab 3

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import your pricing functions
from pricing.vanilla_options import *
from pricing.option_strategies import *
from pricing.utils.option_strategies_greeks import *


def option_strategies_tab():
    """Option Strategies Tab Content"""
    
    # Main title
    st.markdown('<div class="main-header">Options Pricing Suite</div>', unsafe_allow_html=True)
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
        _setup_tab()
    
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
        
        with taa1:
            _single_option_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations)
        
        with taa2:
            _strategy_builder_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations)
        
        with taa3:
            _payoff_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations)
        
        with taa4:
            _greeks_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations)
        
        with taa5:
            _sensitivity_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>Advanced Options Pricing Suite | Built with Streamlit & Python</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


def _setup_tab():
    """Setup and Parameters Tab"""
    st.markdown('<div class="sub-header">Welcome to the Options Pricing Suite!</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="sub-header">Global Market Parameters</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="sub-header">Pricing Model Configuration</div>', unsafe_allow_html=True)
    
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
            "black-scholes": "**Black-Scholes**: Analytical solution for European options. Fast and precise.",
            "binomial": "**Binomial Tree**: Discrete model supporting American options. Flexible but slower.",
            "monte-carlo": "**Monte Carlo**: Simulation-based approach. Handles complex payoffs."
        }
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(model_descriptions[model])
        st.markdown('</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="sub-header">Parameter Summary</div>', unsafe_allow_html=True)
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Spot Price", f"${spot_price:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Strike Price Range", f"${spot_price*0.8:.0f} - ${spot_price*1.2:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with summary_col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Volatility", f"{volatility*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Risk-free Rate", f"{risk_free_rate*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with summary_col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Time to Expiry", f"{time_to_expiry:.3f} years")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Model", model.title())
        st.markdown('</div>', unsafe_allow_html=True)
    
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
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please complete the setup above to unlock all analysis features!")
        st.markdown('</div>', unsafe_allow_html=True)


def _single_option_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Single Option Pricing Tab"""
    st.markdown('<div class="sub-header">Single Option Pricing</div>', unsafe_allow_html=True)
    
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
                
                # Note: This assumes plot_option_price_vs_param exists in your pricing module
                # If not, you'll need to implement this function or create a similar plot here
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


def _strategy_builder_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Strategy Builder Tab"""
    st.markdown('<div class="sub-header">Option Strategy Builder</div>', unsafe_allow_html=True)
    
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
        
        # Store legs in session state for use in other tabs
        st.session_state.current_legs = legs
        st.session_state.current_strategy_exercise = strategy_exercise
    
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
                
                # Store result in session state
                st.session_state.strategy_result = strategy_result
                
                col_price1, col_price2 = st.columns(2)
                with col_price1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Total Strategy Price", f"${strategy_result['strategy_price']:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col_price2:
                    net_premium = strategy_result['strategy_price']
                    strategy_type = "Credit" if net_premium < 0 else "Debit"
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Strategy Type", strategy_type, f"${abs(net_premium):.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Individual leg prices
                st.subheader("Individual Leg Prices")
                for i, (leg, price) in enumerate(zip(legs, strategy_result['individual_prices'])):
                    position = "Long" if leg['qty'] > 0 else "Short"
                    st.write(f"**Leg {i+1}:** {position} {abs(leg['qty'])} {leg['type'].title()} @ {leg['strike']} = ${price:.4f}")
                
            except Exception as e:
                st.error(f"Error pricing strategy: {str(e)}")


def _payoff_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Payoff Analysis Tab"""
    st.markdown('<div class="sub-header">Strategy Payoff Analysis</div>', unsafe_allow_html=True)
    
    if 'current_legs' not in st.session_state:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please build a strategy in the Strategy Builder tab first!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    legs = st.session_state.current_legs
    
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
        if show_profit_loss and 'strategy_result' in st.session_state:
            pnl = payoffs - st.session_state.strategy_result['strategy_price']
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
        if show_breakeven and show_profit_loss and 'strategy_result' in st.session_state:
            pnl = payoffs - st.session_state.strategy_result['strategy_price']
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
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Max Payoff", f"${np.max(payoffs):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_stat2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Min Payoff", f"${np.min(payoffs):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_stat3:
            if show_profit_loss and 'strategy_result' in st.session_state:
                max_profit = np.max(payoffs - st.session_state.strategy_result['strategy_price'])
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Max Profit", f"${max_profit:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Max Profit", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col_stat4:
            if show_profit_loss and 'strategy_result' in st.session_state:
                max_loss = np.min(payoffs - st.session_state.strategy_result['strategy_price'])
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Max Loss", f"${max_loss:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Max Loss", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)


def _greeks_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Greeks Analysis Tab"""
    st.markdown('<div class="sub-header">Greeks Analysis</div>', unsafe_allow_html=True)
    
    if 'current_legs' not in st.session_state:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please build a strategy in the Strategy Builder tab first!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    legs = st.session_state.current_legs
    
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
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"**{greek_name.title()}**: {greek_explanations[greek_name]}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error calculating Greeks: {str(e)}")
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("Greeks analysis requires the Greeks computation module to be properly implemented.")
            st.markdown('</div>', unsafe_allow_html=True)


def _sensitivity_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Advanced Sensitivity Analysis Tab"""
    st.markdown('<div class="sub-header">Advanced Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    if 'current_legs' not in st.session_state:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Please build a strategy in the Strategy Builder tab first!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    legs = st.session_state.current_legs
    strategy_exercise = st.session_state.get('current_strategy_exercise', 'european')
    
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
    
    # Additional analysis sections
    st.markdown("---")
    st.markdown('<div class="sub-header">Risk Metrics & Insights</div>', unsafe_allow_html=True)
    
    # Risk analysis
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Strategy Risk Profile
        
        **Key Risk Factors:**
        - **Delta Risk**: Directional exposure to underlying price moves
        - **Gamma Risk**: Acceleration of delta changes
        - **Theta Decay**: Time value erosion
        - **Vega Risk**: Volatility sensitivity
        - **Rho Risk**: Interest rate sensitivity
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_risk2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Monitoring Guidelines
        
        **Regular Checks:**
        - **P&L Attribution**: Which Greek is driving returns?
        - **Greeks Limits**: Are exposures within acceptable ranges?
        - **Scenario Analysis**: How does strategy perform in stress scenarios?
        - **Hedging Needs**: When to adjust positions?
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational content
    with st.expander("📚 Advanced Strategy Concepts"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Strategy Types & Characteristics
        
        **Directional Strategies:**
        - **Long Call/Put**: Unlimited upside (calls) or downside (puts) protection
        - **Bull/Bear Spreads**: Limited risk, limited reward
        
        **Neutral Strategies:**
        - **Straddles**: Profit from large moves in either direction
        - **Strangles**: Similar to straddles but with different strikes
        - **Iron Condors**: Profit from low volatility
        
        **Volatility Strategies:**
        - **Calendar Spreads**: Profit from time decay differences
        - **Butterfly Spreads**: Profit from price staying near center strike
        
        **Income Strategies:**
        - **Covered Calls**: Generate income on stock holdings
        - **Cash-Secured Puts**: Generate income while potentially acquiring stock
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("⚠️ Risk Management Best Practices"):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Professional Risk Management
        
        **Position Sizing:**
        - Never risk more than you can afford to lose
        - Size positions based on volatility and strategy complexity
        - Consider correlation between multiple strategies
        
        **Monitoring & Adjustments:**
        - Set clear profit/loss targets before entering
        - Monitor Greeks daily, especially for short positions
        - Have adjustment plans for different scenarios
        
        **Execution Considerations:**
        - Account for bid-ask spreads in real trading
        - Consider assignment risk on short options
        - Plan for early exercise on American options
        
        **Market Conditions:**
        - Strategies perform differently in different market regimes
        - Volatility can change rapidly, affecting all strategies
        - Interest rate changes affect longer-term strategies more
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Error handling and warnings
    if model == "black-scholes" and any(leg.get('qty', 1) != int(leg.get('qty', 1)) for leg in legs):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Black-Scholes assumes integer quantities")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if model == "monte-carlo" and n_simulations < 1000:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("⚠️ Low simulation count may affect accuracy")
        st.markdown('</div>', unsafe_allow_html=True)
