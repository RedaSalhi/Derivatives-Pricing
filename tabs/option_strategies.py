"""
tabs.option_strategies1.py
"""

import streamlit as st
import numpy as np
from styles.app_styles import load_theme
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math

# Import pricing modules
from pricing.option_strategies import (
    price_vanilla_option,
    price_option_strategy, 
    get_predefined_strategy,
    analyze_strategy_comprehensive,
    get_strategy_recommendations
)
from pricing.utils.option_strategies_utils import (
    calculate_greeks,
    calculate_strategy_greeks_range as calculate_strategy_greeks,
    compute_strategy_payoff,
    find_breakeven_points,
    validate_strategy_legs,
    calculate_strategy_metrics
)

def option_strategies_tab():
    """Option Strategies Tab Content"""

    load_theme()
    
    
    # Main title
    st.markdown('<div class="main-header">Options Strategy</div>', unsafe_allow_html=True)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Setup & Parameters",
        "Single Option Pricing", 
        "Strategy Builder", 
        "Payoff Analysis", 
        "Greeks Analysis"
    ])
    
    with tab1:
        _setup_tab()
    
    # Extract parameters for other tabs
    if st.session_state.setup_completed:
        params = st.session_state.global_params
        
        with tab2:
            _single_option_tab(params)
        
        with tab3:
            _strategy_builder_tab(params)
        
        with tab4:
            _payoff_analysis_tab(params)
        
        with tab5:
            _greeks_analysis_tab(params)
    
    else:
        # Show warning for incomplete setup in other tabs
        for tab in [tab2, tab3, tab4, tab5]:
            with tab:
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Setup Required</h4>
                    <p>Please complete the setup in the <strong>Setup & Parameters</strong> tab first!</p>
                    <p><strong>Next Step:</strong> Go to Setup & Parameters → Configure your parameters</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; 
                border: 1px solid #dee2e6;'>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2rem;"></span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">Options Strategy</p>
        <p style="margin: 8px 0; color: #6c757d;">Built with Streamlit & Python</p>
        <p style="margin: 0; color: #dc3545; font-weight: bold;">For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


def _setup_tab():
    """Setup and Parameters Tab"""
    st.markdown('<div class="sub-header">Welcome to the Options Strategy Suite!</div>', unsafe_allow_html=True)
    
    # Welcome message
    if not st.session_state.setup_completed:
        st.markdown("""
        <div class="setup-incomplete">
            <h2 style="color: #1f77b4; margin-top: 0px;">Quick Start Guide</h2>
            <p style="font-size: 1.1em; margin-bottom: 0;">
                Configure your market parameters below to unlock advanced options analysis tools!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="setup-complete">
            <h3 style="color: #1f77b4; margin-top: 0px;">Setup Complete!</h3>
            <p style="margin-bottom: 0;">All analysis tools are now available. You can modify parameters anytime.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature overview
    st.markdown("""
    <div class="info-box">
        <h3>Suite Capabilities</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
            <div>
                <h4 style="color: #1f77b4; margin-bottom: 10px;">Pricing & Valuation</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Single option pricing with multiple models</li>
                    <li>Multi-leg strategy construction</li>
                    <li>Real-time parameter sensitivity</li>
                    <li>Model comparison analysis</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1f77b4; margin-bottom: 10px;">Risk Management</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Interactive payoff diagrams</li>
                    <li>Greeks analysis and visualization</li>
                    <li>Multi-parameter sensitivity heatmaps</li>
                    <li>Breakeven and scenario analysis</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Global Parameters Section
    st.markdown('<div class="sub-header">Global Market Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #1f77b4; margin-top: 0;">Asset & Market Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #1f77b4; margin-top: 0;">Option Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
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
            "black-scholes": {
                "title": "Black-Scholes Model",
                "description": "Analytical solution for European options providing instant, precise results.",
                "pros": ["Fastest computation", "Exact analytical solution", "Widely understood"],
                "cons": ["European exercise only", "Constant volatility assumption"]
            },
            "binomial": {
                "title": "Binomial Tree Model", 
                "description": "Discrete time model supporting American options with flexible implementation.",
                "pros": ["American exercise support", "Flexible implementation", "Intuitive approach"],
                "cons": ["Slower computation", "Convergence dependent on steps"]
            },
            "monte-carlo": {
                "title": "Monte Carlo Simulation",
                "description": "Simulation-based approach handling complex payoffs and exotic features.",
                "pros": ["Handles exotic payoffs", "Statistical confidence", "Path-dependent options"],
                "cons": ["Slowest method", "Random error component"]
            }
        }
        
        current_model = model_descriptions[model]
        st.markdown(f"""
        <div class="info-box">
            <h4>{current_model['title']}</h4>
            <p><strong>Description:</strong> {current_model['description']}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                <div>
                    <h5 style="color: #28a745;">✅ Advantages:</h5>
                    <ul style="margin: 0; padding-left: 20px;">
                        {''.join([f'<li>{pro}</li>' for pro in current_model['pros']])}
                    </ul>
                </div>
                <div>
                    <h5 style="color: #dc3545;">⚠️ Limitations:</h5>
                    <ul style="margin: 0; padding-left: 20px;">
                        {''.join([f'<li>{con}</li>' for con in current_model['cons']])}
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
            
            # Performance guidance
            if n_steps <= 50:
                step_performance = "🟢 Fast computation"
            elif n_steps <= 200:
                step_performance = "🟡 Moderate computation time"
            else:
                step_performance = "🔴 Slow computation"
                
            st.markdown(f"""
            <div class="parameter-grid">
                <h5>Performance Impact</h5>
                <p><strong>Current setting:</strong> {step_performance}</p>
                <p><strong>Recommended:</strong> 100-200 steps for good accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
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
            
            # Performance guidance
            if n_simulations <= 1000:
                sim_performance = "🟢 Fast computation"
            elif n_simulations <= 10000:
                sim_performance = "🟡 Moderate computation time"
            else:
                sim_performance = "🔴 Slow computation"
                
            st.markdown(f"""
            <div class="parameter-grid">
                <h5>Performance Impact</h5>
                <p><strong>Current setting:</strong> {sim_performance}</p>
                <p><strong>Recommended:</strong> 10,000+ simulations for accuracy</p>
                <p><strong>Error estimate:</strong> ±{1.96/np.sqrt(n_simulations)*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            n_steps = st.session_state.global_params['n_steps']
            n_simulations = st.session_state.global_params['n_simulations']
            st.markdown("""
            <div class="success-box">
                <h5>Black-Scholes Performance</h5>
                <p>Instantaneous computation with exact analytical results!</p>
                <p>Perfect for rapid analysis and parameter sweeps.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Setup completion button
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    
    with col_button2:
        setup_button_text = "Update Configuration" if st.session_state.setup_completed else "Complete Setup & Start Analysis"
        
        if st.button(setup_button_text, type="primary", use_container_width=True):
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
            st.success("✅ Configuration updated successfully! All analysis tools are now available.")
            if not st.session_state.get('setup_completed_before', False):
                st.balloons()
                st.session_state.setup_completed_before = True
    
    if not st.session_state.setup_completed:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Setup Required</h4>
            <p>Please complete the setup above to unlock all advanced analysis features!</p>
            <p><strong>Next Steps:</strong> Configure your parameters and click the setup button.</p>
        </div>
        """, unsafe_allow_html=True)


def _single_option_tab(params):
    """Single Option Pricing Tab"""
    st.markdown('<div class="sub-header">Single Option Pricing & Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #1f77b4; margin-top: 0;">Option Specification</h4>
        </div>
        """, unsafe_allow_html=True)
        
        option_type = st.selectbox("Option Type", ["call", "put"], help="Call = Right to buy, Put = Right to sell")
        exercise_style = st.selectbox("Exercise Style", ["european", "american"], help="European = Exercise only at expiry")
        strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=0.1, key="single_strike")
        
        # Calculate option price
        try:
            option_price = price_vanilla_option(
                option_type=option_type,
                exercise_style=exercise_style,
                model=params['model'],
                S=params['spot_price'],
                K=strike_price,
                T=params['time_to_expiry'],
                r=params['risk_free_rate'],
                sigma=params['volatility'],
                q=params['dividend_yield'],
                N=params.get('n_steps', 100),
                n_simulations=params.get('n_simulations', 10000)
            )
            
            # Enhanced price display
            intrinsic_value = max(params['spot_price'] - strike_price, 0) if option_type == "call" else max(strike_price - params['spot_price'], 0)
            time_value = option_price - intrinsic_value
            
            st.markdown(f"""
            <div class="success-box">
                <h4 style="margin-top: 0;">Option Valuation</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #28a745;">
                        <td style="padding: 8px; font-weight: bold;">Total Price</td>
                        <td style="padding: 8px; font-family: monospace; color: #2E8B57; font-size: 1.2em; font-weight: bold;">${option_price:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 8px;">Intrinsic Value</td>
                        <td style="padding: 8px; font-family: monospace;">${intrinsic_value:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;">Time Value</td>
                        <td style="padding: 8px; font-family: monospace;">${time_value:.4f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Pricing Error</h4>
                <p>Error calculating option price: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="sub-header">Parameter Sensitivity Analysis</div>', unsafe_allow_html=True)
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            param_to_vary = st.selectbox("Parameter to Vary", ["S", "K", "T", "r", "sigma", "q"])
        
        with param_col2:
            n_points = st.slider("Analysis Points", 20, 100, 50)
        
        # Sensitivity analysis
        current_val = {
            "S": params['spot_price'], "K": strike_price, "T": params['time_to_expiry'],
            "r": params['risk_free_rate'], "sigma": params['volatility'], "q": params['dividend_yield']
        }[param_to_vary]
        
        range_col1, range_col2 = st.columns(2)
        
        with range_col1:
            param_min = st.number_input(f"Min {param_to_vary}", value=current_val * 0.5, step=0.01)
        with range_col2:
            param_max = st.number_input(f"Max {param_to_vary}", value=current_val * 1.5, step=0.01)
        
        if st.button("Generate Sensitivity Analysis", type="primary", use_container_width=True):
            try:
                with st.spinner("Calculating parameter sensitivity..."):
                    fixed_params = {
                        "S": params['spot_price'], "K": strike_price, "T": params['time_to_expiry'],
                        "r": params['risk_free_rate'], "sigma": params['volatility'], "q": params['dividend_yield']
                    }
                    if params['model'] == "binomial":
                        fixed_params["N"] = params['n_steps']
                    elif params['model'] == "monte-carlo":
                        fixed_params["n_simulations"] = params['n_simulations']
                    
                    # Create parameter range
                    param_values = np.linspace(param_min, param_max, n_points)
                    option_prices = []
                    
                    for param_val in param_values:
                        temp_params = fixed_params.copy()
                        temp_params[param_to_vary] = param_val
                        
                        try:
                            price = price_vanilla_option(
                                option_type=option_type,
                                exercise_style=exercise_style,
                                model=params['model'],
                                **temp_params
                            )
                            option_prices.append(price)
                        except:
                            option_prices.append(np.nan)
                    
                    # Create plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=param_values,
                        y=option_prices,
                        mode='lines',
                        name=f'Option Price vs {param_to_vary}',
                        line=dict(color='#1f77b4', width=3)
                    ))
                    
                    # Mark current value
                    fig.add_vline(
                        x=current_val,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Current {param_to_vary}: {current_val:.4f}"
                    )
                    
                    fig.update_layout(
                        title=f'{exercise_style.title()} {option_type.title()} Price Sensitivity to {param_to_vary}',
                        xaxis_title=param_to_vary,
                        yaxis_title='Option Price ($)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>Analysis Error</h4>
                    <p>Error generating sensitivity plot: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)


def _strategy_builder_tab(params):
    """Strategy Builder Tab"""
    st.markdown('<div class="sub-header">Advanced Strategy Builder</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #1f77b4; margin-top: 0;">Strategy Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        strategy_method = st.radio("Choose Construction Method", ["Predefined Strategy", "Custom Strategy"])
        
        if strategy_method == "Predefined Strategy":
            strategy_name = st.selectbox(
                "Select Strategy Template",
                ["straddle", "bull call spread", "bear put spread", "butterfly", "iron condor"]
            )
            
            # Dynamic strike inputs based on strategy
            if strategy_name == "straddle":
                strike1 = st.number_input("Strike Price", value=params['spot_price'], key="pred_k1")
                legs = get_predefined_strategy(strategy_name, strike1)
            elif strategy_name in ["bull call spread", "bear put spread"]:
                strike1 = st.number_input("Lower Strike", value=params['spot_price']-5, key="pred_k1")
                strike2 = st.number_input("Higher Strike", value=params['spot_price']+5, key="pred_k2")
                legs = get_predefined_strategy(strategy_name, strike1, strike2)
            elif strategy_name == "butterfly":
                strike1 = st.number_input("Lower Strike", value=params['spot_price']-10, key="pred_k1")
                strike2 = st.number_input("Middle Strike", value=params['spot_price'], key="pred_k2")
                strike3 = st.number_input("Upper Strike", value=params['spot_price']+10, key="pred_k3")
                legs = get_predefined_strategy(strategy_name, strike1, strike2, strike3)
            elif strategy_name == "iron condor":
                strike1 = st.number_input("Put Long Strike", value=params['spot_price']-15, key="pred_k1")
                strike2 = st.number_input("Put Short Strike", value=params['spot_price']-5, key="pred_k2")
                strike3 = st.number_input("Call Short Strike", value=params['spot_price']+5, key="pred_k3")
                strike4 = st.number_input("Call Long Strike", value=params['spot_price']+15, key="pred_k4")
                legs = get_predefined_strategy(strategy_name, strike1, strike2, strike3, strike4)
                
        else:  # Custom Strategy
            st.markdown("#### Custom Strategy Builder")
            num_legs = st.number_input("Number of Legs", value=2, min_value=1, max_value=10, step=1, key="custom_num_legs")
            
            legs = []
            for i in range(num_legs):
                st.markdown(f"""
                <div class="strategy-leg">
                    <h5 style="margin-top: 0; color: #17a2b8;">Leg {i+1}</h5>
                </div>
                """, unsafe_allow_html=True)
                
                col_type, col_strike, col_qty = st.columns(3)
                with col_type:
                    leg_type = st.selectbox(f"Type", ["call", "put"], key=f"leg_type_{i}")
                with col_strike:
                    leg_strike = st.number_input(f"Strike", value=params['spot_price'], key=f"leg_strike_{i}")
                with col_qty:
                    leg_qty = st.number_input(f"Quantity", value=1.0, step=0.1, key=f"leg_qty_{i}")
                
                legs.append({"type": leg_type, "strike": leg_strike, "qty": leg_qty})
        
        # Exercise style for strategy
        strategy_exercise = st.selectbox("Exercise Style", ["european", "american"], key="strategy_exercise")
        
        # Store legs in session state
        st.session_state.current_legs = legs
        st.session_state.current_strategy_exercise = strategy_exercise
    
    with col2:
        st.markdown('<div class="sub-header">Strategy Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(legs, str):  # Error message from predefined strategy
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Strategy Configuration Error</h4>
                <p>{legs}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Strategy display
            strategy_df = pd.DataFrame(legs)
            strategy_df['Position'] = strategy_df['qty'].apply(lambda x: f"{'Long' if x > 0 else 'Short'} {abs(x)}")
            strategy_df['Option'] = strategy_df.apply(lambda row: f"{row['type'].title()} @ ${row['strike']:.2f}", axis=1)
            
            display_df = strategy_df[['Position', 'Option']].copy()
            display_df.index = [f"Leg {i+1}" for i in range(len(display_df))]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Price the strategy
            try:
                with st.spinner("Pricing strategy..."):
                    strategy_kwargs = {
                        'S': params['spot_price'],
                        'T': params['time_to_expiry'],
                        'r': params['risk_free_rate'],
                        'sigma': params['volatility'],
                        'q': params['dividend_yield']
                    }
                    
                    if params['model'] == "binomial":
                        strategy_kwargs["N"] = params['n_steps']
                    elif params['model'] == "monte-carlo":
                        strategy_kwargs["n_simulations"] = params['n_simulations']
                    
                    strategy_result = price_option_strategy(
                        legs=legs,
                        exercise_style=strategy_exercise,
                        model=params['model'],
                        **strategy_kwargs
                    )
                
                # Store result in session state
                st.session_state.strategy_result = strategy_result
                
                # Results display
                net_premium = strategy_result['strategy_price']
                strategy_type = "Credit" if net_premium < 0 else "Debit"
                strategy_color = "#28a745" if net_premium < 0 else "#dc3545"
                
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0;">Strategy Valuation</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 2px solid #28a745;">
                            <td style="padding: 12px; font-weight: bold;">Metric</td>
                            <td style="padding: 12px; font-weight: bold;">Value</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 10px; font-weight: bold;">Strategy Price</td>
                            <td style="padding: 10px; font-family: monospace; color: {strategy_color}; font-size: 1.2em; font-weight: bold;">${strategy_result['strategy_price']:.4f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 10px; font-weight: bold;">Strategy Type</td>
                            <td style="padding: 10px; font-weight: bold; color: {strategy_color};">{strategy_type} Strategy</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; font-weight: bold;">Net Premium</td>
                            <td style="padding: 10px; font-family: monospace; color: {strategy_color};">{"Received" if net_premium < 0 else "Paid"}: ${abs(net_premium):.4f}</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Strategy Pricing Error</h4>
                    <p>Error pricing strategy: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)


def _payoff_analysis_tab(params):
    """Payoff Analysis Tab"""
    st.markdown('<div class="sub-header">Strategy Payoff Analysis</div>', unsafe_allow_html=True)
    
    if 'current_legs' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ No Strategy Configured</h4>
            <p>Please build a strategy in the <strong>Strategy Builder</strong> tab first!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    legs = st.session_state.current_legs
    
    # Payoff controls
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("#### Price Range")
        strikes = [leg['strike'] for leg in legs]
        min_strike, max_strike = min(strikes), max(strikes)
        
        spot_min = st.number_input("Min Spot Price", value=min_strike * 0.7, step=1.0)
        spot_max = st.number_input("Max Spot Price", value=max_strike * 1.3, step=1.0)
    
    with config_col2:
        st.markdown("#### Analysis Options")
        n_points = st.slider("Resolution", 50, 500, 200)
        show_breakeven = st.checkbox("Show Breakeven Points", value=True)
    
    with config_col3:
        st.markdown("#### P&L Options")
        show_profit_loss = st.checkbox("Include Premium Cost", value=True)
        show_individual_legs = st.checkbox("Show Individual Legs", value=False)
    
    # Calculate and display payoff
    spot_range = np.linspace(spot_min, spot_max, n_points)
    
    try:
        payoffs = compute_strategy_payoff(legs, spot_range)
        
        # Create plotly chart
        fig = go.Figure()
        
        # Main payoff line
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=payoffs,
            mode='lines',
            name='Strategy Payoff at Expiration',
            line=dict(color='#1f77b4', width=4)
        ))
        
        # Add profit/loss line if premium is included
        if show_profit_loss and 'strategy_result' in st.session_state:
            pnl = payoffs - st.session_state.strategy_result['strategy_price']
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=pnl,
                mode='lines',
                name='Net P&L (including premium)',
                line=dict(color='#ff7f0e', width=3, dash='dash')
            ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
        
        # Add current spot line
        fig.add_vline(
            x=params['spot_price'],
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Current Spot: ${params['spot_price']:.2f}"
        )
        
        fig.update_layout(
            title="Strategy Payoff Diagram",
            xaxis_title="Spot Price at Expiration ($)",
            yaxis_title="Payoff ($)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        max_payoff = np.max(payoffs)
        min_payoff = np.min(payoffs)
        
        st.markdown(f"""
        <div class="payoff-stats">
            <h4>Strategy Statistics</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #6c757d;">
                    <td style="padding: 10px; font-weight: bold;">Max Payoff</td>
                    <td style="padding: 10px; font-family: monospace; color: #28a745;">${max_payoff:.2f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Min Payoff</td>
                    <td style="padding: 10px; font-family: monospace; color: #dc3545;">${min_payoff:.2f}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Payoff Calculation Error</h4>
            <p>Error calculating strategy payoff: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)


def _greeks_analysis_tab(params):
    """Greeks Analysis Tab"""
    st.markdown('<div class="sub-header">Greeks Risk Analysis</div>', unsafe_allow_html=True)
    
    if 'current_legs' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ No Strategy Configured</h4>
            <p>Please build a strategy in the <strong>Strategy Builder</strong> tab first!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    legs = st.session_state.current_legs
    
    # Greeks configuration
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        greek_name = st.selectbox("Select Greek to Analyze", ["delta", "gamma", "vega", "theta", "rho"])
        
        # Greek descriptions
        greek_descriptions = {
            "delta": "Delta: Price sensitivity to underlying asset moves",
            "gamma": "Gamma: Rate of change of delta",
            "vega": "Vega: Sensitivity to volatility changes",
            "theta": "Theta: Time decay",
            "rho": "Rho: Interest rate sensitivity"
        }
        
        st.markdown(f"""
        <div class="info-box">
            {greek_descriptions[greek_name]}
        </div>
        """, unsafe_allow_html=True)
    
    with config_col2:
        # Spot range for Greeks
        strikes = [leg['strike'] for leg in legs]
        min_strike, max_strike = min(strikes), max(strikes)
        
        greek_spot_min = st.number_input("Min Spot for Analysis", value=min_strike * 0.8, step=1.0)
        greek_spot_max = st.number_input("Max Spot for Analysis", value=max_strike * 1.2, step=1.0)
    
    with config_col3:
        greek_points = st.slider("Analysis Resolution", 50, 300, 150)
        run_greeks_analysis = st.button("Run Greeks Analysis", type="primary")
    
    if run_greeks_analysis:
        try:
            with st.spinner(f"Calculating {greek_name} analysis..."):
                # Generate spot range
                spot_range = np.linspace(greek_spot_min, greek_spot_max, greek_points)
                
                # Calculate strategy Greeks
                strategy_greeks = calculate_strategy_greeks(
                    legs, spot_range, greek_name, 
                    params['time_to_expiry'], params['risk_free_rate'], 
                    params['volatility'], params['dividend_yield']
                )
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=spot_range,
                    y=strategy_greeks,
                    mode='lines',
                    name=f'Strategy {greek_name.title()}',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                # Add current spot marker
                current_greek = np.interp(params['spot_price'], spot_range, strategy_greeks)
                fig.add_scatter(
                    x=[params['spot_price']],
                    y=[current_greek],
                    mode='markers',
                    name='Current Position',
                    marker=dict(color='red', size=12, symbol='diamond')
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
                
                fig.update_layout(
                    title=f'Strategy {greek_name.title()} vs Spot Price',
                    xaxis_title='Spot Price ($)',
                    yaxis_title=f'{greek_name.title()}',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display current Greek value
                st.markdown(f"""
                <div class="success-box">
                    <h4>Current {greek_name.title()} Analysis</h4>
                    <p><strong>Current {greek_name.title()} value:</strong> {current_greek:.6f}</p>
                    <p><strong>Spot Price:</strong> ${params['spot_price']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Analysis Error</h4>
                <p>Error calculating Greeks: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)


def display_educational_content():
    """Display comprehensive educational content"""

    load_theme()
    
    
    
    st.markdown("---")
    st.markdown('<div class="sub-header">Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("Options Strategy Fundamentals"):
        st.markdown("### Core Strategy Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Directional Strategies")
            st.markdown("""
            - **Long Call/Put:** Unlimited profit potential, limited risk
            - **Bull/Bear Spreads:** Limited risk and reward
            - **Synthetic Positions:** Replicate stock with options
            - **Ratio Spreads:** Unequal long/short positions
            """)
            
            st.markdown("#### Neutral Strategies")
            st.markdown("""
            - **Straddles/Strangles:** Profit from volatility
            - **Iron Condors:** Range-bound profit
            - **Butterflies:** Minimal movement profit
            - **Calendar Spreads:** Time decay profit
            """)
        
        with col2:
            st.markdown("#### Volatility Strategies")
            st.markdown("""
            - **Long Volatility:** Buy straddles/strangles
            - **Short Volatility:** Sell premium strategies
            - **Volatility Arbitrage:** Implied vs realized
            - **Dispersion Trading:** Index vs stocks
            """)
            
            st.markdown("#### Income Strategies")
            st.markdown("""
            - **Covered Calls:** Generate income on holdings
            - **Cash-Secured Puts:** Income while acquiring
            - **Iron Butterflies:** High probability income
            - **Short Strangles:** Range-bound income
            """)
    
    with st.expander("Understanding the Greeks"):
        st.markdown("### The Greeks - Risk Sensitivities Explained")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### First-Order Greeks")
            st.markdown("""
            - **Delta (Δ):** Price sensitivity to underlying moves
            - **Vega (ν):** Sensitivity to volatility changes  
            - **Theta (Θ):** Sensitivity to time (usually negative)
            - **Rho (ρ):** Interest rate sensitivity
            """)
            
            st.markdown("#### Greek Ranges")
            st.markdown("""
            - **Delta:** -1.0 to +1.0 (calls: 0 to 1, puts: -1 to 0)
            - **Gamma:** 0 to ∞ (highest for ATM options)
            - **Theta:** Usually negative (time decay)
            - **Vega:** Always positive (vol increases price)
            """)
        
        with col2:
            st.markdown("#### Second-Order Greeks")
            st.markdown("""
            - **Gamma (Γ):** Rate of change of delta
            - **Vomma:** Rate of change of vega
            - **Charm:** Rate of change of delta over time
            - **Color:** Rate of change of gamma over time
            """)
    
    with st.expander("Mathematical Formulas & Models"):
        st.markdown("### Core Mathematical Foundations")
        
        st.markdown("#### Black-Scholes Formula")
        st.latex(r'''
        \text{Call: } C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
        ''')
        st.latex(r'''
        \text{Put: } P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)
        ''')
        
        st.markdown("**Where:**")
        st.latex(r'''
        d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}
        ''')
        st.latex(r'''
        d_2 = d_1 - \sigma\sqrt{T}
        ''')
        
        st.markdown("#### The Greeks Formulas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**Delta (Δ):**")
            st.latex(r'''
            \text{Call: } \Delta = e^{-qT} N(d_1)
            ''')
            st.latex(r'''
            \text{Put: } \Delta = e^{-qT} [N(d_1) - 1]
            ''')
        
        with col2:
            st.error("**Gamma (Γ):**")
            st.latex(r'''
            \Gamma = \frac{e^{-qT} n(d_1)}{S_0 \sigma \sqrt{T}}
            ''')
            st.markdown("*(Same for calls and puts)*")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.success("**Vega (ν):**")
            st.latex(r'''
            \nu = S_0 e^{-qT} n(d_1) \sqrt{T}
            ''')
            st.markdown("*(Same for calls and puts)*")
        
        with col4:
            st.error("**Theta (Θ):**")
            st.markdown("Complex formula involving time decay components")
    
    with st.expander("Strategy Quick Reference"):
        st.markdown("### Popular Strategy Combinations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Bullish Strategies")
            st.markdown("""
            - Long Call
            - Bull Call Spread
            - Bull Put Spread
            - Covered Call
            - Protective Put
            """)
        
        with col2:
            st.markdown("#### Bearish Strategies")
            st.markdown("""
            - Long Put
            - Bear Put Spread
            - Bear Call Spread
            - Protective Call
            - Collar (Protective)
            """)
        
        with col3:
            st.markdown("#### Neutral Strategies")
            st.markdown("""
            - Long/Short Straddle
            - Long/Short Strangle
            - Iron Condor
            - Butterfly Spread
            - Calendar Spread
            """)
        
        st.success("**Strategy Selection Tips**")
        st.markdown("""
        - **High Volatility Expected:** Buy straddles or strangles
        - **Low Volatility Expected:** Sell premium (iron condors, butterflies)
        - **Moderate Directional View:** Use spreads to reduce cost
        - **Strong Directional View:** Long options or synthetic positions
        - **Income Generation:** Covered calls, cash-secured puts
        """)


def main_option_strategies_tab():
    
    # Main application
    option_strategies_tab()
    
    display_educational_content()
