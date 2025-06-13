

if 'current_legs' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ No Strategy Configured</h4>
            <p>Please build a strategy in the <strong>Strategy Builder</strong> tab first!</p>
            <p><strong>Next Step:</strong> Go to Strategy Builder → Configure your options strategy</p>
        </div>
        """, unsafe_allow_html=True)
        return










# Option Strategies Tab - Tab 3

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import math

# Import your pricing functions
from pricing.vanilla_options import *
from pricing.option_strategies import *
from pricing.utils.option_strategies_greeks import *


def option_strategies_tab():
    """Option Strategies Tab Content"""
    
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
        .error-box {
            background-color: #f8d7da;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #dc3545;
            margin: 1rem 0;
        }
        .setup-incomplete {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #ffc107;
            text-align: center;
        }
        .setup-complete {
            background: linear-gradient(135deg, #d4edda 0%, #a8e6cf 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px solid #28a745;
            text-align: center;
            margin: 1rem 0;
        }
        .parameter-grid {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            margin: 10px 0;
        }
        .strategy-leg {
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 8px;
            margin: 5px 0;
            border-left: 3px solid #17a2b8;
        }
        .greek-analysis {
            background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #17a2b8;
        }
        .payoff-stats {
            background: linear-gradient(135deg, #f0f2f6 0%, #e9ecef 100%);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #6c757d;
        }
        .results-table {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown('<div class="main-header">🎯 Advanced Options Strategy Suite</div>', unsafe_allow_html=True)
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
        "⚙️ Setup & Parameters",
        "📊 Single Option Pricing", 
        "🔧 Strategy Builder", 
        "📈 Payoff Analysis", 
        "🎯 Greeks Analysis",
        "🔬 Sensitivity Analysis"
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
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 1px solid #dee2e6;'>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2rem;">🎯</span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">Advanced Options Strategy Suite</p>
        <p style="margin: 8px 0; color: #6c757d;">Built with Streamlit & Python</p>
        <p style="margin: 0; color: #dc3545; font-weight: bold;">⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


def _setup_tab():
    """Setup and Parameters Tab"""
    st.markdown('<div class="sub-header">🚀 Welcome to the Options Strategy Suite!</div>', unsafe_allow_html=True)
    
    # Enhanced welcome message
    if not st.session_state.setup_completed:
        st.markdown("""
        <div class="setup-incomplete">
            <h2 style="color: #856404; margin-top: 0;">🎯 Quick Start Guide</h2>
            <p style="font-size: 1.1em; margin-bottom: 0;">
                Configure your market parameters below to unlock advanced options analysis tools!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="setup-complete">
            <h3 style="color: #155724; margin-top: 0;">✅ Setup Complete!</h3>
            <p style="margin-bottom: 0;">All analysis tools are now available. You can modify parameters anytime.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature overview
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Suite Capabilities</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
            <div>
                <h4 style="color: #1f77b4; margin-bottom: 10px;">📊 Pricing & Valuation</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li>Single option pricing with multiple models</li>
                    <li>Multi-leg strategy construction</li>
                    <li>Real-time parameter sensitivity</li>
                    <li>Model comparison analysis</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1f77b4; margin-bottom: 10px;">📈 Risk Management</h4>
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
    st.markdown('<div class="sub-header">📊 Global Market Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #1f77b4; margin-top: 0;">💰 Asset & Market Parameters</h4>
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
            <h4 style="color: #1f77b4; margin-top: 0;">📊 Option Parameters</h4>
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
    st.markdown('<div class="sub-header">🔧 Pricing Model Configuration</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        model = st.selectbox(
            "Select Pricing Model", 
            ["black-scholes", "binomial", "monte-carlo"],
            index=["black-scholes", "binomial", "monte-carlo"].index(st.session_state.global_params['model']),
            key="setup_model",
            help="Choose the mathematical model for option pricing"
        )
        
        # Enhanced model descriptions
        model_descriptions = {
            "black-scholes": {
                "title": "🎯 Black-Scholes Model",
                "description": "Analytical solution for European options providing instant, precise results.",
                "pros": ["⚡ Fastest computation", "📐 Exact analytical solution", "🎓 Widely understood"],
                "cons": ["❌ European exercise only", "📊 Constant volatility assumption"]
            },
            "binomial": {
                "title": "🌳 Binomial Tree Model", 
                "description": "Discrete time model supporting American options with flexible implementation.",
                "pros": ["🇺🇸 American exercise support", "🔧 Flexible implementation", "📈 Intuitive approach"],
                "cons": ["⏱️ Slower computation", "🔢 Convergence dependent on steps"]
            },
            "monte-carlo": {
                "title": "🎲 Monte Carlo Simulation",
                "description": "Simulation-based approach handling complex payoffs and exotic features.",
                "pros": ["🌟 Handles exotic payoffs", "📊 Statistical confidence", "🔄 Path-dependent options"],
                "cons": ["⌛ Slowest method", "📈 Random error component"]
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
                <h5>⚡ Performance Impact</h5>
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
                <h5>⚡ Performance Impact</h5>
                <p><strong>Current setting:</strong> {sim_performance}</p>
                <p><strong>Recommended:</strong> 10,000+ simulations for accuracy</p>
                <p><strong>Error estimate:</strong> ±{1.96/np.sqrt(n_simulations)*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <h5>⚡ Black-Scholes Performance</h5>
                <p>Instantaneous computation with exact analytical results!</p>
                <p>Perfect for rapid analysis and parameter sweeps.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Parameter Summary with enhanced visualization
    st.markdown('<div class="sub-header">📋 Configuration Summary</div>', unsafe_allow_html=True)
    
    # Create enhanced summary table
    st.markdown(f"""
    <div class="parameter-grid">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                <td style="padding: 12px; font-weight: bold;">Parameter</td>
                <td style="padding: 12px; font-weight: bold;">Value</td>
                <td style="padding: 12px; font-weight: bold;">Impact</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Spot Price (S)</td>
                <td style="padding: 10px; font-family: monospace; color: #2E8B57;">${spot_price:.2f}</td>
                <td style="padding: 10px; font-style: italic;">Reference price for moneyness</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Volatility (σ)</td>
                <td style="padding: 10px; font-family: monospace; color: #2E8B57;">{volatility*100:.1f}%</td>
                <td style="padding: 10px; font-style: italic;">{"High volatility" if volatility > 0.3 else "Moderate volatility" if volatility > 0.15 else "Low volatility"}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Risk-free Rate (r)</td>
                <td style="padding: 10px; font-family: monospace; color: #2E8B57;">{risk_free_rate*100:.2f}%</td>
                <td style="padding: 10px; font-style: italic;">{"High rate environment" if risk_free_rate > 0.05 else "Low rate environment"}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Time to Expiry (T)</td>
                <td style="padding: 10px; font-family: monospace; color: #2E8B57;">{time_to_expiry:.3f} years ({time_to_expiry*365:.0f} days)</td>
                <td style="padding: 10px; font-style: italic;">{"Long-term" if time_to_expiry > 1 else "Medium-term" if time_to_expiry > 0.25 else "Short-term"}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Dividend Yield (q)</td>
                <td style="padding: 10px; font-family: monospace; color: #2E8B57;">{dividend_yield*100:.2f}%</td>
                <td style="padding: 10px; font-style: italic;">{"High dividend" if dividend_yield > 0.03 else "Low/No dividend"}</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Pricing Model</td>
                <td style="padding: 10px; font-family: monospace; color: #2E8B57;">{model.title()}</td>
                <td style="padding: 10px; font-style: italic;">{model_descriptions[model]['title'].split()[-1]} method</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Setup completion button with enhanced styling
    st.markdown("---")
    
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    
    with col_button2:
        setup_button_text = "🔄 Update Configuration" if st.session_state.setup_completed else "🚀 Complete Setup & Start Analysis"
        
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


def _single_option_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Single Option Pricing Tab"""
    st.markdown('<div class="sub-header">📊 Single Option Pricing & Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #1f77b4; margin-top: 0;">🎯 Option Specification</h4>
        </div>
        """, unsafe_allow_html=True)
        
        option_type = st.selectbox("Option Type", ["call", "put"], help="Call = Right to buy, Put = Right to sell")
        exercise_style = st.selectbox("Exercise Style", ["european", "american"], help="European = Exercise only at expiry, American = Exercise anytime")
        strike_price = st.number_input("Strike Price (K)", value=100.0, min_value=0.1, step=0.1, key="single_strike", help="Exercise price of the option")
        
        # Moneyness analysis
        moneyness = spot_price / strike_price
        if moneyness > 1.05:
            moneyness_desc = "🟢 In-The-Money (ITM)"
            moneyness_color = "#28a745"
        elif moneyness < 0.95:
            moneyness_desc = "🔴 Out-of-The-Money (OTM)"
            moneyness_color = "#dc3545"
        else:
            moneyness_desc = "🟡 At-The-Money (ATM)"
            moneyness_color = "#ffc107"
            
        st.markdown(f"""
        <div style="background-color: {moneyness_color}20; padding: 10px; border-radius: 8px; border-left: 4px solid {moneyness_color};">
            <h5 style="color: {moneyness_color}; margin: 0;">{moneyness_desc}</h5>
            <p style="margin: 5px 0 0 0;">Moneyness: {moneyness:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
        
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
            
            # Enhanced price display
            intrinsic_value = max(spot_price - strike_price, 0) if option_type == "call" else max(strike_price - spot_price, 0)
            time_value = option_price - intrinsic_value
            
            st.markdown(f"""
            <div class="success-box">
                <h4 style="margin-top: 0;">💰 Option Valuation</h4>
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
        st.markdown('<div class="sub-header">📈 Parameter Sensitivity Analysis</div>', unsafe_allow_html=True)
        
        # Enhanced sensitivity analysis
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            param_to_vary = st.selectbox("Parameter to Vary", ["S", "K", "T", "r", "sigma", "q"], 
                                       help="Choose which parameter to analyze")
        
        with param_col2:
            n_points = st.slider("Analysis Points", 20, 100, 50, help="More points = smoother curve")
        
        # Get current parameter value and create range
        current_val = {
            "S": spot_price, "K": strike_price, "T": time_to_expiry,
            "r": risk_free_rate, "sigma": volatility, "q": dividend_yield
        }











# Helper functions for the strategy builder
def get_predefined_strategy(strategy_name, *strikes):
    """Get predefined strategy legs based on strategy name and strikes"""
    try:
        if strategy_name == "straddle":
            strike = strikes[0]
            return [
                {"type": "call", "strike": strike, "qty": 1.0},
                {"type": "put", "strike": strike, "qty": 1.0}
            ]
        elif strategy_name == "bull call spread":
            strike1, strike2 = strikes[0], strikes[1]
            if strike1 >= strike2:
                return "Error: Lower strike must be less than higher strike for bull call spread"
            return [
                {"type": "call", "strike": strike1, "qty": 1.0},
                {"type": "call", "strike": strike2, "qty": -1.0}
            ]
        elif strategy_name == "bear put spread":
            strike1, strike2 = strikes[0], strikes[1]
            if strike1 >= strike2:
                return "Error: Lower strike must be less than higher strike for bear put spread"
            return [
                {"type": "put", "strike": strike2, "qty": 1.0},
                {"type": "put", "strike": strike1, "qty": -1.0}
            ]
        elif strategy_name == "butterfly":
            strike1, strike2, strike3 = strikes[0], strikes[1], strikes[2]
            if not (strike1 < strike2 < strike3):
                return "Error: Strikes must be in ascending order for butterfly spread"
            return [
                {"type": "call", "strike": strike1, "qty": 1.0},
                {"type": "call", "strike": strike2, "qty": -2.0},
                {"type": "call", "strike": strike3, "qty": 1.0}
            ]
        elif strategy_name == "iron condor":
            strike1, strike2, strike3, strike4 = strikes[0], strikes[1], strikes[2], strikes[3]
            if not (strike1 < strike2 < strike3 < strike4):
                return "Error: Strikes must be in ascending order for iron condor"
            return [
                {"type": "put", "strike": strike1, "qty": 1.0},
                {"type": "put", "strike": strike2, "qty": -1.0},
                {"type": "call", "strike": strike3, "qty": -1.0},
                {"type": "call", "strike": strike4, "qty": 1.0}
            ]
        else:
            return f"Error: Unknown strategy '{strategy_name}'"
    except Exception as e:
        return f"Error creating {strategy_name}: {str(e)}"


def compute_strategy_payoff(legs, spot_range):
    """Compute strategy payoff for a range of spot prices"""
    payoffs = np.zeros(len(spot_range))
    
    for leg in legs:
        leg_payoffs = np.zeros(len(spot_range))
        
        for i, spot in enumerate(spot_range):
            if leg['type'] == 'call':
                intrinsic = max(spot - leg['strike'], 0)
            else:  # put
                intrinsic = max(leg['strike'] - spot, 0)
            
            leg_payoffs[i] = intrinsic * leg['qty']
        
        payoffs += leg_payoffs
    
    return payoffs


def calculate_strategy_greeks(legs, spot_range, greek_name, T, r, sigma, q):
    """Calculate strategy Greeks for a range of spot prices (placeholder implementation)"""
    # This is a simplified placeholder - replace with your actual Greeks calculation
    greek_values = np.zeros(len(spot_range))
    
    for i, S in enumerate(spot_range):
        strategy_greek = 0
        
        for leg in legs:
            # Simple Black-Scholes Greeks approximation
            K = leg['strike']
            qty = leg['qty']
            
            # Calculate d1 and d2
            d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            if greek_name == "delta":
                if leg['type'] == 'call':
                    leg_greek = norm.cdf(d1) * math.exp(-q*T)
                else:
                    leg_greek = (norm.cdf(d1) - 1) * math.exp(-q*T)
            elif greek_name == "gamma":
                leg_greek = norm.pdf(d1) * math.exp(-q*T) / (S * sigma * math.sqrt(T))
            elif greek_name == "theta":
                if leg['type'] == 'call':
                    leg_greek = (-S * norm.pdf(d1) * sigma * math.exp(-q*T) / (2 * math.sqrt(T)) 
                                - r * K * math.exp(-r*T) * norm.cdf(d2)
                                + q * S * math.exp(-q*T) * norm.cdf(d1)) / 365
                else:
                    leg_greek = (-S * norm.pdf(d1) * sigma * math.exp(-q*T) / (2 * math.sqrt(T)) 
                                + r * K * math.exp(-r*T) * norm.cdf(-d2)
                                - q * S * math.exp(-q*T) * norm.cdf(-d1)) / 365
            elif greek_name == "vega":
                leg_greek = S * norm.pdf(d1) * math.sqrt(T) * math.exp(-q*T) / 100
            elif greek_name == "rho":
                if leg['type'] == 'call':
                    leg_greek = K * T * math.exp(-r*T) * norm.cdf(d2) / 100
                else:
                    leg_greek = -K * T * math.exp(-r*T) * norm.cdf(-d2) / 100
            else:
                leg_greek = 0
            
            strategy_greek += leg_greek * qty
        
        greek_values[i] = strategy_greek
    
    return greek_values


def price_vanilla_option(option_type, exercise_style, model, **kwargs):
    """Enhanced placeholder for vanilla option pricing function"""
    S = kwargs.get('S', 100)
    K = kwargs.get('K', 100)
    T = kwargs.get('T', 1)
    r = kwargs.get('r', 0.05)
    sigma = kwargs.get('sigma', 0.2)
    q = kwargs.get('q', 0)
    
    try:
        # Black-Scholes calculation
        d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        if option_type == 'call':
            price = S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)
        
        # Add some model-specific adjustments for demonstration
        if model == "binomial":
            # Slight adjustment for binomial (placeholder)
            price *= 1.001
        elif model == "monte-carlo":
            # Add small random component for Monte Carlo (placeholder)
            price *= (1 + np.random.normal(0, 0.002))
        
        return max(price, 0)
    
    except Exception as e:
        raise ValueError(f"Error in option pricing: {str(e)}")


def price_option_strategy(legs, exercise_style, model, **kwargs):
    """Enhanced placeholder for strategy pricing function"""
    individual_prices = []
    total_price = 0
    
    try:
        for leg in legs:
            leg_kwargs = kwargs.copy()
            leg_kwargs['K'] = leg['strike']
            
            leg_price = price_vanilla_option(leg['type'], exercise_style, model, **leg_kwargs)
            individual_prices.append(leg_price)
            total_price += leg_price * leg['qty']
        
        return {
            'strategy_price': total_price,
            'individual_prices': individual_prices
        }
    
    except Exception as e:
        raise ValueError(f"Error in strategy pricing: {str(e)}")


# Additional educational content and final cleanup
def _add_educational_content():
    """Add comprehensive educational content"""
    st.markdown("---")
    st.markdown('<div class="sub-header">📖 Comprehensive Learning Resources</div>', unsafe_allow_html=True)
    
    with st.expander("🔬 Advanced Analysis Techniques"):
        st.markdown("""
        <div class="info-box">
            <h4>🧮 Quantitative Analysis Methods</h4>
            
            <h5>📊 Monte Carlo Simulation</h5>
            <ul>
                <li><strong>Path Simulation:</strong> Generate thousands of price paths</li>
                <li><strong>Option Valuation:</strong> Price complex payoffs statistically</li>
                <li><strong>Risk Assessment:</strong> Calculate VaR and expected shortfall</li>
                <li><strong>Scenario Testing:</strong> Test strategies under various market conditions</li>
            </ul>
            
            <h5>🌡️ Stress Testing</h5>
            <ul>
                <li><strong>Market Crashes:</strong> -20%, -30% overnight moves</li>
                <li><strong>Volatility Spikes:</strong> Vol doubling overnight</li>
                <li><strong>Interest Rate Shocks:</strong> ±200bp rate moves</li>
                <li><strong>Liquidity Crises:</strong> Widened bid-ask spreads</li>
            </ul>
            
            <h5>📈 Advanced Greeks</h5>
            <ul>
                <li><strong>Charm (Delta Decay):</strong> How delta changes with time</li>
                <li><strong>Vomma (Volga):</strong> How vega changes with volatility</li>
                <li><strong>Vanna:</strong> How delta changes with volatility</li>
                <li><strong>Speed:</strong> How gamma changes with spot price</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🛠️ Troubleshooting & Best Practices"):
        st.markdown("""
        <div class="warning-box">
            <h4>❌ Common Issues & Solutions</h4>
            
            <h5>🔧 Strategy Configuration</h5>
            <ul>
                <li><strong>Negative Strikes:</strong> Ensure all strike prices are positive</li>
                <li><strong>Zero Quantities:</strong> Use non-zero option quantities</li>
                <li><strong>Extreme Parameters:</strong> Use realistic market parameters</li>
                <li><strong>Time to Expiry:</strong> Avoid very small time values (< 0.001)</li>
            </ul>
            
            <h5>📊 Calculation Errors</h5>
            <ul>
                <li><strong>Numerical Precision:</strong> Some extreme parameter combinations fail</li>
                <li><strong>Model Limitations:</strong> Each model has convergence constraints</li>
                <li><strong>Memory Issues:</strong> High resolution grids may exceed memory</li>
                <li><strong>Timeout Errors:</strong> Very complex calculations may timeout</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ Best Practices</h4>
            
            <h5>🎯 Parameter Selection</h5>
            <ul>
                <li><strong>Realistic Ranges:</strong> Use market-observed parameter ranges</li>
                <li><strong>Gradual Changes:</strong> Start with small sensitivity ranges</li>
                <li><strong>Model Appropriate:</strong> Choose models suited to your strategy</li>
                <li><strong>Time Management:</strong> Allow sufficient time for complex calculations</li>
            </ul>
            
            <h5>📈 Analysis Strategy</h5>
            <ul>
                <li><strong>Start Simple:</strong> Begin with single-parameter sensitivity</li>
                <li><strong>Build Complexity:</strong> Gradually add more parameters</li>
                <li><strong>Validate Results:</strong> Check results make economic sense</li>
                <li><strong>Document Findings:</strong> Keep track of key insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# Final summary and next steps for users
def _display_final_summary():
    """Display final summary and next steps"""
    st.markdown("---")
    st.markdown('<div class="sub-header">📋 Analysis Summary & Next Steps</div>', unsafe_allow_html=True)
    
    if 'current_legs' in st.session_state and 'strategy_result' in st.session_state:
        strategy_price = st.session_state.strategy_result['strategy_price']
        strategy_type = "Credit" if strategy_price < 0 else "Debit"
        
        st.markdown(f"""
        <div class="success-box">
            <h4>📊 Current Strategy Summary</h4>
            <p><strong>Strategy Type:</strong> {strategy_type} (${abs(strategy_price):.4f})</p>
            <p><strong>Number of Legs:</strong> {len(st.session_state.current_legs)}</p>
            <p><strong>Analysis Status:</strong> Ready for advanced analysis</p>
            
            <h5>🎯 Recommended Next Steps:</h5>
            <ul>
                <li>Analyze payoff diagram for profit/loss scenarios</li>
                <li>Review Greeks exposure for risk management</li>
                <li>Run sensitivity analysis for parameter changes</li>
                <li>Consider position sizing and risk limits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Ready to Begin Advanced Analysis</h4>
            <p>Your Options Strategy Suite is fully configured and ready for use!</p>
            
            <h5>📋 Getting Started Checklist:</h5>
            <ol>
                <li><strong>Setup Parameters:</strong> ✅ Complete</li>
                <li><strong>Build Strategy:</strong> Configure in Strategy Builder tab</li>
                <li><strong>Analyze Payoffs:</strong> Review profit/loss scenarios</li>
                <li><strong>Study Greeks:</strong> Understand risk sensitivities</li>
                <li><strong>Test Sensitivity:</strong> Multi-parameter analysis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


# Add final educational content and summary to the sensitivity tab
if __name__ == "__main__":
    # This section would only run if the file is executed directly
    # In practice, this would be called from your main Streamlit app
    option_strategies_tab()
        current_val = {
            "S": spot_price, "K": strike_price, "T": time_to_expiry,
            "r": risk_free_rate, "sigma": volatility, "q": dividend_yield
        }[param_to_vary]
        
        range_col1, range_col2 = st.columns(2)
        
        with range_col1:
            param_min = st.number_input(f"Min {param_to_vary}", value=current_val * 0.5, step=0.01, key="single_param_min")
        with range_col2:
            param_max = st.number_input(f"Max {param_to_vary}", value=current_val * 1.5, step=0.01, key="single_param_max")
        
        if st.button("🔍 Generate Sensitivity Analysis", type="primary", use_container_width=True):
            try:
                with st.spinner("Calculating parameter sensitivity..."):
                    fixed_params = {
                        "S": spot_price, "K": strike_price, "T": time_to_expiry,
                        "r": risk_free_rate, "sigma": volatility, "q": dividend_yield
                    }
                    if model == "binomial":
                        fixed_params["N"] = n_steps
                    elif model == "monte-carlo":
                        fixed_params["n_simulations"] = n_simulations
                    
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
                                model=model,
                                **temp_params
                            )
                            option_prices.append(price)
                        except:
                            option_prices.append(np.nan)
                    
                    # Create enhanced plotly chart
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
                    
                    # Sensitivity metrics
                    price_range = max(option_prices) - min(option_prices)
                    param_range = param_max - param_min
                    avg_sensitivity = price_range / param_range
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>📊 Sensitivity Metrics</h4>
                        <p><strong>Price Range:</strong> ${min(option_prices):.4f} - ${max(option_prices):.4f}</p>
                        <p><strong>Average Sensitivity:</strong> ${avg_sensitivity:.6f} per unit change in {param_to_vary}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Analysis Error</h4>
                    <p>Error generating sensitivity plot: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)


def _strategy_builder_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Strategy Builder Tab"""
    st.markdown('<div class="sub-header">🔧 Advanced Strategy Builder</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="parameter-grid">
            <h4 style="color: #1f77b4; margin-top: 0;">🎯 Strategy Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        strategy_method = st.radio("Choose Construction Method", ["Predefined Strategy", "Custom Strategy"], 
                                 help="Use predefined templates or build from scratch")
        
        if strategy_method == "Predefined Strategy":
            st.markdown("#### 📋 Predefined Strategies")
            strategy_name = st.selectbox(
                "Select Strategy Template",
                ["straddle", "bull call spread", "bear put spread", "butterfly", "iron condor"],
                help="Choose from common option strategies"
            )
            
            # Strategy descriptions
            strategy_descriptions = {
                "straddle": "🎯 **Long Straddle**: Profit from large moves in either direction. High volatility play.",
                "bull call spread": "📈 **Bull Call Spread**: Limited risk bullish strategy. Buy low strike call, sell high strike call.",
                "bear put spread": "📉 **Bear Put Spread**: Limited risk bearish strategy. Buy high strike put, sell low strike put.",
                "butterfly": "🦋 **Butterfly Spread**: Low volatility strategy. Profit when price stays near center strike.",
                "iron condor": "🦅 **Iron Condor**: Range-bound strategy. Profit when price stays between inner strikes."
            }
            
            st.markdown(f"""
            <div class="info-box">
                {strategy_descriptions[strategy_name]}
            </div>
            """, unsafe_allow_html=True)
            
            # Dynamic strike inputs based on strategy
            if strategy_name == "straddle":
                strike1 = st.number_input("Strike Price", value=spot_price, key="pred_k1", 
                                        help="Both call and put will use this strike")
                legs = get_predefined_strategy(strategy_name, strike1)
            elif strategy_name in ["bull call spread", "bear put spread"]:
                st.markdown("##### Strike Configuration")
                strike1 = st.number_input("Lower Strike (Long)", value=spot_price-5, key="pred_k1")
                strike2 = st.number_input("Higher Strike (Short)", value=spot_price+5, key="pred_k2")
                legs = get_predefined_strategy(strategy_name, strike1, strike2)
            elif strategy_name == "butterfly":
                st.markdown("##### Strike Configuration")
                strike1 = st.number_input("Lower Strike", value=spot_price-10, key="pred_k1")
                strike2 = st.number_input("Middle Strike", value=spot_price, key="pred_k2")
                strike3 = st.number_input("Upper Strike", value=spot_price+10, key="pred_k3")
                legs = get_predefined_strategy(strategy_name, strike1, strike2, strike3)
            elif strategy_name == "iron condor":
                st.markdown("##### Strike Configuration")
                strike1 = st.number_input("Put Long Strike", value=spot_price-15, key="pred_k1")
                strike2 = st.number_input("Put Short Strike", value=spot_price-5, key="pred_k2")
                strike3 = st.number_input("Call Short Strike", value=spot_price+5, key="pred_k3")
                strike4 = st.number_input("Call Long Strike", value=spot_price+15, key="pred_k4")
                legs = get_predefined_strategy(strategy_name, strike1, strike2, strike3, strike4)
                
        else:  # Custom Strategy
            st.markdown("#### 🛠️ Custom Strategy Builder")
            num_legs = st.number_input("Number of Legs", value=2, min_value=1, max_value=10, step=1, key="custom_num_legs",
                                     help="How many option positions in your strategy")
            
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
                    leg_strike = st.number_input(f"Strike", value=spot_price, key=f"leg_strike_{i}")
                with col_qty:
                    leg_qty = st.number_input(f"Quantity", value=1.0, step=0.1, key=f"leg_qty_{i}",
                                            help="Positive = Long, Negative = Short")
                
                legs.append({"type": leg_type, "strike": leg_strike, "qty": leg_qty})
        
        # Exercise style for strategy
        strategy_exercise = st.selectbox("Exercise Style", ["european", "american"], key="strategy_exercise",
                                       help="Apply to all legs in the strategy")
        
        # Store legs in session state for use in other tabs
        st.session_state.current_legs = legs
        st.session_state.current_strategy_exercise = strategy_exercise
    
    with col2:
        st.markdown('<div class="sub-header">📊 Strategy Analysis</div>', unsafe_allow_html=True)
        
        if isinstance(legs, str):  # Error message from predefined strategy
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Strategy Configuration Error</h4>
                <p>{legs}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Enhanced strategy display
            strategy_df = pd.DataFrame(legs)
            strategy_df['Position'] = strategy_df['qty'].apply(lambda x: f"{'Long' if x > 0 else 'Short'} {abs(x)}")
            strategy_df['Option'] = strategy_df.apply(lambda row: f"{row['type'].title()} @ ${row['strike']:.2f}", axis=1)
            
            display_df = strategy_df[['Position', 'Option']].copy()
            display_df.index = [f"Leg {i+1}" for i in range(len(display_df))]
            
            st.markdown('<div class="results-table">', unsafe_allow_html=True)
            st.dataframe(display_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
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
                
                with st.spinner("Pricing strategy..."):
                    strategy_result = price_option_strategy(
                        legs=legs,
                        exercise_style=strategy_exercise,
                        model=model,
                        **kwargs
                    )
                
                # Store result in session state
                st.session_state.strategy_result = strategy_result
                
                # Enhanced results display
                net_premium = strategy_result['strategy_price']
                strategy_type = "Credit" if net_premium < 0 else "Debit"
                strategy_color = "#28a745" if net_premium < 0 else "#dc3545"
                
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="margin-top: 0;">💰 Strategy Valuation</h4>
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
                
                # Strategy type explanation
                if strategy_type == "Credit":
                    st.markdown("""
                    <div class="success-box">
                        <h5>💚 Credit Strategy</h5>
                        <p>You receive premium upfront. Profit if strategy expires worthless or declines in value.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                        <h5>💙 Debit Strategy</h5>
                        <p>You pay premium upfront. Profit if strategy increases in value above your cost.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Individual leg breakdown
                st.markdown("#### 📋 Individual Leg Analysis")
                for i, (leg, price) in enumerate(zip(legs, strategy_result['individual_prices'])):
                    position = "Long" if leg['qty'] > 0 else "Short"
                    position_color = "#28a745" if leg['qty'] > 0 else "#dc3545"
                    total_cost = price * abs(leg['qty'])
                    
                    st.markdown(f"""
                    <div style="background-color: {position_color}20; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid {position_color};">
                        <strong>Leg {i+1}:</strong> <span style="color: {position_color};">{position}</span> {abs(leg['qty'])} {leg['type'].title()} @ ${leg['strike']:.2f}
                        <br><strong>Unit Price:</strong> ${price:.4f} | <strong>Total Cost:</strong> ${total_cost:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Strategy Pricing Error</h4>
                    <p>Error pricing strategy: {str(e)}</p>
                    <p><strong>Tip:</strong> Check that all strikes are positive and parameters are valid.</p>
                </div>
                """, unsafe_allow_html=True)



def _payoff_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Payoff Analysis Tab"""
    st.markdown('<div class="sub-header">📈 Strategy Payoff Analysis</div>', unsafe_allow_html=True)
    
    if 'current_legs' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ No Strategy Configured</h4>
            <p>Please build a strategy in the <strong>Strategy Builder</strong> tab first!</p>
            <p><strong>Next Step:</strong> Go to Strategy Builder → Configure your options strategy</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    legs = st.session_state.current_legs
    
    # Enhanced payoff controls
    st.markdown('<div class="sub-header">🔧 Analysis Configuration</div>', unsafe_allow_html=True)
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("#### 📊 Price Range")
        strikes = [leg['strike'] for leg in legs]
        min_strike, max_strike = min(strikes), max(strikes)
        
        spot_min = st.number_input("Min Spot Price", value=min_strike * 0.7, step=1.0, key="payoff_spot_min")
        spot_max = st.number_input("Max Spot Price", value=max_strike * 1.3, step=1.0, key="payoff_spot_max")
    
    with config_col2:
        st.markdown("#### ⚙️ Analysis Options")
        n_points = st.slider("Resolution", 50, 500, 200, key="payoff_n_points", help="Higher = smoother curves")
        show_breakeven = st.checkbox("Show Breakeven Points", value=True)
    
    with config_col3:
        st.markdown("#### 💰 P&L Options")
        show_profit_loss = st.checkbox("Include Premium Cost", value=True, help="Show net P&L including initial cost")
        show_individual_legs = st.checkbox("Show Individual Legs", value=False, help="Display each leg separately")
    
    # Calculate and display payoff
    spot_range = np.linspace(spot_min, spot_max, n_points)
    
    try:
        payoffs = compute_strategy_payoff(legs, spot_range)
        
        # Create enhanced interactive plotly chart
        fig = go.Figure()
        
        # Main payoff line
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=payoffs,
            mode='lines',
            name='Strategy Payoff at Expiration',
            line=dict(color='#1f77b4', width=4),
            hovertemplate='Spot: $%{x:.2f}<br>Payoff: $%{y:.4f}<extra></extra>'
        ))
        
        # Add profit/loss line if premium is included
        if show_profit_loss and 'strategy_result' in st.session_state:
            pnl = payoffs - st.session_state.strategy_result['strategy_price']
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=pnl,
                mode='lines',
                name='Net P&L (including premium)',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                hovertemplate='Spot: $%{x:.2f}<br>Net P&L: $%{y:.4f}<extra></extra>'
            ))
        
        # Show individual legs if requested
        if show_individual_legs:
            colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            for i, leg in enumerate(legs):
                leg_payoff = []
                for s in spot_range:
                    if leg['type'] == 'call':
                        intrinsic = max(s - leg['strike'], 0)
                    else:
                        intrinsic = max(leg['strike'] - s, 0)
                    leg_payoff.append(intrinsic * leg['qty'])
                
                fig.add_trace(go.Scatter(
                    x=spot_range,
                    y=leg_payoff,
                    mode='lines',
                    name=f"Leg {i+1}: {leg['qty']:.1f} {leg['type']} @ ${leg['strike']:.2f}",
                    line=dict(color=colors[i % len(colors)], width=2, dash='dot'),
                    opacity=0.7
                ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
        
        # Add strike lines
        for i, leg in enumerate(legs):
            fig.add_vline(
                x=leg['strike'],
                line_dash="dot",
                line_color="gray",
                line_width=1,
                annotation_text=f"K{i+1}: ${leg['strike']:.0f}",
                annotation_position="top"
            )
        
        # Add current spot line
        fig.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"Current Spot: ${spot_price:.2f}",
            annotation_position="bottom"
        )
        
        # Breakeven analysis
        breakeven_points = []
        if show_breakeven and show_profit_loss and 'strategy_result' in st.session_state:
            pnl = payoffs - st.session_state.strategy_result['strategy_price']
            # Find breakeven points (where PnL crosses zero)
            for i in range(len(pnl)-1):
                if pnl[i] * pnl[i+1] < 0:  # Sign change
                    # Linear interpolation to find exact crossing
                    x_cross = spot_range[i] + (spot_range[i+1] - spot_range[i]) * (-pnl[i] / (pnl[i+1] - pnl[i]))
                    breakeven_points.append(x_cross)
            
            for i, breakeven in enumerate(breakeven_points):
                fig.add_vline(
                    x=breakeven,
                    line_color="orange",
                    line_width=3,
                    annotation_text=f"Breakeven: ${breakeven:.2f}",
                    annotation_position="top right" if i % 2 == 0 else "top left"
                )
        
        fig.update_layout(
            title=dict(
                text="Strategy Payoff Diagram",
                font=dict(size=20, color='#1f77b4')
            ),
            xaxis_title="Spot Price at Expiration ($)",
            yaxis_title="Payoff ($)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced summary statistics
        st.markdown('<div class="sub-header">📊 Strategy Statistics</div>', unsafe_allow_html=True)
        
        max_payoff = np.max(payoffs)
        min_payoff = np.min(payoffs)
        
        if show_profit_loss and 'strategy_result' in st.session_state:
            pnl = payoffs - st.session_state.strategy_result['strategy_price']
            max_profit = np.max(pnl)
            max_loss = np.min(pnl)
            
            # Find profitable range
            profitable_indices = np.where(pnl > 0)[0]
            if len(profitable_indices) > 0:
                profit_range_min = spot_range[profitable_indices[0]]
                profit_range_max = spot_range[profitable_indices[-1]]
                profitable_range = f"${profit_range_min:.2f} - ${profit_range_max:.2f}"
            else:
                profitable_range = "No profitable range"
        else:
            max_profit = "N/A"
            max_loss = "N/A"
            profitable_range = "N/A"
        
        # Create comprehensive statistics table
        st.markdown(f"""
        <div class="payoff-stats">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #6c757d; background-color: #f8f9fa;">
                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                    <td style="padding: 12px; font-weight: bold;">Value</td>
                    <td style="padding: 12px; font-weight: bold;">Description</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Max Payoff</td>
                    <td style="padding: 10px; font-family: monospace; color: #28a745; font-weight: bold;">${max_payoff:.2f}</td>
                    <td style="padding: 10px; font-style: italic;">Best case scenario at expiration</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Min Payoff</td>
                    <td style="padding: 10px; font-family: monospace; color: #dc3545; font-weight: bold;">${min_payoff:.2f}</td>
                    <td style="padding: 10px; font-style: italic;">Worst case scenario at expiration</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Max Profit</td>
                    <td style="padding: 10px; font-family: monospace; color: #28a745; font-weight: bold;">{"${:.2f}".format(max_profit) if max_profit != "N/A" else "N/A"}</td>
                    <td style="padding: 10px; font-style: italic;">Maximum net profit including premium</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Max Loss</td>
                    <td style="padding: 10px; font-family: monospace; color: #dc3545; font-weight: bold;">{"${:.2f}".format(max_loss) if max_loss != "N/A" else "N/A"}</td>
                    <td style="padding: 10px; font-style: italic;">Maximum net loss including premium</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Breakeven Points</td>
                    <td style="padding: 10px; font-family: monospace; color: #ffc107; font-weight: bold;">{len(breakeven_points) if breakeven_points else 0}</td>
                    <td style="padding: 10px; font-style: italic;">Number of zero P&L points</td>
                </tr>
                <tr>
                    <td style="padding: 10px; font-weight: bold;">Profitable Range</td>
                    <td style="padding: 10px; font-family: monospace; color: #17a2b8; font-weight: bold;">{profitable_range}</td>
                    <td style="padding: 10px; font-style: italic;">Spot price range for profit</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk/Reward analysis
        if show_profit_loss and 'strategy_result' in st.session_state and max_profit != "N/A" and max_loss != "N/A":
            if max_loss < 0:
                risk_reward_ratio = abs(max_profit / max_loss)
                st.markdown(f"""
                <div class="info-box">
                    <h4>⚖️ Risk/Reward Analysis</h4>
                    <p><strong>Risk/Reward Ratio:</strong> {risk_reward_ratio:.2f}:1</p>
                    <p><strong>Interpretation:</strong> For every $1 of potential loss, you can gain ${risk_reward_ratio:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Strategy recommendations
        current_pnl = 0
        if show_profit_loss and 'strategy_result' in st.session_state:
            current_payoff = compute_strategy_payoff(legs, [spot_price])[0]
            current_pnl = current_payoff - st.session_state.strategy_result['strategy_price']
        
        if current_pnl > 0:
            st.markdown("""
            <div class="success-box">
                <h4>✅ Current Position Status</h4>
                <p>Your strategy is currently <strong>profitable</strong> at the current spot price!</p>
            </div>
            """, unsafe_allow_html=True)
        elif current_pnl < 0:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Current Position Status</h4>
                <p>Your strategy is currently <strong>unprofitable</strong> at the current spot price.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>⚖️ Current Position Status</h4>
                <p>Your strategy is currently at <strong>breakeven</strong> at the current spot price.</p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Payoff Calculation Error</h4>
            <p>Error calculating strategy payoff: {str(e)}</p>
            <p><strong>Tip:</strong> Check that your strategy configuration is valid.</p>
        </div>
        """, unsafe_allow_html=True)



def _greeks_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Greeks Analysis Tab"""
    
    st.markdown('<div class="sub-header">🎯 Greeks Risk Analysis</div>', unsafe_allow_html=True)# tabs/option_strategies.py
    
    legs = st.session_state.current_legs
    
    # Enhanced Greeks configuration
    st.markdown('<div class="sub-header">⚙️ Greeks Configuration</div>', unsafe_allow_html=True)
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        st.markdown("""
        <div class="greek-analysis">
            <h5 style="margin-top: 0; color: #17a2b8;">📊 Greek Selection</h5>
        </div>
        """, unsafe_allow_html=True)
        
        greek_name = st.selectbox("Select Greek to Analyze", ["delta", "gamma", "vega", "theta", "rho"],
                                help="Choose which risk sensitivity to analyze")
        
        # Greek descriptions
        greek_descriptions = {
            "delta": "📈 **Delta**: Price sensitivity to underlying asset moves. Range: -1 to +1",
            "gamma": "🔄 **Gamma**: Rate of change of delta. Measures convexity of option price",
            "vega": "🌊 **Vega**: Sensitivity to volatility changes. Higher for ATM options",
            "theta": "⏰ **Theta**: Time decay. Usually negative (options lose value over time)",
            "rho": "💰 **Rho**: Interest rate sensitivity. More important for longer-term options"
        }
        
        st.markdown(f"""
        <div class="info-box">
            {greek_descriptions[greek_name]}
        </div>
        """, unsafe_allow_html=True)
    
    with config_col2:
        st.markdown("""
        <div class="greek-analysis">
            <h5 style="margin-top: 0; color: #17a2b8;">📏 Analysis Range</h5>
        </div>
        """, unsafe_allow_html=True)
        
        # Spot range for Greeks
        strikes = [leg['strike'] for leg in legs]
        min_strike, max_strike = min(strikes), max(strikes)
        
        greek_spot_min = st.number_input("Min Spot for Analysis", value=min_strike * 0.8, step=1.0, key="greeks_spot_min")
        greek_spot_max = st.number_input("Max Spot for Analysis", value=max_strike * 1.2, step=1.0, key="greeks_spot_max")
    
    with config_col3:
        st.markdown("""
        <div class="greek-analysis">
            <h5 style="margin-top: 0; color: #17a2b8;">⚙️ Analysis Options</h5>
        </div>
        """, unsafe_allow_html=True)
        
        greek_points = st.slider("Analysis Resolution", 50, 300, 150, key="greeks_n_points",
                                help="Higher values = smoother curves but slower computation")
        show_individual_greeks = st.checkbox("Show Individual Leg Greeks", value=False,
                                           help="Display Greeks for each option separately")
        
        run_greeks_analysis = st.button("🔍 Run Greeks Analysis", type="primary", use_container_width=True)
    
    if run_greeks_analysis:
        try:
            with st.spinner(f"Calculating {greek_name} analysis..."):
                # Generate spot range
                spot_range = np.linspace(greek_spot_min, greek_spot_max, greek_points)
                
                # Calculate strategy Greeks using placeholder calculation
                strategy_greeks = calculate_strategy_greeks(legs, spot_range, greek_name, 
                                                          time_to_expiry, risk_free_rate, volatility, dividend_yield)
                
                # Create enhanced Plotly chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=spot_range,
                    y=strategy_greeks,
                    mode='lines',
                    name=f'Strategy {greek_name.title()}',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                # Add current spot marker
                current_greek = np.interp(spot_price, spot_range, strategy_greeks)
                fig.add_scatter(
                    x=[spot_price],
                    y=[current_greek],
                    mode='markers',
                    name='Current Position',
                    marker=dict(color='red', size=12, symbol='diamond')
                )
                
                # Add zero line for reference
                fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
                
                # Add current spot line
                fig.add_vline(
                    x=spot_price,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Current Spot: ${spot_price:.2f}"
                )
                
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
                    <h4>📊 Current {greek_name.title()} Analysis</h4>
                    <p><strong>Current {greek_name.title()} value:</strong> {current_greek:.6f}</p>
                    <p><strong>Spot Price:</strong> ${spot_price:.2f}</p>
                    <p><strong>Analysis Range:</strong> ${greek_spot_min:.2f} - ${greek_spot_max:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Greeks interpretation
                interpretations = {
                    "delta": {
                        "positive": "Strategy gains value when underlying price increases",
                        "negative": "Strategy loses value when underlying price increases",
                        "near_zero": "Strategy is relatively insensitive to small price moves"
                    },
                    "gamma": {
                        "positive": "Delta will increase as spot price rises (convex payoff)",
                        "negative": "Delta will decrease as spot price rises (concave payoff)",
                        "near_zero": "Delta is relatively stable across price moves"
                    },
                    "theta": {
                        "positive": "Strategy gains value as time passes (time decay benefit)",
                        "negative": "Strategy loses value as time passes (time decay cost)",
                        "near_zero": "Strategy is relatively insensitive to time decay"
                    },
                    "vega": {
                        "positive": "Strategy gains value when volatility increases",
                        "negative": "Strategy loses value when volatility increases",
                        "near_zero": "Strategy is relatively insensitive to volatility changes"
                    },
                    "rho": {
                        "positive": "Strategy gains value when interest rates increase",
                        "negative": "Strategy loses value when interest rates increase",
                        "near_zero": "Strategy is relatively insensitive to interest rate changes"
                    }
                }
                
                if current_greek > 0.01:
                    interpretation = interpretations[greek_name]["positive"]
                    color = "#28a745"
                elif current_greek < -0.01:
                    interpretation = interpretations[greek_name]["negative"]
                    color = "#dc3545"
                else:
                    interpretation = interpretations[greek_name]["near_zero"]
                    color = "#ffc107"
                
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 15px; border-radius: 8px; border-left: 4px solid {color};">
                    <h5 style="color: {color}; margin-top: 0;">📋 Risk Interpretation</h5>
                    <p>{interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Analysis Error</h4>
                <p>Error calculating Greeks: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Educational content about Greeks
    st.markdown("---")
    st.markdown('<div class="sub-header">📚 Greeks Education</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Understanding the Greeks"):
        st.markdown("""
        <div class="info-box">
            <h4>🎯 The Greeks - Risk Sensitivities Explained</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div>
                    <h5 style="color: #1f77b4;">📈 First-Order Greeks</h5>
                    <ul>
                        <li><strong>Delta (Δ):</strong> Price sensitivity to underlying moves</li>
                        <li><strong>Vega (ν):</strong> Sensitivity to volatility changes</li>
                        <li><strong>Theta (Θ):</strong> Time decay (usually negative)</li>
                        <li><strong>Rho (ρ):</strong> Interest rate sensitivity</li>
                    </ul>
                </div>
                <div>
                    <h5 style="color: #1f77b4;">🔄 Second-Order Greeks</h5>
                    <ul>
                        <li><strong>Gamma (Γ):</strong> Rate of change of delta</li>
                        <li><strong>Vomma:</strong> Rate of change of vega</li>
                        <li><strong>Charm:</strong> Rate of change of delta over time</li>
                        <li><strong>Color:</strong> Rate of change of gamma over time</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("⚖️ Greeks Risk Management"):
        st.markdown("""
        <div class="warning-box">
            <h4>🛡️ Professional Risk Management with Greeks</h4>
            
            <h5>📊 Delta Hedging</h5>
            <ul>
                <li><strong>Delta Neutral:</strong> Hedge with underlying to offset directional risk</li>
                <li><strong>Dynamic Hedging:</strong> Rebalance as delta changes (gamma effect)</li>
                <li><strong>Portfolio Delta:</strong> Sum of all position deltas</li>
            </ul>
            
            <h5>⏰ Time Management</h5>
            <ul>
                <li><strong>Theta Decay:</strong> Short options benefit, long options suffer</li>
                <li><strong>Weekend Effect:</strong> Time decay continues over weekends</li>
                <li><strong>Expiration Risk:</strong> Gamma and theta explode near expiry</li>
            </ul>
            
            <h5>🌊 Volatility Risk</h5>
            <ul>
                <li><strong>Vega Exposure:</strong> Long options are long volatility</li>
                <li><strong>Implied vs Realized:</strong> Trade volatility differences</li>
                <li><strong>Volatility Smile:</strong> Vega varies by strike and time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)





def _sensitivity_analysis_tab(spot_price, risk_free_rate, dividend_yield, volatility, time_to_expiry, model, n_steps, n_simulations):
    """Advanced Sensitivity Analysis Tab"""
    st.markdown('<div class="sub-header">🔬 Advanced Multi-Parameter Sensitivity</div>', unsafe_allow_html=True)
    
    if 'current_legs' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ No Strategy Configured</h4>
            <p>Please build a strategy in the <strong>Strategy Builder</strong> tab first!</p>
            <p><strong>Next Step:</strong> Go to Strategy Builder → Configure your options strategy</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    legs = st.session_state.current_legs
    strategy_exercise = st.session_state.get('current_strategy_exercise', 'european')
    
    st.markdown("#### 🔬 Multi-Parameter Sensitivity Heatmap")
    
    # Enhanced parameter selection
    sens_col1, sens_col2, sens_col3 = st.columns(3)
    
    with sens_col1:
        st.markdown("""
        <div class="parameter-grid">
            <h5 style="margin-top: 0; color: #1f77b4;">📊 Parameter 1 (X-Axis)</h5>
        </div>
        """, unsafe_allow_html=True)
        
        param1 = st.selectbox("First Parameter", ["S", "T", "r", "sigma", "q"], key="param1",
                            help="Parameter to vary along X-axis")
        param1_range = st.slider(
            f"{param1} Range (%)",
            -50, 100, (-20, 20),
            step=5,
            key="param1_range",
            help="Percentage change from current value"
        )
    
    with sens_col2:
        st.markdown("""
        <div class="parameter-grid">
            <h5 style="margin-top: 0; color: #1f77b4;">📈 Parameter 2 (Y-Axis)</h5>
        </div>
        """, unsafe_allow_html=True)
        
        param2 = st.selectbox("Second Parameter", ["S", "T", "r", "sigma", "q"], key="param2",
                            help="Parameter to vary along Y-axis")
        param2_range = st.slider(
            f"{param2} Range (%)",
            -50, 100, (-20, 20),
            step=5,
            key="param2_range",
            help="Percentage change from current value"
        )
    
    with sens_col3:
        st.markdown("""
        <div class="parameter-grid">
            <h5 style="margin-top: 0; color: #1f77b4;">⚙️ Analysis Settings</h5>
        </div>
        """, unsafe_allow_html=True)
        
        resolution = st.slider("Grid Resolution", 10, 50, 20, key="sensitivity_resolution",
                              help="Higher resolution = more detailed but slower")
        
        # Analysis type
        analysis_type = st.selectbox("Analysis Type", 
                                   ["Strategy Price", "Profit/Loss", "% Change"],
                                   help="What to display in the heatmap")
        
        generate_heatmap = st.button("🔥 Generate Sensitivity Heatmap", type="primary", use_container_width=True)
    
    if generate_heatmap:
        if param1 == param2:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Parameter Selection Error</h4>
                <p>Please select different parameters for X and Y axes!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                with st.spinner("Generating multi-parameter sensitivity analysis..."):
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
                    
                    # Base strategy price for comparison
                    base_strategy_price = None
                    if 'strategy_result' in st.session_state:
                        base_strategy_price = st.session_state.strategy_result['strategy_price']
                    
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
                                
                                strategy_price = result["strategy_price"]
                                
                                if analysis_type == "Strategy Price":
                                    price_grid[j, i] = strategy_price
                                elif analysis_type == "Profit/Loss" and base_strategy_price is not None:
                                    price_grid[j, i] = strategy_price - base_strategy_price
                                elif analysis_type == "% Change" and base_strategy_price is not None and base_strategy_price != 0:
                                    price_grid[j, i] = (strategy_price - base_strategy_price) / abs(base_strategy_price) * 100
                                else:
                                    price_grid[j, i] = strategy_price
                                    
                            except:
                                price_grid[j, i] = np.nan
                            
                            iteration += 1
                            progress_bar.progress(iteration / total_iterations)
                    
                    # Create enhanced heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=price_grid,
                        x=param1_vals,
                        y=param2_vals,
                        colorscale='RdYlBu_r',
                        colorbar=dict(
                            title=f"{analysis_type} {'($)' if analysis_type != '% Change' else '(%)'}",
                            titleside="right"
                        ),
                        hovertemplate=f'{param1}: %{{x:.4f}}<br>{param2}: %{{y:.4f}}<br>{analysis_type}: %{{z:.4f}}<extra></extra>'
                    ))
                    
                    # Add current position marker
                    fig.add_scatter(
                        x=[base_values[param1]],
                        y=[base_values[param2]],
                        mode='markers',
                        name='Current Position',
                        marker=dict(
                            symbol='star',
                            size=15,
                            color='white',
                            line=dict(color='black', width=2)
                        )
                    )
                    
                    fig.update_layout(
                        title=f'Strategy {analysis_type} Sensitivity: {param1} vs {param2}',
                        xaxis_title=f'{param1} Value',
                        yaxis_title=f'{param2} Value',
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced analysis summary
                    min_val = np.nanmin(price_grid)
                    max_val = np.nanmax(price_grid)
                    mean_val = np.nanmean(price_grid)
                    std_val = np.nanstd(price_grid)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>📊 Sensitivity Analysis Summary</h4>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid #28a745;">
                                <td style="padding: 8px; font-weight: bold;">Metric</td>
                                <td style="padding: 8px; font-weight: bold;">Value</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 6px;">Minimum {analysis_type}</td>
                                <td style="padding: 6px; font-family: monospace;">{min_val:.4f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 6px;">Maximum {analysis_type}</td>
                                <td style="padding: 6px; font-family: monospace;">{max_val:.4f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 6px;">Range ({analysis_type})</td>
                                <td style="padding: 6px; font-family: monospace;">{max_val - min_val:.4f}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 6px;">Average {analysis_type}</td>
                                <td style="padding: 6px; font-family: monospace;">{mean_val:.4f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px;">Standard Deviation</td>
                                <td style="padding: 6px; font-family: monospace;">{std_val:.4f}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk interpretation
                    volatility_pct = (std_val / abs(mean_val)) * 100 if mean_val != 0 else 0
                    
                    if volatility_pct > 50:
                        risk_level = "🔴 High"
                        risk_color = "#dc3545"
                    elif volatility_pct > 20:
                        risk_level = "🟡 Moderate"
                        risk_color = "#ffc107"
                    else:
                        risk_level = "🟢 Low"
                        risk_color = "#28a745"
                    
                    st.markdown(f"""
                    <div style="background-color: {risk_color}20; padding: 15px; border-radius: 8px; border-left: 4px solid {risk_color};">
                        <h5 style="color: {risk_color}; margin-top: 0;">⚖️ Risk Assessment</h5>
                        <p><strong>Sensitivity Level:</strong> {risk_level}</p>
                        <p><strong>Volatility:</strong> {volatility_pct:.1f}% of mean value</p>
                        <p><strong>Interpretation:</strong> Your strategy shows {risk_level.split()[1].lower()} sensitivity to changes in {param1} and {param2}.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Sensitivity Analysis Error</h4>
                    <p>Error generating sensitivity analysis: {str(e)}</p>
                    <p><strong>Tip:</strong> Try reducing the grid resolution or check parameter ranges.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Educational content
    st.markdown("---")
    st.markdown('<div class="sub-header">📚 Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("🎓 Advanced Strategy Concepts"):
        st.markdown("""
        <div class="info-box">
            <h4>📚 Professional Strategy Classification</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div>
                    <h5 style="color: #1f77b4;">📈 Directional Strategies</h5>
                    <ul>
                        <li><strong>Long Call/Put:</strong> Unlimited profit potential, limited risk</li>
                        <li><strong>Bull/Bear Spreads:</strong> Limited risk and reward</li>
                        <li><strong>Synthetic Positions:</strong> Replicate stock/futures with options</li>
                        <li><strong>Ratio Spreads:</strong> Unequal number of long/short options</li>
                    </ul>
                    
                    <h5 style="color: #1f77b4;">⚖️ Neutral Strategies</h5>
                    <ul>
                        <li><strong>Straddles/Strangles:</strong> Profit from large moves</li>
                        <li><strong>Iron Condors:</strong> Profit from range-bound markets</li>
                        <li><strong>Butterflies:</strong> Profit from minimal movement</li>
                        <li><strong>Calendar Spreads:</strong> Profit from time decay</li>
                    </ul>
                </div>
                <div>
                    <h5 style="color: #1f77b4;">🌊 Volatility Strategies</h5>
                    <ul>
                        <li><strong>Long Volatility:</strong> Buy straddles/strangles</li>
                        <li><strong>Short Volatility:</strong> Sell premium, collect decay</li>
                        <li><strong>Volatility Arbitrage:</strong> Trade implied vs realized vol</li>
                        <li><strong>Dispersion Trading:</strong> Index vs individual stock vol</li>
                    </ul>
                    
                    <h5 style="color: #1f77b4;">💰 Income Strategies</h5>
                    <ul>
                        <li><strong>Covered Calls:</strong> Generate income on stock holdings</li>
                        <li><strong>Cash-Secured Puts:</strong> Generate income while acquiring stock</li>
                        <li><strong>Iron Butterflies:</strong> High probability income trades</li>
                        <li><strong>Short Strangles:</strong> Collect premium from range-bound moves</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("⚠️ Risk Management Framework"):
        st.markdown("""
        <div class="warning-box">
            <h4>🏛️ Institutional Risk Management Framework</h4>
            
            <h5>📊 Position Sizing Guidelines</h5>
            <ul>
                <li><strong>Risk Capital:</strong> Never risk more than 1-2% of capital per trade</li>
                <li><strong>Correlation:</strong> Account for correlation between positions</li>
                <li><strong>Concentration:</strong> Avoid over-concentration in single strategies</li>
                <li><strong>Volatility Scaling:</strong> Size positions inversely to expected volatility</li>
            </ul>
            
            <h5>🎯 Greeks Limits Framework</h5>
            <ul>
                <li><strong>Portfolio Delta:</strong> Maintain within ±10% of portfolio value</li>
                <li><strong>Gamma Limits:</strong> Control convexity exposure for large moves</li>
                <li><strong>Vega Limits:</strong> Manage volatility risk across expiration cycles</li>
                <li><strong>Theta Targets:</strong> Balance time decay income vs risk</li>
            </ul>
            
            <h5>🚨 Stop Loss & Risk Controls</h5>
            <ul>
                <li><strong>Hard Stops:</strong> Automatic position closure at predefined levels</li>
                <li><strong>Profit Targets:</strong> Take profits at 20-50% of maximum potential</li>
                <li><strong>Time Stops:</strong> Close positions before significant theta acceleration</li>
                <li><strong>Volatility Stops:</strong> Exit when implied volatility reaches extremes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

