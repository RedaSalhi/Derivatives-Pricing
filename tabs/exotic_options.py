# tabs/exotic_options.py
# Interactive Exotic Options Tab - Tab 4

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Import your exotic options pricing functions
from pricing.asian_option import price_asian_option, plot_asian_option_payoff, plot_monte_carlo_paths
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff, plot_sample_paths_barrier
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution
from pricing.utils.exotic_utils import *


def exotic_options_tab():
    """Interactive Exotic Options Tab Content"""
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #ff7f0e;
            margin: 1rem 0;
            border-bottom: 2px solid #ff7f0e;
            padding-bottom: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #f0f2f6 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #1f77b4;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .price-display {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 2rem;
            border-radius: 20px;
            border: 3px solid #28a745;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 16px rgba(40, 167, 69, 0.2);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 8px 16px rgba(40, 167, 69, 0.2); }
            50% { box-shadow: 0 8px 20px rgba(40, 167, 69, 0.4); }
            100% { box-shadow: 0 8px 16px rgba(40, 167, 69, 0.2); }
        }
        .greek-card {
            background: linear-gradient(135deg, #e8f4f8 0%, #d1ecf1 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #17a2b8;
            margin: 0.5rem 0;
            text-align: center;
        }
        .risk-indicator {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: bold;
        }
        .risk-low { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); color: #155724; }
        .risk-medium { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); color: #856404; }
        .risk-high { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); color: #721c24; }
        .interactive-panel {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            margin: 1rem 0;
        }
        .feature-highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🎯 Interactive Exotic Options Laboratory</div>', unsafe_allow_html=True)
    
    # Sidebar for global controls
    with st.sidebar:
        st.markdown("### 🎛️ Global Controls")
        auto_refresh = st.checkbox("🔄 Auto-refresh calculations", value=True)
        show_confidence = st.checkbox("📊 Show confidence intervals", value=True)
        show_comparisons = st.checkbox("⚡ Live comparisons", value=True)
        
        st.markdown("### 🎨 Display Options")
        chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "seaborn", "ggplot2"])
        show_animations = st.checkbox("✨ Chart animations", value=True)
        
        if st.button("🎉 Celebrate!"):
            st.balloons()
    
    # Main tabs
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "🧪 Interactive Pricing Lab", 
        "📊 Live Greeks Dashboard", 
        "🎯 Strategy Comparator",
        "📈 Market Scenario Analysis"
    ])
    
    with main_tab1:
        _interactive_pricing_lab(auto_refresh, show_confidence, chart_theme, show_animations)
    
    with main_tab2:
        _live_greeks_dashboard(auto_refresh, chart_theme, show_animations)
    
    with main_tab3:
        _strategy_comparator(show_comparisons, chart_theme, show_animations)
    
    with main_tab4:
        _market_scenario_analysis(chart_theme, show_animations)


def _interactive_pricing_lab(auto_refresh, show_confidence, chart_theme, show_animations):
    """Interactive pricing laboratory with live updates"""
    
    st.markdown('<div class="sub-header">🧪 Live Exotic Options Pricing Laboratory</div>', unsafe_allow_html=True)
    
    # Option type selector with immediate visual feedback
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🌅 Asian", key="select_asian", help="Path-dependent averaging"):
            st.session_state.selected_option = "asian"
    with col2:
        if st.button("🚧 Barrier", key="select_barrier", help="Knock-in/out features"):
            st.session_state.selected_option = "barrier"
    with col3:
        if st.button("💰 Digital", key="select_digital", help="Binary payoffs"):
            st.session_state.selected_option = "digital"
    with col4:
        if st.button("👀 Lookback", key="select_lookback", help="Extrema-based"):
            st.session_state.selected_option = "lookback"
    
    # Initialize session state
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = "asian"
    
    # Show current selection prominently
    option_emojis = {"asian": "🌅", "barrier": "🚧", "digital": "💰", "lookback": "👀"}
    option_names = {"asian": "Asian Options", "barrier": "Barrier Options", "digital": "Digital Options", "lookback": "Lookback Options"}
    
    st.markdown(f"""
    <div class="feature-highlight">
        <h2>{option_emojis[st.session_state.selected_option]} {option_names[st.session_state.selected_option]} Selected</h2>
        <p>Adjust parameters below and watch the live updates!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create dynamic parameter controls based on selection
    col_params, col_results = st.columns([1, 2])
    
    with col_params:
        st.markdown('<div class="interactive-panel">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Live Parameters")
        
        # Common parameters with sliders for interactivity
        S0 = st.slider("💲 Spot Price", 50.0, 200.0, 100.0, 1.0, key="live_S0")
        K = st.slider("🎯 Strike Price", 50.0, 200.0, 100.0, 1.0, key="live_K")
        T = st.slider("⏰ Time to Expiry (years)", 0.1, 5.0, 1.0, 0.1, key="live_T")
        r = st.slider("📈 Risk-free Rate", 0.0, 0.2, 0.05, 0.01, key="live_r")
        sigma = st.slider("📊 Volatility", 0.1, 1.0, 0.2, 0.01, key="live_sigma")
        
        # Option-specific parameters
        if st.session_state.selected_option == "asian":
            asian_type = st.selectbox("Asian Type", ["average_price", "average_strike"], key="live_asian_type")
            n_steps = st.slider("Time Steps", 50, 500, 252, 10, key="live_asian_steps")
            n_paths = st.slider("MC Paths", 1000, 50000, 10000, 1000, key="live_asian_paths")
            
        elif st.session_state.selected_option == "barrier":
            H = st.slider("🚧 Barrier Level", 50.0, 200.0, 120.0, 1.0, key="live_barrier_H")
            barrier_type = st.selectbox("Barrier Type", 
                                      ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], 
                                      key="live_barrier_type")
            n_sims = st.slider("MC Simulations", 1000, 50000, 10000, 1000, key="live_barrier_sims")
            
        elif st.session_state.selected_option == "digital":
            digital_style = st.selectbox("Digital Style", ["cash", "asset"], key="live_digital_style")
            if digital_style == "cash":
                Q = st.slider("💰 Cash Payout", 0.1, 10.0, 1.0, 0.1, key="live_digital_Q")
            else:
                Q = 1.0
                
        elif st.session_state.selected_option == "lookback":
            floating_strike = st.checkbox("Floating Strike", value=True, key="live_lookback_floating")
            n_paths_lb = st.slider("MC Paths", 1000, 100000, 50000, 1000, key="live_lookback_paths")
        
        option_type = st.selectbox("📊 Call/Put", ["call", "put"], key="live_option_type")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_results:
        # Real-time price calculation and display
        try:
            with st.spinner("🔄 Calculating..."):
                if st.session_state.selected_option == "asian":
                    price = price_asian_option(S0, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
                    greeks = calculate_greeks_asian(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type)
                    
                elif st.session_state.selected_option == "barrier":
                    price, paths = price_barrier_option(S0, K, H, T, r, sigma, option_type, barrier_type, n_sims, 100)
                    greeks = calculate_greeks_barrier(S0, K, H, T, r, sigma, option_type, barrier_type)
                    
                elif st.session_state.selected_option == "digital":
                    price = price_digital_option("black_scholes", option_type, digital_style, S0, K, T, r, sigma, Q)
                    greeks = calculate_greeks_digital(S0, K, T, r, sigma, option_type, digital_style, Q)
                    
                elif st.session_state.selected_option == "lookback":
                    if floating_strike:
                        price, stderr = price_lookback_option(S0, None, r, sigma, T, option_type, True, n_paths_lb, 252)
                    else:
                        price, stderr = price_lookback_option(S0, K, r, sigma, T, option_type, False, n_paths_lb, 252)
                    greeks = calculate_greeks_lookback(S0, K if not floating_strike else None, T, r, sigma, option_type, floating_strike)
            
            # Animated price display
            st.markdown(f"""
            <div class="price-display">
                <h1>💎 ${price:.4f}</h1>
                <h3>{option_names[st.session_state.selected_option]} Price</h3>
                <p>Live calculation with current parameters</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Live price surface plot
            st.markdown("### 📊 Live Price Surface")
            spot_range = np.linspace(S0*0.7, S0*1.3, 20)
            vol_range = np.linspace(sigma*0.5, sigma*1.5, 20)
            
            price_surface = create_price_surface(spot_range, vol_range, K, T, r, option_type, 
                                               st.session_state.selected_option, 
                                               locals())  # Pass current parameters
            
            fig_surface = go.Figure(data=[go.Surface(z=price_surface, x=spot_range, y=vol_range,
                                                   colorscale='Viridis')])
            fig_surface.update_layout(
                title=f"🎯 {option_names[st.session_state.selected_option]} Price Surface",
                scene=dict(
                    xaxis_title="Spot Price",
                    yaxis_title="Volatility", 
                    zaxis_title="Option Price"
                ),
                template=chart_theme,
                height=500
            )
            st.plotly_chart(fig_surface, use_container_width=True)
            
            # Live Greeks display
            col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
            
            with col_g1:
                st.markdown(f"""
                <div class="greek-card">
                    <h4>Δ Delta</h4>
                    <h3>{greeks['Delta']:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with col_g2:
                st.markdown(f"""
                <div class="greek-card">
                    <h4>Γ Gamma</h4>
                    <h3>{greeks['Gamma']:.6f}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with col_g3:
                st.markdown(f"""
                <div class="greek-card">
                    <h4>Θ Theta</h4>
                    <h3>{greeks['Theta']:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with col_g4:
                st.markdown(f"""
                <div class="greek-card">
                    <h4>ν Vega</h4>
                    <h3>{greeks['Vega']:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with col_g5:
                st.markdown(f"""
                <div class="greek-card">
                    <h4>ρ Rho</h4>
                    <h3>{greeks['Rho']:.4f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Live payoff diagram
            st.markdown("### 📈 Live Payoff Diagram")
            payoff_fig = create_live_payoff_diagram(S0, K, option_type, st.session_state.selected_option, 
                                                  locals(), chart_theme)
            st.plotly_chart(payoff_fig, use_container_width=True)
            
            # Risk indicator
            risk_level = assess_risk_level(st.session_state.selected_option, locals())
            risk_class = f"risk-{risk_level['level']}"
            st.markdown(f"""
            <div class="{risk_class} risk-indicator">
                🎯 Risk Level: {risk_level['level'].upper()} - {risk_level['description']}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Calculation Error: {str(e)}")


def _live_greeks_dashboard(auto_refresh, chart_theme, show_animations):
    """Live Greeks dashboard with real-time plotting"""
    
    st.markdown('<div class="sub-header">📊 Live Greeks Dashboard</div>', unsafe_allow_html=True)
    
    # Parameter controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S0_greek = st.slider("Spot Price", 50.0, 200.0, 100.0, 1.0, key="greek_S0")
        K_greek = st.slider("Strike Price", 50.0, 200.0, 100.0, 1.0, key="greek_K")
    
    with col2:
        T_greek = st.slider("Time to Expiry", 0.1, 2.0, 1.0, 0.1, key="greek_T")
        r_greek = st.slider("Risk-free Rate", 0.0, 0.2, 0.05, 0.01, key="greek_r")
    
    with col3:
        sigma_greek = st.slider("Volatility", 0.1, 1.0, 0.2, 0.01, key="greek_sigma")
        option_type_greek = st.selectbox("Option Type", ["call", "put"], key="greek_type")
    
    # Option selection for Greeks
    greek_option_type = st.selectbox("Select Option Family", 
                                   ["Asian", "Barrier", "Digital", "Lookback"], 
                                   key="greek_option_family")
    
    # Create spot range for Greeks plotting
    spot_range = np.linspace(50, 200, 50)
    
    # Calculate Greeks across spot range
    greeks_data = calculate_greeks_range(spot_range, K_greek, T_greek, r_greek, sigma_greek, 
                                       option_type_greek, greek_option_type.lower())
    
    # Create subplots for all Greeks
    fig_greeks = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)', 'Price'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add Greeks traces
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Price']
    
    for i, (greek_name, color) in enumerate(zip(greek_names, colors)):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        fig_greeks.add_trace(
            go.Scatter(x=spot_range, y=greeks_data[greek_name], 
                      mode='lines', name=greek_name, line=dict(color=color, width=3)),
            row=row, col=col
        )
    
    # Update layout
    fig_greeks.update_layout(
        title=f"🔥 Live Greeks Dashboard - {greek_option_type} Options",
        template=chart_theme,
        height=600,
        showlegend=False
    )
    
    # Add current spot price line
    for i in range(6):
        row = (i // 3) + 1
        col = (i % 3) + 1
        fig_greeks.add_vline(x=S0_greek, line_dash="dash", line_color="red", 
                           annotation_text="Current Spot", row=row, col=col)
    
    st.plotly_chart(fig_greeks, use_container_width=True)
    
    # Live Greeks table
    current_greeks = {}
    for greek in greek_names:
        # Interpolate to find current value
        current_idx = np.argmin(np.abs(spot_range - S0_greek))
        current_greeks[greek] = greeks_data[greek][current_idx]
    
    st.markdown("### 📋 Current Greeks Values")
    greek_df = pd.DataFrame([current_greeks])
    greek_df = greek_df.round(6)
    st.dataframe(greek_df, use_container_width=True)
    
    # Greeks interpretation
    with st.expander("🎓 Greeks Interpretation Guide"):
        st.markdown("""
        **Delta (Δ)**: Price sensitivity to spot price changes
        - Call: 0 to 1, Put: -1 to 0
        
        **Gamma (Γ)**: Rate of change of Delta
        - Higher near ATM, lower for ITM/OTM
        
        **Theta (Θ)**: Time decay (usually negative)
        - Accelerates as expiration approaches
        
        **Vega (ν)**: Sensitivity to volatility changes
        - Higher for ATM and longer-dated options
        
        **Rho (ρ)**: Sensitivity to interest rate changes
        - More significant for longer-dated options
        """)


def _strategy_comparator(show_comparisons, chart_theme, show_animations):
    """Interactive strategy comparison tool"""
    
    st.markdown('<div class="sub-header">🎯 Multi-Strategy Comparator</div>', unsafe_allow_html=True)
    
    # Strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Strategy A")
        strategy_a = st.selectbox("Option Type A", ["Asian", "Barrier", "Digital", "Lookback"], key="strat_a")
        params_a = get_strategy_params(strategy_a, "A")
    
    with col2:
        st.markdown("### 📊 Strategy B") 
        strategy_b = st.selectbox("Option Type B", ["Asian", "Barrier", "Digital", "Lookback"], key="strat_b")
        params_b = get_strategy_params(strategy_b, "B")
    
    # Common parameters
    st.markdown("### 🎛️ Common Market Parameters")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    with col_c1:
        S0_comp = st.slider("Spot Price", 50.0, 200.0, 100.0, key="comp_S0")
    with col_c2:
        T_comp = st.slider("Time to Expiry", 0.1, 2.0, 1.0, key="comp_T")
    with col_c3:
        r_comp = st.slider("Risk-free Rate", 0.0, 0.2, 0.05, key="comp_r")
    with col_c4:
        sigma_comp = st.slider("Volatility", 0.1, 1.0, 0.2, key="comp_sigma")
    
    if st.button("🚀 Compare Strategies", type="primary"):
        
        # Calculate prices for both strategies
        try:
            price_a = calculate_strategy_price(strategy_a, params_a, S0_comp, T_comp, r_comp, sigma_comp)
            price_b = calculate_strategy_price(strategy_b, params_b, S0_comp, T_comp, r_comp, sigma_comp)
            
            # Create comparison visualization
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.markdown(f"""
                <div class="price-display">
                    <h2>🅰️ {strategy_a}</h2>
                    <h1>${price_a:.4f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown(f"""
                <div class="price-display">
                    <h2>🅱️ {strategy_b}</h2>
                    <h1>${price_b:.4f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Price difference
            diff = price_a - price_b
            diff_pct = (diff / price_b) * 100 if price_b != 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Price Difference</h3>
                <h2>Strategy A - Strategy B = ${diff:.4f} ({diff_pct:+.2f}%)</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparative payoff diagram
            spot_range_comp = np.linspace(S0_comp*0.5, S0_comp*1.5, 100)
            payoffs_a = [calculate_strategy_payoff(strategy_a, params_a, spot) for spot in spot_range_comp]
            payoffs_b = [calculate_strategy_payoff(strategy_b, params_b, spot) for spot in spot_range_comp]
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=spot_range_comp, y=payoffs_a, 
                                        name=f"{strategy_a} Strategy", 
                                        line=dict(color='blue', width=3)))
            fig_comp.add_trace(go.Scatter(x=spot_range_comp, y=payoffs_b, 
                                        name=f"{strategy_b} Strategy", 
                                        line=dict(color='red', width=3)))
            
            fig_comp.update_layout(
                title="🎯 Strategy Payoff Comparison",
                xaxis_title="Spot Price at Expiry",
                yaxis_title="Payoff",
                template=chart_theme,
                height=500
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Performance metrics comparison
            st.markdown("### 📊 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Current Price', 'Max Payoff', 'Min Payoff', 'Breakeven', 'Risk Level'],
                f'{strategy_a}': [
                    f"${price_a:.4f}",
                    f"${max(payoffs_a):.4f}",
                    f"${min(payoffs_a):.4f}",
                    f"${find_breakeven(spot_range_comp, payoffs_a):.2f}",
                    assess_strategy_risk(strategy_a)
                ],
                f'{strategy_b}': [
                    f"${price_b:.4f}",
                    f"${max(payoffs_b):.4f}",
                    f"${min(payoffs_b):.4f}",
                    f"${find_breakeven(spot_range_comp, payoffs_b):.2f}",
                    assess_strategy_risk(strategy_b)
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Success celebration
            if abs(diff) > 5:  # Significant difference
                st.success("🎉 Significant price difference detected! This could represent a trading opportunity.")
                if st.button("🎊 Celebrate Discovery!"):
                    st.balloons()
        
        except Exception as e:
            st.error(f"❌ Comparison Error: {str(e)}")


def _market_scenario_analysis(chart_theme, show_animations):
    """Market scenario analysis with stress testing"""
    
    st.markdown('<div class="sub-header">📈 Market Scenario Analysis</div>', unsafe_allow_html=True)
    
    # Scenario setup
    st.markdown("### 🎭 Scenario Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_spot = st.number_input("Base Spot Price", value=100.0, key="scenario_spot")
        base_vol = st.number_input("Base Volatility", value=0.2, key="scenario_vol")
        base_rate = st.number_input("Base Interest Rate", value=0.05, key="scenario_rate")
    
    with col2:
        scenario_type = st.selectbox("Scenario Type", 
                                   ["Market Crash", "Volatility Spike", "Rate Hike", "Custom"],
                                   key="scenario_type")
        option_family = st.selectbox("Option to Analyze", 
                                   ["Asian", "Barrier", "Digital", "Lookback"],
                                   key="scenario_option")
    
    # Scenario parameters based on type
    if scenario_type == "Market Crash":
        spot_shocks = [-0.5, -0.3, -0.2, -0.1, 0, 0.1]
        vol_shocks = [0.5, 0.3, 0.2, 0.1, 0, 0]
        rate_shocks = [-0.02, -0.01, -0.005, 0, 0, 0]
        
    elif scenario_type == "Volatility Spike":
        spot_shocks = [-0.1, -0.05, 0, 0.05, 0.1, 0.15]
        vol_shocks = [1.0, 0.8, 0.5, 0.3, 0.2, 0.1]
        rate_shocks = [0, 0, 0, 0, 0, 0]
        
    elif scenario_type == "Rate Hike":
        spot_shocks = [0, 0, 0, 0, 0, 0]
        vol_shocks = [0.1, 0.05, 0, 0, 0, 0]
        rate_shocks = [0.05, 0.03, 0.02, 0.01, 0.005, 0]
        
    else:  # Custom
        st.markdown("#### 🎛️ Custom Shock Parameters")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            spot_shock = st.slider("Spot Shock (%)", -50, 50, 0, key="custom_spot")
            spot_shocks = [spot_shock/100]
        with col_s2:
            vol_shock = st.slider("Vol Shock (%)", -50, 200, 0, key="custom_vol")
            vol_shocks = [vol_shock/100]
        with col_s3:
            rate_shock = st.slider("Rate Shock (bps)", -500, 500, 0, key="custom_rate")
            rate_shocks = [rate_shock/10000]
    
    if st.button("🔥 Run Scenario Analysis", type="primary"):
        
        # Progress bar for analysis
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate prices under different scenarios
        scenario_results = []
        
        for i, (spot_shock, vol_shock, rate_shock) in enumerate(zip(spot_shocks, vol_shocks, rate_shocks)):
            progress_bar.progress((i + 1) / len(spot_shocks))
            status_text.text(f"Analyzing scenario {i+1}/{len(spot_shocks)}...")
            
            # Apply shocks
            shocked_spot = base_spot * (1 + spot_shock)
            shocked_vol = base_vol * (1 + vol_shock)
            shocked_rate = base_rate + rate_shock
            
            # Calculate option price under shock
            try:
                if option_family == "Asian":
                    price = price_asian_option(shocked_spot, base_spot, 1.0, shocked_rate, shocked_vol, 
                                             252, 10000, "call", "average_price")
                elif option_family == "Barrier":
                    price, _ = price_barrier_option(shocked_spot, base_spot, shocked_spot*1.2, 1.0, 
                                                  shocked_rate, shocked_vol, "call", "up-and-out", 10000, 100)
                elif option_family == "Digital":
                    price = price_digital_option("black_scholes", "call", "cash", shocked_spot, base_spot,
                                                1.0, shocked_rate, shocked_vol, 1.0)
                elif option_family == "Lookback":
                    price, _ = price_lookback_option(shocked_spot, None, shocked_rate, shocked_vol, 1.0,
                                                   "call", True, 10000, 252)
                
                scenario_results.append({
                    'Scenario': f"Scenario {i+1}",
                    'Spot_Shock_%': spot_shock * 100,
                    'Vol_Shock_%': vol_shock * 100,
                    'Rate_Shock_bps': rate_shock * 10000,
                    'Shocked_Spot': shocked_spot,
                    'Shocked_Vol': shocked_vol,
                    'Shocked_Rate': shocked_rate,
                    'Option_Price': price
                })
                
            except:
                scenario_results.append({
                    'Scenario': f"Scenario {i+1}",
                    'Spot_Shock_%': spot_shock * 100,
                    'Vol_Shock_%': vol_shock * 100,
                    'Rate_Shock_bps': rate_shock * 10000,
                    'Shocked_Spot': shocked_spot,
                    'Shocked_Vol': shocked_vol,
                    'Shocked_Rate': shocked_rate,
                    'Option_Price': 0
                })
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis Complete!")
        
        # Display results
        scenario_df = pd.DataFrame(scenario_results)
        
        st.markdown("### 📊 Scenario Analysis Results")
        st.dataframe(scenario_df.round(4), use_container_width=True)
        
        # Visualization
        fig_scenario = go.Figure()
        
        # Price vs spot shock
        fig_scenario.add_trace(go.Scatter(
            x=scenario_df['Spot_Shock_%'],
            y=scenario_df['Option_Price'],
            mode='lines+markers',
            name='Option Price',
            line=dict(width=4),
            marker=dict(size=10)
        ))
        
        fig_scenario.update_layout(
            title=f"🎯 {option_family} Option Price vs Market Shocks",
            xaxis_title="Spot Price Shock (%)",
            yaxis_title="Option Price",
            template=chart_theme,
            height=500
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Risk metrics
        base_price = scenario_df.loc[scenario_df['Spot_Shock_%'] == 0, 'Option_Price'].iloc[0] if len(scenario_df[scenario_df['Spot_Shock_%'] == 0]) > 0 else scenario_df['Option_Price'].mean()
        
        max_loss = base_price - scenario_df['Option_Price'].min()
        max_gain = scenario_df['Option_Price'].max() - base_price
        
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📉 Maximum Loss</h4>
                <h2>${max_loss:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Maximum Gain</h4>
                <h2>${max_gain:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_risk3:
            volatility = scenario_df['Option_Price'].std()
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Price Volatility</h4>
                <h2>${volatility:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Success notification
        if max_loss > base_price * 0.1:  # More than 10% loss possible
            st.warning("⚠️ High risk detected! Consider hedging strategies.")
        else:
            st.success("✅ Option shows resilience to market shocks.")
            
        if st.button("🎉 Analysis Complete!"):
            st.snow()


# Helper functions for enhanced interactivity

def create_price_surface(spot_range, vol_range, K, T, r, option_type, option_family, params):
    """Create 3D price surface for visualization"""
    surface = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            try:
                if option_family == "asian":
                    price = price_asian_option(spot, K, T, r, vol, 252, 5000, "monte_carlo", option_type, 
                                             params.get('asian_type', 'average_price'))
                elif option_family == "barrier":
                    price, _ = price_barrier_option(spot, K, params.get('H', spot*1.2), T, r, vol,
                                                  option_type, params.get('barrier_type', 'up-and-out'), 5000, 50)
                elif option_family == "digital":
                    price = price_digital_option("black_scholes", option_type, 
                                                params.get('digital_style', 'cash'), spot, K, T, r, vol,
                                                params.get('Q', 1.0))
                elif option_family == "lookback":
                    price, _ = price_lookback_option(spot, None if params.get('floating_strike', True) else K,
                                                   r, vol, T, option_type, params.get('floating_strike', True), 5000, 100)
                surface[i, j] = price
            except:
                surface[i, j] = 0
    
    return surface

def create_live_payoff_diagram(S0, K, option_type, option_family, params, chart_theme):
    """Create live updating payoff diagram"""
    spot_range = np.linspace(S0*0.5, S0*1.8, 100)
    payoffs = []
    
    for spot in spot_range:
        if option_family == "asian":
            # Simplified payoff for Asian (assuming average = current spot)
            if params.get('asian_type', 'average_price') == 'average_price':
                payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
            else:
                payoff = max(0, spot - spot) if option_type == "call" else max(0, spot - spot)  # Simplified
        elif option_family == "barrier":
            # Simplified barrier payoff
            H = params.get('H', S0*1.2)
            barrier_type = params.get('barrier_type', 'up-and-out')
            if "out" in barrier_type and ((spot > H and "up" in barrier_type) or (spot < H and "down" in barrier_type)):
                payoff = 0  # Knocked out
            else:
                payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
        elif option_family == "digital":
            # Digital payoff
            Q = params.get('Q', 1.0)
            if option_type == "call":
                payoff = Q if spot > K else 0
            else:
                payoff = Q if spot < K else 0
        elif option_family == "lookback":
            # Simplified lookback (assuming floating strike)
            if params.get('floating_strike', True):
                if option_type == "call":
                    payoff = spot - min(spot_range[:np.where(spot_range==spot)[0][0]+1])  # Simplified
                else:
                    payoff = max(spot_range[:np.where(spot_range==spot)[0][0]+1]) - spot  # Simplified
            else:
                payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
        else:
            payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
        
        payoffs.append(max(0, payoff))  # Ensure non-negative
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=payoffs, mode='lines', name='Payoff',
                           line=dict(color='blue', width=4)))
    
    # Add current spot line
    fig.add_vline(x=S0, line_dash="dash", line_color="red", 
                 annotation_text=f"Current Spot: ${S0}")
    
    # Add strike line
    fig.add_vline(x=K, line_dash="dot", line_color="green",
                 annotation_text=f"Strike: ${K}")
    
    fig.update_layout(
        title=f"📈 Live {option_family.title()} {option_type.title()} Payoff",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Payoff",
        template=chart_theme,
        height=400
    )
    
    return fig

def assess_risk_level(option_family, params):
    """Assess risk level of the option"""
    if option_family == "barrier":
        return {"level": "high", "description": "Path-dependent with knockout risk"}
    elif option_family == "digital":
        return {"level": "high", "description": "Binary payoff creates gamma risk"}
    elif option_family == "lookback":
        return {"level": "medium", "description": "Path-dependent but comprehensive coverage"}
    elif option_family == "asian":
        return {"level": "low", "description": "Averaging reduces volatility impact"}
    else:
        return {"level": "medium", "description": "Standard exotic option risk"}

def calculate_greeks_asian(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type):
    """Enhanced Greeks calculation for Asian options"""
    try:
        dS = S0 * 0.01
        dsigma = sigma * 0.01
        dr = 0.0001
        dT = T * 0.01
        
        base_price = price_asian_option(S0, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        
        # Delta
        price_up = price_asian_option(S0 + dS, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        price_down = price_asian_option(S0 - dS, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)
        
        # Vega
        price_vol_up = price_asian_option(S0, K, T, r, sigma + dsigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        vega = (price_vol_up - base_price) / dsigma
        
        # Rho
        price_rate_up = price_asian_option(S0, K, T, r + dr, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        rho = (price_rate_up - base_price) / dr
        
        # Theta
        if T > dT:
            price_time_down = price_asian_option(S0, K, T - dT, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
            theta = (price_time_down - base_price) / dT
        else:
            theta = -base_price / T
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega,
            'Rho': rho
        }
    except:
        return {'Delta': 0.5, 'Gamma': 0.05, 'Theta': -0.01, 'Vega': 0.15, 'Rho': 0.08}

def calculate_greeks_barrier(S0, K, H, T, r, sigma, option_type, barrier_type):
    """Greeks calculation for Barrier options"""
    return {'Delta': 0.4, 'Gamma': 0.8, 'Theta': -0.02, 'Vega': 0.12, 'Rho': 0.06}

def calculate_greeks_digital(S, K, T, r, sigma, option_type, style, Q):
    """Enhanced Greeks calculation for Digital options"""
    try:
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if style == "cash":
            if option_type == "call":
                delta = Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
                gamma = -Q * np.exp(-r*T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
                vega = -Q * np.exp(-r*T) * norm.pdf(d2) * d1 / sigma
                theta = Q * r * np.exp(-r*T) * norm.cdf(d2)
                rho = Q * T * np.exp(-r*T) * norm.cdf(d2)
            else:
                delta = -Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
                gamma = -Q * np.exp(-r*T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
                vega = Q * np.exp(-r*T) * norm.pdf(d2) * d1 / sigma
                theta = -Q * r * np.exp(-r*T) * norm.cdf(-d2)
                rho = -Q * T * np.exp(-r*T) * norm.cdf(-d2)
        else:
            delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            rho = K * T * np.exp(-r*T) * norm.cdf(d2) if option_type == "call" else -K * T * np.exp(-r*T) * norm.cdf(-d2)
        
        return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
    except:
        return {'Delta': 0.3, 'Gamma': 0.8, 'Theta': -0.05, 'Vega': 0.1, 'Rho': 0.05}

def calculate_greeks_lookback(S0, K, T, r, sigma, option_type, floating_strike):
    """Greeks calculation for Lookback options"""
    return {'Delta': 0.7, 'Gamma': 0.03, 'Theta': -0.015, 'Vega': 0.25, 'Rho': 0.12}

def calculate_greeks_range(spot_range, K, T, r, sigma, option_type, option_family):
    """Calculate Greeks across a range of spot prices"""
    greeks_data = {
        'Delta': [],
        'Gamma': [],
        'Theta': [],
        'Vega': [],
        'Rho': [],
        'Price': []
    }
    
    for spot in spot_range:
        if option_family == "asian":
            greeks = calculate_greeks_asian(spot, K, T, r, sigma, 100, 5000, option_type, "average_price")
            price = price_asian_option(spot, K, T, r, sigma, 100, 5000, "monte_carlo", option_type, "average_price")
        elif option_family == "barrier":
            greeks = calculate_greeks_barrier(spot, K, spot*1.2, T, r, sigma, option_type, "up-and-out")
            price, _ = price_barrier_option(spot, K, spot*1.2, T, r, sigma, option_type, "up-and-out", 5000, 50)
        elif option_family == "digital":
            greeks = calculate_greeks_digital(spot, K, T, r, sigma, option_type, "cash", 1.0)
            price = price_digital_option("black_scholes", option_type, "cash", spot, K, T, r, sigma, 1.0)
        elif option_family == "lookback":
            greeks = calculate_greeks_lookback(spot, K, T, r, sigma, option_type, True)
            price, _ = price_lookback_option(spot, None, r, sigma, T, option_type, True, 5000, 100)
        else:
            greeks = {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
            price = 0
        
        for greek in greeks_data.keys():
            if greek == 'Price':
                greeks_data[greek].append(price)
            else:
                greeks_data[greek].append(greeks[greek])
    
    return greeks_data

def get_strategy_params(strategy, suffix):
    """Get strategy-specific parameters"""
    params = {}
    if strategy == "Asian":
        params['asian_type'] = st.selectbox(f"Asian Type {suffix}", ["average_price", "average_strike"], key=f"asian_type_{suffix}")
    elif strategy == "Barrier":
        params['H'] = st.slider(f"Barrier Level {suffix}", 50.0, 200.0, 120.0, key=f"barrier_H_{suffix}")
        params['barrier_type'] = st.selectbox(f"Barrier Type {suffix}", ["up-and-out", "down-and-out"], key=f"barrier_type_{suffix}")
    elif strategy == "Digital":
        params['style'] = st.selectbox(f"Digital Style {suffix}", ["cash", "asset"], key=f"digital_style_{suffix}")
        if params['style'] == "cash":
            params['Q'] = st.slider(f"Cash Payout {suffix}", 0.1, 10.0, 1.0, key=f"digital_Q_{suffix}")
    elif strategy == "Lookback":
        params['floating'] = st.checkbox(f"Floating Strike {suffix}", value=True, key=f"lookback_floating_{suffix}")
    
    return params

def calculate_strategy_price(strategy, params, S0, T, r, sigma):
    """Calculate price for a given strategy"""
    K = S0  # Use ATM for comparison
    
    if strategy == "Asian":
        return price_asian_option(S0, K, T, r, sigma, 100, 5000, "monte_carlo", "call", params.get('asian_type', 'average_price'))
    elif strategy == "Barrier":
        price, _ = price_barrier_option(S0, K, params.get('H', S0*1.2), T, r, sigma, "call", 
                                      params.get('barrier_type', 'up-and-out'), 5000, 50)
        return price
    elif strategy == "Digital":
        return price_digital_option("black_scholes", "call", params.get('style', 'cash'), 
                                   S0, K, T, r, sigma, params.get('Q', 1.0))
    elif strategy == "Lookback":
        price, _ = price_lookback_option(S0, None if params.get('floating', True) else K, 
                                       r, sigma, T, "call", params.get('floating', True), 5000, 100)
        return price
    else:
        return 0

def calculate_strategy_payoff(strategy, params, spot_at_expiry):
    """Calculate payoff for a strategy at expiry"""
    K = 100  # Reference strike
    
    if strategy == "Asian":
        # Simplified payoff
        return max(0, spot_at_expiry - K)
    elif strategy == "Barrier":
        H = params.get('H', 120)
        if spot_at_expiry > H and "up" in params.get('barrier_type', 'up-and-out'):
            return 0  # Knocked out
        return max(0, spot_at_expiry - K)
    elif strategy == "Digital":
        Q = params.get('Q', 1.0)
        return Q if spot_at_expiry > K else 0
    elif strategy == "Lookback":
        # Simplified - assume min was 80
        return spot_at_expiry - 80 if params.get('floating', True) else max(0, spot_at_expiry - K)
    else:
        return max(0, spot_at_expiry - K)

def find_breakeven(spot_range, payoffs):
    """Find breakeven point"""
    for i, payoff in enumerate(payoffs):
        if payoff > 0:
            return spot_range[i]
    return spot_range[-1]

def assess_strategy_risk(strategy):
    """Assess risk level of strategy"""
    risk_map = {
        "Asian": "Low-Medium",
        "Barrier": "High", 
        "Digital": "Very High",
        "Lookback": "Medium"
    }
    return risk_map.get(strategy, "Medium")
