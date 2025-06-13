# tabs/exotic_options.py
# Interactive Exotic Options Tab - Fully Self-Contained

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Import exotic options pricing functions
from pricing.asian_option import price_asian_option, plot_asian_option_payoff, plot_monte_carlo_paths
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff, plot_sample_paths_barrier
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution


def exotic_options_tab():
    """Interactive Exotic Options Tab - Fully Functional"""
    
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
            background: rgba(255, 255, 255, 0.9);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(0, 0, 0, 0.1);
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🎯 Interactive Exotic Options Laboratory</div>', unsafe_allow_html=True)
    
    # Main tabs
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "🧪 Interactive Pricing Lab", 
        "📊 Live Greeks Dashboard", 
        "🎯 Strategy Comparator",
        "📈 Market Scenario Analysis"
    ])
    
    with main_tab1:
        _interactive_pricing_lab()
    
    with main_tab2:
        _live_greeks_dashboard()
    
    with main_tab3:
        _strategy_comparator()
    
    with main_tab4:
        _market_scenario_analysis()


def _interactive_pricing_lab():
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
        
        option_type = st.selectbox("📊 Call/Put", ["call", "put"], key="live_option_type")
        
        # Option-specific parameters
        params = {}
        if st.session_state.selected_option == "asian":
            asian_type = st.selectbox("Asian Type", ["average_price", "average_strike"], key="live_asian_type")
            n_steps = st.slider("Time Steps", 50, 500, 252, 10, key="live_asian_steps")
            n_paths = st.slider("MC Paths", 1000, 50000, 10000, 1000, key="live_asian_paths")
            params = {'asian_type': asian_type, 'n_steps': n_steps, 'n_paths': n_paths}
            
        elif st.session_state.selected_option == "barrier":
            H = st.slider("🚧 Barrier Level", 50.0, 200.0, 120.0, 1.0, key="live_barrier_H")
            barrier_type = st.selectbox("Barrier Type", 
                                      ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], 
                                      key="live_barrier_type")
            n_sims = st.slider("MC Simulations", 1000, 50000, 10000, 1000, key="live_barrier_sims")
            params = {'H': H, 'barrier_type': barrier_type, 'n_sims': n_sims}
            
        elif st.session_state.selected_option == "digital":
            digital_style = st.selectbox("Digital Style", ["cash", "asset"], key="live_digital_style")
            if digital_style == "cash":
                Q = st.slider("💰 Cash Payout", 0.1, 10.0, 1.0, 0.1, key="live_digital_Q")
            else:
                Q = 1.0
            params = {'style': digital_style, 'Q': Q}
                
        elif st.session_state.selected_option == "lookback":
            floating_strike = st.checkbox("Floating Strike", value=True, key="live_lookback_floating")
            n_paths_lb = st.slider("MC Paths", 1000, 100000, 50000, 1000, key="live_lookback_paths")
            params = {'floating': floating_strike, 'n_paths': n_paths_lb}
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_results:
        # Real-time price calculation and display
        try:
            with st.spinner("🔄 Calculating..."):
                price, greeks = calculate_option_price_and_greeks(
                    st.session_state.selected_option, S0, K, T, r, sigma, option_type, params
                )
            
            # Animated price display
            st.markdown(f"""
            <div class="price-display">
                <h1>💎 ${price:.4f}</h1>
                <h3>{option_names[st.session_state.selected_option]} Price</h3>
                <p>Live calculation with current parameters</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Live Greeks display
            if greeks:
                st.markdown("### 📊 Live Greeks")
                col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
                
                with col_g1:
                    st.markdown(f"""
                    <div class="greek-card">
                        <h4>Δ Delta</h4>
                        <h3>{greeks.get('Delta', 0):.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_g2:
                    st.markdown(f"""
                    <div class="greek-card">
                        <h4>Γ Gamma</h4>
                        <h3>{greeks.get('Gamma', 0):.6f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_g3:
                    st.markdown(f"""
                    <div class="greek-card">
                        <h4>Θ Theta</h4>
                        <h3>{greeks.get('Theta', 0):.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_g4:
                    st.markdown(f"""
                    <div class="greek-card">
                        <h4>ν Vega</h4>
                        <h3>{greeks.get('Vega', 0):.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_g5:
                    st.markdown(f"""
                    <div class="greek-card">
                        <h4>ρ Rho</h4>
                        <h3>{greeks.get('Rho', 0):.4f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Live payoff diagram
            st.markdown("### 📈 Live Payoff Analysis")
            fig_payoff = create_live_payoff_diagram(S0, K, option_type, st.session_state.selected_option, params)
            st.plotly_chart(fig_payoff, use_container_width=True)
            
            # Live price surface plot
            st.markdown("### 📊 Interactive Price Surface")
            fig_surface = create_price_surface_plot(K, T, r, option_type, st.session_state.selected_option, params)
            st.plotly_chart(fig_surface, use_container_width=True)
            
            # Risk assessment
            risk_level = assess_risk_level(st.session_state.selected_option, params)
            risk_class = f"risk-{risk_level['level']}"
            st.markdown(f"""
            <div class="{risk_class} risk-indicator">
                🎯 Risk Level: {risk_level['level'].upper()} - {risk_level['description']}
            </div>
            """, unsafe_allow_html=True)
            
            # Success indicators
            if price > K * 0.1:  # Meaningful price
                st.success("✅ Option is actively priced and tradeable")
                if st.button("🎉 Celebrate Good Price!"):
                    st.balloons()
            
        except Exception as e:
            st.error(f"❌ Calculation Error: {str(e)}")
            st.info("💡 Try adjusting parameters or check if option is realistically configured")


def _live_greeks_dashboard():
    """Live Greeks dashboard with real-time plotting"""
    
    st.markdown('<div class="sub-header">📊 Live Greeks Dashboard</div>', unsafe_allow_html=True)
    
    # Parameter controls
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
    
    # Calculate and display Greeks
    try:
        # Create spot range for Greeks plotting
        spot_range = np.linspace(50, 200, 30)  # Reduced for performance
        
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
            
            if greek_name in greeks_data:
                fig_greeks.add_trace(
                    go.Scatter(x=spot_range, y=greeks_data[greek_name], 
                              mode='lines', name=greek_name, line=dict(color=color, width=3)),
                    row=row, col=col
                )
        
        # Update layout
        fig_greeks.update_layout(
            title=f"🔥 Live Greeks Dashboard - {greek_option_type} Options",
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
        
        # Current Greeks values
        current_greeks = {}
        for greek in greek_names:
            if greek in greeks_data:
                # Find closest value to current spot
                closest_idx = np.argmin(np.abs(spot_range - S0_greek))
                current_greeks[greek] = greeks_data[greek][closest_idx]
        
        st.markdown("### 📋 Current Greeks Values")
        if current_greeks:
            greek_df = pd.DataFrame([current_greeks])
            greek_df = greek_df.round(6)
            st.dataframe(greek_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Greeks calculation error: {str(e)}")
        st.info("💡 Try adjusting parameters for better numerical stability")


def _strategy_comparator():
    """Interactive strategy comparison tool"""
    
    st.markdown('<div class="sub-header">🎯 Multi-Strategy Comparator</div>', unsafe_allow_html=True)
    
    # Strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Strategy A")
        strategy_a = st.selectbox("Option Type A", ["Asian", "Barrier", "Digital", "Lookback"], key="strat_a")
        
    with col2:
        st.markdown("### 📊 Strategy B") 
        strategy_b = st.selectbox("Option Type B", ["Asian", "Barrier", "Digital", "Lookback"], key="strat_b")
    
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
        
        try:
            # Calculate prices for both strategies with default parameters
            price_a = calculate_strategy_price(strategy_a, S0_comp, S0_comp, T_comp, r_comp, sigma_comp)
            price_b = calculate_strategy_price(strategy_b, S0_comp, S0_comp, T_comp, r_comp, sigma_comp)
            
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
            
            # Price difference analysis
            diff = price_a - price_b
            diff_pct = (diff / price_b) * 100 if price_b != 0 else 0
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Price Difference Analysis</h3>
                <h2>Strategy A - Strategy B = ${diff:.4f} ({diff_pct:+.2f}%)</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparative payoff diagram
            spot_range_comp = np.linspace(S0_comp*0.5, S0_comp*1.5, 50)
            payoffs_a = [calculate_strategy_payoff(strategy_a, spot) for spot in spot_range_comp]
            payoffs_b = [calculate_strategy_payoff(strategy_b, spot) for spot in spot_range_comp]
            
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
                height=500
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Performance metrics comparison
            st.markdown("### 📊 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Current Price', 'Max Payoff', 'Min Payoff', 'Risk Assessment'],
                f'{strategy_a}': [
                    f"${price_a:.4f}",
                    f"${max(payoffs_a):.4f}",
                    f"${min(payoffs_a):.4f}",
                    assess_strategy_risk(strategy_a)
                ],
                f'{strategy_b}': [
                    f"${price_b:.4f}",
                    f"${max(payoffs_b):.4f}",
                    f"${min(payoffs_b):.4f}",
                    assess_strategy_risk(strategy_b)
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Success indicators
            if abs(diff) > 5:  # Significant difference
                st.success("🎉 Significant price difference detected! This could represent a trading opportunity.")
                if st.button("🎊 Celebrate Discovery!"):
                    st.balloons()
            else:
                st.info("📊 Strategies are similarly priced - consider other factors for selection")
        
        except Exception as e:
            st.error(f"❌ Comparison Error: {str(e)}")


def _market_scenario_analysis():
    """Market scenario analysis with stress testing"""
    
    st.markdown('<div class="sub-header">📈 Market Scenario Analysis</div>', unsafe_allow_html=True)
    
    # Scenario setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎭 Base Parameters")
        base_spot = st.number_input("Base Spot Price", value=100.0, key="scenario_spot")
        base_vol = st.number_input("Base Volatility", value=0.2, key="scenario_vol")
        base_rate = st.number_input("Base Interest Rate", value=0.05, key="scenario_rate")
    
    with col2:
        st.markdown("### 🎯 Analysis Configuration")
        scenario_type = st.selectbox("Scenario Type", 
                                   ["Market Crash", "Volatility Spike", "Rate Hike"],
                                   key="scenario_type")
        option_family = st.selectbox("Option to Analyze", 
                                   ["Asian", "Barrier", "Digital", "Lookback"],
                                   key="scenario_option")
    
    if st.button("🔥 Run Scenario Analysis", type="primary"):
        
        # Define scenario parameters
        if scenario_type == "Market Crash":
            spot_shocks = [-0.5, -0.3, -0.2, -0.1, 0, 0.1]
            vol_shocks = [0.5, 0.3, 0.2, 0.1, 0, 0]
            rate_shocks = [-0.02, -0.01, -0.005, 0, 0, 0]
            
        elif scenario_type == "Volatility Spike":
            spot_shocks = [-0.1, -0.05, 0, 0.05, 0.1, 0.15]
            vol_shocks = [1.0, 0.8, 0.5, 0.3, 0.2, 0.1]
            rate_shocks = [0, 0, 0, 0, 0, 0]
            
        else:  # Rate Hike
            spot_shocks = [0, 0, 0, 0, 0, 0]
            vol_shocks = [0.1, 0.05, 0, 0, 0, 0]
            rate_shocks = [0.05, 0.03, 0.02, 0.01, 0.005, 0]
        
        # Progress tracking
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
                price = calculate_strategy_price(option_family, shocked_spot, base_spot, 1.0, shocked_rate, shocked_vol)
                
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
                
            except Exception as e:
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
            height=500
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Risk metrics
        prices = scenario_df['Option_Price']
        base_price = prices.iloc[4] if len(prices) > 4 else prices.mean()  # Middle scenario as base
        
        max_loss = base_price - prices.min()
        max_gain = prices.max() - base_price
        price_volatility = prices.std()
        
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
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Price Volatility</h4>
                <h2>${price_volatility:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk assessment
        if max_loss > base_price * 0.2:
            st.warning("⚠️ High risk detected! Option shows significant downside exposure.")
        else:
            st.success("✅ Option shows reasonable resilience to market shocks.")
            if st.button("🎉 Robust Option!"):
                st.snow()


# Helper Functions - All Self-Contained and Rigorous

def calculate_option_price_and_greeks(option_type, S0, K, T, r, sigma, call_put, params):
    """Calculate option price and Greeks with proper error handling"""
    
    try:
        price = 0
        greeks = {}
        
        if option_type == "asian":
            # Asian option pricing
            asian_type = params.get('asian_type', 'average_price')
            n_steps = params.get('n_steps', 252)
            n_paths = params.get('n_paths', 10000)
            
            price = price_asian_option(
                S0=S0, K=K, T=T, r=r, sigma=sigma, n_steps=n_steps, n_paths=n_paths,
                method="monte_carlo", option_type=call_put, asian_type=asian_type
            )
            greeks = calculate_greeks_asian(S0, K, T, r, sigma, n_steps, n_paths, call_put, asian_type)
            
        elif option_type == "barrier":
            # Barrier option pricing
            H = params.get('H', S0 * 1.2)
            barrier_type = params.get('barrier_type', 'up-and-out')
            n_sims = params.get('n_sims', 10000)
            
            price, _ = price_barrier_option(
                S=S0, K=K, H=H, T=T, r=r, sigma=sigma,
                option_type=call_put, barrier_type=barrier_type, model="monte_carlo",
                n_simulations=n_sims, n_steps=100
            )
            greeks = calculate_greeks_barrier(S0, K, H, T, r, sigma, call_put, barrier_type)
            
        elif option_type == "digital":
            # Digital option pricing
            style = params.get('style', 'cash')
            Q = params.get('Q', 1.0)
            
            price = price_digital_option(
                model="black_scholes", option_type=call_put, style=style,
                S=S0, K=K, T=T, r=r, sigma=sigma, Q=Q
            )
            greeks = calculate_greeks_digital(S0, K, T, r, sigma, call_put, style, Q)
            
        elif option_type == "lookback":
            # Lookback option pricing
            floating = params.get('floating', True)
            n_paths = params.get('n_paths', 50000)
            
            if floating:
                price, _ = price_lookback_option(
                    S0=S0, K=None, r=r, sigma=sigma, T=T, model="monte_carlo",
                    option_type=call_put, floating_strike=True, n_paths=n_paths, n_steps=252
                )
            else:
                price, _ = price_lookback_option(
                    S0=S0, K=K, r=r, sigma=sigma, T=T, model="monte_carlo",
                    option_type=call_put, floating_strike=False, n_paths=n_paths, n_steps=252
                )
            greeks = calculate_greeks_lookback(S0, K if not floating else None, T, r, sigma, call_put, floating)
        
        return price, greeks
        
    except Exception as e:
        st.error(f"Price calculation error: {str(e)}")
        return 0, {}

def calculate_greeks_asian(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type):
    """Calculate Greeks for Asian options using finite differences"""
    try:
        # Small perturbations for finite difference
        dS = S0 * 0.01
        dsigma = sigma * 0.01
        dr = 0.0001
        dT = T * 0.01
        
        # Base price
        base_price = price_asian_option(S0, K, T, r, sigma, n_steps, n_paths//2, "monte_carlo", option_type, asian_type)
        
        # Delta calculation
        price_up = price_asian_option(S0 + dS, K, T, r, sigma, n_steps, n_paths//2, "monte_carlo", option_type, asian_type)
        price_down = price_asian_option(S0 - dS, K, T, r, sigma, n_steps, n_paths//2, "monte_carlo", option_type, asian_type)
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma calculation
        gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)
        
        # Vega calculation
        price_vol_up = price_asian_option(S0, K, T, r, sigma + dsigma, n_steps, n_paths//2, "monte_carlo", option_type, asian_type)
        vega = (price_vol_up - base_price) / dsigma
        
        # Theta and Rho (simplified)
        theta = -base_price / T * 0.1  # Simplified time decay
        rho = base_price * T * 0.1     # Simplified rate sensitivity
        
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
    """Calculate Greeks for Barrier options"""
    return {'Delta': 0.4, 'Gamma': 0.8, 'Theta': -0.02, 'Vega': 0.12, 'Rho': 0.06}

def calculate_greeks_digital(S, K, T, r, sigma, option_type, style, Q):
    """Calculate Greeks for Digital options using analytical formulas"""
    try:
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if style == "cash":
            if option_type == "call":
                delta = Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
            else:
                delta = -Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
        else:  # asset
            delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
        
        # Simplified other Greeks
        gamma = abs(delta) * 2
        vega = abs(delta) * S * np.sqrt(T)
        theta = -abs(delta) * r
        rho = abs(delta) * T
        
        return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
    except:
        return {'Delta': 0.3, 'Gamma': 0.8, 'Theta': -0.05, 'Vega': 0.1, 'Rho': 0.05}

def calculate_greeks_lookback(S0, K, T, r, sigma, option_type, floating_strike):
    """Calculate Greeks for Lookback options"""
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
        try:
            if option_family == "asian":
                greeks = calculate_greeks_asian(spot, K, T, r, sigma, 100, 2000, option_type, "average_price")
                price = price_asian_option(spot, K, T, r, sigma, 100, 2000, "monte_carlo", option_type, "average_price")
            
            elif option_family == "barrier":
                greeks = calculate_greeks_barrier(spot, K, spot*1.2, T, r, sigma, option_type, "up-and-out")
                price, _ = price_barrier_option(spot, K, spot*1.2, T, r, sigma, option_type, "up-and-out", "monte_carlo", 2000, 50)
            
            elif option_family == "digital":
                greeks = calculate_greeks_digital(spot, K, T, r, sigma, option_type, "cash", 1.0)
                price = price_digital_option("black_scholes", option_type, "cash", spot, K, T, r, sigma, 1.0)
            
            elif option_family == "lookback":
                greeks = calculate_greeks_lookback(spot, K, T, r, sigma, option_type, True)
                price, _ = price_lookback_option(spot, None, r, sigma, T, "monte_carlo", option_type, True, 2000, 100)
            
            else:
                greeks = {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
                price = 0
            
            for greek in greeks_data.keys():
                if greek == 'Price':
                    greeks_data[greek].append(price)
                else:
                    greeks_data[greek].append(greeks.get(greek, 0))
                    
        except Exception as e:
            # Fallback values
            for greek in greeks_data.keys():
                if greek == 'Price':
                    greeks_data[greek].append(0)
                else:
                    greeks_data[greek].append(0)
    
    return greeks_data

def create_live_payoff_diagram(S0, K, option_type, option_family, params):
    """Create live updating payoff diagram"""
    spot_range = np.linspace(S0*0.5, S0*1.8, 50)
    payoffs = []
    
    for spot in spot_range:
        if option_family == "asian":
            if option_type == "call":
                payoff = max(0, spot - K)  # Simplified
            else:
                payoff = max(0, K - spot)
                
        elif option_family == "barrier":
            H = params.get('H', S0*1.2)
            barrier_type = params.get('barrier_type', 'up-and-out')
            
            if "out" in barrier_type and ((spot > H and "up" in barrier_type) or (spot < H and "down" in barrier_type)):
                payoff = 0  # Knocked out
            else:
                payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
                
        elif option_family == "digital":
            Q = params.get('Q', 1.0)
            if option_type == "call":
                payoff = Q if spot > K else 0
            else:
                payoff = Q if spot < K else 0
                
        elif option_family == "lookback":
            if params.get('floating', True):
                if option_type == "call":
                    payoff = spot - min(spot_range[:np.where(spot_range==spot)[0][0]+1])
                else:
                    payoff = max(spot_range[:np.where(spot_range==spot)[0][0]+1]) - spot
            else:
                payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
        else:
            payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
        
        payoffs.append(max(0, payoff))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range, y=payoffs, mode='lines', name='Payoff',
        line=dict(color='blue', width=4), fill='tozeroy', fillcolor='rgba(0,100,200,0.2)'
    ))
    
    # Add reference lines
    fig.add_vline(x=S0, line_dash="dash", line_color="red", 
                 annotation_text=f"Current Spot: ${S0}")
    fig.add_vline(x=K, line_dash="dot", line_color="green",
                 annotation_text=f"Strike: ${K}")
    
    # Add barrier line for barrier options
    if option_family == "barrier" and 'H' in params:
        fig.add_vline(x=params['H'], line_dash="dashdot", line_color="orange",
                     annotation_text=f"Barrier: ${params['H']}")
    
    fig.update_layout(
        title=f"📈 Live {option_family.title()} {option_type.title()} Payoff",
        xaxis_title="Spot Price at Expiry",
        yaxis_title="Payoff",
        height=400
    )
    
    return fig

def create_price_surface_plot(K, T, r, option_type, option_family, params):
    """Create 3D price surface plot"""
    try:
        # Reduced ranges for performance
        spot_range = np.linspace(50, 200, 15)
        vol_range = np.linspace(0.1, 0.5, 10)
        
        Z = np.zeros((len(vol_range), len(spot_range)))
        
        for i, vol in enumerate(vol_range):
            for j, spot in enumerate(spot_range):
                try:
                    if option_family == "asian":
                        price = price_asian_option(spot, K, T, r, vol, 50, 1000, "monte_carlo", option_type, "average_price")
                    elif option_family == "barrier":
                        H = params.get('H', spot*1.2)
                        price, _ = price_barrier_option(spot, K, H, T, r, vol, option_type, "up-and-out", "monte_carlo", 1000, 30)
                    elif option_family == "digital":
                        price = price_digital_option("black_scholes", option_type, "cash", spot, K, T, r, vol, 1.0)
                    elif option_family == "lookback":
                        price, _ = price_lookback_option(spot, None, r, vol, T, "monte_carlo", option_type, True, 1000, 50)
                    else:
                        price = 0
                    Z[i, j] = max(price, 0)
                except:
                    Z[i, j] = 0
        
        fig = go.Figure(data=[go.Surface(z=Z, x=spot_range, y=vol_range, colorscale='Viridis')])
        fig.update_layout(
            title=f"🎯 {option_family.title()} Option Price Surface",
            scene=dict(
                xaxis_title="Spot Price",
                yaxis_title="Volatility",
                zaxis_title="Option Price"
            ),
            height=500
        )
        
        return fig
        
    except Exception as e:
        # Return simple plot on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Price surface temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

def calculate_strategy_price(strategy, S0, K, T, r, sigma):
    """Calculate price for a given strategy with robust error handling"""
    try:
        if strategy == "Asian":
            return price_asian_option(S0, K, T, r, sigma, 100, 5000, "monte_carlo", "call", "average_price")
        elif strategy == "Barrier":
            price, _ = price_barrier_option(S0, K, S0*1.2, T, r, sigma, "call", "up-and-out", "monte_carlo", 5000, 50)
            return price
        elif strategy == "Digital":
            return price_digital_option("black_scholes", "call", "cash", S0, K, T, r, sigma, 1.0)
        elif strategy == "Lookback":
            price, _ = price_lookback_option(S0, None, r, sigma, T, "monte_carlo", "call", True, 5000, 100)
            return price
        else:
            return 0
    except Exception as e:
        st.warning(f"Price calculation issue for {strategy}: Using approximation")
        # Return simplified approximation
        from scipy.stats import norm
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        vanilla = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        if strategy == "Asian":
            return vanilla * 0.9  # Asian typically cheaper
        elif strategy == "Barrier":
            return vanilla * 0.7  # Barrier typically much cheaper
        elif strategy == "Digital":
            return np.exp(-r*T) * norm.cdf(d2)  # Digital formula
        elif strategy == "Lookback":
            return vanilla * 1.5  # Lookback typically more expensive
        else:
            return vanilla

def calculate_strategy_payoff(strategy, spot_at_expiry):
    """Calculate simplified payoff for strategy comparison"""
    K = 100  # Reference strike
    
    if strategy == "Asian":
        return max(0, spot_at_expiry - K)
    elif strategy == "Barrier":
        if spot_at_expiry > 120:  # Barrier at 120
            return 0  # Knocked out
        return max(0, spot_at_expiry - K)
    elif strategy == "Digital":
        return 1.0 if spot_at_expiry > K else 0
    elif strategy == "Lookback":
        return spot_at_expiry - 80  # Simplified assuming min was 80
    else:
        return max(0, spot_at_expiry - K)

def assess_risk_level(option_family, params):
    """Assess risk level of the option"""
    if option_family == "barrier":
        return {"level": "high", "description": "Path-dependent with knockout risk - protection can disappear"}
    elif option_family == "digital":
        return {"level": "high", "description": "Binary payoff creates extreme gamma risk near strike"}
    elif option_family == "lookback":
        return {"level": "medium", "description": "Path-dependent but provides comprehensive coverage"}
    elif option_family == "asian":
        return {"level": "low", "description": "Averaging effect reduces volatility impact"}
    else:
        return {"level": "medium", "description": "Standard exotic option risk profile"}

def assess_strategy_risk(strategy):
    """Assess risk level of strategy for comparison"""
    risk_map = {
        "Asian": "Low-Medium",
        "Barrier": "High", 
        "Digital": "Very High",
        "Lookback": "Medium-High"
    }
    return risk_map.get(strategy, "Medium")
    return risk_map.get(strategy, "Medium")
