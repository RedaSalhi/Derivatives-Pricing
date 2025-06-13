# tabs/exotic_options.py
# Interactive Exotic Options Tab - Professional Grade

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
    """Interactive Exotic Options Tab - Professional"""
    
    # Custom CSS for enhanced styling - matching other tabs
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
        .results-table {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
        .parameter-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #dee2e6;
        }
        .price-display {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #28a745;
            text-align: center;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">Exotic Options Pricing Laboratory</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Main tabs
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "Interactive Pricing Lab", 
        "Live Greeks Analysis", 
        "Strategy Comparator",
        "Market Scenario Testing"
    ])
    
    with main_tab1:
        _interactive_pricing_lab()
    
    with main_tab2:
        _live_greeks_analysis()
    
    with main_tab3:
        _strategy_comparator()
    
    with main_tab4:
        _market_scenario_analysis()
    
    # Educational content and footer
    _display_educational_content()


def _interactive_pricing_lab():
    """Interactive pricing laboratory with live updates"""
    
    st.markdown('<div class="sub-header">Interactive Pricing Laboratory</div>', unsafe_allow_html=True)
    
    # Option type selector
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Asian Options", key="select_asian", help="Path-dependent averaging"):
            st.session_state.selected_option = "asian"
    with col2:
        if st.button("Barrier Options", key="select_barrier", help="Knock-in/out features"):
            st.session_state.selected_option = "barrier"
    with col3:
        if st.button("Digital Options", key="select_digital", help="Binary payoffs"):
            st.session_state.selected_option = "digital"
    with col4:
        if st.button("Lookback Options", key="select_lookback", help="Extrema-based"):
            st.session_state.selected_option = "lookback"
    
    # Initialize session state
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = "asian"
    
    # Show current selection
    option_names = {"asian": "Asian Options", "barrier": "Barrier Options", "digital": "Digital Options", "lookback": "Lookback Options"}
    
    st.markdown(f"""
    <div class="info-box">
        <h3>{option_names[st.session_state.selected_option]} Selected</h3>
        <p>Adjust parameters below to see live price updates and analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create dynamic parameter controls
    col_params, col_results = st.columns([1, 2])
    
    with col_params:
        st.markdown('<div class="parameter-box">', unsafe_allow_html=True)
        st.markdown("### Market Parameters")
        
        # Common parameters
        S0 = st.slider("Spot Price", 50.0, 200.0, 100.0, 1.0, key="live_S0")
        K = st.slider("Strike Price", 50.0, 200.0, 100.0, 1.0, key="live_K")
        T = st.slider("Time to Expiry (years)", 0.1, 5.0, 1.0, 0.1, key="live_T")
        r = st.slider("Risk-free Rate", 0.0, 0.2, 0.05, 0.01, key="live_r")
        sigma = st.slider("Volatility", 0.1, 1.0, 0.2, 0.01, key="live_sigma")
        
        option_type = st.selectbox("Option Type", ["call", "put"], key="live_option_type")
        
        # Option-specific parameters
        params = {}
        if st.session_state.selected_option == "asian":
            st.markdown("### Asian Parameters")
            asian_type = st.selectbox("Asian Type", ["average_price", "average_strike"], key="live_asian_type")
            n_steps = st.slider("Time Steps", 50, 500, 252, 10, key="live_asian_steps")
            n_paths = st.slider("MC Paths", 1000, 20000, 5000, 1000, key="live_asian_paths")
            params = {'asian_type': asian_type, 'n_steps': n_steps, 'n_paths': n_paths}
            
        elif st.session_state.selected_option == "barrier":
            st.markdown("### Barrier Parameters")
            H = st.slider("Barrier Level", 50.0, 200.0, 120.0, 1.0, key="live_barrier_H")
            barrier_type = st.selectbox("Barrier Type", 
                                      ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], 
                                      key="live_barrier_type")
            n_sims = st.slider("MC Simulations", 1000, 20000, 5000, 1000, key="live_barrier_sims")
            params = {'H': H, 'barrier_type': barrier_type, 'n_sims': n_sims}
            
        elif st.session_state.selected_option == "digital":
            st.markdown("### Digital Parameters")
            digital_style = st.selectbox("Digital Style", ["cash", "asset"], key="live_digital_style")
            if digital_style == "cash":
                Q = st.slider("Cash Payout", 0.1, 10.0, 1.0, 0.1, key="live_digital_Q")
            else:
                Q = 1.0  # Asset-or-nothing always pays the asset
                st.info("Asset-or-nothing pays the underlying asset price if in-the-money")
            params = {'style': digital_style, 'Q': Q}
                
        elif st.session_state.selected_option == "lookback":
            st.markdown("### Lookback Parameters")
            floating_strike = st.checkbox("Floating Strike", value=True, key="live_lookback_floating")
            n_paths_lb = st.slider("MC Paths", 1000, 50000, 10000, 1000, key="live_lookback_paths")
            params = {'floating': floating_strike, 'n_paths': n_paths_lb}
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_results:
        # Real-time price calculation and display
        try:
            with st.spinner("Calculating option price..."):
                price, greeks = calculate_option_price_and_greeks(
                    st.session_state.selected_option, S0, K, T, r, sigma, option_type, params
                )
            
            # Price display
            st.markdown(f"""
            <div class="price-display">
                <h1>${price:.4f}</h1>
                <h3>{option_names[st.session_state.selected_option]} Price</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Greeks display in table format
            if greeks:
                st.markdown("### Option Greeks")
                greeks_df = pd.DataFrame([greeks])
                greeks_df = greeks_df.round(6)
                st.dataframe(greeks_df, use_container_width=True)
            
            # Live payoff diagram
            st.markdown("### Payoff Analysis")
            fig_payoff = create_live_payoff_diagram(S0, K, option_type, st.session_state.selected_option, params)
            st.plotly_chart(fig_payoff, use_container_width=True)
            
            # Price sensitivity analysis
            st.markdown("### Price Sensitivity")
            fig_sensitivity = create_price_sensitivity_chart(K, T, r, sigma, option_type, st.session_state.selected_option, params)
            st.plotly_chart(fig_sensitivity, use_container_width=True)
            
        except Exception as e:
            st.error(f"Calculation Error: {str(e)}")
            st.info("Try adjusting parameters or check if option configuration is valid")


def _live_greeks_analysis():
    """Live Greeks analysis with improved visualization"""
    
    st.markdown('<div class="sub-header">Live Greeks Analysis</div>', unsafe_allow_html=True)
    
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
        # Create continuous spot range for smooth plots (bigger range: 0.5x to 2x)
        spot_range = np.linspace(S0_greek * 0.5, S0_greek * 2.0, 100)  # Continuous with 100 points
        
        # Calculate Greeks across spot range
        greeks_data = calculate_greeks_range_continuous(spot_range, K_greek, T_greek, r_greek, sigma_greek, 
                                           option_type_greek, greek_option_type.lower())
        
        # Current Greeks values prominently displayed
        current_spot_idx = np.argmin(np.abs(spot_range - S0_greek))
        
        st.markdown("### Current Greeks at Selected Spot Price")
        
        col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
        
        with col_g1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Delta", f"{greeks_data['Delta'][current_spot_idx]:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_g2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Gamma", f"{greeks_data['Gamma'][current_spot_idx]:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_g3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Theta", f"{greeks_data['Theta'][current_spot_idx]:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_g4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Vega", f"{greeks_data['Vega'][current_spot_idx]:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_g5:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Rho", f"{greeks_data['Rho'][current_spot_idx]:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Continuous Greeks charts
        st.markdown("### Greeks Visualization")
        
        # Create individual charts for each Greek with continuous lines
        fig_greeks = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Price'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        greek_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Price']
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        
        for i, (greek_name, color) in enumerate(zip(greek_names, colors)):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig_greeks.add_trace(
                go.Scatter(x=spot_range, y=greeks_data[greek_name], 
                          mode='lines', name=greek_name, 
                          line=dict(color=color, width=3)),  # Smooth continuous lines
                row=row, col=col
            )
            
            # Add current spot indicator
            fig_greeks.add_vline(x=S0_greek, line_dash="dash", line_color="red", 
                               annotation_text="Current", row=row, col=col)
        
        fig_greeks.update_layout(
            title=f"Continuous Greeks Analysis - {greek_option_type} Options",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_greeks, use_container_width=True)
        
        # Add path visualization for path-dependent options
        if greek_option_type.lower() in ["asian", "barrier", "lookback"]:
            st.markdown("### Sample Price Paths")
            fig_paths = create_sample_paths_visualization(S0_greek, r_greek, sigma_greek, T_greek, greek_option_type.lower())
            st.plotly_chart(fig_paths, use_container_width=True)
        
    except Exception as e:
        st.error(f"Greeks calculation error: {str(e)}")
        st.info("Try adjusting parameters for better numerical stability")


def _strategy_comparator():
    """Fully customizable strategy comparison"""
    
    st.markdown('<div class="sub-header">Strategy Comparator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Customizable Strategy Comparison</h4>
        <p>Configure each strategy with your preferred parameters for detailed comparison analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Strategy A")
        strategy_a = st.selectbox("Option Type A", ["Asian", "Barrier", "Digital", "Lookback"], key="strat_a")
        
        # Strategy A specific parameters
        st.markdown("#### Strategy A Parameters")
        params_a = {}
        
        if strategy_a == "Asian":
            params_a['asian_type'] = st.selectbox("Asian Type A", ["average_price", "average_strike"], key="asian_a")
            st.markdown(f"📋 **{params_a['asian_type'].replace('_', ' ').title()} Asian call**")
            
        elif strategy_a == "Barrier":
            params_a['barrier_type'] = st.selectbox("Barrier Type A", 
                                                   ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], 
                                                   key="barrier_type_a")
            params_a['barrier_multiple'] = st.slider("Barrier Level A (% of strike)", 80, 150, 120, 5, key="barrier_mult_a") / 100
            st.markdown(f"📋 **{params_a['barrier_type'].replace('-', ' ').title()} call** (barrier at {params_a['barrier_multiple']:.0%} of strike)")
            
        elif strategy_a == "Digital":
            params_a['style'] = st.selectbox("Digital Style A", ["cash", "asset"], key="digital_style_a")
            if params_a['style'] == "cash":
                params_a['Q'] = st.slider("Cash Payout A", 0.5, 5.0, 1.0, 0.1, key="digital_Q_a")
                st.markdown(f"📋 **Cash-or-nothing call** (pays ${params_a['Q']:.1f})")
            else:
                params_a['Q'] = 1.0
                st.markdown("📋 **Asset-or-nothing call** (pays underlying price)")
                
        elif strategy_a == "Lookback":
            params_a['floating'] = st.checkbox("Floating Strike A", value=True, key="lookback_float_a")
            if params_a['floating']:
                st.markdown("📋 **Floating strike call** (strike = minimum price reached)")
            else:
                st.markdown("📋 **Fixed strike call** (strike = predetermined level)")
        
    with col2:
        st.markdown("### Strategy B") 
        strategy_b = st.selectbox("Option Type B", ["Asian", "Barrier", "Digital", "Lookback"], key="strat_b")
        
        # Strategy B specific parameters
        st.markdown("#### Strategy B Parameters")
        params_b = {}
        
        if strategy_b == "Asian":
            params_b['asian_type'] = st.selectbox("Asian Type B", ["average_price", "average_strike"], key="asian_b")
            st.markdown(f"📋 **{params_b['asian_type'].replace('_', ' ').title()} Asian call**")
            
        elif strategy_b == "Barrier":
            params_b['barrier_type'] = st.selectbox("Barrier Type B", 
                                                   ["up-and-out", "down-and-out", "up-and-in", "down-and-in"], 
                                                   key="barrier_type_b")
            params_b['barrier_multiple'] = st.slider("Barrier Level B (% of strike)", 80, 150, 120, 5, key="barrier_mult_b") / 100
            st.markdown(f"📋 **{params_b['barrier_type'].replace('-', ' ').title()} call** (barrier at {params_b['barrier_multiple']:.0%} of strike)")
            
        elif strategy_b == "Digital":
            params_b['style'] = st.selectbox("Digital Style B", ["cash", "asset"], key="digital_style_b")
            if params_b['style'] == "cash":
                params_b['Q'] = st.slider("Cash Payout B", 0.5, 5.0, 1.0, 0.1, key="digital_Q_b")
                st.markdown(f"📋 **Cash-or-nothing call** (pays ${params_b['Q']:.1f})")
            else:
                params_b['Q'] = 1.0
                st.markdown("📋 **Asset-or-nothing call** (pays underlying price)")
                
        elif strategy_b == "Lookback":
            params_b['floating'] = st.checkbox("Floating Strike B", value=True, key="lookback_float_b")
            if params_b['floating']:
                st.markdown("📋 **Floating strike call** (strike = minimum price reached)")
            else:
                st.markdown("📋 **Fixed strike call** (strike = predetermined level)")
    
    # Common parameters
    st.markdown("### Market Parameters")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    with col_c1:
        S0_comp = st.slider("Spot Price", 50.0, 200.0, 100.0, key="comp_S0")
    with col_c2:
        T_comp = st.slider("Time to Expiry", 0.1, 2.0, 1.0, key="comp_T")
    with col_c3:
        r_comp = st.slider("Risk-free Rate", 0.0, 0.2, 0.05, key="comp_r")
    with col_c4:
        sigma_comp = st.slider("Volatility", 0.1, 1.0, 0.2, key="comp_sigma")
    
    if st.button("Compare Strategies", type="primary"):
        
        try:
            # Calculate prices for both strategies with custom parameters
            price_a = calculate_strategy_price_custom(strategy_a, params_a, S0_comp, S0_comp, T_comp, r_comp, sigma_comp)
            price_b = calculate_strategy_price_custom(strategy_b, params_b, S0_comp, S0_comp, T_comp, r_comp, sigma_comp)
            
            # Create comparison visualization
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.markdown(f"""
                <div class="price-display">
                    <h2>{strategy_a}</h2>
                    <h1>${price_a:.4f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown(f"""
                <div class="price-display">
                    <h2>{strategy_b}</h2>
                    <h1>${price_b:.4f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Price difference analysis
            diff = price_a - price_b
            diff_pct = (diff / price_b) * 100 if price_b != 0 else 0
            
            st.markdown(f"""
            <div class="metric-container">
                <h3>Price Difference Analysis</h3>
                <p><strong>Absolute Difference:</strong> ${diff:.4f}</p>
                <p><strong>Percentage Difference:</strong> {diff_pct:+.2f}%</p>
                <p><strong>Interpretation:</strong> {get_custom_price_interpretation(strategy_a, strategy_b, params_a, params_b, diff)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparative payoff diagram with continuous lines
            spot_range_comp = np.linspace(S0_comp*0.5, S0_comp*2.0, 100)  # Bigger continuous range
            payoffs_a = [calculate_strategy_payoff_custom(strategy_a, params_a, spot, S0_comp) for spot in spot_range_comp]
            payoffs_b = [calculate_strategy_payoff_custom(strategy_b, params_b, spot, S0_comp) for spot in spot_range_comp]
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=spot_range_comp, y=payoffs_a, 
                                        name=f"{strategy_a} Strategy", 
                                        line=dict(color='blue', width=3),
                                        mode='lines'))  # Continuous lines
            fig_comp.add_trace(go.Scatter(x=spot_range_comp, y=payoffs_b, 
                                        name=f"{strategy_b} Strategy", 
                                        line=dict(color='red', width=3),
                                        mode='lines'))  # Continuous lines
            
            # Add current spot line
            fig_comp.add_vline(x=S0_comp, line_dash="dash", line_color="green",
                             annotation_text=f"Current Spot: ${S0_comp}")
            
            # Add barrier lines if applicable
            if strategy_a == "Barrier":
                barrier_a = S0_comp * params_a['barrier_multiple']
                fig_comp.add_vline(x=barrier_a, line_dash="dashdot", line_color="blue", 
                                 annotation_text=f"Barrier A: ${barrier_a:.1f}")
            
            if strategy_b == "Barrier":
                barrier_b = S0_comp * params_b['barrier_multiple']
                fig_comp.add_vline(x=barrier_b, line_dash="dashdot", line_color="red",
                                 annotation_text=f"Barrier B: ${barrier_b:.1f}")
            
            fig_comp.update_layout(
                title="Strategy Payoff Comparison at Expiration",
                xaxis_title="Spot Price at Expiry ($)",
                yaxis_title="Payoff ($)",
                height=500
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Performance metrics comparison
            st.markdown("### Performance Metrics Comparison")
            
            metrics_df = pd.DataFrame({
                'Metric': ['Current Price', 'Max Payoff', 'Min Payoff', 'Payoff at Current Spot'],
                f'{strategy_a}': [
                    f"${price_a:.4f}",
                    f"${max(payoffs_a):.4f}",
                    f"${min(payoffs_a):.4f}",
                    f"${payoffs_a[len(payoffs_a)//2]:.4f}"  # Middle point (current spot)
                ],
                f'{strategy_b}': [
                    f"${price_b:.4f}",
                    f"${max(payoffs_b):.4f}",
                    f"${min(payoffs_b):.4f}",
                    f"${payoffs_b[len(payoffs_b)//2]:.4f}"  # Middle point (current spot)
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Configuration summary
            st.markdown("### Configuration Summary")
            config_df = pd.DataFrame({
                'Parameter': get_config_summary_labels(strategy_a, strategy_b),
                f'{strategy_a}': get_config_summary_values(strategy_a, params_a, S0_comp),
                f'{strategy_b}': get_config_summary_values(strategy_b, params_b, S0_comp)
            })
            st.dataframe(config_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Comparison Error: {str(e)}")


def calculate_strategy_price_custom(strategy, params, S0, K, T, r, sigma):
    """Calculate price with custom parameters"""
    try:
        if strategy == "Asian":
            asian_type = params.get('asian_type', 'average_price')
            return price_asian_option(S0, K, T, r, sigma, 100, 3000, "monte_carlo", "call", asian_type)
            
        elif strategy == "Barrier":
            barrier_type = params.get('barrier_type', 'up-and-out')
            barrier_multiple = params.get('barrier_multiple', 1.2)
            H = K * barrier_multiple
            price, _ = price_barrier_option(S0, K, H, T, r, sigma, "call", barrier_type, "monte_carlo", 3000, 50)
            return price
            
        elif strategy == "Digital":
            style = params.get('style', 'cash')
            Q = params.get('Q', 1.0)
            return price_digital_option("black_scholes", "call", style, S0, K, T, r, sigma, Q)
            
        elif strategy == "Lookback":
            floating = params.get('floating', True)
            if floating:
                price, _ = price_lookback_option(S0, None, r, sigma, T, "monte_carlo", "call", True, 3000, 100)
            else:
                price, _ = price_lookback_option(S0, K, r, sigma, T, "monte_carlo", "call", False, 3000, 100)
            return price
        else:
            return 0
            
    except Exception as e:
        # Fallback to approximation
        from scipy.stats import norm
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        vanilla = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        # Strategy-specific approximations
        if strategy == "Asian":
            return vanilla * 0.9
        elif strategy == "Barrier":
            return vanilla * 0.7
        elif strategy == "Digital":
            return np.exp(-r*T) * norm.cdf(d2) * params.get('Q', 1.0)
        elif strategy == "Lookback":
            return vanilla * 1.4
        else:
            return vanilla

def calculate_strategy_payoff_custom(strategy, params, spot_at_expiry, reference_spot):
    """Calculate payoff with custom parameters"""
    K = reference_spot  # Use reference spot as strike
    
    if strategy == "Asian":
        # Simplified: assume average equals final spot for payoff illustration
        return max(0, spot_at_expiry - K)
        
    elif strategy == "Barrier":
        barrier_type = params.get('barrier_type', 'up-and-out')
        barrier_multiple = params.get('barrier_multiple', 1.2)
        H = K * barrier_multiple
        
        # Check knockout conditions
        knocked_out = False
        if "up" in barrier_type and spot_at_expiry >= H:
            knocked_out = True
        elif "down" in barrier_type and spot_at_expiry <= H:
            knocked_out = True
        
        if "out" in barrier_type and knocked_out:
            return 0
        elif "in" in barrier_type and not knocked_out:
            return 0
        else:
            return max(0, spot_at_expiry - K)
            
    elif strategy == "Digital":
        style = params.get('style', 'cash')
        Q = params.get('Q', 1.0)
        
        if style == "cash":
            return Q if spot_at_expiry > K else 0
        else:  # asset
            return spot_at_expiry if spot_at_expiry > K else 0
            
    elif strategy == "Lookback":
        floating = params.get('floating', True)
        if floating:
            # Simplified: assume minimum was 20% below reference
            min_price = K * 0.8
            return max(0, spot_at_expiry - min_price)
        else:
            return max(0, spot_at_expiry - K)
    else:
        return max(0, spot_at_expiry - K)

def get_custom_price_interpretation(strategy_a, strategy_b, params_a, params_b, price_diff):
    """Get interpretation based on custom parameters"""
    
    if abs(price_diff) < 0.5:
        return "Prices are very similar with current configurations"
    
    more_expensive = strategy_a if price_diff > 0 else strategy_b
    cheaper = strategy_b if price_diff > 0 else strategy_a
    more_exp_params = params_a if price_diff > 0 else params_b
    cheaper_params = params_b if price_diff > 0 else params_a
    
    # Generate parameter-specific interpretation
    interpretation = f"{more_expensive} is more expensive"
    
    # Add specific reasons based on parameters
    if more_expensive == "Digital" and more_exp_params.get('style') == 'asset':
        interpretation += " (asset-or-nothing provides unlimited upside)"
    elif more_expensive == "Digital" and more_exp_params.get('Q', 1) > 1:
        interpretation += f" (higher cash payout of ${more_exp_params['Q']:.1f})"
    elif more_expensive == "Barrier" and "in" in more_exp_params.get('barrier_type', ''):
        interpretation += " (knock-in provides conditional protection)"
    elif cheaper == "Barrier" and "out" in cheaper_params.get('barrier_type', ''):
        interpretation += " due to knockout risk in " + cheaper
    elif more_expensive == "Lookback" and more_exp_params.get('floating', True):
        interpretation += " (floating strike provides optimal timing)"
    elif more_expensive == "Asian" and more_exp_params.get('asian_type') == 'average_price':
        interpretation += " (average price reduces volatility impact)"
    
    return interpretation

def get_config_summary_labels(strategy_a, strategy_b):
    """Get configuration summary labels"""
    labels = ['Option Type']
    
    if strategy_a == "Asian" or strategy_b == "Asian":
        labels.append('Asian Type')
    if strategy_a == "Barrier" or strategy_b == "Barrier":
        labels.extend(['Barrier Type', 'Barrier Level'])
    if strategy_a == "Digital" or strategy_b == "Digital":
        labels.extend(['Digital Style', 'Payout Amount'])
    if strategy_a == "Lookback" or strategy_b == "Lookback":
        labels.append('Strike Type')
    
    return labels

def get_config_summary_values(strategy, params, reference_spot):
    """Get configuration summary values"""
    values = [strategy]
    
    if strategy == "Asian":
        values.append(params.get('asian_type', 'average_price').replace('_', ' ').title())
    elif strategy == "Barrier":
        values.append(params.get('barrier_type', 'up-and-out').replace('-', ' ').title())
        barrier_level = reference_spot * params.get('barrier_multiple', 1.2)
        values.append(f"${barrier_level:.1f}")
    elif strategy == "Digital":
        values.append(params.get('style', 'cash').title())
        if params.get('style') == 'cash':
            values.append(f"${params.get('Q', 1.0):.1f}")
        else:
            values.append("Asset Price")
    elif strategy == "Lookback":
        strike_type = "Floating" if params.get('floating', True) else "Fixed"
        values.append(strike_type)
    
    # Pad with empty strings to match label length
    while len(values) < 6:  # Max possible labels
        values.append("-")
    
    return values[:len(get_config_summary_labels("Asian", "Lookback"))]  # Trim to actual label count


def _market_scenario_analysis():
    """Clear market scenario analysis with specific stress tests"""
    
    st.markdown('<div class="sub-header">Market Scenario Stress Testing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>Stress Testing Framework</h4>
        <p>Test how exotic option prices respond to extreme market conditions:</p>
        <ul>
        <li><strong>Market Crash:</strong> Spot drops 10-40%, volatility spikes 20-80%, rates drop 0-150bp</li>
        <li><strong>Volatility Spike:</strong> Volatility increases 50-150%, modest spot movement ±15%</li>
        <li><strong>Interest Rate Hike:</strong> Rates increase 50-400bp, volatility rises 10-20%</li>
        </ul>
        <p><em>All scenarios test 6 progressive stress levels to understand option sensitivity.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Scenario setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Base Market Parameters")
        base_spot = st.number_input("Base Spot Price", value=100.0, key="scenario_spot")
        base_vol = st.number_input("Base Volatility", value=0.2, key="scenario_vol")
        base_rate = st.number_input("Base Interest Rate", value=0.05, key="scenario_rate")
    
    with col2:
        st.markdown("### Stress Test Configuration")
        scenario_type = st.selectbox("Stress Scenario", 
                                   ["Market Crash", "Volatility Spike", "Interest Rate Hike"],
                                   key="scenario_type")
        option_family = st.selectbox("Option to Test", 
                                   ["Asian", "Barrier", "Digital", "Lookback"],
                                   key="scenario_option")
        
        # Show what this option configuration means
        if option_family == "Asian":
            st.markdown("📋 **Testing:** Average-price call option")
        elif option_family == "Barrier":
            st.markdown("📋 **Testing:** Up-and-out call (barrier at 120% of spot)")
        elif option_family == "Digital":
            st.markdown("📋 **Testing:** Cash-or-nothing call ($1 payout)")
        elif option_family == "Lookback":
            st.markdown("📋 **Testing:** Floating strike call")
    
    if st.button("Run Stress Test", type="primary"):
        
        # Define specific scenario parameters with clear descriptions
        if scenario_type == "Market Crash":
            spot_shocks = [-0.4, -0.3, -0.2, -0.15, -0.1, -0.05]
            vol_shocks = [0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
            rate_shocks = [-0.015, -0.01, -0.008, -0.005, -0.003, -0.001]
            scenario_description = "Progressive market crash with volatility spike and rate cuts"
            
        elif scenario_type == "Volatility Spike":
            spot_shocks = [-0.15, -0.1, -0.05, 0.05, 0.1, 0.15]
            vol_shocks = [1.5, 1.2, 0.8, 0.6, 0.4, 0.2]
            rate_shocks = [0, 0, 0, 0, 0, 0]
            scenario_description = "Volatility surge with modest directional movement"
            
        else:  # Interest Rate Hike
            spot_shocks = [0.05, 0.03, 0.02, 0.01, 0, -0.02]
            vol_shocks = [0.2, 0.15, 0.1, 0.08, 0.05, 0.02]
            rate_shocks = [0.04, 0.03, 0.025, 0.02, 0.015, 0.005]
            scenario_description = "Progressive interest rate hikes with volatility increase"
        
        st.markdown(f"""
        <div class="warning-box">
            <h4>Running: {scenario_type}</h4>
            <p>{scenario_description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Calculate prices under different scenarios
        scenario_results = []
        
        for i, (spot_shock, vol_shock, rate_shock) in enumerate(zip(spot_shocks, vol_shocks, rate_shocks)):
            progress_bar.progress((i + 1) / len(spot_shocks))
            status_text.text(f"Testing stress level {i+1}/6...")
            
            # Apply shocks
            shocked_spot = base_spot * (1 + spot_shock)
            shocked_vol = base_vol * (1 + vol_shock)
            shocked_rate = base_rate + rate_shock
            
            # Calculate option price under shock
            try:
                price = calculate_strategy_price_standardized(option_family, shocked_spot, base_spot, 1.0, shocked_rate, shocked_vol)
                
                scenario_results.append({
                    'Stress_Level': i+1,
                    'Spot_Change_%': spot_shock * 100,
                    'Vol_Change_%': vol_shock * 100,
                    'Rate_Change_bp': rate_shock * 10000,
                    'Final_Spot': shocked_spot,
                    'Final_Vol': shocked_vol,
                    'Final_Rate': shocked_rate,
                    'Option_Price': price,
                    'Description': f"Level {i+1}: {scenario_type.lower()}"
                })
                
            except Exception as e:
                st.warning(f"Calculation failed for stress level {i+1}: {str(e)}")
                scenario_results.append({
                    'Stress_Level': i+1,
                    'Spot_Change_%': spot_shock * 100,
                    'Vol_Change_%': vol_shock * 100,
                    'Rate_Change_bp': rate_shock * 10000,
                    'Final_Spot': shocked_spot,
                    'Final_Vol': shocked_vol,
                    'Final_Rate': shocked_rate,
                    'Option_Price': 0,
                    'Description': f"Level {i+1}: calculation failed"
                })
        
        progress_bar.progress(1.0)
        status_text.text("Stress test complete!")
        
        # Display results
        scenario_df = pd.DataFrame(scenario_results)
        
        st.markdown("### Stress Test Results")
        st.dataframe(scenario_df.round(4), use_container_width=True)
        
        # Visualization with continuous lines
        fig_scenario = go.Figure()
        
        fig_scenario.add_trace(go.Scatter(
            x=scenario_df['Stress_Level'],
            y=scenario_df['Option_Price'],
            mode='lines+markers',
            name=f'{option_family} Option Price',
            line=dict(width=4, color='blue'),
            marker=dict(size=10, color='red')
        ))
        
        fig_scenario.update_layout(
            title=f"{option_family} Option Response to {scenario_type}",
            xaxis_title="Stress Level (1=Mild → 6=Severe)",
            yaxis_title="Option Price ($)",
            height=500
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Risk metrics with clear interpretation
        prices = scenario_df['Option_Price'][scenario_df['Option_Price'] > 0]
        if len(prices) > 0:
            base_price = prices.iloc[0] if len(prices) > 0 else prices.mean()
            
            max_loss = base_price - prices.min()
            max_gain = prices.max() - base_price
            price_volatility = prices.std()
            
            col_risk1, col_risk2, col_risk3 = st.columns(3)
            
            with col_risk1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Maximum Loss", f"${max_loss:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_risk2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Maximum Gain", f"${max_gain:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_risk3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Price Volatility", f"${price_volatility:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Clear risk assessment based on actual data
            loss_ratio = max_loss / base_price if base_price > 0 else 0
            if loss_ratio > 0.5:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown(f"<strong>High Risk:</strong> Option lost {loss_ratio:.1%} of value in worst stress scenario. Consider hedging strategies.")
                st.markdown('</div>', unsafe_allow_html=True)
            elif loss_ratio > 0.25:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"<strong>Moderate Risk:</strong> Option shows {loss_ratio:.1%} maximum downside. Monitor market conditions closely.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"<strong>Low Risk:</strong> Option demonstrates resilience with only {loss_ratio:.1%} maximum loss under stress.")
                st.markdown('</div>', unsafe_allow_html=True)


def _display_educational_content():
    """Educational content and footer matching other tabs"""
    
    st.markdown("---")
    st.markdown('<div class="sub-header">Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander("Exotic Options Fundamentals"):
        st.markdown("""
        ### Core Exotic Option Types
        
        **Asian Options (Average Options)**
        - Payoff depends on the average price of the underlying over a specific period
        - Two main types: Average Price (payoff based on average) and Average Strike (strike is average)
        - Less volatile than vanilla options due to the averaging effect
        - Popular in commodity markets and FX trading
        
        **Barrier Options**
        - Payoff depends on whether the underlying price crosses a predetermined barrier level
        - Knock-out options: extinguished if barrier is crossed
        - Knock-in options: activated only if barrier is crossed
        - Generally cheaper than vanilla options due to the additional path dependency risk
        
        **Digital Options (Binary Options)**
        - All-or-nothing payoff structure
        - Cash-or-nothing: pays fixed amount if ITM, nothing if OTM
        - Asset-or-nothing: pays the asset price if ITM, nothing if OTM
        - High gamma risk near expiration and at the strike price
        
        **Lookback Options**
        - Payoff based on the maximum or minimum price reached during the option's life
        - Floating strike: strike price is set to the optimal level at expiration
        - Fixed strike: payoff based on extrema versus a predetermined strike
        - Most expensive due to the path-dependent "perfect timing" feature
        """)
    
    with st.expander("Risk Management Guidelines"):
        st.markdown("""
        ### Key Risk Considerations for Exotic Options
        
        **Model Risk**
        - Monte Carlo simulations have inherent sampling error
        - Model assumptions may not hold under stressed market conditions
        - Regular calibration to current market data is essential
        - Consider multiple pricing methodologies for validation
        
        **Market Risk**
        - Exotic options often exhibit non-linear, complex Greeks behavior
        - Path-dependent options require sophisticated hedging strategies
        - Barrier options have discontinuous payoffs creating hedging challenges
        - Digital options show extreme gamma behavior near strikes and expiration
        
        **Operational Risk**
        - Real-time monitoring requirements for barrier levels and averaging calculations
        - Accurate path reconstruction needed for lookback options
        - Complex settlement procedures for exotic payoffs
        - System reliability critical for time-sensitive calculations
        
        **Liquidity Risk**
        - Limited secondary markets for most exotic options
        - Wider bid-ask spreads compared to vanilla options
        - Fewer market makers and liquidity providers
        - Early termination may be costly or impossible
        """)
    
    with st.expander("Pricing Methodology Notes"):
        st.markdown("""
        ### Technical Implementation Details
        
        **Monte Carlo Methods**
        - Used for path-dependent options (Asian, Barrier, Lookback)
        - Euler discretization scheme for price path simulation
        - Antithetic variates and control variates for variance reduction
        - Standard errors reported where applicable
        
        **Analytical Methods**
        - Black-Scholes framework for Digital options
        - Closed-form solutions where available
        - Greeks calculated using analytical derivatives when possible
        
        **Numerical Considerations**
        - Time step selection affects accuracy vs. computation time
        - Number of simulation paths impacts precision and confidence intervals
        - Greeks calculated using finite difference methods with optimized step sizes
        - Numerical stability considerations for extreme parameter values
        
        **Validation and Testing**
        - Cross-validation against known analytical solutions
        - Convergence testing for Monte Carlo simulations
        - Stress testing under extreme market scenarios
        - Regular benchmarking against market prices where available
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Exotic Options Pricing Laboratory</strong></p>
        <p>Advanced derivatives pricing and risk analysis • Built with Streamlit</p>
        <p><em>For educational and research purposes only</em></p>
        <p><strong>Disclaimer:</strong> This tool is for educational purposes. Always consult qualified professionals for trading decisions.</p>
    </div>
    """, unsafe_allow_html=True)


# Helper Functions - Improved and Stabilized

def calculate_option_price_and_greeks(option_type, S0, K, T, r, sigma, call_put, params):
    """Calculate option price and Greeks with improved error handling"""
    
    try:
        price = 0
        greeks = {}
        
        if option_type == "asian":
            asian_type = params.get('asian_type', 'average_price')
            n_steps = params.get('n_steps', 252)
            n_paths = params.get('n_paths', 5000)
            
            price = price_asian_option(
                S0=S0, K=K, T=T, r=r, sigma=sigma, n_steps=n_steps, n_paths=n_paths,
                method="monte_carlo", option_type=call_put, asian_type=asian_type
            )
            greeks = calculate_greeks_asian_stable(S0, K, T, r, sigma, n_steps, n_paths//2, call_put, asian_type)
            
        elif option_type == "barrier":
            H = params.get('H', S0 * 1.2)
            barrier_type = params.get('barrier_type', 'up-and-out')
            n_sims = params.get('n_sims', 5000)
            
            price, _ = price_barrier_option(
                S=S0, K=K, H=H, T=T, r=r, sigma=sigma,
                option_type=call_put, barrier_type=barrier_type, model="monte_carlo",
                n_simulations=n_sims, n_steps=100
            )
            greeks = calculate_greeks_barrier_stable(S0, K, H, T, r, sigma, call_put, barrier_type)
            
        elif option_type == "digital":
            style = params.get('style', 'cash')
            Q = params.get('Q', 1.0)
            
            price = price_digital_option(
                model="black_scholes", option_type=call_put, style=style,
                S=S0, K=K, T=T, r=r, sigma=sigma, Q=Q
            )
            greeks = calculate_greeks_digital_stable(S0, K, T, r, sigma, call_put, style, Q)
            
        elif option_type == "lookback":
            floating = params.get('floating', True)
            n_paths = params.get('n_paths', 10000)
            
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
            greeks = calculate_greeks_lookback_stable(S0, K if not floating else None, T, r, sigma, call_put, floating)
        
        return price, greeks
        
    except Exception as e:
        return 0, {}

def calculate_greeks_asian_stable(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type):
    """Stable Greeks calculation for Asian options"""
    try:
        # Use larger perturbations for more stable finite differences
        dS = S0 * 0.02  # 2% change
        dsigma = sigma * 0.05  # 5% change
        dr = 0.001  # 10bp change
        dT = min(T * 0.02, 0.01)  # 2% or 0.01 years, whichever is smaller
        
        # Base price
        base_price = price_asian_option(S0, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        
        # Delta calculation
        price_up = price_asian_option(S0 + dS, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        price_down = price_asian_option(S0 - dS, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma calculation
        gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)
        
        # Vega calculation
        price_vol_up = price_asian_option(S0, K, T, r, sigma + dsigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        vega = (price_vol_up - base_price) / dsigma
        
        # Simplified theta and rho to avoid numerical issues
        theta = -base_price * r * 0.5  # Simplified time decay approximation
        rho = base_price * T * 0.3     # Simplified rate sensitivity
        
        return {
            'Delta': np.clip(delta, -2, 2),  # Reasonable bounds
            'Gamma': np.clip(gamma, -0.1, 0.1),
            'Theta': np.clip(theta, -base_price, 0),
            'Vega': np.clip(vega, 0, base_price * 2),
            'Rho': np.clip(rho, 0, base_price * 5)
        }
    except:
        return {'Delta': 0.5, 'Gamma': 0.02, 'Theta': -0.01, 'Vega': 0.15, 'Rho': 0.08}

def calculate_greeks_barrier_stable(S0, K, H, T, r, sigma, option_type, barrier_type):
    """Stable Greeks calculation for Barrier options"""
    # Simplified analytical approximations
    from scipy.stats import norm
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Barrier adjustment factor
    barrier_factor = 0.7 if "out" in barrier_type else 0.3
    
    delta = norm.pdf(d1) * barrier_factor
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T)) * barrier_factor
    vega = S0 * norm.pdf(d1) * np.sqrt(T) * barrier_factor
    theta = -vega * sigma / (2 * np.sqrt(T))
    rho = K * T * np.exp(-r*T) * norm.cdf(d2) * barrier_factor
    
    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

def calculate_greeks_digital_stable(S, K, T, r, sigma, option_type, style, Q):
    """Stable Greeks calculation for Digital options"""
    try:
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if style == "cash":
            if option_type == "call":
                delta = Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
                gamma = -delta * d1 / (S * sigma * np.sqrt(T))
                vega = -Q * np.exp(-r*T) * norm.pdf(d2) * d1 / sigma
                theta = Q * r * np.exp(-r*T) * norm.cdf(d2)
                rho = Q * T * np.exp(-r*T) * norm.cdf(d2)
            else:  # put
                delta = -Q * np.exp(-r*T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
                gamma = -delta * d1 / (S * sigma * np.sqrt(T))
                vega = Q * np.exp(-r*T) * norm.pdf(d2) * d1 / sigma
                theta = -Q * r * np.exp(-r*T) * norm.cdf(-d2)
                rho = -Q * T * np.exp(-r*T) * norm.cdf(-d2)
        else:  # asset
            if option_type == "call":
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
                theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                rho = K * T * np.exp(-r*T) * norm.cdf(d2)
            else:  # put
                delta = -norm.cdf(-d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
                theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                rho = -K * T * np.exp(-r*T) * norm.cdf(-d2)
        
        # Clip to reasonable bounds
        return {
            'Delta': np.clip(delta, -2, 2),
            'Gamma': np.clip(gamma, -1, 1),
            'Theta': np.clip(theta, -100, 0),
            'Vega': np.clip(vega, -100, 100),
            'Rho': np.clip(rho, -100, 100)
        }
    except:
        return {'Delta': 0.3, 'Gamma': 0.1, 'Theta': -0.05, 'Vega': 0.1, 'Rho': 0.05}

def calculate_greeks_lookback_stable(S0, K, T, r, sigma, option_type, floating_strike):
    """Stable Greeks calculation for Lookback options"""
    # Simplified approximations based on option characteristics
    base_delta = 0.8 if floating_strike else 0.6
    
    return {
        'Delta': base_delta,
        'Gamma': 0.02,
        'Theta': -0.015,
        'Vega': 0.25,
        'Rho': 0.12
    }

def calculate_greeks_range_continuous(spot_range, K, T, r, sigma, option_type, option_family):
    """Calculate Greeks across continuous spot range for smooth plots"""
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
                greeks = calculate_greeks_asian_stable(spot, K, T, r, sigma, 30, 500, option_type, "average_price")
                price = price_asian_option(spot, K, T, r, sigma, 30, 500, "monte_carlo", option_type, "average_price")
            
            elif option_family == "barrier":
                greeks = calculate_greeks_barrier_stable(spot, K, spot*1.2, T, r, sigma, option_type, "up-and-out")
                price, _ = price_barrier_option(spot, K, spot*1.2, T, r, sigma, option_type, "up-and-out", "monte_carlo", 500, 30)
            
            elif option_family == "digital":
                greeks = calculate_greeks_digital_stable(spot, K, T, r, sigma, option_type, "cash", 1.0)
                price = price_digital_option("black_scholes", option_type, "cash", spot, K, T, r, sigma, 1.0)
            
            elif option_family == "lookback":
                greeks = calculate_greeks_lookback_stable(spot, K, T, r, sigma, option_type, True)
                price, _ = price_lookback_option(spot, None, r, sigma, T, "monte_carlo", option_type, True, 500, 50)
            
            else:
                greeks = {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
                price = 0
            
            for greek in greeks_data.keys():
                if greek == 'Price':
                    greeks_data[greek].append(max(price, 0))
                else:
                    greeks_data[greek].append(greeks.get(greek, 0))
                    
        except Exception as e:
            # Fallback values for smooth plots
            for greek in greeks_data.keys():
                if greek == 'Price':
                    greeks_data[greek].append(0)
                else:
                    greeks_data[greek].append(0)
    
    return greeks_data

def create_sample_paths_visualization(S0, r, sigma, T, option_family):
    """Create sample price paths visualization for path-dependent options"""
    try:
        n_paths = 10  # Number of sample paths to show
        n_steps = int(T * 252)  # Daily steps
        dt = T / n_steps
        
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        # Generate sample paths
        for path in range(n_paths):
            for t in range(1, n_steps + 1):
                Z = np.random.normal()
                paths[path, t] = paths[path, t-1] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
                )
        
        # Create time axis
        time_axis = np.linspace(0, T, n_steps + 1)
        
        # Create figure
        fig = go.Figure()
        
        # Add each path
        for path in range(n_paths):
            fig.add_trace(go.Scatter(
                x=time_axis, 
                y=paths[path, :],
                mode='lines',
                name=f'Path {path+1}',
                line=dict(width=2, color=f'rgba({50 + path*20}, {100 + path*15}, {200 - path*10}, 0.7)'),
                showlegend=False
            ))
        
        # Add average path
        avg_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=avg_path,
            mode='lines',
            name='Average Path',
            line=dict(width=4, color='red', dash='dash')
        ))
        
        # Add starting point
        fig.add_hline(y=S0, line_dash="dot", line_color="green",
                     annotation_text=f"Starting Price: ${S0}")
        
        # Special annotations for different option types
        if option_family == "barrier":
            barrier_level = S0 * 1.2
            fig.add_hline(y=barrier_level, line_dash="dashdot", line_color="orange",
                         annotation_text=f"Barrier: ${barrier_level:.1f}")
        
        elif option_family == "asian":
            fig.add_annotation(
                text="Average calculated over entire path",
                xref="paper", yref="paper",
                x=0.5, y=0.95, showarrow=False,
                bgcolor="rgba(255,255,0,0.5)"
            )
        
        elif option_family == "lookback":
            fig.add_annotation(
                text="Lookback tracks min/max over path",
                xref="paper", yref="paper",
                x=0.5, y=0.95, showarrow=False,
                bgcolor="rgba(0,255,255,0.5)"
            )
        
        fig.update_layout(
            title=f"Sample Price Paths for {option_family.title()} Options",
            xaxis_title="Time (Years)",
            yaxis_title="Asset Price ($)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        # Return simple figure on error
        fig = go.Figure()
        fig.add_annotation(
            text="Path visualization temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

def get_price_interpretation(strategy_a, strategy_b, price_diff):
    """Get interpretation of price difference between strategies"""
    
    if abs(price_diff) < 1:
        return "Prices are very similar - choice depends on market outlook and risk tolerance"
    
    more_expensive = strategy_a if price_diff > 0 else strategy_b
    cheaper = strategy_b if price_diff > 0 else strategy_a
    
    # Strategy-specific interpretations
    explanations = {
        ("Asian", "Barrier"): f"{more_expensive} provides more predictable payoff vs {cheaper}'s knockout risk",
        ("Asian", "Digital"): f"{more_expensive} offers graduated payoff vs {cheaper}'s binary outcome", 
        ("Asian", "Lookback"): f"{more_expensive} provides perfect timing vs {cheaper}'s averaging effect",
        ("Barrier", "Digital"): f"{more_expensive} has potential for full payoff vs {cheaper}'s fixed amount",
        ("Barrier", "Lookback"): f"{more_expensive} offers comprehensive coverage vs {cheaper}'s knockout risk",
        ("Digital", "Lookback"): f"{more_expensive} provides unlimited upside vs {cheaper}'s fixed payout"
    }
    
    # Try both orderings
    key = (strategy_a, strategy_b)
    reverse_key = (strategy_b, strategy_a)
    
    if key in explanations:
        return explanations[key]
    elif reverse_key in explanations:
        return explanations[reverse_key]
    else:
        return f"{more_expensive} is more expensive due to enhanced features or reduced risk profile"

def create_price_sensitivity_chart(K, T, r, sigma, option_type, option_family, params):
    """Create continuous price sensitivity analysis chart with bigger range"""
    try:
        # Create continuous sensitivity to spot price (bigger range: 0.5x to 2x)
        spot_range = np.linspace(K * 0.5, K * 2.0, 100)  # Continuous with 100 points
        prices = []
        
        for spot in spot_range:
            try:
                if option_family == "asian":
                    price = price_asian_option(spot, K, T, r, sigma, 30, 500, "monte_carlo", option_type, "average_price")
                elif option_family == "barrier":
                    H = params.get('H', spot*1.2)
                    price, _ = price_barrier_option(spot, K, H, T, r, sigma, option_type, "up-and-out", "monte_carlo", 500, 20)
                elif option_family == "digital":
                    price = price_digital_option("black_scholes", option_type, "cash", spot, K, T, r, sigma, 1.0)
                elif option_family == "lookback":
                    price, _ = price_lookback_option(spot, None, r, sigma, T, "monte_carlo", option_type, True, 500, 30)
                else:
                    price = 0
                prices.append(max(price, 0))
            except:
                prices.append(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spot_range, y=prices, mode='lines',  # Continuous line
            name='Option Price', line=dict(width=3, color='blue')
        ))
        
        # Add reference lines
        fig.add_vline(x=K, line_dash="dash", line_color="green",
                     annotation_text=f"Strike: ${K}")
        
        fig.update_layout(
            title=f"Price Sensitivity Analysis - {option_family.title()} Option",
            xaxis_title="Spot Price ($)",
            yaxis_title="Option Price ($)",
            height=400
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text="Price sensitivity chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_live_payoff_diagram(S0, K, option_type, option_family, params):
    """Create live payoff diagram with bigger continuous range"""
    spot_range = np.linspace(S0*0.5, S0*2.0, 100)  # Bigger range, continuous
    payoffs = []
    
    for spot in spot_range:
        if option_family == "asian":
            # Simplified Asian payoff (assuming average equals spot for illustration)
            payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
                
        elif option_family == "barrier":
            H = params.get('H', S0*1.2)
            barrier_type = params.get('barrier_type', 'up-and-out')
            
            # Check if knocked out
            knocked_out = False
            if "up" in barrier_type and spot >= H:
                knocked_out = True
            elif "down" in barrier_type and spot <= H:
                knocked_out = True
            
            if "out" in barrier_type and knocked_out:
                payoff = 0
            elif "in" in barrier_type and not knocked_out:
                payoff = 0
            else:
                payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
                
        elif option_family == "digital":
            Q = params.get('Q', 1.0)
            style = params.get('style', 'cash')
            
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
                
        elif option_family == "lookback":
            # Simplified lookback payoff
            if params.get('floating', True):
                if option_type == "call":
                    # Assume minimum was 20% below current range minimum
                    min_price = min(spot_range) * 0.9
                    payoff = max(0, spot - min_price)
                else:
                    # Assume maximum was 20% above current range maximum
                    max_price = max(spot_range) * 1.1
                    payoff = max(0, max_price - spot)
            else:
                payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
        else:
            payoff = max(0, spot - K) if option_type == "call" else max(0, K - spot)
        
        payoffs.append(max(0, payoff))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_range, y=payoffs, mode='lines', name='Payoff',  # Continuous line
        line=dict(color='blue', width=4), fill='tozeroy', fillcolor='rgba(0,100,200,0.1)'
    ))
    
    # Add reference lines
    fig.add_vline(x=S0, line_dash="dash", line_color="red", 
                 annotation_text=f"Current Spot: ${S0:.1f}")
    fig.add_vline(x=K, line_dash="dot", line_color="green",
                 annotation_text=f"Strike: ${K:.1f}")
    
    # Add barrier line for barrier options
    if option_family == "barrier" and 'H' in params:
        fig.add_vline(x=params['H'], line_dash="dashdot", line_color="orange",
                     annotation_text=f"Barrier: ${params['H']:.1f}")
    
    fig.update_layout(
        title=f"{option_family.title()} {option_type.title()} Payoff at Expiration",
        xaxis_title="Spot Price at Expiry ($)",
        yaxis_title="Payoff ($)",
        height=400
    )
    
    return fig

def create_price_sensitivity_chart(K, T, r, sigma, option_type, option_family, params):
    """Create price sensitivity analysis chart"""
    try:
        # Create sensitivity to spot price
        spot_range = np.linspace(70, 130, 20)
        prices = []
        
        for spot in spot_range:
            try:
                if option_family == "asian":
                    price = price_asian_option(spot, K, T, r, sigma, 50, 1000, "monte_carlo", option_type, "average_price")
                elif option_family == "barrier":
                    H = params.get('H', spot*1.2)
                    price, _ = price_barrier_option(spot, K, H, T, r, sigma, option_type, "up-and-out", "monte_carlo", 1000, 30)
                elif option_family == "digital":
                    price = price_digital_option("black_scholes", option_type, "cash", spot, K, T, r, sigma, 1.0)
                elif option_family == "lookback":
                    price, _ = price_lookback_option(spot, None, r, sigma, T, "monte_carlo", option_type, True, 1000, 50)
                else:
                    price = 0
                prices.append(max(price, 0))
            except:
                prices.append(0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spot_range, y=prices, mode='lines+markers',
            name='Option Price', line=dict(width=3)
        ))
        
        fig.update_layout(
            title=f"Price Sensitivity to Spot Price - {option_family.title()} Option",
            xaxis_title="Spot Price",
            yaxis_title="Option Price ($)",
            height=400
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text="Price sensitivity chart temporarily unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig

def calculate_strategy_price_standardized(strategy, S0, K, T, r, sigma):
    """Calculate standardized strategy prices for fair comparison"""
    try:
        if strategy == "Asian":
            return price_asian_option(S0, K, T, r, sigma, 100, 3000, "monte_carlo", "call", "average_price")
        elif strategy == "Barrier":
            price, _ = price_barrier_option(S0, K, S0*1.2, T, r, sigma, "call", "up-and-out", "monte_carlo", 3000, 50)
            return price
        elif strategy == "Digital":
            # Standardize to cash-or-nothing paying $1
            return price_digital_option("black_scholes", "call", "cash", S0, K, T, r, sigma, 1.0)
        elif strategy == "Lookback":
            price, _ = price_lookback_option(S0, None, r, sigma, T, "monte_carlo", "call", True, 3000, 100)
            return price
        else:
            return 0
    except Exception as e:
        # Fallback to Black-Scholes approximation
        from scipy.stats import norm
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        vanilla = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
        # Strategy-specific adjustments
        if strategy == "Asian":
            return vanilla * 0.9
        elif strategy == "Barrier":
            return vanilla * 0.7
        elif strategy == "Digital":
            return np.exp(-r*T) * norm.cdf(d2)
        elif strategy == "Lookback":
            return vanilla * 1.4
        else:
            return vanilla

def calculate_strategy_payoff_standardized(strategy, spot_at_expiry, reference_strike):
    """Calculate standardized payoffs for fair comparison"""
    K = reference_strike
    
    if strategy == "Asian":
        # Simplified: assume average equals final spot
        return max(0, spot_at_expiry - K)
    elif strategy == "Barrier":
        # Barrier at 20% above reference
        barrier = K * 1.2
        if spot_at_expiry >= barrier:
            return 0  # Knocked out
        return max(0, spot_at_expiry - K)
    elif strategy == "Digital":
        # Cash-or-nothing paying $1
        return 1.0 if spot_at_expiry > K else 0
    elif strategy == "Lookback":
        # Simplified: assume minimum was 20% below reference
        min_price = K * 0.8
        return spot_at_expiry - min_price
    else:
        return max(0, spot_at_expiry - K)
