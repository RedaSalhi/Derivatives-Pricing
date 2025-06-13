# pricing/utils/enhanced_exotic_utils.py
# Enhanced utilities for exotic options with fixes for all reported issues

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import math

def create_interactive_parameter_widget(param_name, default_value, min_val=None, max_val=None, step=None, param_type="number"):
    """Create interactive parameter widgets with validation"""
    
    if param_type == "slider":
        return st.slider(
            param_name,
            min_value=min_val or default_value * 0.1,
            max_value=max_val or default_value * 3.0,
            value=default_value,
            step=step or default_value * 0.01,
            key=f"widget_{param_name.lower().replace(' ', '_')}"
        )
    elif param_type == "text":
        return st.text_input(
            param_name,
            value=str(default_value),
            key=f"text_{param_name.lower().replace(' ', '_')}"
        )
    else:
        return st.number_input(
            param_name,
            value=default_value,
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=f"num_{param_name.lower().replace(' ', '_')}"
        )

def create_continuous_sensitivity_heatmap(option_function, base_params, param1, param2, 
                                        param1_range, param2_range, resolution=30):
    """Create continuous sensitivity heatmap for any two parameters"""
    
    # Create parameter grids
    p1_values = np.linspace(param1_range[0], param1_range[1], resolution)
    p2_values = np.linspace(param2_range[0], param2_range[1], resolution)
    
    # Calculate option prices for each combination
    price_matrix = np.zeros((resolution, resolution))
    
    for i, p1 in enumerate(p1_values):
        for j, p2 in enumerate(p2_values):
            temp_params = base_params.copy()
            temp_params[param1] = p1
            temp_params[param2] = p2
            
            try:
                price = option_function(**temp_params)
                price_matrix[j, i] = price
            except:
                price_matrix[j, i] = np.nan
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=price_matrix,
        x=p1_values,
        y=p2_values,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title="Option Price")
    ))
    
    fig.update_layout(
        title=f"Sensitivity Heatmap: {param1} vs {param2}",
        xaxis_title=param1,
        yaxis_title=param2,
        template="plotly_white"
    )
    
    return fig

def create_enhanced_greeks_dashboard(greeks_dict, option_type=""):
    """Create comprehensive Greeks dashboard with visualizations"""
    
    # Greeks summary cards
    cols = st.columns(len(greeks_dict))
    
    greek_colors = {
        'Delta': '#FF6B6B',
        'Gamma': '#4ECDC4', 
        'Theta': '#45B7D1',
        'Vega': '#96CEB4',
        'Rho': '#FFEAA7'
    }
    
    for i, (greek_name, value) in enumerate(greeks_dict.items()):
        if greek_name != 'Price':
            with cols[i]:
                color = greek_colors.get(greek_name, '#666666')
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}CC 100%); 
                           color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 5px;">
                    <h4 style="margin: 0; color: white;">{greek_name}</h4>
                    <p style="font-size: 1.2em; font-weight: bold; margin: 5px 0;">{value:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Greeks radar chart
    greek_names = [k for k in greeks_dict.keys() if k != 'Price']
    greek_values = [abs(greeks_dict[k]) for k in greek_names]  # Use absolute values for radar
    
    # Normalize values for better visualization
    max_val = max(greek_values) if max(greek_values) > 0 else 1
    normalized_values = [v / max_val for v in greek_values]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],  # Close the polygon
        theta=greek_names + [greek_names[0]],
        fill='toself',
        name=f'{option_type} Greeks Profile',
        line=dict(color='rgb(102, 126, 234)', width=3),
        fillcolor='rgba(102, 126, 234, 0.25)'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Greeks Risk Profile (Normalized)",
        template="plotly_white"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

def create_payoff_3d_surface(option_function, base_params, param1="S", param2="sigma", 
                           param1_range=(50, 150), param2_range=(0.1, 0.5), resolution=25):
    """Create 3D payoff surface for any option"""
    
    # Create parameter grids
    p1_values = np.linspace(param1_range[0], param1_range[1], resolution)
    p2_values = np.linspace(param2_range[0], param2_range[1], resolution)
    
    # Calculate option prices
    price_surface = np.zeros((resolution, resolution))
    
    for i, p1 in enumerate(p1_values):
        for j, p2 in enumerate(p2_values):
            temp_params = base_params.copy()
            temp_params[param1] = p1
            temp_params[param2] = p2
            
            try:
                price = option_function(**temp_params)
                price_surface[j, i] = price
            except:
                price_surface[j, i] = 0
    
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        z=price_surface,
        x=p1_values,
        y=p2_values,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(title="Option Price")
    )])
    
    fig.update_layout(
        title=f'3D Price Surface: {param1} vs {param2}',
        scene=dict(
            xaxis_title=param1,
            yaxis_title=param2,
            zaxis_title='Option Price',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            )
        ),
        height=600,
        template="plotly_white"
    )
    
    return fig

def create_monte_carlo_convergence_plot(option_function, base_params, max_paths=50000, step=2500):
    """Create Monte Carlo convergence analysis"""
    
    path_counts = list(range(step, max_paths + step, step))
    prices = []
    std_errors = []
    
    # Run multiple simulations for each path count
    for n_paths in path_counts:
        temp_params = base_params.copy()
        temp_params['n_paths'] = n_paths
        
        trial_prices = []
        for _ in range(5):  # 5 trials for each path count
            try:
                price = option_function(**temp_params)
                trial_prices.append(price)
            except:
                trial_prices.append(np.nan)
        
        clean_prices = [p for p in trial_prices if not np.isnan(p)]
        if clean_prices:
            prices.append(np.mean(clean_prices))
            std_errors.append(np.std(clean_prices) / np.sqrt(len(clean_prices)))
        else:
            prices.append(np.nan)
            std_errors.append(np.nan)
    
    # Create convergence plot
    fig = go.Figure()
    
    # Main convergence line
    fig.add_trace(go.Scatter(
        x=path_counts,
        y=prices,
        mode='lines+markers',
        name='Estimated Price',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    # Confidence intervals
    upper_bound = [p + 1.96 * se if not np.isnan(p) and not np.isnan(se) else np.nan 
                   for p, se in zip(prices, std_errors)]
    lower_bound = [p - 1.96 * se if not np.isnan(p) and not np.isnan(se) else np.nan 
                   for p, se in zip(prices, std_errors)]
    
    fig.add_trace(go.Scatter(
        x=path_counts + path_counts[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title='Monte Carlo Convergence Analysis',
        xaxis_title='Number of Simulation Paths',
        yaxis_title='Estimated Option Price',
        template="plotly_white",
        height=500
    )
    
    return fig

def create_time_decay_animation(option_function, base_params, time_points=20):
    """Create animated time decay visualization"""
    
    T_initial = base_params.get('T', 1.0)
    time_values = np.linspace(T_initial, 0.01, time_points)
    
    fig = go.Figure()
    
    # Create frames for animation
    frames = []
    spot_range = np.linspace(base_params.get('S', 100) * 0.5, 
                           base_params.get('S', 100) * 1.5, 50)
    
    for i, T in enumerate(time_values):
        prices = []
        temp_params = base_params.copy()
        temp_params['T'] = T
        
        for S in spot_range:
            temp_params['S'] = S
            try:
                price = option_function(**temp_params)
                prices.append(price)
            except:
                prices.append(0)
        
        frame = go.Frame(
            data=[go.Scatter(
                x=spot_range,
                y=prices,
                mode='lines',
                line=dict(color='blue', width=3),
                name=f'T = {T:.3f}'
            )],
            name=f'frame_{i}'
        )
        frames.append(frame)
    
    # Initial frame
    temp_params = base_params.copy()
    temp_params['T'] = T_initial
    initial_prices = []
    
    for S in spot_range:
        temp_params['S'] = S
        try:
            price = option_function(**temp_params)
            initial_prices.append(price)
        except:
            initial_prices.append(0)
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=initial_prices,
        mode='lines',
        line=dict(color='blue', width=3),
        name=f'T = {T_initial:.3f}'
    ))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title="Time Decay Animation",
        xaxis_title="Spot Price",
        yaxis_title="Option Price",
        template="plotly_white",
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 300}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    return fig

def create_volatility_smile_impact(option_function, base_params, vol_range=(0.1, 0.6)):
    """Analyze impact of volatility smile on exotic options"""
    
    strike_multipliers = np.linspace(0.8, 1.2, 20)
    base_strike = base_params.get('K', 100)
    strikes = [base_strike * mult for mult in strike_multipliers]
    
    # Different volatility scenarios
    vol_scenarios = {
        'Flat Vol': [base_params.get('sigma', 0.2)] * len(strikes),
        'Volatility Smile': [base_params.get('sigma', 0.2) * (1 + 0.5 * (mult - 1)**2) 
                           for mult in strike_multipliers],
        'Volatility Skew': [base_params.get('sigma', 0.2) * (1.2 - 0.4 * mult) 
                          for mult in strike_multipliers]
    }
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green']
    
    for i, (scenario_name, vols) in enumerate(vol_scenarios.items()):
        prices = []
        
        for K, vol in zip(strikes, vols):
            temp_params = base_params.copy()
            temp_params['K'] = K
            temp_params['sigma'] = vol
            
            try:
                price = option_function(**temp_params)
                prices.append(price)
            except:
                prices.append(np.nan)
        
        fig.add_trace(go.Scatter(
            x=strike_multipliers,
            y=prices,
            mode='lines+markers',
            name=scenario_name,
            line=dict(color=colors[i], width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title='Impact of Volatility Smile on Option Prices',
        xaxis_title='Strike / Spot Ratio',
        yaxis_title='Option Price',
        template="plotly_white",
        height=500
    )
    
    return fig

def create_risk_scenario_analysis(option_function, base_params, scenarios):
    """Create comprehensive risk scenario analysis"""
    
    scenario_results = {}
    base_price = None
    
    try:
        base_price = option_function(**base_params)
    except:
        base_price = 0
    
    for scenario_name, scenario_params in scenarios.items():
        temp_params = base_params.copy()
        temp_params.update(scenario_params)
        
        try:
            scenario_price = option_function(**temp_params)
            pnl = scenario_price - base_price
            pnl_pct = (pnl / base_price * 100) if base_price != 0 else 0
            
            scenario_results[scenario_name] = {
                'Price': scenario_price,
                'PnL': pnl,
                'PnL (%)': pnl_pct
            }
        except:
            scenario_results[scenario_name] = {
                'Price': np.nan,
                'PnL': np.nan,
                'PnL (%)': np.nan
            }
    
    # Create visualization
    scenario_names = list(scenario_results.keys())
    prices = [scenario_results[name]['Price'] for name in scenario_names]
    pnls = [scenario_results[name]['PnL'] for name in scenario_names]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Scenario Prices', 'P&L Impact'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Prices chart
    fig.add_trace(
        go.Bar(x=scenario_names, y=prices, name='Scenario Prices', 
               marker_color='lightblue'),
        row=1, col=1
    )
    
    # P&L chart with color coding
    colors = ['red' if pnl < 0 else 'green' for pnl in pnls]
    fig.add_trace(
        go.Bar(x=scenario_names, y=pnls, name='P&L', 
               marker_color=colors),
        row=1, col=2
    )
    
    # Add base price line
    fig.add_hline(y=base_price, line_dash="dash", line_color="black", 
                  annotation_text=f"Base: {base_price:.4f}", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", 
                  annotation_text="Breakeven", row=1, col=2)
    
    fig.update_layout(
        title='Risk Scenario Analysis',
        template="plotly_white",
        height=500
    )
    
    return fig, scenario_results

def create_correlation_matrix(option_functions, base_params_list, param_ranges):
    """Create correlation matrix between different options under parameter changes"""
    
    correlations = {}
    
    # Generate price series for each option under different parameter values
    for param_name, param_range in param_ranges.items():
        param_values = np.linspace(param_range[0], param_range[1], 20)
        
        option_price_series = {}
        
        for i, (option_name, option_func) in enumerate(option_functions.items()):
            prices = []
            base_params = base_params_list[i] if isinstance(base_params_list, list) else base_params_list
            
            for param_val in param_values:
                temp_params = base_params.copy()
                temp_params[param_name] = param_val
                
                try:
                    price = option_func(**temp_params)
                    prices.append(price)
                except:
                    prices.append(np.nan)
            
            option_price_series[option_name] = prices
        
        # Calculate correlation matrix for this parameter
        df = pd.DataFrame(option_price_series)
        corr_matrix = df.corr()
        correlations[param_name] = corr_matrix
    
    return correlations

def create_interactive_comparison_dashboard(option_data, metrics=['Price', 'Delta', 'Gamma']):
    """Create interactive comparison dashboard with multiple metrics"""
    
    # Parameter controls
    st.markdown("### 🎛️ Interactive Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_metric = st.selectbox("Metric to Compare", metrics, key="comp_metric_select")
    
    with col2:
        chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Radar"], key="comp_chart_type")
    
    with col3:
        normalize_data = st.checkbox("Normalize Data", key="comp_normalize")
    
    # Extract data for selected metric
    comparison_data = {}
    for option_name, data in option_data.items():
        if selected_metric in data:
            comparison_data[option_name] = data[selected_metric]
    
    if not comparison_data:
        st.warning(f"No data available for metric: {selected_metric}")
        return
    
    # Create visualization based on chart type
    fig = go.Figure()
    
    if chart_type == "Line":
        x_values = list(range(len(next(iter(comparison_data.values())))))
        
        for option_name, values in comparison_data.items():
            if normalize_data and max(values) > 0:
                values = [v / max(values) for v in values]
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=values,
                mode='lines+markers',
                name=option_name,
                line=dict(width=3),
                marker=dict(size=6)
            ))
    
    elif chart_type == "Bar":
        option_names = list(comparison_data.keys())
        values = [np.mean(data) for data in comparison_data.values()]
        
        if normalize_data and max(values) > 0:
            values = [v / max(values) for v in values]
        
        fig.add_trace(go.Bar(
            x=option_names,
            y=values,
            marker_color=px.colors.qualitative.Set3[:len(option_names)]
        ))
    
    elif chart_type == "Radar":
        option_names = list(comparison_data.keys())
        values = [np.mean(data) for data in comparison_data.values()]
        
        if normalize_data and max(values) > 0:
            values = [v / max(values) for v in values]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=option_names + [option_names[0]],
            fill='toself',
            name=selected_metric
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) if values else 1]
                )),
        )
    
    fig.update_layout(
        title=f'{selected_metric} Comparison - {chart_type} Chart',
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_advanced_stress_test_matrix(option_functions, base_scenarios, stress_factors):
    """Create advanced stress test matrix with multiple factors"""
    
    stress_results = {}
    
    # Base case
    base_results = {}
    for option_name, option_func in option_functions.items():
        try:
            base_price = option_func(**base_scenarios[option_name])
            base_results[option_name] = base_price
        except:
            base_results[option_name] = np.nan
    
    # Stress scenarios
    for stress_name, stress_params in stress_factors.items():
        stress_results[stress_name] = {}
        
        for option_name, option_func in option_functions.items():
            stressed_params = base_scenarios[option_name].copy()
            
            # Apply stress factors
            for param, factor in stress_params.items():
                if param in stressed_params:
                    if isinstance(factor, (int, float)):
                        stressed_params[param] *= factor
                    else:
                        stressed_params[param] = factor
            
            try:
                stressed_price = option_func(**stressed_params)
                pnl = stressed_price - base_results[option_name]
                pnl_pct = (pnl / base_results[option_name] * 100) if base_results[option_name] != 0 else 0
                
                stress_results[stress_name][option_name] = {
                    'Price': stressed_price,
                    'PnL': pnl,
                    'PnL_Pct': pnl_pct
                }
            except:
                stress_results[stress_name][option_name] = {
                    'Price': np.nan,
                    'PnL': np.nan,
                    'PnL_Pct': np.nan
                }
    
    return stress_results, base_results

def export_analysis_results(results_dict, filename="exotic_options_analysis"):
    """Export analysis results to downloadable formats"""
    
    # Create summary DataFrame
    summary_data = []
    
    for analysis_type, data in results_dict.items():
        if isinstance(data, dict):
            for key, value in data.items():
                summary_data.append({
                    'Analysis Type': analysis_type,
                    'Parameter/Option': key,
                    'Value': value if not isinstance(value, (list, dict)) else str(value)
                })
        else:
            summary_data.append({
                'Analysis Type': analysis_type,
                'Parameter/Option': 'Result',
                'Value': str(data)
            })
    
    df = pd.DataFrame(summary_data)
    
    # Convert to CSV for download
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📁 Download Analysis Results (CSV)",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
        key="download_results"
    )
    
    return df

# Utility functions for mathematical operations
def black_scholes_call(S, K, T, r, sigma):
    """Standard Black-Scholes call price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    """Standard Black-Scholes put price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_volatility_newton(market_price, S, K, T, r, option_type='call', max_iterations=100):
    """Calculate implied volatility using Newton-Raphson method"""
    
    # Initial guess
    sigma = 0.2
    
    for i in range(max_iterations):
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)
        
        # Vega calculation
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        # Price difference
        price_diff = price - market_price
        
        # Check convergence
        if abs(price_diff) < 1e-6:
            return sigma
        
        # Newton-Raphson update
        if vega != 0:
            sigma = sigma - price_diff / vega
        else:
            break
        
        # Ensure sigma stays positive
        sigma = max(sigma, 1e-6)
    
    return sigma

def monte_carlo_asian_optimized(S, K, T, r, sigma, n_steps, n_paths, option_type, asian_type, use_variance_reduction=True):
    """Optimized Monte Carlo for Asian options with variance reduction"""
    
    dt = T / n_steps
    discount_factor = np.exp(-r * T)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate correlated random numbers for variance reduction
    if use_variance_reduction:
        # Antithetic variates
        n_paths_half = n_paths // 2
        Z1 = np.random.standard_normal((n_paths_half, n_steps))
        Z2 = -Z1
        Z = np.vstack([Z1, Z2])
    else:
        Z = np.random.standard_normal((n_paths, n_steps))
    
    # Generate price paths
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S)
    
    for i in range(n_steps):
        log_S[:, i + 1] = log_S[:, i] + (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i]
    
    S_paths = np.exp(log_S)
    
    # Calculate payoffs based on Asian type
    if asian_type == "average_price":
        avg_prices = np.mean(S_paths[:, 1:], axis=1)
        if option_type == "call":
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
    else:  # average_strike
        avg_strikes = np.mean(S_paths[:, 1:], axis=1)
        final_prices = S_paths[:, -1]
        if option_type == "call":
            payoffs = np.maximum(final_prices - avg_strikes, 0)
        else:
            payoffs = np.maximum(avg_strikes - final_prices, 0)
    
    # Control variate using geometric average
    if use_variance_reduction:
        geo_avg = np.exp(np.mean(np.log(S_paths[:, 1:]), axis=1))
        
        if option_type == "call":
            geo_payoffs = np.maximum(geo_avg - K, 0)
        else:
            geo_payoffs = np.maximum(K - geo_avg, 0)
        
        # Theoretical geometric Asian price (approximate)
        sigma_geo = sigma / np.sqrt(3)
        mu_geo = (r + sigma**2 / 6) / 2
        
        d1 = (np.log(S / K) + (mu_geo + 0.5 * sigma_geo**2) * T) / (sigma_geo * np.sqrt(T))
        d2 = d1 - sigma_geo * np.sqrt(T)
        
        if option_type == "call":
            theo_geo = S * np.exp((mu_geo - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theo_geo = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((mu_geo - r) * T) * norm.cdf(-d1)
        
        # Control variate adjustment
        cov_matrix = np.cov(payoffs, geo_payoffs)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
        
        adjusted_payoffs = payoffs - beta * (geo_payoffs - theo_geo)
        return discount_factor * np.mean(adjusted_payoffs)
    
    return discount_factor * np.mean(payoffs)

def calculate_portfolio_greeks(positions, option_prices, option_greeks):
    """Calculate portfolio-level Greeks for multiple positions"""
    
    portfolio_greeks = {
        'Delta': 0,
        'Gamma': 0,
        'Theta': 0,
        'Vega': 0,
        'Rho': 0
    }
    
    total_value = 0
    
    for i, (position_size, option_data) in enumerate(positions.items()):
        option_price = option_prices[i]
        greeks = option_greeks[i]
        
        position_value = position_size * option_price
        total_value += position_value
        
        for greek_name in portfolio_greeks.keys():
            if greek_name in greeks:
                portfolio_greeks[greek_name] += position_size * greeks[greek_name]
    
    # Calculate percentage Greeks
    portfolio_greeks_pct = {}
    for greek_name, value in portfolio_greeks.items():
        if total_value != 0:
            portfolio_greeks_pct[f'{greek_name}_pct'] = (value * 100) / total_value
        else:
            portfolio_greeks_pct[f'{greek_name}_pct'] = 0
    
    return portfolio_greeks, portfolio_greeks_pct, total_value

def validate_option_parameters(params, option_type="generic"):
    """Validate option parameters and provide warnings"""
    
    warnings = []
    errors = []
    
    # Common validations
    S = params.get('S', 100)
    K = params.get('K', 100)
    T = params.get('T', 1)
    r = params.get('r', 0.05)
    sigma = params.get('sigma', 0.2)
    
    if S <= 0:
        errors.append("Stock price must be positive")
    
    if K <= 0:
        errors.append("Strike price must be positive")
    
    if T <= 0:
        errors.append("Time to maturity must be positive")
    
    if r < 0:
        warnings.append("Negative interest rates detected")
    
    if sigma <= 0:
        errors.append("Volatility must be positive")
    elif sigma > 2.0:
        warnings.append("Very high volatility (>200%) - results may be unrealistic")
    
    if T > 10:
        warnings.append("Very long time to maturity (>10 years) - consider model limitations")
    
    # Option-specific validations
    if option_type == "barrier":
        B = params.get('B', 120)
        if B <= 0:
            errors.append("Barrier level must be positive")
        
        barrier_type = params.get('barrier_type', '')
        if 'up' in barrier_type and B <= max(S, K):
            warnings.append("Up barrier should typically be above current spot and strike")
        elif 'down' in barrier_type and B >= min(S, K):
            warnings.append("Down barrier should typically be below current spot and strike")
    
    elif option_type == "digital":
        Q = params.get('Q', 1)
        if Q <= 0:
            warnings.append("Digital payout should typically be positive")
    
    return warnings, errors
