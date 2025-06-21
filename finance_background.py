import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
from plotly.subplots import make_subplots
import warnings
from styles.app_styles import (
    load_theme, 
    apply_global_styles, 
    get_component_styles,
    render_app_header,
    render_page_header,
    render_section_title,
    render_info_box,
    COLORS
)
warnings.filterwarnings('ignore')

# Apply the comprehensive styling system
st.set_page_config(
    page_title="Quantitative Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the global theme and styles
load_theme()
apply_global_styles()

# Inject component styles
st.markdown(get_component_styles(), unsafe_allow_html=True)

# Core pricing functions (unchanged)
class OptionPricer:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        if T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(S, K, T, r, sigma):
        if T <= 0:
            return max(K - S, 0)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type='call'):
        if T <= 0:
            return 0, 0, 0, 0, 0
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
            theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        else:  # put
            delta = norm.cdf(d1) - 1
            rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
            theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return delta, gamma, theta, vega, rho

    @staticmethod
    def binomial_tree(S, K, T, r, sigma, n_steps=100, option_type='call', american=False):
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize arrays
        prices = np.zeros(n_steps + 1)
        option_values = np.zeros(n_steps + 1)
        
        # Calculate final stock prices
        for j in range(n_steps + 1):
            prices[j] = S * (u ** j) * (d ** (n_steps - j))
        
        # Calculate option values at expiration
        for j in range(n_steps + 1):
            if option_type == 'call':
                option_values[j] = max(0, prices[j] - K)
            else:
                option_values[j] = max(0, K - prices[j])
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                # Calculate stock price at this node
                current_price = S * (u ** j) * (d ** (i - j))
                
                # Calculate discounted expected value
                option_values[j] = np.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])
                
                # For American options, check early exercise
                if american:
                    if option_type == 'call':
                        intrinsic_value = max(0, current_price - K)
                        option_values[j] = max(option_values[j], intrinsic_value)
                    else:
                        intrinsic_value = max(0, K - current_price)
                        option_values[j] = max(option_values[j], intrinsic_value)
        
        return option_values[0]

    @staticmethod
    def monte_carlo_option(S, K, T, r, sigma, n_simulations=10000, n_steps=252, option_type='call', path_dependent=False):
        dt = T / n_steps
        discount_factor = np.exp(-r * T)
        
        if not path_dependent:
            # European option - only need final price
            z = np.random.normal(0, 1, n_simulations)
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
            
            if option_type == 'call':
                payoffs = np.maximum(ST - K, 0)
            else:
                payoffs = np.maximum(K - ST, 0)
        else:
            # Path-dependent option
            payoffs = []
            for _ in range(n_simulations):
                prices = [S]
                for _ in range(n_steps):
                    z = np.random.normal()
                    price = prices[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
                    prices.append(price)
                
                if option_type == 'asian_call':
                    avg_price = np.mean(prices)
                    payoffs.append(max(avg_price - K, 0))
                elif option_type == 'asian_put':
                    avg_price = np.mean(prices)
                    payoffs.append(max(K - avg_price, 0))
                elif option_type == 'lookback_call':
                    max_price = max(prices)
                    payoffs.append(max_price - K)
                elif option_type == 'lookback_put':
                    min_price = min(prices)
                    payoffs.append(K - min_price)
            
            payoffs = np.array(payoffs)
        
        option_price = discount_factor * np.mean(payoffs)
        standard_error = discount_factor * np.std(payoffs) / np.sqrt(n_simulations)
        
        return option_price, standard_error

# Strategy classes (unchanged)
class OptionsStrategy:
    @staticmethod
    def bull_call_spread(S, K1, K2, T, r, sigma):
        """Bull Call Spread: Long call at K1, short call at K2 (K1 < K2)"""
        long_call = OptionPricer.black_scholes_call(S, K1, T, r, sigma)
        short_call = OptionPricer.black_scholes_call(S, K2, T, r, sigma)
        return long_call - short_call
    
    @staticmethod
    def bear_put_spread(S, K1, K2, T, r, sigma):
        """Bear Put Spread: Long put at K2, short put at K1 (K1 < K2)"""
        long_put = OptionPricer.black_scholes_put(S, K2, T, r, sigma)
        short_put = OptionPricer.black_scholes_put(S, K1, T, r, sigma)
        return long_put - short_put
    
    @staticmethod
    def straddle(S, K, T, r, sigma):
        """Long Straddle: Long call + long put at same strike"""
        call = OptionPricer.black_scholes_call(S, K, T, r, sigma)
        put = OptionPricer.black_scholes_put(S, K, T, r, sigma)
        return call + put
    
    @staticmethod
    def strangle(S, K1, K2, T, r, sigma):
        """Long Strangle: Long call at K2, long put at K1 (K1 < K2)"""
        call = OptionPricer.black_scholes_call(S, K2, T, r, sigma)
        put = OptionPricer.black_scholes_put(S, K1, T, r, sigma)
        return call + put

# Initialize default parameters
default_params = {
    'S': 100,
    'K': 100,
    'T': 1.0,
    'r': 0.05,
    'sigma': 0.20
}

# Use the styled header function
render_app_header(
    "Quantitative Finance Platform",
    "Advanced Derivatives Pricing & Risk Management System"
)

# Interactive parameter controls with enhanced styling
st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
render_section_title("🎛️ Controls")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    spot_price = st.number_input("Spot Price (S₀)", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
with col2:
    strike_price = st.number_input("Strike Price (K)", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
with col3:
    volatility = st.number_input("Volatility (σ)", min_value=0.05, max_value=1.0, value=0.20, step=0.05, format="%.2f")
with col4:
    time_to_maturity = st.number_input("Time to Maturity (T)", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
with col5:
    risk_free_rate = st.number_input("Risk-free Rate (r)", min_value=0.001, max_value=0.20, value=0.05, step=0.01, format="%.3f")

st.markdown('</div>', unsafe_allow_html=True)

# Quick metrics display with enhanced styling
call_price = OptionPricer.black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
put_price = OptionPricer.black_scholes_put(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
delta_call, gamma, theta_call, vega, rho_call = OptionPricer.calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')

# Enhanced Quick Metrics Display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <div style="color: {COLORS['success']}; font-size: 1.2rem; font-weight: 600;">Call Option</div>
        <div style="font-size: 2rem; font-weight: bold; color: {COLORS['gray_800']};">${call_price:.4f}</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.9rem;">Current Value</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card animate-fade-in-delay">
        <div style="color: {COLORS['danger']}; font-size: 1.2rem; font-weight: 600;">Put Option</div>
        <div style="font-size: 2rem; font-weight: bold; color: {COLORS['gray_800']};">${put_price:.4f}</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.9rem;">Current Value</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <div style="color: {COLORS['primary']}; font-size: 1.2rem; font-weight: 600;">Delta (Δ)</div>
        <div style="font-size: 2rem; font-weight: bold; color: {COLORS['gray_800']};">{delta_call:.4f}</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.9rem;">Price Sensitivity</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card animate-fade-in-delay">
        <div style="color: {COLORS['info']}; font-size: 1.2rem; font-weight: 600;">Vega (ν)</div>
        <div style="font-size: 2rem; font-weight: bold; color: {COLORS['gray_800']};">{vega:.2f}</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.9rem;">Vol Sensitivity</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Objective Section with enhanced styling
# ----------------------
st.markdown("""
<div class="objective-box animate-fade-in">
    <div class="section-title">🎯 Platform Overview</div>
    <div class="content-text">
        This comprehensive quantitative finance platform provides state-of-the-art tools for derivatives pricing, risk management, and portfolio analysis. Designed for quants, researchers, and students.
    </div>
    <div class="highlight-box">
        <strong>Core Capabilities:</strong>
        <ul style="margin-top: 1rem;">
            <li><strong>Multi-Model Pricing:</strong> Black-Scholes, Binomial Trees, Monte Carlo simulations</li>
            <li><strong>Complete Greeks Suite:</strong> Delta, Gamma, Theta, Vega, Rho with sensitivity analysis</li>
            <li><strong>Strategy Builder:</strong> 10+ options strategies with P&L visualization</li>
            <li><strong>Risk Management:</strong> VaR, Expected Shortfall, stress testing</li>
            <li><strong>Portfolio Analytics:</strong> Multi-asset portfolio optimization and analysis</li>
            <li><strong>Volatility Modeling:</strong> Surface construction and implied volatility analysis</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------
# Pricing Models Section with enhanced styling
# ----------------------
st.markdown("""
<div class="model-box animate-fade-in">
    <div class="section-title">📊 Pricing Models Comparison</div>
    <div class="content-text">
        Compare different pricing methodologies and understand their applications in modern finance.
    </div>
</div>
""", unsafe_allow_html=True)

# Model comparison with enhanced tabs
tab1, tab2, tab3 = st.tabs(["🔢 Black-Scholes Analysis", "🌳 Binomial Trees", "🎲 Monte Carlo Methods"])

with tab1:
    st.markdown("""
    <div class="model-card animate-fade-in">
        <div class="subsection-title">Black-Scholes Framework</div>
        <div class="content-text">
            The foundation of modern derivatives pricing, providing analytical solutions for European options.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced formula display
    st.markdown("""
    <div class="formula">
        C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)
        <br><br>
        <small>where d₁ = [ln(S₀/K) + (r + ½σ²)T] / (σ√T)</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Sensitivity analysis with enhanced colors
    S_range = np.linspace(spot_price * 0.5, spot_price * 1.5, 100)
    call_prices = [OptionPricer.black_scholes_call(S, strike_price, time_to_maturity, risk_free_rate, volatility) for S in S_range]
    put_prices = [OptionPricer.black_scholes_put(S, strike_price, time_to_maturity, risk_free_rate, volatility) for S in S_range]
    
    fig_bs = go.Figure()
    fig_bs.add_trace(go.Scatter(
        x=S_range, y=call_prices, 
        name='Call Option', 
        line=dict(color=COLORS['success'], width=3),
        hovertemplate="Stock Price: $%{x:.2f}<br>Call Value: $%{y:.4f}<extra></extra>"
    ))
    fig_bs.add_trace(go.Scatter(
        x=S_range, y=put_prices, 
        name='Put Option', 
        line=dict(color=COLORS['danger'], width=3),
        hovertemplate="Stock Price: $%{x:.2f}<br>Put Value: $%{y:.4f}<extra></extra>"
    ))
    fig_bs.add_vline(
        x=spot_price, 
        line_dash="dash", 
        line_color=COLORS['primary'], 
        annotation_text=f"Current S₀ = ${spot_price}",
        annotation_position="top"
    )
    fig_bs.add_vline(
        x=strike_price, 
        line_dash="dot", 
        line_color=COLORS['secondary'], 
        annotation_text=f"Strike K = ${strike_price}",
        annotation_position="top"
    )
    
    fig_bs.update_layout(
        title="Black-Scholes Option Prices vs Underlying",
        xaxis_title="Underlying Price ($)",
        yaxis_title="Option Value ($)",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_bs, use_container_width=True)

with tab2:
    st.markdown("""
    <div class="model-card animate-fade-in">
        <div class="subsection-title">Binomial Tree Model</div>
        <div class="content-text">
            Discrete-time model supporting both European and American options with early exercise features.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        n_steps = st.slider("Tree Steps", 10, 200, 50)
        american_option = st.checkbox("American Option")
    
    with col2:
        option_type = st.selectbox("Option Type", ["call", "put"])
    
    # Calculate binomial price
    binomial_price = OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, n_steps, option_type, american_option)
    bs_price = OptionPricer.black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility) if option_type == 'call' else OptionPricer.black_scholes_put(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div style="color: {COLORS['primary']}; font-weight: 600;">Binomial Price</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${binomial_price:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div style="color: {COLORS['success']}; font-weight: 600;">Black-Scholes Price</div>
            <div style="font-size: 1.5rem; font-weight: bold;">${bs_price:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        difference = binomial_price - bs_price
        color = COLORS['success'] if abs(difference) < 0.01 else COLORS['warning']
        st.markdown(f"""
        <div class="metric-container">
            <div style="color: {color}; font-weight: 600;">Difference</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">${difference:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Convergence analysis
    steps_range = range(10, 101, 10)
    binomial_prices = [OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, n, option_type, american_option) for n in steps_range]
    
    fig_binomial = go.Figure()
    fig_binomial.add_trace(go.Scatter(
        x=list(steps_range), y=binomial_prices, 
        name='Binomial Price', 
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8)
    ))
    fig_binomial.add_hline(
        y=bs_price, 
        line_dash="dash", 
        line_color=COLORS['danger'], 
        annotation_text=f"BS Price: ${bs_price:.4f}"
    )
    
    fig_binomial.update_layout(
        title="Binomial Tree Convergence to Black-Scholes",
        xaxis_title="Number of Steps",
        yaxis_title="Option Price ($)",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_binomial, use_container_width=True)

with tab3:
    st.markdown("""
    <div class="model-card animate-fade-in">
        <div class="subsection-title">Monte Carlo Simulation</div>
        <div class="content-text">
            Stochastic simulation method ideal for complex, path-dependent derivatives.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_simulations = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=2)
    with col2:
        mc_option_type = st.selectbox("MC Option Type", ["call", "put", "asian_call", "asian_put", "lookback_call", "lookback_put"])
    with col3:
        path_dependent = mc_option_type.startswith(('asian', 'lookback'))
    
    # Run Monte Carlo
    if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            mc_price, mc_se = OptionPricer.monte_carlo_option(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, n_simulations, option_type=mc_option_type, path_dependent=path_dependent)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="success-box">
                    <strong>MC Price: ${mc_price:.4f}</strong>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="info-box">
                    <strong>Standard Error: ${mc_se:.4f}</strong>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                confidence_interval = 1.96 * mc_se
                st.markdown(f"""
                <div class="warning-box">
                    <strong>95% CI: ±${confidence_interval:.4f}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    # Path visualization for standard options
    if not path_dependent:
        np.random.seed(42)
        n_paths = 100
        n_time_steps = 252
        dt = time_to_maturity / n_time_steps
        
        paths = np.zeros((n_time_steps + 1, n_paths))
        paths[0] = spot_price
        
        for i in range(n_paths):
            for t in range(1, n_time_steps + 1):
                z = np.random.normal()
                paths[t, i] = paths[t-1, i] * np.exp((risk_free_rate - 0.5*volatility**2)*dt + volatility*np.sqrt(dt)*z)
        
        time_axis = np.linspace(0, time_to_maturity, n_time_steps + 1)
        
        fig_mc = go.Figure()
        
        # Plot sample paths (first 20 for clarity)
        for i in range(min(20, n_paths)):
            fig_mc.add_trace(go.Scatter(
                x=time_axis, y=paths[:, i], 
                name=f'Path {i+1}' if i < 5 else '', 
                line=dict(width=1.5, color=f'rgba({49 + i*10}, {130 + i*5}, {206}, 0.6)'),
                showlegend=i < 5,
                hovertemplate="Time: %{x:.3f}<br>Price: $%{y:.2f}<extra></extra>"
            ))
        
        # Add strike line
        fig_mc.add_hline(
            y=strike_price, 
            line_dash="dash", 
            line_color=COLORS['danger'], 
            annotation_text=f"Strike = ${strike_price}"
        )
        
        fig_mc.update_layout(
            title="Monte Carlo Simulation Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Asset Price ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_mc, use_container_width=True)

# ----------------------
# Greeks Analysis Section with enhanced styling
# ----------------------
st.markdown("""
<div class="greeks-box animate-fade-in">
    <div class="section-title">📈 Greeks & Risk Sensitivities</div>
    <div class="content-text">
        Comprehensive analysis of option sensitivities to market parameters.
    </div>
</div>
""", unsafe_allow_html=True)

# Greeks calculation for current parameters
delta_call, gamma, theta_call, vega, rho_call = OptionPricer.calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')
delta_put, _, theta_put, _, rho_put = OptionPricer.calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'put')

# Enhanced Greeks display
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📞 Call Option Greeks")
    st.markdown(f"""
    <div class="greeks-delta">
        <strong>Delta (Δ): {delta_call:.4f}</strong><br>
        <small>Price sensitivity to underlying movement</small>
    </div>
    <div class="greeks-gamma">
        <strong>Gamma (Γ): {gamma:.4f}</strong><br>
        <small>Rate of change of Delta</small>
    </div>
    <div class="greeks-theta">
        <strong>Theta (Θ): ${theta_call:.2f}</strong><br>
        <small>Time decay per day</small>
    </div>
    <div class="greeks-vega">
        <strong>Vega (ν): {vega:.2f}</strong><br>
        <small>Volatility sensitivity (per 1%)</small>
    </div>
    <div class="greeks-rho">
        <strong>Rho (ρ): {rho_call:.2f}</strong><br>
        <small>Interest rate sensitivity (per 1%)</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📉 Put Option Greeks")
    st.markdown(f"""
    <div class="greeks-delta">
        <strong>Delta (Δ): {delta_put:.4f}</strong><br>
        <small>Price sensitivity to underlying movement</small>
    </div>
    <div class="greeks-gamma">
        <strong>Gamma (Γ): {gamma:.4f}</strong><br>
        <small>Rate of change of Delta (same as call)</small>
    </div>
    <div class="greeks-theta">
        <strong>Theta (Θ): ${theta_put:.2f}</strong><br>
        <small>Time decay per day</small>
    </div>
    <div class="greeks-vega">
        <strong>Vega (ν): {vega:.2f}</strong><br>
        <small>Volatility sensitivity (same as call)</small>
    </div>
    <div class="greeks-rho">
        <strong>Rho (ρ): {rho_put:.2f}</strong><br>
        <small>Interest rate sensitivity (per 1%)</small>
    </div>
    """, unsafe_allow_html=True)

# Greeks visualization
st.markdown("### 📊 Greeks Sensitivity Charts")

S_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 50)
deltas_call = []
deltas_put = []
gammas = []
vegas = []
thetas_call = []

for S in S_range:
    dc, g, tc, v, _ = OptionPricer.calculate_greeks(S, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')
    dp, _, _, _, _ = OptionPricer.calculate_greeks(S, strike_price, time_to_maturity, risk_free_rate, volatility, 'put')
    deltas_call.append(dc)
    deltas_put.append(dp)
    gammas.append(g)
    vegas.append(v)
    thetas_call.append(tc)

# Create enhanced subplots for Greeks
fig_greeks = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Delta (Δ)', 'Gamma (Γ)', 'Vega (ν)', 'Theta (Θ)'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Delta with enhanced colors
fig_greeks.add_trace(go.Scatter(
    x=S_range, y=deltas_call, 
    name='Call Delta', 
    line=dict(color=COLORS['success'], width=3)
), row=1, col=1)
fig_greeks.add_trace(go.Scatter(
    x=S_range, y=deltas_put, 
    name='Put Delta', 
    line=dict(color=COLORS['danger'], width=3)
), row=1, col=1)

# Gamma
fig_greeks.add_trace(go.Scatter(
    x=S_range, y=gammas, 
    name='Gamma', 
    line=dict(color=COLORS['primary'], width=3), 
    showlegend=False
), row=1, col=2)

# Vega
fig_greeks.add_trace(go.Scatter(
    x=S_range, y=vegas, 
    name='Vega', 
    line=dict(color=COLORS['secondary'], width=3), 
    showlegend=False
), row=2, col=1)

# Theta
fig_greeks.add_trace(go.Scatter(
    x=S_range, y=thetas_call, 
    name='Call Theta', 
    line=dict(color=COLORS['info'], width=3), 
    showlegend=False
), row=2, col=2)

# Add current spot price line to all subplots
for row in range(1, 3):
    for col in range(1, 3):
        fig_greeks.add_vline(
            x=spot_price, 
            line_dash="dash", 
            line_color=COLORS['gray_500'], 
            row=row, col=col
        )

fig_greeks.update_layout(height=600, title_text="Greeks Sensitivity Analysis")
fig_greeks.update_xaxes(title_text="Underlying Price ($)")

st.plotly_chart(fig_greeks, use_container_width=True)

# ----------------------
# Options Strategies Section with enhanced styling
# ----------------------
st.markdown("""
<div class="strategy-box animate-fade-in">
    <div class="section-title">🎯 Options Strategies</div>
    <div class="content-text">
        Build and analyze complex multi-leg options strategies with interactive P&L visualization.
    </div>
</div>
""", unsafe_allow_html=True)

# Strategy selection with enhanced tabs
strategy_tabs = st.tabs(["📈 Directional", "📊 Volatility", "🔧 Complex", "🏗️ Custom Builder"])

with strategy_tabs[0]:  # Directional strategies
    st.markdown("### 📈 Directional Strategies")
    
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        st.markdown("#### 🐂 Bull Call Spread")
        K1_bull = st.number_input("Lower Strike (K1)", value=strike_price-10, key="bull_k1")
        K2_bull = st.number_input("Higher Strike (K2)", value=strike_price+10, key="bull_k2")
        
        bull_call_price = OptionsStrategy.bull_call_spread(spot_price, K1_bull, K2_bull, time_to_maturity, risk_free_rate, volatility)
        
        st.markdown(f"""
        <div class="success-box">
            <strong>Strategy Cost: ${bull_call_price:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Bull call spread payoff
        S_exp = np.linspace(spot_price*0.7, spot_price*1.3, 100)
        bull_payoffs = []
        for S in S_exp:
            long_call_payoff = max(S - K1_bull, 0)
            short_call_payoff = max(S - K2_bull, 0)
            bull_payoffs.append(long_call_payoff - short_call_payoff - bull_call_price)
        
        fig_bull = go.Figure()
        fig_bull.add_trace(go.Scatter(
            x=S_exp, y=bull_payoffs, 
            name='Bull Call Spread P&L', 
            fill='tonexty', 
            line=dict(color=COLORS['success'], width=3),
            hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
        ))
        fig_bull.add_hline(y=0, line_dash="dash", line_color=COLORS['gray_500'])
        fig_bull.add_vline(x=spot_price, line_dash="dot", line_color=COLORS['primary'], annotation_text="Current Price")
        
        fig_bull.update_layout(
            title="Bull Call Spread P&L", 
            xaxis_title="Stock Price at Expiration", 
            yaxis_title="Profit/Loss ($)", 
            template="plotly_white", 
            height=300
        )
        st.plotly_chart(fig_bull, use_container_width=True)
    
    with strategy_col2:
        st.markdown("#### 🐻 Bear Put Spread")
        K1_bear = st.number_input("Lower Strike (K1)", value=strike_price-10, key="bear_k1")
        K2_bear = st.number_input("Higher Strike (K2)", value=strike_price+10, key="bear_k2")
        
        bear_put_price = OptionsStrategy.bear_put_spread(spot_price, K1_bear, K2_bear, time_to_maturity, risk_free_rate, volatility)
        
        st.markdown(f"""
        <div class="danger-box">
            <strong>Strategy Cost: ${bear_put_price:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Bear put spread payoff
        bear_payoffs = []
        for S in S_exp:
            long_put_payoff = max(K2_bear - S, 0)
            short_put_payoff = max(K1_bear - S, 0)
            bear_payoffs.append(long_put_payoff - short_put_payoff - bear_put_price)
        
        fig_bear = go.Figure()
        fig_bear.add_trace(go.Scatter(
            x=S_exp, y=bear_payoffs, 
            name='Bear Put Spread P&L', 
            fill='tonexty', 
            line=dict(color=COLORS['danger'], width=3),
            hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
        ))
        fig_bear.add_hline(y=0, line_dash="dash", line_color=COLORS['gray_500'])
        fig_bear.add_vline(x=spot_price, line_dash="dot", line_color=COLORS['primary'], annotation_text="Current Price")
        
        fig_bear.update_layout(
            title="Bear Put Spread P&L", 
            xaxis_title="Stock Price at Expiration", 
            yaxis_title="Profit/Loss ($)", 
            template="plotly_white", 
            height=300
        )
        st.plotly_chart(fig_bear, use_container_width=True)

with strategy_tabs[1]:  # Volatility strategies
    st.markdown("### 📊 Volatility Strategies")
    
    vol_col1, vol_col2 = st.columns(2)
    
    with vol_col1:
        st.markdown("#### 🎯 Long Straddle")
        straddle_strike = st.number_input("Strike Price", value=strike_price, key="straddle_k")
        
        straddle_price = OptionsStrategy.straddle(spot_price, straddle_strike, time_to_maturity, risk_free_rate, volatility)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Strategy Cost: ${straddle_price:.2f}</strong><br>
            <small>Profit if |movement| > ${straddle_price:.2f}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Straddle payoff
        S_exp = np.linspace(spot_price*0.6, spot_price*1.4, 100)
        straddle_payoffs = []
        for S in S_exp:
            call_payoff = max(S - straddle_strike, 0)
            put_payoff = max(straddle_strike - S, 0)
            straddle_payoffs.append(call_payoff + put_payoff - straddle_price)
        
        fig_straddle = go.Figure()
        fig_straddle.add_trace(go.Scatter(
            x=S_exp, y=straddle_payoffs, 
            name='Long Straddle P&L', 
            fill='tozeroy', 
            line=dict(color=COLORS['info'], width=3),
            hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
        ))
        fig_straddle.add_hline(y=0, line_dash="dash", line_color=COLORS['gray_500'])
        fig_straddle.add_vline(x=spot_price, line_dash="dot", line_color=COLORS['primary'], annotation_text="Current Price")
        fig_straddle.add_vline(x=straddle_strike, line_dash="dot", line_color=COLORS['secondary'], annotation_text="Strike")
        
        # Add breakeven lines
        breakeven_up = straddle_strike + straddle_price
        breakeven_down = straddle_strike - straddle_price
        fig_straddle.add_vline(x=breakeven_up, line_dash="dash", line_color=COLORS['success'], annotation_text="Upper BE")
        fig_straddle.add_vline(x=breakeven_down, line_dash="dash", line_color=COLORS['success'], annotation_text="Lower BE")
        
        fig_straddle.update_layout(
            title="Long Straddle P&L", 
            xaxis_title="Stock Price at Expiration", 
            yaxis_title="Profit/Loss ($)", 
            template="plotly_white", 
            height=300
        )
        st.plotly_chart(fig_straddle, use_container_width=True)
    
    with vol_col2:
        st.markdown("#### 🎪 Long Strangle")
        K1_strangle = st.number_input("Put Strike (K1)", value=strike_price-10, key="strangle_k1")
        K2_strangle = st.number_input("Call Strike (K2)", value=strike_price+10, key="strangle_k2")
        
        strangle_price = OptionsStrategy.strangle(spot_price, K1_strangle, K2_strangle, time_to_maturity, risk_free_rate, volatility)
        
        st.markdown(f"""
        <div class="warning-box">
            <strong>Strategy Cost: ${strangle_price:.2f}</strong><br>
            <small>Lower cost than straddle, wider breakeven range</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Strangle payoff
        strangle_payoffs = []
        for S in S_exp:
            call_payoff = max(S - K2_strangle, 0)
            put_payoff = max(K1_strangle - S, 0)
            strangle_payoffs.append(call_payoff + put_payoff - strangle_price)
        
        fig_strangle = go.Figure()
        fig_strangle.add_trace(go.Scatter(
            x=S_exp, y=strangle_payoffs, 
            name='Long Strangle P&L', 
            fill='tozeroy', 
            line=dict(color=COLORS['warning'], width=3),
            hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
        ))
        fig_strangle.add_hline(y=0, line_dash="dash", line_color=COLORS['gray_500'])
        fig_strangle.add_vline(x=spot_price, line_dash="dot", line_color=COLORS['primary'], annotation_text="Current Price")
        fig_strangle.add_vline(x=K1_strangle, line_dash="dot", line_color=COLORS['danger'], annotation_text="Put Strike")
        fig_strangle.add_vline(x=K2_strangle, line_dash="dot", line_color=COLORS['success'], annotation_text="Call Strike")
        
        # Add breakeven lines
        breakeven_up = K2_strangle + strangle_price
        breakeven_down = K1_strangle - strangle_price
        fig_strangle.add_vline(x=breakeven_up, line_dash="dash", line_color=COLORS['success'], annotation_text="Upper BE")
        fig_strangle.add_vline(x=breakeven_down, line_dash="dash", line_color=COLORS['success'], annotation_text="Lower BE")
        
        fig_strangle.update_layout(
            title="Long Strangle P&L", 
            xaxis_title="Stock Price at Expiration", 
            yaxis_title="Profit/Loss ($)", 
            template="plotly_white", 
            height=300
        )
        st.plotly_chart(fig_strangle, use_container_width=True)

with strategy_tabs[2]:  # Complex strategies
    st.markdown("### 🔧 Complex Strategies")
    
    complex_col1, complex_col2 = st.columns(2)
    
    with complex_col1:
        st.markdown("#### 🦋 Butterfly Spread")
        st.markdown("*Long 2 middle strikes, Short 1 lower + 1 higher*")
        
        K1_butterfly = st.number_input("Lower Strike", value=strike_price-15, key="butterfly_k1")
        K2_butterfly = st.number_input("Middle Strike", value=strike_price, key="butterfly_k2")
        K3_butterfly = st.number_input("Higher Strike", value=strike_price+15, key="butterfly_k3")
        
        # Calculate butterfly spread cost
        long_call_low = OptionPricer.black_scholes_call(spot_price, K1_butterfly, time_to_maturity, risk_free_rate, volatility)
        short_call_mid = OptionPricer.black_scholes_call(spot_price, K2_butterfly, time_to_maturity, risk_free_rate, volatility)
        long_call_high = OptionPricer.black_scholes_call(spot_price, K3_butterfly, time_to_maturity, risk_free_rate, volatility)
        butterfly_cost = long_call_low - 2*short_call_mid + long_call_high
        
        st.markdown(f"""
        <div class="primary-box">
            <strong>Net Cost: ${butterfly_cost:.2f}</strong><br>
            <small>Max Profit: ${(K2_butterfly - K1_butterfly) - butterfly_cost:.2f}</small><br>
            <small>At Stock Price: ${K2_butterfly}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Butterfly payoff
        butterfly_payoffs = []
        for S in S_exp:
            payoff_1 = max(S - K1_butterfly, 0)  # Long lower call
            payoff_2 = -2 * max(S - K2_butterfly, 0)  # Short 2 middle calls
            payoff_3 = max(S - K3_butterfly, 0)  # Long higher call
            butterfly_payoffs.append(payoff_1 + payoff_2 + payoff_3 - butterfly_cost)
        
        fig_butterfly = go.Figure()
        fig_butterfly.add_trace(go.Scatter(
            x=S_exp, y=butterfly_payoffs, 
            name='Butterfly Spread P&L', 
            fill='tozeroy', 
            line=dict(color=COLORS['primary'], width=3),
            hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
        ))
        fig_butterfly.add_hline(y=0, line_dash="dash", line_color=COLORS['gray_500'])
        fig_butterfly.add_vline(x=spot_price, line_dash="dot", line_color=COLORS['info'], annotation_text="Current")
        fig_butterfly.add_vline(x=K2_butterfly, line_dash="dot", line_color=COLORS['success'], annotation_text="Max Profit")
        
        fig_butterfly.update_layout(
            title="Butterfly Spread P&L", 
            xaxis_title="Stock Price at Expiration", 
            yaxis_title="Profit/Loss ($)", 
            template="plotly_white", 
            height=300
        )
        st.plotly_chart(fig_butterfly, use_container_width=True)
    
    with complex_col2:
        st.markdown("#### 🪃 Iron Condor")
        st.markdown("*Short Strangle + Long Strangle (wider)*")
        
        K1_condor = st.number_input("Long Put Strike", value=strike_price-20, key="condor_k1")
        K2_condor = st.number_input("Short Put Strike", value=strike_price-10, key="condor_k2")
        K3_condor = st.number_input("Short Call Strike", value=strike_price+10, key="condor_k3")
        K4_condor = st.number_input("Long Call Strike", value=strike_price+20, key="condor_k4")
        
        # Calculate iron condor cost (net credit)
        long_put_low = OptionPricer.black_scholes_put(spot_price, K1_condor, time_to_maturity, risk_free_rate, volatility)
        short_put_mid = OptionPricer.black_scholes_put(spot_price, K2_condor, time_to_maturity, risk_free_rate, volatility)
        short_call_mid = OptionPricer.black_scholes_call(spot_price, K3_condor, time_to_maturity, risk_free_rate, volatility)
        long_call_high = OptionPricer.black_scholes_call(spot_price, K4_condor, time_to_maturity, risk_free_rate, volatility)
        condor_credit = short_put_mid + short_call_mid - long_put_low - long_call_high
        
        st.markdown(f"""
        <div class="secondary-box">
            <strong>Net Credit: ${condor_credit:.2f}</strong><br>
            <small>Max Profit: ${condor_credit:.2f}</small><br>
            <small>If price stays between ${K2_condor}-${K3_condor}</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Iron condor payoff
        condor_payoffs = []
        for S in S_exp:
            payoff_1 = -max(K1_condor - S, 0)  # Short long put
            payoff_2 = max(K2_condor - S, 0)   # Long short put
            payoff_3 = max(S - K3_condor, 0)   # Long short call
            payoff_4 = -max(S - K4_condor, 0)  # Short long call
            condor_payoffs.append(payoff_1 + payoff_2 + payoff_3 + payoff_4 + condor_credit)
        
        fig_condor = go.Figure()
        fig_condor.add_trace(go.Scatter(
            x=S_exp, y=condor_payoffs, 
            name='Iron Condor P&L', 
            fill='tozeroy', 
            line=dict(color=COLORS['secondary'], width=3),
            hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
        ))
        fig_condor.add_hline(y=0, line_dash="dash", line_color=COLORS['gray_500'])
        fig_condor.add_vline(x=spot_price, line_dash="dot", line_color=COLORS['info'], annotation_text="Current")
        fig_condor.add_vrect(x0=K2_condor, x1=K3_condor, fillcolor=COLORS['success'], opacity=0.2, annotation_text="Max Profit Zone")
        
        fig_condor.update_layout(
            title="Iron Condor P&L", 
            xaxis_title="Stock Price at Expiration", 
            yaxis_title="Profit/Loss ($)", 
            template="plotly_white", 
            height=300
        )
        st.plotly_chart(fig_condor, use_container_width=True)

with strategy_tabs[3]:  # Custom Builder
    st.markdown("### 🏗️ Custom Strategy Builder")
    st.markdown("Build your own multi-leg options strategy")
    
    # Initialize session state for legs
    if 'strategy_legs' not in st.session_state:
        st.session_state.strategy_legs = []
    
    # Add new leg section
    st.markdown("#### ➕ Add Strategy Leg")
    
    add_col1, add_col2, add_col3, add_col4, add_col5 = st.columns(5)
    
    with add_col1:
        leg_type = st.selectbox("Type", ["Call", "Put"], key="leg_type")
    with add_col2:
        leg_position = st.selectbox("Position", ["Long", "Short"], key="leg_position")
    with add_col3:
        leg_strike = st.number_input("Strike", value=strike_price, key="leg_strike")
    with add_col4:
        leg_quantity = st.number_input("Quantity", value=1, min_value=1, max_value=10, key="leg_quantity")
    with add_col5:
        if st.button("➕ Add Leg", type="primary"):
            # Calculate option price
            if leg_type == "Call":
                option_price = OptionPricer.black_scholes_call(spot_price, leg_strike, time_to_maturity, risk_free_rate, volatility)
            else:
                option_price = OptionPricer.black_scholes_put(spot_price, leg_strike, time_to_maturity, risk_free_rate, volatility)
            
            # Add to strategy
            leg = {
                'type': leg_type,
                'position': leg_position,
                'strike': leg_strike,
                'quantity': leg_quantity,
                'price': option_price
            }
            st.session_state.strategy_legs.append(leg)
            st.success(f"Added {leg_position} {leg_quantity}x {leg_type} @ ${leg_strike}")
    
    # Display current strategy
    if st.session_state.strategy_legs:
        st.markdown("#### 📋 Current Strategy")
        
        # Create strategy summary
        strategy_df = pd.DataFrame(st.session_state.strategy_legs)
        strategy_df['Net Cost'] = strategy_df.apply(lambda row: 
            row['price'] * row['quantity'] * (-1 if row['position'] == 'Long' else 1), axis=1)
        
        # Display strategy table
        st.dataframe(strategy_df[['position', 'type', 'quantity', 'strike', 'price', 'Net Cost']], 
                    use_container_width=True)
        
        # Calculate total cost
        total_cost = strategy_df['Net Cost'].sum()
        net_position = "Net Credit" if total_cost > 0 else "Net Debit"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="{'success-box' if total_cost > 0 else 'danger-box'}">
                <strong>{net_position}: ${abs(total_cost):.2f}</strong>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("🗑️ Clear Strategy", type="secondary"):
                st.session_state.strategy_legs = []
                st.rerun()
        with col3:
            if st.button("📊 Analyze Strategy", type="primary"):
                # Calculate custom strategy P&L
                S_exp = np.linspace(spot_price*0.6, spot_price*1.4, 100)
                custom_payoffs = []
                
                for S in S_exp:
                    total_payoff = -total_cost  # Start with net cost
                    
                    for leg in st.session_state.strategy_legs:
                        if leg['type'] == 'Call':
                            intrinsic = max(S - leg['strike'], 0)
                        else:
                            intrinsic = max(leg['strike'] - S, 0)
                        
                        if leg['position'] == 'Long':
                            total_payoff += intrinsic * leg['quantity']
                        else:
                            total_payoff -= intrinsic * leg['quantity']
                    
                    custom_payoffs.append(total_payoff)
                
                # Plot custom strategy
                fig_custom = go.Figure()
                fig_custom.add_trace(go.Scatter(
                    x=S_exp, y=custom_payoffs, 
                    name='Custom Strategy P&L', 
                    fill='tozeroy', 
                    line=dict(color=COLORS['primary'], width=4),
                    hovertemplate="Stock Price: $%{x:.2f}<br>P&L: $%{y:.2f}<extra></extra>"
                ))
                fig_custom.add_hline(y=0, line_dash="dash", line_color=COLORS['gray_500'])
                fig_custom.add_vline(x=spot_price, line_dash="dot", line_color=COLORS['info'], annotation_text="Current Price")
                
                # Add strike lines
                strikes = [leg['strike'] for leg in st.session_state.strategy_legs]
                for strike in set(strikes):
                    fig_custom.add_vline(x=strike, line_dash="dot", line_color=COLORS['gray_500'], opacity=0.7)
                
                fig_custom.update_layout(
                    title="Custom Strategy P&L Analysis", 
                    xaxis_title="Stock Price at Expiration", 
                    yaxis_title="Profit/Loss ($)", 
                    template="plotly_white", 
                    height=400
                )
                st.plotly_chart(fig_custom, use_container_width=True)
                
                # Strategy analytics
                max_profit = max(custom_payoffs)
                max_loss = min(custom_payoffs)
                breakeven_points = []
                
                # Find breakeven points (where P&L crosses zero)
                for i in range(len(custom_payoffs)-1):
                    if (custom_payoffs[i] <= 0 <= custom_payoffs[i+1]) or (custom_payoffs[i] >= 0 >= custom_payoffs[i+1]):
                        # Linear interpolation to find exact breakeven
                        if custom_payoffs[i+1] != custom_payoffs[i]:
                            be_price = S_exp[i] + (S_exp[i+1] - S_exp[i]) * (-custom_payoffs[i]) / (custom_payoffs[i+1] - custom_payoffs[i])
                            breakeven_points.append(be_price)
                
                # Display analytics
                analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
                
                with analytics_col1:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>Max Profit: ${max_profit:.2f}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                with analytics_col2:
                    st.markdown(f"""
                    <div class="danger-box">
                        <strong>Max Loss: ${max_loss:.2f}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                with analytics_col3:
                    if breakeven_points:
                        be_text = ", ".join([f"${be:.2f}" for be in breakeven_points])
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Breakeven: {be_text}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>No Breakeven Found</strong>
                        </div>
                        """, unsafe_allow_html=True)
    
    else:
        st.info("No strategy legs added yet. Add your first leg above to get started!")
        
        # Show some example strategies
        st.markdown("#### 💡 Example Strategies")
        
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("📈 Load Bull Call Spread"):
                st.session_state.strategy_legs = [
                    {'type': 'Call', 'position': 'Long', 'strike': strike_price-5, 'quantity': 1, 
                     'price': OptionPricer.black_scholes_call(spot_price, strike_price-5, time_to_maturity, risk_free_rate, volatility)},
                    {'type': 'Call', 'position': 'Short', 'strike': strike_price+5, 'quantity': 1, 
                     'price': OptionPricer.black_scholes_call(spot_price, strike_price+5, time_to_maturity, risk_free_rate, volatility)}
                ]
                st.rerun()
        
        with example_col2:
            if st.button("🎯 Load Long Straddle"):
                call_price = OptionPricer.black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
                put_price = OptionPricer.black_scholes_put(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
                st.session_state.strategy_legs = [
                    {'type': 'Call', 'position': 'Long', 'strike': strike_price, 'quantity': 1, 'price': call_price},
                    {'type': 'Put', 'position': 'Long', 'strike': strike_price, 'quantity': 1, 'price': put_price}
                ]
                st.rerun()
        
        with example_col3:
            if st.button("🦋 Load Iron Butterfly"):
                st.session_state.strategy_legs = [
                    {'type': 'Put', 'position': 'Long', 'strike': strike_price-10, 'quantity': 1, 
                     'price': OptionPricer.black_scholes_put(spot_price, strike_price-10, time_to_maturity, risk_free_rate, volatility)},
                    {'type': 'Put', 'position': 'Short', 'strike': strike_price, 'quantity': 1, 
                     'price': OptionPricer.black_scholes_put(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)},
                    {'type': 'Call', 'position': 'Short', 'strike': strike_price, 'quantity': 1, 
                     'price': OptionPricer.black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)},
                    {'type': 'Call', 'position': 'Long', 'strike': strike_price+10, 'quantity': 1, 
                     'price': OptionPricer.black_scholes_call(spot_price, strike_price+10, time_to_maturity, risk_free_rate, volatility)}
                ]
                st.rerun()

# ----------------------
# Footer with enhanced styling
# ----------------------
st.markdown("""
<div class="footer-section animate-fade-in">
    <div style="font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; color: #1a365d;">
        📊 Quantitative Finance Platform
    </div>
    <div style="color: #4a5568; font-style: italic; margin-bottom: 1rem;">
        Professional-Grade Derivatives Pricing & Risk Management
    </div>
    <div style="color: #718096; font-size: 0.9rem;">
        © 2025 | SALHI Reda | Financial Engineering Research | Advanced Analytics Suite
    </div>
    <div style="margin-top: 1rem; color: #718096; font-size: 0.8rem;">
        <strong>Disclaimer:</strong> This platform is for educational and research purposes. 
        All models are theoretical and should not be used for actual trading without proper validation.
    </div>
</div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
