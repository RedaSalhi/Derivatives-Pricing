import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Professional CSS with solid colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #1a365d;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        animation: fadeInUp 1s ease-out;
    }
    
    .subtitle {
        font-size: 1.4rem;
        text-align: center;
        color: #2d3748;
        margin-bottom: 3rem;
        font-weight: 500;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Professional sections */
    .objective-box {
        background: #f7fafc;
        border: 2px solid #e2e8f0;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #3182ce;
    }
    
    .methodology-box {
        background: #fffaf0;
        border: 2px solid #fed7aa;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #d69e2e;
    }
    
    .model-box {
        background: #f0fff4;
        border: 2px solid #c6f6d5;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #38a169;
    }
    
    .greeks-box {
        background: #fff5f5;
        border: 2px solid #fed7d7;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #e53e3e;
    }
    
    .strategy-box {
        background: #f0f8ff;
        border: 2px solid #bee3f8;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #3182ce;
    }
    
    .risk-box {
        background: #fafafa;
        border: 2px solid #e2e8f0;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #718096;
    }
    
    .portfolio-box {
        background: #f7fafc;
        border: 2px solid #cbd5e0;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #4a5568;
    }
    
    .volatility-box {
        background: #fef5e7;
        border: 2px solid #f6e05e;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #d69e2e;
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 2rem;
        color: #1a365d;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        border-bottom: 3px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    .subsection-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 2rem 0 1.5rem 0;
        padding: 1rem 1.5rem;
        background: #edf2f7;
        border-left: 4px solid #3182ce;
        border-radius: 4px;
        color: #2d3748;
    }
    
    .content-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2d3748;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    .highlight-box {
        background: #ebf8ff;
        border: 1px solid #bee3f8;
        border-left: 4px solid #3182ce;
        padding: 1.5rem 2rem;
        margin: 2rem 0;
        border-radius: 4px;
    }
    
    .greek-item, .strategy-item, .model-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .greek-item:hover, .strategy-item:hover, .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-color: #3182ce;
    }
    
    .metric-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        border-color: #3182ce;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .parameter-card {
        background: #f7fafc;
        border: 1px solid #cbd5e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .strategy-button {
        background: #3182ce;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.25rem;
    }
    
    .strategy-button:hover {
        background: #2c5282;
        transform: translateY(-1px);
    }
    
    .footer-section {
        text-align: center;
        margin-top: 4rem;
        padding: 3rem;
        background: #f7fafc;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
    }
    
    .warning-box {
        background: #fffaf0;
        border: 1px solid #fed7aa;
        border-left: 4px solid #d69e2e;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #f0fff4;
        border: 1px solid #c6f6d5;
        border-left: 4px solid #38a169;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #ebf8ff;
        border: 1px solid #bee3f8;
        border-left: 4px solid #3182ce;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
        }
        .section-title {
            font-size: 1.6rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Core pricing functions
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

# Strategy classes
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
    
    @staticmethod
    def butterfly_call(S, K1, K2, K3, T, r, sigma):
        """Butterfly Spread: Long calls at K1,K3, short 2 calls at K2"""
        call1 = OptionPricer.black_scholes_call(S, K1, T, r, sigma)
        call2 = OptionPricer.black_scholes_call(S, K2, T, r, sigma)
        call3 = OptionPricer.black_scholes_call(S, K3, T, r, sigma)
        return call1 - 2*call2 + call3
    
    @staticmethod
    def iron_condor(S, K1, K2, K3, K4, T, r, sigma):
        """Iron Condor: Bull put spread + bear call spread"""
        bull_put = OptionsStrategy.bear_put_spread(S, K1, K2, T, r, sigma)
        bear_call = -OptionsStrategy.bull_call_spread(S, K3, K4, T, r, sigma)
        return -(bull_put + bear_call)  # Net credit received

# Risk management functions
class RiskManager:
    @staticmethod
    def value_at_risk(returns, confidence_level=0.05):
        """Calculate Value at Risk"""
        sorted_returns = np.sort(returns)
        index = int(confidence_level * len(sorted_returns))
        return abs(sorted_returns[index])
    
    @staticmethod
    def expected_shortfall(returns, confidence_level=0.05):
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = RiskManager.value_at_risk(returns, confidence_level)
        return abs(np.mean(returns[returns <= -var]))
    
    @staticmethod
    def portfolio_var(weights, cov_matrix, confidence_level=0.05):
        """Portfolio VaR using variance-covariance method"""
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        var = norm.ppf(confidence_level) * portfolio_std
        return abs(var)

# Initialize default parameters
default_params = {
    'S': 100,
    'K': 100,
    'T': 1.0,
    'r': 0.05,
    'sigma': 0.20
}

# Main header
st.markdown('<div class="main-header">Quantitative Finance Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Derivatives Pricing & Risk Management System</div>', unsafe_allow_html=True)

# Interactive parameter controls at the top
st.markdown("## 🎛️ Interactive Controls")
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

# Quick metrics display
st.markdown("### 📊 Quick Pricing Overview")
call_price = OptionPricer.black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
put_price = OptionPricer.black_scholes_put(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
delta_call, gamma, theta_call, vega, rho_call = OptionPricer.calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Call Price</h4>
        <h3>${call_price:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Put Price</h4>
        <h3>${put_price:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Delta</h4>
        <h3>{delta_call:.3f}</h3>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Gamma</h4>
        <h3>{gamma:.3f}</h3>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Vega</h4>
        <h3>{vega:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Objective Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="objective-box">
        <div class="section-title">🎯 Platform Overview</div>
        <div class="content-text">
            This comprehensive quantitative finance platform provides state-of-the-art tools for derivatives pricing, risk management, and portfolio analysis. Designed for practitioners, researchers, and students, it combines theoretical rigor with practical implementation.
        </div>
        <div class="highlight-box">
            <strong>Core Capabilities:</strong>
            <ul>
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
# Pricing Models Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="model-box">
        <div class="section-title">⚙️ Pricing Models Comparison</div>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison
    tab1, tab2, tab3 = st.tabs(["📊 Black-Scholes Analysis", "🌳 Binomial Trees", "🎲 Monte Carlo Methods"])
    
    with tab1:
        st.markdown("""
        <div class="model-card">
            <div class="subsection-title">Black-Scholes Framework</div>
            <div class="content-text">
                The foundation of modern derivatives pricing, providing analytical solutions for European options.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r'''
        C = S_0 N(d_1) - K e^{-rT} N(d_2) \quad \text{where} \quad
        d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left( r + \frac{1}{2}\sigma^2 \right) T}{\sigma \sqrt{T}}
        ''')
        
        # Sensitivity analysis
        S_range = np.linspace(spot_price * 0.5, spot_price * 1.5, 100)
        call_prices = [OptionPricer.black_scholes_call(S, strike_price, time_to_maturity, risk_free_rate, volatility) for S in S_range]
        put_prices = [OptionPricer.black_scholes_put(S, strike_price, time_to_maturity, risk_free_rate, volatility) for S in S_range]
        
        fig_bs = go.Figure()
        fig_bs.add_trace(go.Scatter(x=S_range, y=call_prices, name='Call Option', line=dict(color='#38a169', width=3)))
        fig_bs.add_trace(go.Scatter(x=S_range, y=put_prices, name='Put Option', line=dict(color='#e53e3e', width=3)))
        fig_bs.add_vline(x=spot_price, line_dash="dash", line_color="#3182ce", annotation_text=f"Current S₀ = ${spot_price}")
        fig_bs.add_vline(x=strike_price, line_dash="dot", line_color="#d69e2e", annotation_text=f"Strike K = ${strike_price}")
        
        fig_bs.update_layout(
            title="Black-Scholes Option Prices vs Underlying",
            xaxis_title="Underlying Price ($)",
            yaxis_title="Option Value ($)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_bs, use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="model-card">
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
            st.metric("Binomial Price", f"${binomial_price:.4f}")
        with col2:
            st.metric("Black-Scholes Price", f"${bs_price:.4f}")
        with col3:
            difference = binomial_price - bs_price
            st.metric("Difference", f"${difference:.4f}")
        
        # Convergence analysis
        steps_range = range(10, 101, 10)
        binomial_prices = [OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, n, option_type, american_option) for n in steps_range]
        
        fig_binomial = go.Figure()
        fig_binomial.add_trace(go.Scatter(x=list(steps_range), y=binomial_prices, name='Binomial Price', mode='lines+markers'))
        fig_binomial.add_hline(y=bs_price, line_dash="dash", line_color="#e53e3e", annotation_text=f"BS Price: ${bs_price:.4f}")
        
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
        <div class="model-card">
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
        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                mc_price, mc_se = OptionPricer.monte_carlo_option(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, n_simulations, option_type=mc_option_type, path_dependent=path_dependent)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MC Price", f"${mc_price:.4f}")
                with col2:
                    st.metric("Standard Error", f"${mc_se:.4f}")
                with col3:
                    confidence_interval = 1.96 * mc_se
                    st.metric("95% CI", f"±${confidence_interval:.4f}")
        
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
                fig_mc.add_trace(go.Scatter(x=time_axis, y=paths[:, i], 
                                           name=f'Path {i+1}' if i < 5 else '', 
                                           line=dict(width=1, color='rgba(49, 130, 206, 0.3)'),
                                           showlegend=i < 5))
            
            # Add strike line
            fig_mc.add_hline(y=strike_price, line_dash="dash", line_color="#e53e3e", 
                            annotation_text=f"Strike = ${strike_price}")
            
            fig_mc.update_layout(
                title="Monte Carlo Simulation Paths",
                xaxis_title="Time (Years)",
                yaxis_title="Asset Price ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)

# ----------------------
# Greeks Analysis Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="greeks-box">
        <div class="section-title">🇬🇷 Greeks & Risk Sensitivities</div>
        <div class="content-text">
            Comprehensive analysis of option sensitivities to market parameters.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Greeks calculation for current parameters
    delta_call, gamma, theta_call, vega, rho_call = OptionPricer.calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')
    delta_put, _, theta_put, _, rho_put = OptionPricer.calculate_greeks(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'put')
    
    # Greeks display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Call Option Greeks")
        st.markdown(f"""
        <div class="greek-item">
            <strong>Delta (Δ): {delta_call:.4f}</strong><br>
            <small>Price sensitivity to underlying movement</small>
        </div>
        <div class="greek-item">
            <strong>Gamma (Γ): {gamma:.4f}</strong><br>
            <small>Rate of change of Delta</small>
        </div>
        <div class="greek-item">
            <strong>Theta (Θ): ${theta_call:.2f}</strong><br>
            <small>Time decay per day</small>
        </div>
        <div class="greek-item">
            <strong>Vega (ν): ${vega:.2f}</strong><br>
            <small>Volatility sensitivity (per 1%)</small>
        </div>
        <div class="greek-item">
            <strong>Rho (ρ): ${rho_call:.2f}</strong><br>
            <small>Interest rate sensitivity (per 1%)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Put Option Greeks")
        st.markdown(f"""
        <div class="greek-item">
            <strong>Delta (Δ): {delta_put:.4f}</strong><br>
            <small>Price sensitivity to underlying movement</small>
        </div>
        <div class="greek-item">
            <strong>Gamma (Γ): {gamma:.4f}</strong><br>
            <small>Rate of change of Delta (same as call)</small>
        </div>
        <div class="greek-item">
            <strong>Theta (Θ): ${theta_put:.2f}</strong><br>
            <small>Time decay per day</small>
        </div>
        <div class="greek-item">
            <strong>Vega (ν): ${vega:.2f}</strong><br>
            <small>Volatility sensitivity (same as call)</small>
        </div>
        <div class="greek-item">
            <strong>Rho (ρ): ${rho_put:.2f}</strong><br>
            <small>Interest rate sensitivity (per 1%)</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Greeks visualization
    st.markdown("### Greeks Sensitivity Charts")
    
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
    
    # Create subplots for Greeks
    from plotly.subplots import make_subplots
    
    fig_greeks = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Delta
    fig_greeks.add_trace(go.Scatter(x=S_range, y=deltas_call, name='Call Delta', line=dict(color='#38a169')), row=1, col=1)
    fig_greeks.add_trace(go.Scatter(x=S_range, y=deltas_put, name='Put Delta', line=dict(color='#e53e3e')), row=1, col=1)
    
    # Gamma
    fig_greeks.add_trace(go.Scatter(x=S_range, y=gammas, name='Gamma', line=dict(color='#3182ce'), showlegend=False), row=1, col=2)
    
    # Vega
    fig_greeks.add_trace(go.Scatter(x=S_range, y=vegas, name='Vega', line=dict(color='#d69e2e'), showlegend=False), row=2, col=1)
    
    # Theta
    fig_greeks.add_trace(go.Scatter(x=S_range, y=thetas_call, name='Call Theta', line=dict(color='#9f7aea'), showlegend=False), row=2, col=2)
    
    # Add current spot price line to all subplots
    for row in range(1, 3):
        for col in range(1, 3):
            fig_greeks.add_vline(x=spot_price, line_dash="dash", line_color="#718096", row=row, col=col)
    
    fig_greeks.update_layout(height=600, title_text="Greeks Sensitivity Analysis")
    fig_greeks.update_xaxes(title_text="Underlying Price ($)")
    
    st.plotly_chart(fig_greeks, use_container_width=True)

# ----------------------
# Options Strategies Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="strategy-box">
        <div class="section-title">🔄 Options Strategies</div>
        <div class="content-text">
            Build and analyze complex multi-leg options strategies with interactive P&L visualization.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy selection
    strategy_tabs = st.tabs(["📈 Directional", "📊 Volatility", "🦋 Complex", "💰 Custom Builder"])
    
    with strategy_tabs[0]:  # Directional strategies
        st.markdown("### Directional Strategies")
        
        strategy_col1, strategy_col2 = st.columns(2)
        
        with strategy_col1:
            st.markdown("#### Bull Call Spread")
            K1_bull = st.number_input("Lower Strike (K1)", value=strike_price-10, key="bull_k1")
            K2_bull = st.number_input("Higher Strike (K2)", value=strike_price+10, key="bull_k2")
            
            bull_call_price = OptionsStrategy.bull_call_spread(spot_price, K1_bull, K2_bull, time_to_maturity, risk_free_rate, volatility)
            st.metric("Strategy Cost", f"${bull_call_price:.2f}")
            
            # Bull call spread payoff
            S_exp = np.linspace(spot_price*0.7, spot_price*1.3, 100)
            bull_payoffs = []
            for S in S_exp:
                long_call_payoff = max(S - K1_bull, 0)
                short_call_payoff = max(S - K2_bull, 0)
                bull_payoffs.append(long_call_payoff - short_call_payoff - bull_call_price)
            
            fig_bull = go.Figure()
            fig_bull.add_trace(go.Scatter(x=S_exp, y=bull_payoffs, name='Bull Call Spread P&L', 
                                        fill='tonexty', line=dict(color='#38a169', width=3)))
            fig_bull.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_bull.add_vline(x=spot_price, line_dash="dot", line_color="#3182ce", annotation_text="Current Price")
            
            fig_bull.update_layout(title="Bull Call Spread P&L", xaxis_title="Stock Price at Expiration", 
                                 yaxis_title="Profit/Loss ($)", template="plotly_white", height=300)
            st.plotly_chart(fig_bull, use_container_width=True)
        
        with strategy_col2:
            st.markdown("#### Bear Put Spread")
            K1_bear = st.number_input("Lower Strike (K1)", value=strike_price-10, key="bear_k1")
            K2_bear = st.number_input("Higher Strike (K2)", value=strike_price+10, key="bear_k2")
            
            bear_put_price = OptionsStrategy.bear_put_spread(spot_price, K1_bear, K2_bear, time_to_maturity, risk_free_rate, volatility)
            st.metric("Strategy Cost", f"${bear_put_price:.2f}")
            
            # Bear put spread payoff
            bear_payoffs = []
            for S in S_exp:
                long_put_payoff = max(K2_bear - S, 0)
                short_put_payoff = max(K1_bear - S, 0)
                bear_payoffs.append(long_put_payoff - short_put_payoff - bear_put_price)
            
            fig_bear = go.Figure()
            fig_bear.add_trace(go.Scatter(x=S_exp, y=bear_payoffs, name='Bear Put Spread P&L', 
                                        fill='tonexty', line=dict(color='#e53e3e', width=3)))
            fig_bear.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_bear.add_vline(x=spot_price, line_dash="dot", line_color="#3182ce", annotation_text="Current Price")
            
            fig_bear.update_layout(title="Bear Put Spread P&L", xaxis_title="Stock Price at Expiration", 
                                 yaxis_title="Profit/Loss ($)", template="plotly_white", height=300)
            st.plotly_chart(fig_bear, use_container_width=True)
    
    with strategy_tabs[1]:  # Volatility strategies
        st.markdown("### Volatility Strategies")
        
        vol_col1, vol_col2 = st.columns(2)
        
        with vol_col1:
            st.markdown("#### Long Straddle")
            K_straddle = st.number_input("Strike Price", value=strike_price, key="straddle_k")
            
            straddle_price = OptionsStrategy.straddle(spot_price, K_straddle, time_to_maturity, risk_free_rate, volatility)
            st.metric("Strategy Cost", f"${straddle_price:.2f}")
            
            # Straddle payoff
            straddle_payoffs = []
            for S in S_exp:
                call_payoff = max(S - K_straddle, 0)
                put_payoff = max(K_straddle - S, 0)
                straddle_payoffs.append(call_payoff + put_payoff - straddle_price)
            
            fig_straddle = go.Figure()
            fig_straddle.add_trace(go.Scatter(x=S_exp, y=straddle_payoffs, name='Long Straddle P&L', 
                                            fill='tonexty', line=dict(color='#9f7aea', width=3)))
            fig_straddle.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_straddle.add_vline(x=spot_price, line_dash="dot", line_color="#3182ce", annotation_text="Current Price")
            
            fig_straddle.update_layout(title="Long Straddle P&L", xaxis_title="Stock Price at Expiration", 
                                     yaxis_title="Profit/Loss ($)", template="plotly_white", height=300)
            st.plotly_chart(fig_straddle, use_container_width=True)
        
        with vol_col2:
            st.markdown("#### Long Strangle")
            K1_strangle = st.number_input("Put Strike (K1)", value=strike_price-10, key="strangle_k1")
            K2_strangle = st.number_input("Call Strike (K2)", value=strike_price+10, key="strangle_k2")
            
            strangle_price = OptionsStrategy.strangle(spot_price, K1_strangle, K2_strangle, time_to_maturity, risk_free_rate, volatility)
            st.metric("Strategy Cost", f"${strangle_price:.2f}")
            
            # Strangle payoff
            strangle_payoffs = []
            for S in S_exp:
                call_payoff = max(S - K2_strangle, 0)
                put_payoff = max(K1_strangle - S, 0)
                strangle_payoffs.append(call_payoff + put_payoff - strangle_price)
            
            fig_strangle = go.Figure()
            fig_strangle.add_trace(go.Scatter(x=S_exp, y=strangle_payoffs, name='Long Strangle P&L', 
                                            fill='tonexty', line=dict(color='#d69e2e', width=3)))
            fig_strangle.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_strangle.add_vline(x=spot_price, line_dash="dot", line_color="#3182ce", annotation_text="Current Price")
            
            fig_strangle.update_layout(title="Long Strangle P&L", xaxis_title="Stock Price at Expiration", 
                                     yaxis_title="Profit/Loss ($)", template="plotly_white", height=300)
            st.plotly_chart(fig_strangle, use_container_width=True)
    
    with strategy_tabs[2]:  # Complex strategies
        st.markdown("### Complex Strategies")
        
        complex_col1, complex_col2 = st.columns(2)
        
        with complex_col1:
            st.markdown("#### Butterfly Spread")
            K1_butterfly = st.number_input("Lower Strike", value=strike_price-10, key="butterfly_k1")
            K2_butterfly = st.number_input("Middle Strike", value=strike_price, key="butterfly_k2")
            K3_butterfly = st.number_input("Upper Strike", value=strike_price+10, key="butterfly_k3")
            
            butterfly_price = OptionsStrategy.butterfly_call(spot_price, K1_butterfly, K2_butterfly, K3_butterfly, time_to_maturity, risk_free_rate, volatility)
            st.metric("Strategy Cost", f"${butterfly_price:.2f}")
            
            # Butterfly payoff
            butterfly_payoffs = []
            for S in S_exp:
                payoff1 = max(S - K1_butterfly, 0)
                payoff2 = max(S - K2_butterfly, 0)
                payoff3 = max(S - K3_butterfly, 0)
                butterfly_payoffs.append(payoff1 - 2*payoff2 + payoff3 - butterfly_price)
            
            fig_butterfly = go.Figure()
            fig_butterfly.add_trace(go.Scatter(x=S_exp, y=butterfly_payoffs, name='Butterfly Spread P&L', 
                                             fill='tonexty', line=dict(color='#f56565', width=3)))
            fig_butterfly.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_butterfly.add_vline(x=spot_price, line_dash="dot", line_color="#3182ce", annotation_text="Current Price")
            
            fig_butterfly.update_layout(title="Butterfly Spread P&L", xaxis_title="Stock Price at Expiration", 
                                       yaxis_title="Profit/Loss ($)", template="plotly_white", height=300)
            st.plotly_chart(fig_butterfly, use_container_width=True)
        
        with complex_col2:
            st.markdown("#### Iron Condor")
            K1_condor = st.number_input("Put Strike 1", value=strike_price-20, key="condor_k1")
            K2_condor = st.number_input("Put Strike 2", value=strike_price-10, key="condor_k2")
            K3_condor = st.number_input("Call Strike 1", value=strike_price+10, key="condor_k3")
            K4_condor = st.number_input("Call Strike 2", value=strike_price+20, key="condor_k4")
            
            condor_price = OptionsStrategy.iron_condor(spot_price, K1_condor, K2_condor, K3_condor, K4_condor, time_to_maturity, risk_free_rate, volatility)
            st.metric("Net Credit", f"${condor_price:.2f}")
            
            # Iron condor payoff
            condor_payoffs = []
            for S in S_exp:
                put_spread = max(K2_condor - S, 0) - max(K1_condor - S, 0)
                call_spread = max(S - K3_condor, 0) - max(S - K4_condor, 0)
                condor_payoffs.append(condor_price - put_spread - call_spread)
            
            fig_condor = go.Figure()
            fig_condor.add_trace(go.Scatter(x=S_exp, y=condor_payoffs, name='Iron Condor P&L', 
                                          fill='tonexty', line=dict(color='#805ad5', width=3)))
            fig_condor.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_condor.add_vline(x=spot_price, line_dash="dot", line_color="#3182ce", annotation_text="Current Price")
            
            fig_condor.update_layout(title="Iron Condor P&L", xaxis_title="Stock Price at Expiration", 
                                   yaxis_title="Profit/Loss ($)", template="plotly_white", height=300)
            st.plotly_chart(fig_condor, use_container_width=True)
    
    with strategy_tabs[3]:  # Custom builder
        st.markdown("### Custom Strategy Builder")
        st.markdown("Build your own multi-leg strategy by adding individual options positions.")
        
        if 'strategy_legs' not in st.session_state:
            st.session_state.strategy_legs = []
        
        # Add new leg
        st.markdown("#### Add New Leg")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            leg_type = st.selectbox("Type", ["Call", "Put"])
        with col2:
            leg_position = st.selectbox("Position", ["Long", "Short"])
        with col3:
            leg_strike = st.number_input("Strike", value=strike_price, key="custom_strike")
        with col4:
            leg_quantity = st.number_input("Quantity", value=1, min_value=1, key="custom_quantity")
        with col5:
            if st.button("Add Leg"):
                leg = {
                    'type': leg_type.lower(),
                    'position': leg_position.lower(),
                    'strike': leg_strike,
                    'quantity': leg_quantity
                }
                st.session_state.strategy_legs.append(leg)
                st.success(f"Added {leg_position} {leg_quantity} {leg_type}(s) at ${leg_strike}")
        
        # Display current strategy
        if st.session_state.strategy_legs:
            st.markdown("#### Current Strategy")
            
            strategy_df = pd.DataFrame(st.session_state.strategy_legs)
            st.dataframe(strategy_df)
            
            # Calculate custom strategy payoff
            custom_payoffs = []
            total_cost = 0
            
            for S in S_exp:
                payoff = 0
                for leg in st.session_state.strategy_legs:
                    if leg['type'] == 'call':
                        option_payoff = max(S - leg['strike'], 0)
                        option_price = OptionPricer.black_scholes_call(spot_price, leg['strike'], time_to_maturity, risk_free_rate, volatility)
                    else:
                        option_payoff = max(leg['strike'] - S, 0)
                        option_price = OptionPricer.black_scholes_put(spot_price, leg['strike'], time_to_maturity, risk_free_rate, volatility)
                    
                    if leg['position'] == 'long':
                        payoff += leg['quantity'] * option_payoff
                        if S == S_exp[0]:  # Calculate cost only once
                            total_cost += leg['quantity'] * option_price
                    else:
                        payoff -= leg['quantity'] * option_payoff
                        if S == S_exp[0]:  # Calculate cost only once
                            total_cost -= leg['quantity'] * option_price
                
                custom_payoffs.append(payoff - total_cost)
            
            # Plot custom strategy
            fig_custom = go.Figure()
            fig_custom.add_trace(go.Scatter(x=S_exp, y=custom_payoffs, name='Custom Strategy P&L', 
                                          fill='tonexty', line=dict(color='#3182ce', width=3)))
            fig_custom.add_hline(y=0, line_dash="dash", line_color="#718096")
            fig_custom.add_vline(x=spot_price, line_dash="dot", line_color="#d69e2e", annotation_text="Current Price")
            
            fig_custom.update_layout(title="Custom Strategy P&L", xaxis_title="Stock Price at Expiration", 
                                   yaxis_title="Profit/Loss ($)", template="plotly_white", height=400)
            st.plotly_chart(fig_custom, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Cost", f"${total_cost:.2f}")
            with col2:
                max_profit = max(custom_payoffs)
                st.metric("Max Profit", f"${max_profit:.2f}" if max_profit < 1000 else "Unlimited")
            with col3:
                max_loss = min(custom_payoffs)
                st.metric("Max Loss", f"${max_loss:.2f}" if max_loss > -1000 else "Unlimited")
            
            if st.button("Clear Strategy"):
                st.session_state.strategy_legs = []
                st.experimental_rerun()

# ----------------------
# Interactive Volatility Analysis
# ----------------------
with st.container():
    st.markdown("""
    <div class="volatility-box">
        <div class="section-title">📈 Interactive Volatility Analysis</div>
        <div class="content-text">
            Explore how volatility affects option prices and discover volatility smile patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    vol_tab1, vol_tab2 = st.tabs(["🎯 Volatility Impact", "😊 Volatility Smile Builder"])
    
    with vol_tab1:
        st.markdown("### How Volatility Affects Your Options")
        
        # Interactive volatility analysis
        col1, col2 = st.columns(2)
        
        with col1:
            vol_range_min = st.slider("Min Volatility", 0.05, 0.50, 0.10, 0.05)
            vol_range_max = st.slider("Max Volatility", vol_range_min + 0.05, 1.0, 0.40, 0.05)
        
        with col2:
            analyze_option = st.selectbox("Analyze", ["Call Option", "Put Option", "Both"])
        
        # Volatility sensitivity analysis
        vol_range = np.linspace(vol_range_min, vol_range_max, 50)
        call_prices_vol = [OptionPricer.black_scholes_call(spot_price, strike_price, time_to_maturity, risk_free_rate, vol) for vol in vol_range]
        put_prices_vol = [OptionPricer.black_scholes_put(spot_price, strike_price, time_to_maturity, risk_free_rate, vol) for vol in vol_range]
        
        fig_vol_impact = go.Figure()
        
        if analyze_option in ["Call Option", "Both"]:
            fig_vol_impact.add_trace(go.Scatter(x=vol_range*100, y=call_prices_vol, name='Call Price', 
                                              line=dict(color='#38a169', width=3)))
        
        if analyze_option in ["Put Option", "Both"]:
            fig_vol_impact.add_trace(go.Scatter(x=vol_range*100, y=put_prices_vol, name='Put Price', 
                                              line=dict(color='#e53e3e', width=3)))
        
        fig_vol_impact.add_vline(x=volatility*100, line_dash="dash", line_color="#3182ce", 
                                annotation_text=f"Current Vol: {volatility:.1%}")
        
        fig_vol_impact.update_layout(
            title="Option Price vs Volatility",
            xaxis_title="Volatility (%)",
            yaxis_title="Option Price ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_vol_impact, use_container_width=True)
        
        # Show current vega
        current_vega = vega
        st.markdown(f"""
        <div class="info-box">
            <strong>Current Vega: ${current_vega:.2f}</strong><br>
            For every 1% increase in volatility, your option value changes by ${current_vega:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    with vol_tab2:
        st.markdown("### Build a Volatility Smile")
        st.markdown("Create different volatility patterns across strikes to see market-realistic pricing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smile_type = st.selectbox("Smile Pattern", 
                                    ["Flat (Constant Vol)", "U-Shape (Volatility Smile)", "Skew (Put Skew)", "Custom"])
            atm_vol = st.slider("ATM Volatility", 0.10, 0.50, 0.20, 0.01)
        
        with col2:
            if smile_type != "Flat (Constant Vol)":
                smile_intensity = st.slider("Pattern Intensity", 0.0, 0.15, 0.05, 0.01)
            else:
                smile_intensity = 0.0
        
        # Generate strikes around current spot
        strikes_smile = np.linspace(spot_price * 0.8, spot_price * 1.2, 15)
        
        def generate_vol_pattern(K, S, pattern_type, base_vol, intensity):
            moneyness = np.log(K / S)
            
            if pattern_type == "Flat (Constant Vol)":
                return base_vol
            elif pattern_type == "U-Shape (Volatility Smile)":
                return base_vol + intensity * moneyness**2
            elif pattern_type == "Skew (Put Skew)":
                return base_vol + intensity * moneyness
            else:  # Custom - you can modify this
                return base_vol + intensity * (moneyness**2 + 0.5 * moneyness)
        
        # Calculate implied vols and option prices for each strike
        implied_vols = [generate_vol_pattern(K, spot_price, smile_type, atm_vol, smile_intensity) for K in strikes_smile]
        call_prices_smile = [OptionPricer.black_scholes_call(spot_price, K, time_to_maturity, risk_free_rate, vol) 
                           for K, vol in zip(strikes_smile, implied_vols)]
        
        # Plot volatility smile
        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(x=strikes_smile, y=[vol*100 for vol in implied_vols], 
                                     mode='lines+markers', name='Implied Volatility',
                                     line=dict(color='#3182ce', width=3), marker=dict(size=8)))
        fig_smile.add_vline(x=spot_price, line_dash="dash", line_color="#e53e3e", annotation_text="ATM")
        
        fig_smile.update_layout(
            title=f'{smile_type} Pattern',
            xaxis_title='Strike Price ($)',
            yaxis_title='Implied Volatility (%)',
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig_smile, use_container_width=True)
        
        # Plot resulting option prices
        fig_prices_smile = go.Figure()
        fig_prices_smile.add_trace(go.Scatter(x=strikes_smile, y=call_prices_smile, 
                                            mode='lines+markers', name='Call Prices',
                                            line=dict(color='#38a169', width=3), marker=dict(size=8)))
        fig_prices_smile.add_vline(x=spot_price, line_dash="dash", line_color="#e53e3e", annotation_text="ATM")
        
        fig_prices_smile.update_layout(
            title='Resulting Call Option Prices',
            xaxis_title='Strike Price ($)',
            yaxis_title='Call Price ($)',
            template="plotly_white",
            height=300
        )
        
        st.plotly_chart(fig_prices_smile, use_container_width=True)

# ----------------------
# Interactive Risk Analysis
# ----------------------
with st.container():
    st.markdown("""
    <div class="risk-box">
        <div class="section-title">⚠️ Interactive Risk Analysis</div>
        <div class="content-text">
            Understand the risks of your options positions through interactive stress testing.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    risk_tab1, risk_tab2, risk_tab3 = st.tabs(["📊 Position Risk", "🎯 Stress Testing", "⏰ Time Decay Analysis"])
    
    with risk_tab1:
        st.markdown("### Your Current Position Risk")
        
        # Position size input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            position_size = st.number_input("Position Size", min_value=1, max_value=1000, value=10, step=1)
        with col2:
            position_type = st.selectbox("Position Type", ["Long Call", "Short Call", "Long Put", "Short Put"])
        with col3:
            portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=100000, step=1000)
        
        # Calculate position metrics
        if "Call" in position_type:
            option_price = call_price
            option_delta = delta_call
        else:
            option_price = put_price
            option_delta = delta_call - 1  # Put delta
        
        if "Short" in position_type:
            option_price *= -1
            option_delta *= -1
        
        total_position_value = position_size * option_price
        total_delta = position_size * option_delta
        position_percentage = abs(total_position_value) / portfolio_value * 100
        
        # Risk metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Position Value", f"${total_position_value:.0f}")
        with col2:
            st.metric("Total Delta", f"{total_delta:.1f}")
        with col3:
            st.metric("Portfolio %", f"{position_percentage:.1f}%")
        with col4:
            daily_theta = position_size * theta_call
            st.metric("Daily Theta", f"${daily_theta:.0f}")
        
        # Position P&L chart
        price_changes = np.linspace(-0.3, 0.3, 50)
        position_pnl = []
        
        for change in price_changes:
            new_spot = spot_price * (1 + change)
            if "Call" in position_type:
                new_option_price = OptionPricer.black_scholes_call(new_spot, strike_price, time_to_maturity, risk_free_rate, volatility)
                current_option_price = call_price
            else:
                new_option_price = OptionPricer.black_scholes_put(new_spot, strike_price, time_to_maturity, risk_free_rate, volatility)
                current_option_price = put_price
            
            pnl = position_size * (new_option_price - current_option_price)
            if "Short" in position_type:
                pnl *= -1
            
            position_pnl.append(pnl)
        
        fig_position_risk = go.Figure()
        fig_position_risk.add_trace(go.Scatter(x=[change*100 for change in price_changes], y=position_pnl,
                                             name='Position P&L', line=dict(color='#3182ce', width=3),
                                             fill='tonexty'))
        fig_position_risk.add_hline(y=0, line_dash="dash", line_color="#718096")
        
        fig_position_risk.update_layout(
            title=f'{position_type} Position P&L ({position_size} contracts)',
            xaxis_title='Underlying Price Change (%)',
            yaxis_title='P&L ($)',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_position_risk, use_container_width=True)
    
    with risk_tab2:
        st.markdown("### Stress Test Your Position")
        st.markdown("See how extreme market moves would affect your options.")
        
        # Stress test scenarios
        scenarios = {
            'Small Move Up': {'price_change': 0.05, 'vol_change': 0.0},
            'Small Move Down': {'price_change': -0.05, 'vol_change': 0.0},
            'Large Move Up': {'price_change': 0.20, 'vol_change': -0.05},
            'Large Move Down': {'price_change': -0.20, 'vol_change': 0.05},
            'Vol Spike': {'price_change': 0.0, 'vol_change': 0.10},
            'Vol Crush': {'price_change': 0.0, 'vol_change': -0.10},
            'Market Crash': {'price_change': -0.30, 'vol_change': 0.15},
            'Bull Rally': {'price_change': 0.25, 'vol_change': -0.03}
        }
        
        stress_results = []
        
        for scenario_name, changes in scenarios.items():
            new_spot = spot_price * (1 + changes['price_change'])
            new_vol = max(0.05, volatility + changes['vol_change'])
            
            call_new = OptionPricer.black_scholes_call(new_spot, strike_price, time_to_maturity, risk_free_rate, new_vol)
            put_new = OptionPricer.black_scholes_put(new_spot, strike_price, time_to_maturity, risk_free_rate, new_vol)
            
            call_pnl = (call_new - call_price) * (position_size if "Long Call" in position_type else -position_size if "Short Call" in position_type else 0)
            put_pnl = (put_new - put_price) * (position_size if "Long Put" in position_type else -position_size if "Short Put" in position_type else 0)
            
            total_pnl = call_pnl + put_pnl
            
            stress_results.append({
                'Scenario': scenario_name,
                'Price Change': f"{changes['price_change']:+.1%}",
                'Vol Change': f"{changes['vol_change']:+.1%}",
                'New Spot': f"${new_spot:.2f}",
                'New Vol': f"{new_vol:.1%}",
                'P&L': f"${total_pnl:.0f}",
                'P&L %': f"{total_pnl/abs(total_position_value)*100:+.1f}%" if total_position_value != 0 else "N/A"
            })
        
        stress_df = pd.DataFrame(stress_results)
        st.dataframe(stress_df, use_container_width=True)
        
        # Stress test visualization
        pnl_values = [float(row['P&L'].replace('

# ----------------------
# Educational Hub
# ----------------------
with st.container():
    st.markdown("""
    <div class="risk-box">
        <div class="section-title">🎓 Educational Hub</div>
        <div class="content-text">
            Learn the fundamentals and advanced concepts behind options pricing.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    edu_tab1, edu_tab2, edu_tab3 = st.tabs(["📚 Core Concepts", "⚖️ Risk-Neutral Pricing", "🧮 Model Comparison"])
    
    with edu_tab1:
        st.markdown("### Options Pricing Fundamentals")
        
        concept_choice = st.selectbox("Choose a concept to explore:", 
                                    ["Moneyness", "Intrinsic vs Time Value", "Put-Call Parity", "Exercise Styles"])
        
        if concept_choice == "Moneyness":
            moneyness = spot_price / strike_price
            
            if moneyness > 1.05:
                moneyness_desc = "**In-the-Money (ITM)** 💰"
                explanation = "Your call option has intrinsic value and would be profitable if exercised today."
            elif moneyness < 0.95:
                moneyness_desc = "**Out-of-the-Money (OTM)** 📉"
                explanation = "Your call option has no intrinsic value - it needs the stock to rise to become profitable."
            else:
                moneyness_desc = "**At-the-Money (ATM)** 🎯"
                explanation = "Your call option is right at the strike - maximum time value and gamma."
            
            st.markdown(f"""
            <div class="info-box">
                <strong>Current Moneyness: {moneyness:.3f}</strong><br>
                Status: {moneyness_desc}<br><br>
                {explanation}
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive moneyness chart
            spot_range = np.linspace(strike_price * 0.7, strike_price * 1.3, 100)
            call_intrinsic = np.maximum(spot_range - strike_price, 0)
            call_prices_moneyness = [OptionPricer.black_scholes_call(S, strike_price, time_to_maturity, risk_free_rate, volatility) for S in spot_range]
            time_values = np.array(call_prices_moneyness) - call_intrinsic
            
            fig_moneyness = go.Figure()
            fig_moneyness.add_trace(go.Scatter(x=spot_range, y=call_intrinsic, name='Intrinsic Value', 
                                             fill='tonexty', line=dict(color='#e53e3e', width=2)))
            fig_moneyness.add_trace(go.Scatter(x=spot_range, y=time_values, name='Time Value',
                                             fill='tonexty', line=dict(color='#3182ce', width=2)))
            fig_moneyness.add_trace(go.Scatter(x=spot_range, y=call_prices_moneyness, name='Total Option Value',
                                             line=dict(color='#38a169', width=3)))
            
            fig_moneyness.add_vline(x=spot_price, line_dash="dash", line_color="#d69e2e", annotation_text="Current Spot")
            fig_moneyness.add_vline(x=strike_price, line_dash="dot", line_color="#718096", annotation_text="Strike")
            
            fig_moneyness.update_layout(
                title='Call Option Value Components',
                xaxis_title='Stock Price ($)',
                yaxis_title='Value ($)',
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_moneyness, use_container_width=True)
        
        elif concept_choice == "Intrinsic vs Time Value":
            call_intrinsic = max(spot_price - strike_price, 0)
            call_time_value = call_price - call_intrinsic
            
            put_intrinsic = max(strike_price - spot_price, 0)
            put_time_value = put_price - put_intrinsic
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Call Option")
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Total Value: ${call_price:.2f}</strong><br>
                    Intrinsic: ${call_intrinsic:.2f}<br>
                    Time Value: ${call_time_value:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Put Option")
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Total Value: ${put_price:.2f}</strong><br>
                    Intrinsic: ${put_intrinsic:.2f}<br>
                    Time Value: ${put_time_value:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>Key Insight:</strong> Time value represents the premium investors pay for the possibility 
                that the option could become more valuable before expiration. This value decreases as 
                expiration approaches (time decay/theta).
            </div>
            """, unsafe_allow_html=True)
        
        elif concept_choice == "Put-Call Parity":
            # Put-call parity: C - P = S - K*e^(-rT)
            parity_lhs = call_price - put_price
            parity_rhs = spot_price - strike_price * np.exp(-risk_free_rate * time_to_maturity)
            parity_difference = abs(parity_lhs - parity_rhs)
            
            st.markdown("""
            ### Put-Call Parity Relationship
            For European options: **C - P = S - K·e^(-rT)**
            """)
            
            st.latex(r'C - P = S_0 - K \cdot e^{-rT}')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Left Side (C - P)", f"${parity_lhs:.4f}")
            with col2:
                st.metric("Right Side (S - PV(K))", f"${parity_rhs:.4f}")
            with col3:
                st.metric("Difference", f"${parity_difference:.4f}")
            
            if parity_difference < 0.01:
                st.markdown("""
                <div class="success-box">
                    ✅ <strong>Put-call parity holds!</strong> The relationship is satisfied within rounding errors.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>Parity violation detected.</strong> This could indicate an arbitrage opportunity 
                    in real markets (though here it's likely due to model assumptions).
                </div>
                """, unsafe_allow_html=True)
        
        else:  # Exercise Styles
            st.markdown("### American vs European Options")
            
            european_call = call_price
            american_call = OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity, 
                                                      risk_free_rate, volatility, n_steps=100, 
                                                      option_type='call', american=True)
            
            european_put = put_price
            american_put = OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity, 
                                                     risk_free_rate, volatility, n_steps=100, 
                                                     option_type='put', american=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Call Options")
                st.markdown(f"""
                <div class="metric-card">
                    European: ${european_call:.4f}<br>
                    American: ${american_call:.4f}<br>
                    Difference: ${american_call - european_call:.4f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Put Options")
                st.markdown(f"""
                <div class="metric-card">
                    European: ${european_put:.4f}<br>
                    American: ${american_put:.4f}<br>
                    Difference: ${american_put - european_put:.4f}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>Why the difference?</strong><br>
                • <strong>American calls on non-dividend stocks:</strong> Usually not optimal to exercise early<br>
                • <strong>American puts:</strong> May be optimal to exercise early, especially when deep ITM<br>
                • <strong>Early exercise premium:</strong> The additional value from the flexibility to exercise anytime
            </div>
            """, unsafe_allow_html=True)
    
    with edu_tab2:
        st.markdown("### Risk-Neutral Valuation")
        st.markdown("The mathematical foundation of modern derivatives pricing.")
        
        st.latex(r'V(t) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r(T - t)} \cdot \text{Payoff}(S_T) \mid \mathcal{F}_t \right]')
        
        # Interactive demonstration
        demo_choice = st.selectbox("Choose demonstration:", 
                                 ["Real-World vs Risk-Neutral", "Monte Carlo Pricing", "Drift Impact"])
        
        if demo_choice == "Real-World vs Risk-Neutral":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Real-World Measure (ℙ)")
                real_drift = st.slider("Expected Return (μ)", 0.05, 0.25, 0.12, 0.01)
                st.latex(f'dS_t = {real_drift:.2f} S_t dt + {volatility:.2f} S_t dW_t^{{\\mathbb{{P}}}}')
                
                st.markdown("""
                <div class="warning-box">
                    <strong>Problem:</strong> Option price would depend on investor risk preferences 
                    and expected returns, making pricing subjective.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Risk-Neutral Measure (ℚ)")
                st.latex(f'dS_t = {risk_free_rate:.2f} S_t dt + {volatility:.2f} S_t dW_t^{{\\mathbb{{Q}}}}')
                
                st.markdown("""
                <div class="success-box">
                    <strong>Solution:</strong> Under ℚ, all assets earn the risk-free rate, 
                    making pricing objective and arbitrage-free.
                </div>
                """, unsafe_allow_html=True)
        
        elif demo_choice == "Monte Carlo Pricing":
            st.markdown("#### Risk-Neutral Monte Carlo Simulation")
            
            mc_sims = st.slider("Number of Simulations", 1000, 10000, 5000, 1000)
            
            if st.button("Run Risk-Neutral Simulation"):
                np.random.seed(42)
                
                # Simulate under risk-neutral measure
                Z = np.random.normal(0, 1, mc_sims)
                S_T = spot_price * np.exp((risk_free_rate - 0.5*volatility**2)*time_to_maturity + 
                                        volatility*np.sqrt(time_to_maturity)*Z)
                
                # Calculate payoffs
                call_payoffs = np.maximum(S_T - strike_price, 0)
                put_payoffs = np.maximum(strike_price - S_T, 0)
                
                # Discount back
                mc_call_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(call_payoffs)
                mc_put_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(put_payoffs)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MC Call Price", f"${mc_call_price:.4f}")
                with col2:
                    st.metric("BS Call Price", f"${call_price:.4f}")
                with col3:
                    st.metric("MC Put Price", f"${mc_put_price:.4f}")
                with col4:
                    st.metric("BS Put Price", f"${put_price:.4f}")
                
                # Show distribution of final stock prices
                fig_mc_dist = go.Figure()
                fig_mc_dist.add_trace(go.Histogram(x=S_T, nbinsx=50, name='Final Stock Prices',
                                                 histnorm='probability density', opacity=0.7))
                fig_mc_dist.add_vline(x=strike_price, line_dash="dash", line_color="#e53e3e", 
                                     annotation_text="Strike Price")
                fig_mc_dist.add_vline(x=np.mean(S_T), line_dash="dot", line_color="#38a169", 
                                     annotation_text=f"Mean: ${np.mean(S_T):.2f}")
                
                fig_mc_dist.update_layout(
                    title='Distribution of Stock Prices at Expiration (Risk-Neutral)',
                    xaxis_title='Stock Price ($)',
                    yaxis_title='Probability Density',
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig_mc_dist, use_container_width=True)
        
        else:  # Drift Impact
            st.markdown("#### Impact of Different Drift Assumptions")
            
            drifts = np.linspace(0.0, 0.20, 10)
            call_prices_drift = []
            
            # This demonstrates why we can't use real-world drift for pricing
            for drift in drifts:
                # Simulate with different drifts (this is wrong for pricing!)
                np.random.seed(42)
                Z = np.random.normal(0, 1, 5000)
                S_T = spot_price * np.exp((drift - 0.5*volatility**2)*time_to_maturity + 
                                        volatility*np.sqrt(time_to_maturity)*Z)
                call_payoffs = np.maximum(S_T - strike_price, 0)
                wrong_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(call_payoffs)
                call_prices_drift.append(wrong_price)
            
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(x=[d*100 for d in drifts], y=call_prices_drift,
                                         mode='lines+markers', name='"Option Price"',
                                         line=dict(color='#e53e3e', width=3)))
            fig_drift.add_hline(y=call_price, line_dash="dash", line_color="#38a169",
                               annotation_text=f"Correct BS Price: ${call_price:.2f}")
            fig_drift.add_vline(x=risk_free_rate*100, line_dash="dot", line_color="#3182ce",
                               annotation_text="Risk-free rate")
            
            fig_drift.update_layout(
                title='Why We Need Risk-Neutral Pricing',
                xaxis_title='Assumed Drift Rate (%)',
                yaxis_title='Calculated Option Price ($)',
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_drift, use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
                <strong>Key Insight:</strong> If we used real-world expected returns for pricing, 
                the option price would vary dramatically based on subjective market views. 
                Risk-neutral pricing eliminates this problem!
            </div>
            """, unsafe_allow_html=True)
    
    with edu_tab3:
        st.markdown("### Model Comparison & When to Use Each")
        
        comparison_choice = st.selectbox("Compare models for:", 
                                       ["European Options", "American Options", "Path-Dependent Options"])
        
        if comparison_choice == "European Options":
            # Compare BS vs Binomial for European options
            steps_range = [10, 25, 50, 100, 200]
            binomial_prices = []
            
            for steps in steps_range:
                bin_price = OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity,
                                                      risk_free_rate, volatility, steps, 'call', False)
                binomial_prices.append(bin_price)
            
            fig_convergence = go.Figure()
            fig_convergence.add_trace(go.Scatter(x=steps_range, y=binomial_prices,
                                               mode='lines+markers', name='Binomial Tree',
                                               line=dict(color='#3182ce', width=3)))
            fig_convergence.add_hline(y=call_price, line_dash="dash", line_color="#38a169",
                                    annotation_text=f"Black-Scholes: ${call_price:.4f}")
            
            fig_convergence.update_layout(
                title='Binomial Tree Convergence to Black-Scholes',
                xaxis_title='Number of Steps',
                yaxis_title='Call Option Price ($)',
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_convergence, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>For European Options:</strong><br>
                • <strong>Black-Scholes:</strong> Fastest, analytical solution, perfect for vanilla options<br>
                • <strong>Binomial:</strong> More flexible, good for understanding, converges to BS<br>
                • <strong>Monte Carlo:</strong> Overkill for vanilla Europeans, but useful for path-dependent features
            </div>
            """, unsafe_allow_html=True)
        
        elif comparison_choice == "American Options":
            # Compare European vs American pricing
            euro_call = call_price
            amer_call = OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity,
                                                  risk_free_rate, volatility, 100, 'call', True)
            euro_put = put_price
            amer_put = OptionPricer.binomial_tree(spot_price, strike_price, time_to_maturity,
                                                 risk_free_rate, volatility, 100, 'put', True)
            
            comparison_data = {
                'Option Type': ['Call', 'Put'],
                'European Price': [f"${euro_call:.4f}", f"${euro_put:.4f}"],
                'American Price': [f"${amer_call:.4f}", f"${amer_put:.4f}"],
                'Early Exercise Premium': [f"${amer_call - euro_call:.4f}", f"${amer_put - euro_put:.4f}"]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>For American Options:</strong><br>
                • <strong>Binomial Trees:</strong> Natural choice, handles early exercise easily<br>
                • <strong>Monte Carlo:</strong> Possible with Longstaff-Schwartz algorithm, but complex<br>
                • <strong>Black-Scholes:</strong> Cannot handle American features directly
            </div>
            """, unsafe_allow_html=True)
        
        else:  # Path-Dependent Options
            st.markdown("#### Path-Dependent Options Require Different Approaches")
            
            path_option_type = st.selectbox("Path-dependent type:", 
                                          ["Asian (Average Price)", "Lookback (Extreme Value)", "Barrier (Knock-out)"])
            
            if path_option_type == "Asian (Average Price)":
                st.markdown("""
                **Asian Call Payoff:** max(Average(S) - K, 0)
                
                **Why different models matter:**
                """)
                
                # Monte Carlo for Asian option
                mc_asian, mc_se = OptionPricer.monte_carlo_option(spot_price, strike_price, time_to_maturity,
                                                                 risk_free_rate, volatility, 5000,
                                                                 option_type='asian_call', path_dependent=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("European Call", f"${call_price:.4f}")
                with col2:
                    st.metric("Asian Call (MC)", f"${mc_asian:.4f}")
                
                st.markdown("""
                <div class="info-box">
                    <strong>Asian options are typically cheaper</strong> than European options because 
                    averaging reduces volatility. Only Monte Carlo (or complex binomial modifications) 
                    can accurately price these.
                </div>
                """, unsafe_allow_html=True)
            
            elif path_option_type == "Lookback (Extreme Value)":
                st.markdown("""
                **Lookback Call Payoff:** max(Max(S) - K, 0)
                
                This option pays based on the highest price reached during the option's life.
                """)
                
                # Monte Carlo for Lookback option
                mc_lookback, mc_se = OptionPricer.monte_carlo_option(spot_price, strike_price, time_to_maturity,
                                                                    risk_free_rate, volatility, 5000,
                                                                    option_type='lookback_call', path_dependent=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("European Call", f"${call_price:.4f}")
                with col2:
                    st.metric("Lookback Call (MC)", f"${mc_lookback:.4f}")
                
                st.markdown("""
                <div class="info-box">
                    <strong>Lookback options are more expensive</strong> than European options because 
                    they guarantee the best possible exercise price. Analytical formulas exist but 
                    Monte Carlo is more intuitive.
                </div>
                """, unsafe_allow_html=True)
            
            else:  # Barrier
                st.markdown("""
                **Barrier Options** have payoffs that depend on whether the underlying crosses a barrier level.
                
                Example: **Knock-out Call** - becomes worthless if stock price hits barrier.
                """)
                
                barrier_level = st.slider("Barrier Level", spot_price * 0.7, spot_price * 1.3, spot_price * 1.2)
                
                st.markdown(f"""
                <div class="warning-box">
                    <strong>Barrier at ${barrier_level:.2f}</strong><br>
                    If the stock price reaches this level, a knock-out option becomes worthless.
                    This requires path monitoring - impossible with standard Black-Scholes.
                </div>
                """, unsafe_allow_html=True)

# ----------------------
# Footer
# ----------------------
with st.container():
    st.markdown("""
    <div class="footer-section">
        <div style="font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; color: #1a365d;">
            Quantitative Finance Platform
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
, '').replace(',', '')) for row in stress_results]
        scenario_names = [row['Scenario'] for row in stress_results]
        
        fig_stress = go.Figure(data=[
            go.Bar(x=scenario_names, y=pnl_values,
                  marker_color=['#38a169' if x > 0 else '#e53e3e' for x in pnl_values])
        ])
        
        fig_stress.update_layout(
            title='Stress Test Results',
            xaxis_title='Scenario',
            yaxis_title='P&L ($)',
            template="plotly_white",
            height=400,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig_stress, use_container_width=True)
    
    with risk_tab3:
        st.markdown("### Time Decay Analysis")
        st.markdown("Watch how your options lose value over time.")
        
        # Time decay analysis
        days_to_analyze = st.slider("Days to Analyze", 1, 180, 30)
        
        time_points = np.linspace(time_to_maturity, 0.01, days_to_analyze)
        call_values_time = []
        put_values_time = []
        
        for T_remaining in time_points:
            call_val = OptionPricer.black_scholes_call(spot_price, strike_price, T_remaining, risk_free_rate, volatility)
            put_val = OptionPricer.black_scholes_put(spot_price, strike_price, T_remaining, risk_free_rate, volatility)
            call_values_time.append(call_val)
            put_values_time.append(put_val)
        
        days_remaining = [T * 365 for T in time_points]
        
        fig_time_decay = go.Figure()
        fig_time_decay.add_trace(go.Scatter(x=days_remaining, y=call_values_time, name='Call Value',
                                          line=dict(color='#38a169', width=3)))
        fig_time_decay.add_trace(go.Scatter(x=days_remaining, y=put_values_time, name='Put Value',
                                          line=dict(color='#e53e3e', width=3)))
        
        current_days = time_to_maturity * 365
        fig_time_decay.add_vline(x=current_days, line_dash="dash", line_color="#3182ce", 
                               annotation_text=f"Current: {current_days:.0f} days")
        
        fig_time_decay.update_layout(
            title='Option Value vs Time to Expiration',
            xaxis_title='Days to Expiration',
            yaxis_title='Option Value ($)',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_time_decay, use_container_width=True)
        
        # Weekly theta impact
        weeks_ahead = min(8, int(time_to_maturity * 52))
        weekly_decay = []
        
        for week in range(weeks_ahead + 1):
            T_week = max(0.01, time_to_maturity - week/52)
            if "Call" in position_type:
                week_value = OptionPricer.black_scholes_call(spot_price, strike_price, T_week, risk_free_rate, volatility)
                current_value = call_price
            else:
                week_value = OptionPricer.black_scholes_put(spot_price, strike_price, T_week, risk_free_rate, volatility)
                current_value = put_price
            
            decay_amount = (current_value - week_value) * position_size
            if "Short" in position_type:
                decay_amount *= -1
            
            weekly_decay.append(decay_amount)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Weekly Time Decay Impact")
            for i, decay in enumerate(weekly_decay[:4]):
                if decay >= 0:
                    st.markdown(f"**Week {i+1}:** <span style='color: #38a169'>+${decay:.0f}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Week {i+1}:** <span style='color: #e53e3e'>${decay:.0f}</span>", unsafe_allow_html=True)
        
        with col2:
            if len(weekly_decay) > 4:
                st.markdown("#### Extended Decay")
                for i, decay in enumerate(weekly_decay[4:8]):
                    week_num = i + 5
                    if decay >= 0:
                        st.markdown(f"**Week {week_num}:** <span style='color: #38a169'>+${decay:.0f}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Week {week_num}:** <span style='color: #e53e3e'>${decay:.0f}</span>", unsafe_allow_html=True)

# ----------------------
# Advanced Calculator
# ----------------------
with st.container():
    st.markdown("""
    <div class="portfolio-box">
        <div class="section-title">🧮 Advanced Options Calculator</div>
        <div class="content-text">
            Professional tools for complex options analysis and what-if scenarios.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    calc_tab1, calc_tab2, calc_tab3 = st.tabs(["🎯 Breakeven Analysis", "💰 Profit Target Tool", "⚡ Quick Comparisons"])
    
    with calc_tab1:
        st.markdown("### Breakeven Analysis")
        st.markdown("Find the exact price levels where your position breaks even.")
        
        selected_strategy = st.selectbox("Select Strategy for Analysis", 
                                       ["Long Call", "Long Put", "Bull Call Spread", "Straddle"])
        
        if selected_strategy == "Long Call":
            breakeven = strike_price + call_price
            st.markdown(f"""
            <div class="success-box">
                <strong>Breakeven Price: ${breakeven:.2f}</strong><br>
                Your long call breaks even when the stock reaches ${breakeven:.2f} at expiration.<br>
                That's a {((breakeven/spot_price - 1)*100):+.1f}% move from current price.
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_strategy == "Long Put":
            breakeven = strike_price - put_price
            st.markdown(f"""
            <div class="success-box">
                <strong>Breakeven Price: ${breakeven:.2f}</strong><br>
                Your long put breaks even when the stock falls to ${breakeven:.2f} at expiration.<br>
                That's a {((breakeven/spot_price - 1)*100):+.1f}% move from current price.
            </div>
            """, unsafe_allow_html=True)
            
        elif selected_strategy == "Bull Call Spread":
            K1 = strike_price - 5
            K2 = strike_price + 5
            spread_cost = OptionsStrategy.bull_call_spread(spot_price, K1, K2, time_to_maturity, risk_free_rate, volatility)
            breakeven = K1 + spread_cost
            st.markdown(f"""
            <div class="success-box">
                <strong>Breakeven Price: ${breakeven:.2f}</strong><br>
                Your bull call spread (${K1}-${K2}) breaks even when the stock reaches ${breakeven:.2f}.<br>
                Maximum profit of ${K2-K1-spread_cost:.2f} achieved above ${K2}.
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Straddle
            straddle_cost = call_price + put_price
            breakeven_up = strike_price + straddle_cost
            breakeven_down = strike_price - straddle_cost
            st.markdown(f"""
            <div class="success-box">
                <strong>Breakeven Prices: ${breakeven_down:.2f} and ${breakeven_up:.2f}</strong><br>
                Your straddle breaks even when the stock moves to either ${breakeven_down:.2f} or ${breakeven_up:.2f}.<br>
                Required move: {((straddle_cost/spot_price)*100):.1f}% in either direction.
            </div>
            """, unsafe_allow_html=True)
    
    with calc_tab2:
        st.markdown("### Profit Target Calculator")
        st.markdown("Set a profit target and see what needs to happen to achieve it.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_profit = st.number_input("Target Profit ($)", min_value=10, value=500, step=50)
            target_option = st.selectbox("Option to Analyze", ["Call", "Put"])
        
        with col2:
            target_timeframe = st.selectbox("Time Frame", ["1 week", "2 weeks", "1 month", "At expiration"])
        
        # Calculate required stock price for target profit
        time_mapping = {"1 week": 1/52, "2 weeks": 2/52, "1 month": 1/12, "At expiration": 0.01}
        target_time = time_to_maturity - time_mapping[target_timeframe]
        target_time = max(0.01, target_time)
        
        current_option_price = call_price if target_option == "Call" else put_price
        target_option_price = current_option_price + target_profit
        
        # Binary search for required stock price
        def find_required_price(target_price, option_type, K, T, r, sigma, is_call=True):
            from scipy.optimize import brentq
            
            def price_difference(S):
                if is_call:
                    calculated_price = OptionPricer.black_scholes_call(S, K, T, r, sigma)
                else:
                    calculated_price = OptionPricer.black_scholes_put(S, K, T, r, sigma)
                return calculated_price - target_price
            
            try:
                if is_call:
                    # For calls, price increases with spot price
                    required_S = brentq(price_difference, 0.1, 1000)
                else:
                    # For puts, price decreases with spot price  
                    required_S = brentq(price_difference, 0.1, 1000)
                return required_S
            except:
                return None
        
        required_price = find_required_price(target_option_price, target_option, strike_price, 
                                           target_time, risk_free_rate, volatility, target_option == "Call")
        
        if required_price:
            price_change = (required_price / spot_price - 1) * 100
            st.markdown(f"""
            <div class="info-box">
                <strong>Required Stock Price: ${required_price:.2f}</strong><br>
                To achieve ${target_profit} profit on your {target_option.lower()} option in {target_timeframe},<br>
                the stock needs to move to ${required_price:.2f} ({price_change:+.1f}% change).
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>Target may not be achievable</strong><br>
                The profit target might be too high for the given timeframe and option.
            </div>
            """, unsafe_allow_html=True)
    
    with calc_tab3:
        st.markdown("### Quick Strategy Comparisons")
        st.markdown("Compare different strategies side by side.")
        
        # Strategy comparison
        strategies_to_compare = st.multiselect(
            "Select strategies to compare:",
            ["Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle"],
            default=["Long Call", "Long Put"]
        )
        
        if len(strategies_to_compare) >= 2:
            comparison_data = []
            
            for strategy in strategies_to_compare:
                if strategy == "Long Call":
                    cost = call_price
                    max_profit = "Unlimited"
                    max_loss = call_price
                    breakeven = strike_price + call_price
                    
                elif strategy == "Long Put":
                    cost = put_price
                    max_profit = strike_price - put_price
                    max_loss = put_price
                    breakeven = strike_price - put_price
                    
                elif strategy == "Bull Call Spread":
                    K1, K2 = strike_price - 5, strike_price + 5
                    cost = OptionsStrategy.bull_call_spread(spot_price, K1, K2, time_to_maturity, risk_free_rate, volatility)
                    max_profit = (K2 - K1) - cost
                    max_loss = cost
                    breakeven = K1 + cost
                    
                elif strategy == "Bear Put Spread":
                    K1, K2 = strike_price - 5, strike_price + 5
                    cost = OptionsStrategy.bear_put_spread(spot_price, K1, K2, time_to_maturity, risk_free_rate, volatility)
                    max_profit = (K2 - K1) - cost
                    max_loss = cost
                    breakeven = K2 - cost
                    
                elif strategy == "Straddle":
                    cost = call_price + put_price
                    max_profit = "Unlimited"
                    max_loss = cost
                    breakeven = f"${strike_price - cost:.2f} / ${strike_price + cost:.2f}"
                    
                else:  # Strangle
                    K1, K2 = strike_price - 10, strike_price + 10
                    cost = OptionsStrategy.strangle(spot_price, K1, K2, time_to_maturity, risk_free_rate, volatility)
                    max_profit = "Unlimited"
                    max_loss = cost
                    breakeven = f"${K1 - cost:.2f} / ${K2 + cost:.2f}"
                
                comparison_data.append({
                    'Strategy': strategy,
                    'Cost': f"${cost:.2f}",
                    'Max Profit': max_profit if isinstance(max_profit, str) else f"${max_profit:.2f}",
                    'Max Loss': f"${max_loss:.2f}",
                    'Breakeven': breakeven if isinstance(breakeven, str) else f"${breakeven:.2f}",
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Quick visual comparison of costs
            costs = []
            strategy_names = []
            for strategy in strategies_to_compare:
                if strategy == "Long Call":
                    costs.append(call_price)
                elif strategy == "Long Put":
                    costs.append(put_price)
                elif strategy == "Bull Call Spread":
                    costs.append(OptionsStrategy.bull_call_spread(spot_price, strike_price-5, strike_price+5, time_to_maturity, risk_free_rate, volatility))
                elif strategy == "Bear Put Spread":
                    costs.append(OptionsStrategy.bear_put_spread(spot_price, strike_price-5, strike_price+5, time_to_maturity, risk_free_rate, volatility))
                elif strategy == "Straddle":
                    costs.append(call_price + put_price)
                else:  # Strangle
                    costs.append(OptionsStrategy.strangle(spot_price, strike_price-10, strike_price+10, time_to_maturity, risk_free_rate, volatility))
                strategy_names.append(strategy)
            
            fig_comparison = go.Figure(data=[
                go.Bar(x=strategy_names, y=costs, marker_color='#3182ce')
            ])
            
            fig_comparison.update_layout(
                title='Strategy Cost Comparison',
                xaxis_title='Strategy',
                yaxis_title='Initial Cost ($)',
                template="plotly_white",
                height=300,
                xaxis={'tickangle': 45}
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)

# ----------------------
# Risk-Neutral Framework
# ----------------------
with st.container():
    st.markdown("""
    <div class="risk-box">
        <div class="section-title">⚖️ Risk-Neutral Valuation Framework</div>
        <div class="content-text">
            The mathematical foundation underlying all pricing models in this platform.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    V(t) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r(T - t)} \cdot \text{Payoff}(S_T) \mid \mathcal{F}_t \right]
    ''')
    
    st.markdown("""
    <div class="info-box">
        <strong>Key Principles:</strong>
        <ul>
            <li><strong>Risk-Neutral Measure (ℚ):</strong> Under this probability measure, all assets earn the risk-free rate</li>
            <li><strong>Martingale Property:</strong> Discounted asset prices are martingales under ℚ</li>
            <li><strong>No-Arbitrage:</strong> The fundamental assumption preventing risk-free profit opportunities</li>
            <li><strong>Complete Markets:</strong> Any derivative can be replicated using the underlying and bonds</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Demonstrate risk-neutral vs real-world drift
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Real-World Measure (ℙ)")
        real_world_drift = st.slider("Expected Return (μ)", 0.05, 0.20, 0.12, 0.01)
        st.latex(f'dS_t = {real_world_drift:.2f} \\cdot S_t dt + {volatility:.2f} \\cdot S_t dW_t^{{\\mathbb{{P}}}}')
        
        st.markdown("""
        <div class="warning-box">
            <strong>Note:</strong> Real-world probabilities incorporate risk premiums and investor preferences.
            Options pricing under ℙ would require knowledge of market price of risk.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Risk-Neutral Measure (ℚ)")
        st.latex(f'dS_t = {risk_free_rate:.2f} \\cdot S_t dt + {volatility:.2f} \\cdot S_t dW_t^{{\\mathbb{{Q}}}}')
        
        st.markdown("""
        <div class="success-box">
            <strong>Advantage:</strong> Risk-neutral valuation eliminates the need to estimate risk premiums,
            making derivative pricing model-independent of investor risk preferences.
        </div>
        """, unsafe_allow_html=True)

# ----------------------
# Footer
# ----------------------
with st.container():
    st.markdown("""
    <div class="footer-section">
        <div style="font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; color: #1a365d;">
            Quantitative Finance Platform
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
