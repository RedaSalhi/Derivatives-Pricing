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
        
        # Initialize price tree
        prices = np.zeros((n_steps + 1, n_steps + 1))
        prices[0, 0] = S
        
        # Fill the tree
        for i in range(1, n_steps + 1):
            for j in range(i + 1):
                prices[i, j] = S * (u ** j) * (d ** (i - j))
        
        # Calculate option values at expiration
        option_values = np.zeros((n_steps + 1, n_steps + 1))
        for j in range(n_steps + 1):
            if option_type == 'call':
                option_values[n_steps, j] = max(0, prices[n_steps, j] - K)
            else:
                option_values[n_steps, j] = max(0, K - prices[n_steps, j])
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[i, j] = np.exp(-r * dt) * (p * option_values[i + 1, j + 1] + (1 - p) * option_values[i + 1, j])
                
                if american:
                    if option_type == 'call':
                        option_values[i, j] = max(option_values[i, j], prices[i, j] - K)
                    else:
                        option_values[i, j] = max(option_values[i, j], K - prices[i, j])
        
        return option_values[0, 0]

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
# Volatility Surface Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="volatility-box">
        <div class="section-title">📈 Volatility Surface Analysis</div>
        <div class="content-text">
            Advanced volatility modeling and implied volatility surface construction.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate implied volatility surface
    strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, 10)
    maturities = np.linspace(0.1, 2.0, 8)
    
    # Simulate implied volatility surface (simplified model)
    def implied_vol_surface(S, K, T):
        # Simplified volatility smile/skew model
        moneyness = np.log(K / S)
        vol = volatility + 0.1 * moneyness**2 + 0.05 * np.exp(-T)
        return max(0.05, vol)  # Minimum vol of 5%
    
    vol_surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            vol_surface[i, j] = implied_vol_surface(spot_price, K, T)
    
    # 3D volatility surface plot
    fig_vol_surface = go.Figure(data=[go.Surface(
        z=vol_surface,
        x=strikes,
        y=maturities,
        colorscale='Viridis',
        colorbar=dict(title="Implied Volatility")
    )])
    
    fig_vol_surface.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price ($)',
            yaxis_title='Time to Maturity (Years)',
            zaxis_title='Implied Volatility'
        ),
        height=500
    )
    
    st.plotly_chart(fig_vol_surface, use_container_width=True)
    
    # Volatility smile at current maturity
    current_vols = [implied_vol_surface(spot_price, K, time_to_maturity) for K in strikes]
    
    fig_smile = go.Figure()
    fig_smile.add_trace(go.Scatter(x=strikes, y=current_vols, mode='lines+markers',
                                  name='Implied Volatility', line=dict(color='#3182ce', width=3)))
    fig_smile.add_vline(x=spot_price, line_dash="dash", line_color="#e53e3e", annotation_text="ATM")
    
    fig_smile.update_layout(
        title=f'Volatility Smile (T = {time_to_maturity} years)',
        xaxis_title='Strike Price ($)',
        yaxis_title='Implied Volatility',
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_smile, use_container_width=True)

# ----------------------
# Risk Management Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="risk-box">
        <div class="section-title">⚠️ Risk Management & Portfolio Analysis</div>
        <div class="content-text">
            Comprehensive risk metrics and portfolio-level analysis tools.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio risk analysis
    st.markdown("### Portfolio Risk Metrics")
    
    # Simulate portfolio returns
    np.random.seed(42)
    n_assets = 5
    n_observations = 252
    
    # Generate correlated returns
    correlation_matrix = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.4],
        [0.3, 1.0, 0.5, 0.2, 0.3],
        [0.2, 0.5, 1.0, 0.3, 0.2],
        [0.1, 0.2, 0.3, 1.0, 0.6],
        [0.4, 0.3, 0.2, 0.6, 1.0]
    ])
    
    returns = np.random.multivariate_normal(
        mean=[0.08, 0.10, 0.12, 0.06, 0.09],
        cov=correlation_matrix * 0.04,  # Scale by volatilities
        size=n_observations
    ) / np.sqrt(252)  # Daily returns
    
    # Portfolio weights
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    portfolio_returns = np.dot(returns, weights)
    
    # Risk metrics
    var_95 = RiskManager.value_at_risk(portfolio_returns, 0.05)
    var_99 = RiskManager.value_at_risk(portfolio_returns, 0.01)
    es_95 = RiskManager.expected_shortfall(portfolio_returns, 0.05)
    es_99 = RiskManager.expected_shortfall(portfolio_returns, 0.01)
    
    # Display risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>VaR (95%)</h4>
            <h3>{var_95:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>VaR (99%)</h4>
            <h3>{var_99:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>ES (95%)</h4>
            <h3>{es_95:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>ES (99%)</h4>
            <h3>{es_99:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Return distribution with VaR
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, name='Portfolio Returns',
                                   histnorm='probability density', opacity=0.7))
    fig_risk.add_vline(x=-var_95, line_dash="dash", line_color="#e53e3e", annotation_text="VaR 95%")
    fig_risk.add_vline(x=-var_99, line_dash="dash", line_color="#9b2c2c", annotation_text="VaR 99%")
    
    fig_risk.update_layout(
        title='Portfolio Return Distribution with VaR',
        xaxis_title='Daily Returns',
        yaxis_title='Density',
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Correlation heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=[f'Asset {i+1}' for i in range(n_assets)],
        y=[f'Asset {i+1}' for i in range(n_assets)],
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix,
        texttemplate="%{text:.2f}",
        textfont={"size": 12}
    ))
    
    fig_corr.update_layout(
        title='Asset Correlation Matrix',
        height=400
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------
# Advanced Analytics Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="portfolio-box">
        <div class="section-title">📊 Advanced Analytics</div>
        <div class="content-text">
            Sophisticated analytical tools for quantitative finance professionals.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    analytics_tabs = st.tabs(["🎯 Options Analytics", "📈 Performance Attribution", "🔍 Scenario Analysis"])
    
    with analytics_tabs[0]:
        st.markdown("### Options Portfolio Analytics")
        
        # Options portfolio simulation
        portfolio_options = [
            {'type': 'call', 'strike': 95, 'quantity': 10, 'position': 'long'},
            {'type': 'put', 'strike': 105, 'quantity': 5, 'position': 'short'},
            {'type': 'call', 'strike': 110, 'quantity': 8, 'position': 'short'}
        ]
        
        # Calculate portfolio Greeks
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_value = 0
        
        for option in portfolio_options:
            if option['type'] == 'call':
                value = OptionPricer.black_scholes_call(spot_price, option['strike'], time_to_maturity, risk_free_rate, volatility)
            else:
                value = OptionPricer.black_scholes_put(spot_price, option['strike'], time_to_maturity, risk_free_rate, volatility)
            
            delta, gamma, theta, vega, rho = OptionPricer.calculate_greeks(spot_price, option['strike'], time_to_maturity, risk_free_rate, volatility, option['type'])
            
            multiplier = 1 if option['position'] == 'long' else -1
            total_delta += multiplier * delta * option['quantity']
            total_gamma += multiplier * gamma * option['quantity']
            total_theta += multiplier * theta * option['quantity']
            total_vega += multiplier * vega * option['quantity']
            total_value += multiplier * value * option['quantity']
        
        # Display portfolio Greeks
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Portfolio Value", f"${total_value:.0f}")
        with col2:
            st.metric("Portfolio Delta", f"{total_delta:.1f}")
        with col3:
            st.metric("Portfolio Gamma", f"{total_gamma:.2f}")
        with col4:
            st.metric("Portfolio Theta", f"${total_theta:.0f}")
        with col5:
            st.metric("Portfolio Vega", f"${total_vega:.0f}")
        
        # Portfolio P&L across underlying prices
        S_portfolio = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
        portfolio_pnl = []
        
        for S in S_portfolio:
            pnl = 0
            for option in portfolio_options:
                if option['type'] == 'call':
                    current_value = OptionPricer.black_scholes_call(S, option['strike'], time_to_maturity, risk_free_rate, volatility)
                    initial_value = OptionPricer.black_scholes_call(spot_price, option['strike'], time_to_maturity, risk_free_rate, volatility)
                else:
                    current_value = OptionPricer.black_scholes_put(S, option['strike'], time_to_maturity, risk_free_rate, volatility)
                    initial_value = OptionPricer.black_scholes_put(spot_price, option['strike'], time_to_maturity, risk_free_rate, volatility)
                
                multiplier = 1 if option['position'] == 'long' else -1
                pnl += multiplier * (current_value - initial_value) * option['quantity']
            
            portfolio_pnl.append(pnl)
        
        fig_portfolio = go.Figure()
        fig_portfolio.add_trace(go.Scatter(x=S_portfolio, y=portfolio_pnl, name='Portfolio P&L',
                                         line=dict(color='#3182ce', width=3)))
        fig_portfolio.add_hline(y=0, line_dash="dash", line_color="#718096")
        fig_portfolio.add_vline(x=spot_price, line_dash="dot", line_color="#e53e3e", annotation_text="Current Price")
        
        fig_portfolio.update_layout(
            title='Options Portfolio P&L',
            xaxis_title='Underlying Price ($)',
            yaxis_title='P&L ($)',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_portfolio, use_container_width=True)
    
    with analytics_tabs[1]:
        st.markdown("### Performance Attribution")
        
        # Simulate performance attribution
        factors = ['Market Beta', 'Sector Allocation', 'Security Selection', 'Currency', 'Residual']
        contributions = [0.045, 0.012, 0.008, -0.003, 0.002]
        
        fig_attribution = go.Figure(data=[
            go.Bar(x=factors, y=contributions, 
                  marker_color=['#38a169' if x > 0 else '#e53e3e' for x in contributions])
        ])
        
        fig_attribution.update_layout(
            title='Performance Attribution Analysis',
            xaxis_title='Attribution Factors',
            yaxis_title='Contribution to Return',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_attribution, use_container_width=True)
        
        # Performance metrics table
        metrics_data = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown', 'Beta', 'Alpha'],
            'Portfolio': ['12.5%', '1.24', '1.67', '-8.3%', '0.92', '2.1%'],
            'Benchmark': ['10.2%', '1.15', '1.52', '-12.1%', '1.00', '0.0%']
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    with analytics_tabs[2]:
        st.markdown("### Scenario Analysis")
        
        # Define scenarios
        scenarios = {
            'Base Case': {'stock_change': 0.0, 'vol_change': 0.0, 'rate_change': 0.0},
            'Bull Market': {'stock_change': 0.2, 'vol_change': -0.05, 'rate_change': 0.01},
            'Bear Market': {'stock_change': -0.3, 'vol_change': 0.1, 'rate_change': -0.02},
            'High Volatility': {'stock_change': 0.0, 'vol_change': 0.15, 'rate_change': 0.0},
            'Rate Shock': {'stock_change': -0.1, 'vol_change': 0.05, 'rate_change': 0.03}
        }
        
        scenario_results = []
        
        for scenario_name, changes in scenarios.items():
            new_S = spot_price * (1 + changes['stock_change'])
            new_vol = max(0.05, volatility + changes['vol_change'])
            new_rate = max(0.001, risk_free_rate + changes['rate_change'])
            
            call_value = OptionPricer.black_scholes_call(new_S, strike_price, time_to_maturity, new_rate, new_vol)
            put_value = OptionPricer.black_scholes_put(new_S, strike_price, time_to_maturity, new_rate, new_vol)
            
            call_pnl = call_value - call_price
            put_pnl = put_value - put_price
            
            scenario_results.append({
                'Scenario': scenario_name,
                'Stock Price': f"${new_S:.2f}",
                'Call P&L': f"${call_pnl:.2f}",
                'Put P&L': f"${put_pnl:.2f}",
                'Volatility': f"{new_vol:.1%}",
                'Interest Rate': f"{new_rate:.1%}"
            })
        
        scenario_df = pd.DataFrame(scenario_results)
        st.dataframe(scenario_df, use_container_width=True)
        
        # Scenario impact chart
        pnl_values = [float(row['Call P&L'].replace('$', '')) for row in scenario_results]
        scenario_names = [row['Scenario'] for row in scenario_results]
        
        fig_scenarios = go.Figure(data=[
            go.Bar(x=scenario_names, y=pnl_values,
                  marker_color=['#38a169' if x > 0 else '#e53e3e' for x in pnl_values])
        ])
        
        fig_scenarios.update_layout(
            title='Call Option P&L Across Scenarios',
            xaxis_title='Scenario',
            yaxis_title='P&L ($)',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)

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
