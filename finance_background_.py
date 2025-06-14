import streamlit as st

# Enhanced CSS for finance background page with LaTeX support
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #1f77b4, #2e86de);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #ff7f0e;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Objective Section */
    .objective-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Methodology Section */
    .methodology-box {
        background: linear-gradient(135deg, #fff3cd 0%, #fef9e7 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Model Sections */
    .model-box {
        background: linear-gradient(135deg, #d4edda 0%, #e8f5e8 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Greeks Section */
    .greeks-box {
        background: linear-gradient(135deg, #f8d7da 0%, #fdeaea 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #dc3545;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk-Neutral Section */
    .risk-neutral-box {
        background: linear-gradient(135deg, #e2e3e5 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #6c757d;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Strategy Section */
    .strategy-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #e8f6f9 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #17a2b8;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Engineering Section */
    .engineering-box {
        background: linear-gradient(135deg, #d6d8db 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #343a40;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        color: #1f77b4;
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .subsection-title {
        color: #28a745;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #28a745;
        display: inline-block;
    }
    
    .content-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .highlight-box {
        background: rgba(31, 119, 180, 0.1);
        border-left: 4px solid #1f77b4;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .greek-item {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 3px solid #dc3545;
    }
    
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #28a745;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .footer-section {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Finance Background & Methodology</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Mathematical Models & Quantitative Framework</div>', unsafe_allow_html=True)

# ----------------------
# Objective Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="objective-box">
        <div class="section-title">Objective</div>
        <div class="content-text">
            The <strong>Derivatives Pricing App</strong> is built to help users understand and visualize how various financial derivatives are priced and analyzed using different mathematical models. Whether you're a student, a quant, or a curious learner, this app aims to bridge theory and implementation through interactive tools and comprehensive visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Methodology Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="methodology-box">
        <div class="section-title">Methodology & Pricing Models</div>
        <div class="content-text">
            We implement core pricing methodologies under the <strong>risk-neutral measure</strong> (ℚ), where the present value of any derivative is the discounted expected payoff:
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    V(t) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r(T - t)} \cdot \text{Payoff}(S_T) \mid \mathcal{F}_t \right]
    ''')

# ----------------------
# Models Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="model-box">
        <div class="section-title">Pricing Models</div>
    </div>
    """, unsafe_allow_html=True)

    # Black-Scholes Model
    st.markdown("""
    <div class="model-card">
        <div class="subsection-title">1. Black–Scholes Model</div>
        <div class="content-text">
            Used for <strong>European options</strong>, assumes:
            <ul>
                <li>Geometric Brownian Motion (GBM)</li>
                <li>Constant volatility σ</li>
                <li>No arbitrage, no dividends</li>
            </ul>
            <strong>European Call Price:</strong><br>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    C = S_0 N(d_1) - K e^{-rT} N(d_2)
    ''')
    
    st.markdown("with:")
    
    st.latex(r'''
    d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left( r + \frac{1}{2}\sigma^2 \right) T}{\sigma \sqrt{T}}, \quad
    d_2 = d_1 - \sigma \sqrt{T}
    ''')

    # Binomial Tree Model
    st.markdown("""
    <div class="model-card">
        <div class="subsection-title">2. Binomial Tree Model</div>
        <div class="content-text">
            Used for <strong>European and American options</strong>:
            <ul>
                <li>Builds a discrete recombining tree with $N$ steps</li>
                <li>Up/down factors: u = e<sup>σ√Δt</sup>, d = 1/u</li>
                <li>Risk-neutral probability:</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    p = \frac{e^{r \Delta t} - d}{u - d}
    ''')
    
    st.markdown("Backward induction yields the fair price. Useful for early-exercise features and American options. Adapted for **barrier** and **lookback** options.")

    # Monte Carlo Simulation
    st.markdown("""
    <div class="model-card">
        <div class="subsection-title">3. Monte Carlo Simulation</div>
        <div class="content-text">
            Used for <strong>path-dependent options</strong> (e.g., Asian, Lookback). Simulates M sample paths of the underlying:
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    S_{t+\Delta t} = S_t \cdot \exp\left( \left( r - \frac{1}{2}\sigma^2 \right)\Delta t + \sigma \sqrt{\Delta t} \cdot Z \right), \quad Z \sim \mathcal{N}(0,1)
    ''')
    
    st.markdown("Option value estimated as:")
    
    st.latex(r'''
    V_0 \approx e^{-rT} \cdot \frac{1}{M} \sum_{i=1}^M \text{Payoff}^{(i)}
    ''')
    
    st.markdown("Extended with **Longstaff-Schwartz** regression for American options.")

# ----------------------
# Greeks Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="greeks-box">
        <div class="section-title">Greeks & Sensitivities</div>
        <div class="content-text">
            The app computes standard <strong>Greeks</strong> for <strong>vanilla options</strong> under the <strong>Black-Scholes</strong> model.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Greeks with LaTeX
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="greek-item">
            <strong>🔹 Delta (Δ)</strong><br>
            Represents sensitivity of the option price V to changes in the underlying asset price S:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'\Delta = \frac{\partial V}{\partial S}')
        
        st.markdown("""
        <div class="greek-item">
            <strong>🔹 Gamma (Γ)</strong><br>
            Second derivative of the option price with respect to S. Measures the curvature of V(S):
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'\Gamma = \frac{\partial^2 V}{\partial S^2}')
        
        st.markdown("""
        <div class="greek-item">
            <strong>🔹 Vega (ν)</strong><br>
            Sensitivity of the option price to volatility σ:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'\nu = \frac{\partial V}{\partial \sigma}')
    
    with col2:
        st.markdown("""
        <div class="greek-item">
            <strong>🔹 Theta (Θ)</strong><br>
            Rate of change of the option price with respect to time t:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'\Theta = \frac{\partial V}{\partial t}')
        
        st.markdown("""
        <div class="greek-item">
            <strong>🔹 Rho (ρ)</strong><br>
            Sensitivity of the option price to the risk-free interest rate r:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r'\rho = \frac{\partial V}{\partial r}')

    st.markdown("""
    <div class="content-text">
        For each option, these sensitivities are computed numerically and visualized to show how risk exposure evolves across strike prices and underlying values.
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Strategy Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="strategy-box">
        <div class="section-title">🔄 Multi-Leg Strategies</div>
        <div class="content-text">
            In real-world trading, investors often combine multiple options into <strong>sophisticated strategies</strong>:
        </div>
        <div class="highlight-box">
            <strong>Available Strategies:</strong>
            <ul>
                <li><strong>Spreads:</strong> Vertical, horizontal, diagonal</li>
                <li><strong>Volatility Plays:</strong> Straddles & Strangles</li>
                <li><strong>Complex Strategies:</strong> Butterflies & Condors</li>
            </ul>
        </div>
        <div class="content-text">
            These strategies can be custom-built and visualized, with <strong>net payoff</strong> and <strong>Greek sensitivity charts</strong> for comprehensive risk analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Risk-Neutral Framework
# ----------------------
with st.container():
    st.markdown("""
    <div class="risk-neutral-box">
        <div class="section-title">⚖️ Risk-Neutral Framework</div>
        <div class="content-text">
            All pricing models in this app are based on the <strong>risk-neutral measure</strong> ℚ, under which the discounted price of a traded asset is a <strong>martingale</strong>. This leads to the fundamental pricing formula:
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r'''
    V(t) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r(T - t)} \cdot \text{Payoff}(S_T) \mid \mathcal{F}_t \right]
    ''')
    
    st.markdown("""
    <div class="highlight-box">
        <strong>Where:</strong>
        <ul>
            <li>$V(t)$ is the value of the derivative at time $t$</li>
            <li>$r$ is the risk-free interest rate</li>
            <li>$T$ is the maturity</li>
            <li>S<sub>T is the underlying asset price at time $T$</li>
            <li>$ℱ<sub>t$ is the information available at time $t$</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Application in Models")
    
    st.markdown("- In **Black-Scholes**, we transform real-world drift $\\mu$ to $r$, and simulate:")
    
    st.latex(r'''
    dS_t = r S_t dt + \sigma S_t dW_t^{\mathbb{Q}}
    ''')
    
    st.markdown("- In **Binomial models**, we construct a **risk-neutral probability**:")
    
    st.latex(r'''
    p = \frac{e^{r \Delta t} - d}{u - d}
    ''')
    
    st.markdown("- In **Monte Carlo**, each simulated payoff is discounted:")
    
    st.latex(r'''
    V_0 \approx e^{-rT} \cdot \frac{1}{M} \sum_{i=1}^M \text{Payoff}^{(i)}
    ''')
    
    st.markdown("This unified framework enables the consistent pricing of a wide range of derivatives.")

# ----------------------
# Engineering Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="engineering-box">
        <div class="section-title">Engineering & Design</div>
        <div class="content-text">
            All code is written in <strong>Python</strong> and available on <strong>GitHub</strong>. The application features:
        </div>
        <div class="highlight-box">
            <ul>
                <li><strong>Modular Architecture:</strong> Easy to extend and maintain</li>
                <li><strong>Interactive Visualizations:</strong> Real-time parameter adjustments</li>
                <li><strong>Robust Calculations:</strong> Validated against industry standards</li>
                <li><strong>User-Friendly Interface:</strong> Streamlit-powered dashboard</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Footer
# ----------------------
with st.container():
    st.markdown("""
    <div class="footer-section">
        <div style="font-size: 1.2rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.5rem;">
            Thank you for exploring the quantitative finance framework!
        </div>
        <div style="color: #6c757d; font-style: italic;">
            © 2025 | SALHI Reda | Financial Engineering Research
        </div>
    </div>
    """, unsafe_allow_html=True)
