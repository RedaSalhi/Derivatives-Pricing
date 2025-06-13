import streamlit as st

# Enhanced CSS for finance background page
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
    
    .formula-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
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
    
    .model-comparison {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #28a745;
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
        <div class="formula-box">
            <div style="text-align: center; font-size: 1.2rem;">
                <strong>V(t) = 𝔼<sup>ℚ</sup>[e<sup>-r(T-t)</sup> · Payoff(S<sub>T</sub>) | ℱ<sub>t</sub>]</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Models Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="model-box">
        <div class="section-title">Pricing Models</div>
        <div class="model-comparison">
    """, unsafe_allow_html=True)
    
    # Model cards
    st.markdown("""
        <div class="model-card">
            <div class="subsection-title">1. Black–Scholes Model</div>
            <div class="content-text">
                <strong>Used for:</strong> European options<br>
                <strong>Assumptions:</strong>
                <ul>
                    <li>Geometric Brownian Motion (GBM)</li>
                    <li>Constant volatility σ</li>
                    <li>No arbitrage, no dividends</li>
                </ul>
                <strong>European Call Price:</strong><br>
                C = S₀N(d₁) - Ke<sup>-rT</sup>N(d₂)
            </div>
        </div>
        
        <div class="model-card">
            <div class="subsection-title">2. Binomial Tree Model</div>
            <div class="content-text">
                <strong>Used for:</strong> European & American options<br>
                <strong>Features:</strong>
                <ul>
                    <li>Discrete recombining tree with N steps</li>
                    <li>Up/down factors: u = e<sup>σ√Δt</sup>, d = 1/u</li>
                    <li>Backward induction pricing</li>
                </ul>
                <strong>Risk-neutral probability:</strong><br>
                p = (e<sup>rΔt</sup> - d)/(u - d)
            </div>
        </div>
        
        <div class="model-card">
            <div class="subsection-title">3. Monte Carlo Simulation</div>
            <div class="content-text">
                <strong>Used for:</strong> Path-dependent options<br>
                <strong>Applications:</strong>
                <ul>
                    <li>Asian options</li>
                    <li>Lookback options</li>
                    <li>Barrier options</li>
                </ul>
                <strong>Price Estimation:</strong><br>
                V₀ ≈ e<sup>-rT</sup> · (1/M)∑Payoff<sup>(i)</sup>
            </div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Greeks Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="greeks-box">
        <div class="section-title">Greeks & Sensitivities</div>
        <div class="content-text">
            The app computes standard <strong>Greeks</strong> — partial derivatives of the option price with respect to key parameters — for <strong>vanilla options</strong> under the <strong>Black-Scholes</strong> model.
        </div>
    """, unsafe_allow_html=True)
    
    # Greeks grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="greek-item">
            <strong>🔹 Delta (Δ)</strong><br>
            Sensitivity to underlying price changes<br>
            <em>Δ = ∂V/∂S</em>
        </div>
        
        <div class="greek-item">
            <strong>🔹 Gamma (Γ)</strong><br>
            Second derivative w.r.t. underlying price<br>
            <em>Γ = ∂²V/∂S²</em>
        </div>
        
        <div class="greek-item">
            <strong>🔹 Vega (ν)</strong><br>
            Sensitivity to volatility changes<br>
            <em>ν = ∂V/∂σ</em>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="greek-item">
            <strong>🔹 Theta (Θ)</strong><br>
            Time decay sensitivity<br>
            <em>Θ = ∂V/∂t</em>
        </div>
        
        <div class="greek-item">
            <strong>🔹 Rho (ρ)</strong><br>
            Interest rate sensitivity<br>
            <em>ρ = ∂V/∂r</em>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Strategy Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="strategy-box">
        <div class="section-title">Multi-Leg Strategies</div>
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
        <div class="section-title">Risk-Neutral Framework</div>
        <div class="content-text">
            All pricing models in this app are based on the <strong>risk-neutral measure</strong> ℚ, under which the discounted price of a traded asset is a <strong>martingale</strong>. This leads to the fundamental pricing formula:
        </div>
        <div class="formula-box">
            <div style="text-align: center; font-size: 1.1rem;">
                <strong>V(t) = 𝔼<sup>ℚ</sup>[e<sup>-r(T-t)</sup> · Payoff(S<sub>T</sub>) | ℱ<sub>t</sub>]</strong>
            </div>
        </div>
        <div class="highlight-box">
            <strong>Key Components:</strong>
            <ul>
                <li><strong>V(t):</strong> Derivative value at time t</li>
                <li><strong>r:</strong> Risk-free interest rate</li>
                <li><strong>T:</strong> Time to maturity</li>
                <li><strong>S<sub>T</sub>:</strong> Underlying asset price at maturity</li>
                <li><strong>ℱ<sub>t</sub>:</strong> Information available at time t</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
