# pricer_minim.py - Enhanced with Comprehensive Styling Integration
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

import sys
import os
import streamlit as st
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import enhanced styling system
from styles.app_styles import (
    load_theme, 
    apply_global_styles, 
    get_component_styles,
    render_app_header,
    render_section_title,
    COLORS
)

# Configure Streamlit page
st.set_page_config(
    page_title="Derivatives Pricer Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar since we're not using it
)

# Apply comprehensive styling
load_theme()
apply_global_styles()
st.markdown(get_component_styles(), unsafe_allow_html=True)

# Import tab modules
try:
    from tabs.vanilla_options import vanilla_options_tab
    from tabs.forward_contracts import forward_contracts_tab
    from tabs.option_strategies import main_option_strategies_tab
    from tabs.exotic_options import exotic_options_tab
    from tabs.swaps import swaps_tab
    from tabs.interest_rate_instruments import interest_rate_instruments_tab
except ImportError as e:
    st.error(f"Error importing tabs: {e}")
    st.stop()

# Enhanced main header with professional styling
render_app_header(
    "Professional Derivatives Pricing Suite",
    "Advanced Quantitative Finance Tools for Options, Forwards, Swaps & Interest Rate Instruments"
)

# Add platform overview with enhanced styling
st.markdown("""
<div class="objective-box animate-fade-in">
    <div class="section-title">🚀 Platform Capabilities</div>
    <div class="content-text">
        Comprehensive derivatives pricing platform combining theoretical accuracy with practical usability.
        Built for quantitative analysts, traders, risk managers, and finance students.
    </div>
    <div class="highlight-box">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 15px;">
            <div>
                <h4 style="color: #2563eb; margin-bottom: 10px;">🎯 Pricing Models</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Black-Scholes:</strong> Analytical solutions for European options</li>
                    <li><strong>Binomial Trees:</strong> American options & flexible modeling</li>
                    <li><strong>Monte Carlo:</strong> Complex payoffs & path dependencies</li>
                    <li><strong>Vasicek Model:</strong> Interest rate modeling & bond pricing</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #10b981; margin-bottom: 10px;">📊 Risk Analytics</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Complete Greeks:</strong> Delta, Gamma, Theta, Vega, Rho</li>
                    <li><strong>Scenario Analysis:</strong> Stress testing & what-if scenarios</li>
                    <li><strong>P&L Attribution:</strong> Real-time mark-to-market</li>
                    <li><strong>Portfolio Analytics:</strong> Multi-instrument analysis</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #f59e0b; margin-bottom: 10px;">🛠️ Professional Tools</h4>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Strategy Builder:</strong> Multi-leg option strategies</li>
                    <li><strong>Payoff Diagrams:</strong> Interactive visualizations</li>
                    <li><strong>Parameter Sensitivity:</strong> Real-time analysis</li>
                    <li><strong>Educational Content:</strong> Formulas & explanations</li>
                </ul>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick stats display
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <div style="color: {COLORS['success']}; font-size: 1rem; font-weight: 600;">Vanilla Options</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">BS/Binomial/MC</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.8rem;">European & American</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card animate-fade-in-delay">
        <div style="color: {COLORS['primary']}; font-size: 1rem; font-weight: 600;">Forward Contracts</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">Cost of Carry</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.8rem;">Mark-to-Market</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <div style="color: {COLORS['info']}; font-size: 1rem; font-weight: 600;">Option Strategies</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">Multi-Leg</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.8rem;">P&L Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card animate-fade-in-delay">
        <div style="color: {COLORS['warning']}; font-size: 1rem; font-weight: 600;">Exotic Options</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">Path-Dependent</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.8rem;">Asian/Barrier/Digital</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <div style="color: {COLORS['danger']}; font-size: 1rem; font-weight: 600;">Swaps</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">IRS/FX/Equity</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.8rem;">DCF/LMM Models</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card animate-fade-in-delay">
        <div style="color: {COLORS['secondary']}; font-size: 1rem; font-weight: 600;">Interest Rates</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">Vasicek Model</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.8rem;">Bond Pricing</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced tabs with better styling and icons
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Vanilla Options", 
    "📈 Forward Contracts", 
    "🔧 Option Strategies", 
    "🌟 Exotic Options",
    "🔄 Swaps",
    "📊 Interest Rate Instruments"
])

# Execute each tab with enhanced error handling and styling
with tab1:
    try:
        st.markdown('<div class="tab-content animate-fade-in">', unsafe_allow_html=True)
        vanilla_options_tab()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="danger-box">
            <h4>❌ Vanilla Options Tab Error</h4>
            <p>Error loading vanilla options: {str(e)}</p>
            <p><strong>Possible Solutions:</strong></p>
            <ul>
                <li>Check if all pricing modules are properly installed</li>
                <li>Verify pricing.vanilla_options module exists</li>
                <li>Ensure dependencies are correctly imported</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    try:
        st.markdown('<div class="tab-content animate-fade-in">', unsafe_allow_html=True)
        forward_contracts_tab()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="danger-box">
            <h4>❌ Forward Contracts Tab Error</h4>
            <p>Error loading forward contracts: {str(e)}</p>
            <p><strong>Possible Solutions:</strong></p>
            <ul>
                <li>Check if pricing.forward module exists</li>
                <li>Verify forward pricing functions are available</li>
                <li>Ensure numpy and plotly dependencies are installed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    try:
        st.markdown('<div class="tab-content animate-fade-in">', unsafe_allow_html=True)
        main_option_strategies_tab()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="danger-box">
            <h4>❌ Option Strategies Tab Error</h4>
            <p>Error loading option strategies: {str(e)}</p>
            <p><strong>Possible Solutions:</strong></p>
            <ul>
                <li>Check if pricing.option_strategies module exists</li>
                <li>Verify strategy pricing functions are available</li>
                <li>Ensure plotting utilities are properly imported</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    try:
        st.markdown('<div class="tab-content animate-fade-in">', unsafe_allow_html=True)
        exotic_options_tab()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="danger-box">
            <h4>❌ Exotic Options Tab Error</h4>
            <p>Error loading exotic options: {str(e)}</p>
            <p><strong>Possible Solutions:</strong></p>
            <ul>
                <li>Check if exotic pricing modules exist (asian_option, barrier_option, etc.)</li>
                <li>Verify Monte Carlo simulation functions are available</li>
                <li>Ensure scipy and numerical libraries are installed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab5:
    try:
        st.markdown('<div class="tab-content animate-fade-in">', unsafe_allow_html=True)
        swaps_tab()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="danger-box">
            <h4>❌ Swaps Tab Error</h4>
            <p>Error loading swaps: {str(e)}</p>
            <p><strong>Possible Solutions:</strong></p>
            <ul>
                <li>Check if pricing.swaps module exists</li>
                <li>Verify swap pricing models are available</li>
                <li>Ensure interest rate curve building functions exist</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab6:
    try:
        st.markdown('<div class="tab-content animate-fade-in">', unsafe_allow_html=True)
        interest_rate_instruments_tab()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="danger-box">
            <h4>❌ Interest Rate Instruments Tab Error</h4>
            <p>Error loading interest rate instruments: {str(e)}</p>
            <p><strong>Possible Solutions:</strong></p>
            <ul>
                <li>Check if Vasicek model modules exist</li>
                <li>Verify bond pricing functions are available</li>
                <li>Ensure parameter estimation utilities are installed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Model status indicator in main area
st.markdown("### 🔧 Available Models & Performance Guide")

col_model1, col_model2, col_model3 = st.columns(3)

with col_model1:
    st.markdown(f"""
    <div class="success-box">
        <h4>⚡ Analytical Models</h4>
        <p><strong>Black-Scholes:</strong> ✅ Active - Instant calculations</p>
        <p><strong>Vasicek Model:</strong> ✅ Active - Bond pricing</p>
    </div>
    """, unsafe_allow_html=True)

with col_model2:
    st.markdown(f"""
    <div class="info-box">
        <h4>🔄 Numerical Models</h4>
        <p><strong>Binomial Trees:</strong> ✅ Active - American options</p>
        <p><strong>Monte Carlo:</strong> ✅ Active - Complex payoffs</p>
        <p><strong>LMM (Basic):</strong> ✅ Active - Interest rates</p>
    </div>
    """, unsafe_allow_html=True)

with col_model3:
    st.markdown(f"""
    <div class="warning-box">
        <h4>🚧 Coming Soon</h4>
        <p><strong>Hull-White:</strong> Advanced interest rate model</p>
        <p><strong>CIR Model:</strong> Cox-Ingersoll-Ross rates</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer with comprehensive information
st.markdown("---")
st.markdown("""
<div class="footer-section animate-fade-in">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px; margin-bottom: 2rem;">
        <div>
            <h4 style="color: #2563eb; margin-bottom: 10px;">📈 Platform Features</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; color: #4a5568;">
                <li>Real-time option pricing across multiple models</li>
                <li>Interactive Greeks analysis with live updates</li>
                <li>Advanced strategy construction & backtesting</li>
                <li>Comprehensive risk scenario analysis</li>
                <li>Professional-grade visualizations</li>
            </ul>
        </div>
        <div>
            <h4 style="color: #10b981; margin-bottom: 10px;">🎯 Use Cases</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; color: #4a5568;">
                <li><strong>Trading:</strong> Strategy evaluation & optimization</li>
                <li><strong>Risk Management:</strong> Portfolio hedging analysis</li>
                <li><strong>Education:</strong> Learning derivatives theory</li>
                <li><strong>Research:</strong> Model validation & comparison</li>
                <li><strong>Compliance:</strong> Risk reporting & documentation</li>
            </ul>
        </div>
        <div>
            <h4 style="color: #f59e0b; margin-bottom: 10px;">🛡️ Risk Disclaimer</h4>
            <div style="font-size: 0.9rem; color: #4a5568;">
                <p><strong>Educational Purpose:</strong> This platform is designed for educational and research purposes.</p>
                <p><strong>Not Financial Advice:</strong> All calculations and analyses should be independently verified before use in trading or investment decisions.</p>
                <p><strong>Model Risk:</strong> All models are theoretical approximations of market behavior.</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
