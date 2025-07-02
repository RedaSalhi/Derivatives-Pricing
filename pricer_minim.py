# pricer_minim.py
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
    initial_sidebar_state="expanded"  # Collapse sidebar since we're not using it
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
    from tabs.interest_rate_instruments import interest_rate_instruments_tab
except ImportError as e:
    st.error(f"Error importing tabs: {e}")
    st.stop()

# Enhanced main header with professional styling
render_app_header(
    "Derivatives Pricing Suite",
    "Quantitative Finance Tools for Options, Forwards & Interest Rate Instruments"
)

# Quick stats display
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card animate-fade-in">
        <div style="color: {COLORS['success']}; font-size: 1rem; font-weight: 600;">Vanilla Options</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">BS/ Binomial/ MC</div>
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
    <div class="metric-card animate-fade-in-delay">
        <div style="color: {COLORS['secondary']}; font-size: 1rem; font-weight: 600;">Interest Rates</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: {COLORS['gray_800']};">Vasicek Model</div>
        <div style="color: {COLORS['gray_500']}; font-size: 0.8rem;">Bond Pricing</div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced tabs with better styling and icons
st.markdown("---")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Vanilla Options", 
    "Forward Contracts", 
    "Option Strategies", 
    "Exotic Options",
    "Interest Rate Instruments"
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
st.markdown("### Available Models & Performance Guide")

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


# Footer
st.markdown("""
<div class="footer-section animate-fade-in">
    <div style="font-size: 1.4rem; font-weight: 600; margin-bottom: 1rem; color: #1a365d;">
        📊 Quantitative Finance Platform
    </div>
    <div style="color: #4a5568; font-style: italic; margin-bottom: 1rem;">
        Derivatives Pricing & Risk Management
    </div>
    <div style="color: #718096; font-size: 0.9rem;">
        © 2025 | SALHI Reda | Financial Engineering Research | Advanced Analytics
    </div>
    <div style="margin-top: 1rem; color: #718096; font-size: 0.8rem;">
        <strong>Disclaimer:</strong> This platform is for educational and research purposes. 
        All models are theoretical and should not be used for actual trading without proper validation.
    </div>
</div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
