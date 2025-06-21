import streamlit as st
import streamlit.components.v1 as components
import os
import sys

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
    page_title="About Me",
    page_icon="📈",
    layout="wide",
)

# Apply the theme and global styles
load_theme()
apply_global_styles()

# Page header with proper CSS classes
render_app_header(
    "About Me",
    "Advanced Quantitative Finance Tools for Options, Forwards, Swaps & Interest Rate Instruments"
)

# Professional Profile section using enhanced CSS
st.markdown("""
<div class="profile-box animate-fade-in">
    <h2 class="section-title">Professional Profile</h2>
    <div class="profile-info">
        <strong>Welcome to my Derivatives Pricing Application!</strong> I'm passionate about quantitative finance and building tools that make complex financial concepts accessible to students, practitioners, and enthusiasts alike.
        <br><br>
        This application demonstrates <strong>advanced derivatives pricing models</strong> including Black-Scholes for European options, binomial trees for American options, and Monte Carlo simulations for exotic derivatives. Each model includes educational insights to help users understand the underlying mathematics and financial theory.
        <br><br>
        I believe in making <strong>quantitative finance education</strong> more interactive and practical, bridging the gap between academic theory and real-world application through clean, professional tools.
    </div>
</div>
""", unsafe_allow_html=True)

# Skills and Interests section with proper CSS classes
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="skills-section animate-fade-in-delay">
        <h3 class="section-title">Technical Skills</h3>
        <div class="content-text">
            <ul>
                <li><strong>Programming:</strong> Python, R, SQL, MATLAB</li>
                <li><strong>Finance:</strong> Derivatives Pricing, Risk Management</li>
                <li><strong>Analytics:</strong> Monte Carlo Simulation, Statistical Modeling</li>
                <li><strong>Tools:</strong> Streamlit, NumPy, SciPy, Pandas</li>
                <li><strong>Visualization:</strong> Plotly, Matplotlib, Financial Charts</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="skills-section animate-fade-in-delay">
        <h3 class="section-title">Interests & Focus</h3>
        <div class="content-text">
            <ul>
                <li><strong>Quantitative Finance:</strong> Options pricing and risk metrics</li>
                <li><strong>Financial Engineering:</strong> Model development and validation</li>
                <li><strong>Education:</strong> Making finance accessible through technology</li>
                <li><strong>Open Source:</strong> Building tools for the finance community</li>
                <li><strong>Research:</strong> Market volatility and pricing accuracy</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Links section with enhanced styling
st.markdown("""
<div class="links-box animate-fade-in-delay">
    <h2 class="section-title">Connect & Explore</h2>
    <div class="link-item shadow-hover">
        <a href="https://github.com/RedaSalhi" target="_blank">GitHub Portfolio</a>
    </div>
    <div class="link-item shadow-hover">
        <a href="https://www.linkedin.com/in/reda-salhi-195297290/" target="_blank">LinkedIn Profile</a>
    </div>
    <div class="link-item shadow-hover">
        <a href="mailto:salhi.reda47@gmail.com" class="email-link">Email</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Contact form section with enhanced styling
st.markdown("""
<div class="contact-box animate-fade-in-delay">
    <h2 class="section-title">Get In Touch</h2>
    <div class="content-text">
        Have questions about derivatives pricing, want to collaborate, or interested in discussing quantitative finance? I'd love to hear from you!
    </div>
</div>
""", unsafe_allow_html=True)

# Enhanced contact form using FormSubmit with proper CSS classes
contact_form = """
<form action="https://formsubmit.co/salhi.reda47@gmail.com" method="POST" class="contact-form">
    <input type="hidden" name="_captcha" value="false">
    <input type="hidden" name="_subject" value="New Contact from Derivatives Pricing App">
    <input type="hidden" name="_next" value="https://derivatives-pricing.streamlit.app">
    
    <input type="text" name="name" placeholder="Your Name" required class="form-input">
    <input type="email" name="email" placeholder="Your Email" required class="form-input">
    <input type="text" name="subject" placeholder="Subject" required class="form-input">
    <textarea name="message" placeholder="Your Message" rows="5" required class="form-input"></textarea>
    
    <button type="submit" class="form-button">Send Message 📧</button>
</form>
"""

components.html(contact_form, height=450)

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
