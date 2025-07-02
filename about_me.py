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
    "Quantitative Finance Tools for Options, Forwards, Swaps & Interest Rate Instruments"
)

# Professional Profile section using enhanced CSS
st.markdown("""
<div class="profile-box animate-fade-in">
    <h2 class="section-title">Professional Profile</h2>
    <div class="profile-info">
        <strong>Welcome to my Derivatives Pricing Application!</strong> I'm passionate about quantitative finance and building tools that make complex financial concepts accessible to students, and enthusiasts.
        <br><br>
        This application demonstrates <strong>advanced derivatives pricing models</strong> including Black-Scholes for European options, binomial trees for American options, and Monte Carlo simulations for exotic derivatives. Each model includes educational insights to help users understand the underlying mathematics and financial theory.
        <br><br>
        I believe in making <strong>quantitative finance education</strong> more interactive and practical. My goal is to bridge the gap between theory and real-world practice by using professional tools.
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
                <li><strong>Programming:</strong> Python, VBA, SQL, MATLAB</li>
                <li><strong>Finance:</strong> Derivatives Pricing, Risk Management</li>
                <li><strong>Analytics:</strong> Monte Carlo Simulation, Statistical Modeling</li>
                <li><strong>Tools:</strong> Streamlit, NumPy, SciPy, Pandas, Bloomberg, S&P Capital IQ, GitHub</li>
                <li><strong>Visualization:</strong> Plotly, Matplotlib</li>
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
                <li><strong>Quantitative Finance:</strong> Options pricing, Portfolio Theory and risk metrics</li>
                <li><strong>Financial Engineering:</strong> Model development and validation</li>
                <li><strong>Open Source:</strong> Building tools for the finance community</li>
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
        <a href="mailto:salhi.reda47@gmail.com" target="_blank">Email</a>
    </div>
</div>
""", unsafe_allow_html=True)


# Contact form section with enhanced styling
with st.container():
    st.subheader("Contact Me")
    st.markdown("Have questions about derivatives pricing, want to collaborate, or interested in discussing quantitative finance? I'd love to hear from you!")
    
    formsubmit_email = "salhi.reda47@gmail.com"
    
    # CSS personnalisé pour un look moderne et sobre
    form_css = """
    <style>
    .contact-form {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .contact-form input, .contact-form textarea {
        width: 100%;
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #ced4da;
        border-radius: 8px;
        font-size: 16px;
        background: white;
        transition: all 0.3s ease;
        color: #495057;
    }
    .contact-form input:focus, .contact-form textarea:focus {
        outline: none;
        border-color: #6c757d;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(108, 117, 125, 0.15);
    }
    .contact-form button {
        background: linear-gradient(45deg, #6c757d, #495057);
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.5px;
    }
    .contact-form button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(108, 117, 125, 0.3);
        background: linear-gradient(45deg, #495057, #343a40);
    }
    </style>
    """
    
    form_code = f"""
    {form_css}
    <div class="contact-form">
        <form action="https://formsubmit.co/{formsubmit_email}" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="hidden" name="_template" value="table">
            <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">
            <input type="text" name="name" placeholder="Your Name" required>
            <input type="email" name="email" placeholder="Your Email" required>
            <textarea name="message" placeholder="Your Message" rows="5" required></textarea>
            <button type="submit">Send Message</button>
        </form>
    </div>
    """
    
    st.markdown(form_code, unsafe_allow_html=True)



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
