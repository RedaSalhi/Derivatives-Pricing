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


    
# Page header with animation
st.markdown("""
<div class="animate-fade-in">
    <h1 class="page-title">👋 About Me</h1>
</div>
""", unsafe_allow_html=True)

# Profile section
st.markdown("""
<div class="profile-box animate-fade-in">
    <h2 class="section-title">🎯 Professional Profile</h2>
    <div class="profile-info">
        <strong>Welcome to my Derivatives Pricing Application!</strong> I'm passionate about quantitative finance and building tools that make complex financial concepts accessible to students, practitioners, and enthusiasts alike.
        <br><br>
        This application demonstrates <strong>advanced derivatives pricing models</strong> including Black-Scholes for European options, binomial trees for American options, and Monte Carlo simulations for exotic derivatives. Each model includes educational insights to help users understand the underlying mathematics and financial theory.
        <br><br>
        I believe in making <strong>quantitative finance education</strong> more interactive and practical, bridging the gap between academic theory and real-world application through clean, professional tools.
    </div>
</div>
""", unsafe_allow_html=True)

# Skills and Interests section
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="skills-section animate-fade-in-delay">
        <h3>🔧 Technical Skills</h3>
        <ul>
            <li><strong>Programming:</strong> Python, R, SQL, MATLAB</li>
            <li><strong>Finance:</strong> Derivatives Pricing, Risk Management</li>
            <li><strong>Analytics:</strong> Monte Carlo Simulation, Statistical Modeling</li>
            <li><strong>Tools:</strong> Streamlit, NumPy, SciPy, Pandas</li>
            <li><strong>Visualization:</strong> Plotly, Matplotlib, Financial Charts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="skills-section animate-fade-in-delay">
        <h3>💡 Interests & Focus</h3>
        <ul>
            <li><strong>Quantitative Finance:</strong> Options pricing and risk metrics</li>
            <li><strong>Financial Engineering:</strong> Model development and validation</li>
            <li><strong>Education:</strong> Making finance accessible through technology</li>
            <li><strong>Open Source:</strong> Building tools for the finance community</li>
            <li><strong>Research:</strong> Market volatility and pricing accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Downloads section
st.markdown("""
<div class="download-section animate-fade-in-delay">
    <h2 class="section-title">📄 Download Resources</h2>
    <p>Get my CV and project documentation to learn more about my background and this application.</p>
</div>
""", unsafe_allow_html=True)

# Create download buttons - with error handling
col1, col2 = st.columns(2)

with col1:
    # CV Download - create a simple text file as placeholder if no CV exists
    try:
        if os.path.exists("CV_Reda_Salhi.pdf"):
            with open("CV_Reda_Salhi.pdf", "rb") as file:
                st.download_button(
                    label="📋 Download CV",
                    data=file,
                    file_name="CV_Reda_Salhi.pdf",
                    mime="application/pdf",
                    help="Download my professional CV"
                )
        elif os.path.exists("README.md"):
            with open("README.md", "rb") as file:
                st.download_button(
                    label="📋 Download CV",
                    data=file,
                    file_name="CV_Reda_Salhi.pdf",
                    mime="application/pdf",
                    help="Download my professional CV (placeholder)"
                )
        else:
            st.info("CV download will be available soon")
    except Exception as e:
        st.info("CV download temporarily unavailable")

with col2:
    # Project documentation
    try:
        if os.path.exists("README.md"):
            with open("README.md", "rb") as file:
                st.download_button(
                    label="📖 Project Documentation",
                    data=file,
                    file_name="Derivatives_Pricing_Guide.md",
                    mime="text/markdown",
                    help="Comprehensive guide to this derivatives pricing tool"
                )
        else:
            st.info("Documentation download will be available soon")
    except Exception as e:
        st.info("Documentation download temporarily unavailable")

# Links section
st.markdown("""
<div class="links-box animate-fade-in-delay">
    <h2 class="section-title">🔗 Connect & Explore</h2>
    <div class="link-item">
        <a href="https://github.com/your-username" target="_blank">🐙 GitHub Portfolio</a>
    </div>
    <div class="link-item">
        <a href="https://linkedin.com/in/your-profile" target="_blank">💼 LinkedIn Profile</a>
    </div>
    <div class="link-item">
        <a href="https://derivatives-pricing.streamlit.app" target="_blank">🚀 Live Application</a>
    </div>
    <div class="link-item">
        <a href="mailto:salhi.reda47@gmail.com" class="email-link">📧 Professional Email</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Contact form section
st.markdown("""
<div class="contact-form animate-fade-in-delay">
    <h2 class="section-title">💬 Get In Touch</h2>
    <p>Have questions about derivatives pricing, want to collaborate, or interested in discussing quantitative finance? I'd love to hear from you!</p>
</div>
""", unsafe_allow_html=True)

# Contact form using FormSubmit
contact_form = f"""
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

components.html(contact_form, height=400)

# Application insights
st.markdown("""
<div class="info-box animate-fade-in-delay">
    <h2 class="section-title">🔍 About This Application</h2>
    <p><strong>Purpose:</strong> This derivatives pricing tool was built to provide accurate, educational, and accessible financial modeling for students, professionals, and enthusiasts.</p>
    
    <p><strong>Key Features:</strong></p>
    <ul>
        <li>Multiple pricing models (Black-Scholes, Binomial, Monte Carlo)</li>
        <li>Real-time Greeks calculation and visualization</li>
        <li>Interactive parameter adjustment with instant results</li>
        <li>Educational content explaining each model's assumptions</li>
        <li>Professional-grade accuracy with academic transparency</li>
    </ul>
    
    <p><strong>Technology Stack:</strong> Built with Python, Streamlit, NumPy, and SciPy for robust mathematical computations and an intuitive user interface.</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-section animate-fade-in-delay">
    <div>
        🎯 "Making quantitative finance accessible through technology and education"
    </div>
    <br>
    <div style="font-size: 0.9em; color: #666;">
        Built with ❤️ using Streamlit | © 2024 Reda Salhi
    </div>
</div>
""", unsafe_allow_html=True)

