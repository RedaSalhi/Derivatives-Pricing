"""
Refactored about_me.py using the unified styling system
"""

import streamlit as st
import sys
import os

# Import the unified styling system
from styles.app_styles import (
    apply_global_styles, 
    get_component_styles, 
    render_page_header,
    render_section_title,
    render_info_box
)

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Apply unified styles
apply_global_styles()
st.markdown(get_component_styles(), unsafe_allow_html=True)

# Page header using unified styles
render_page_header(
    "About Me",
    "Financial Engineering Student | Quant Researcher"
)

# ----------------------
# Profile Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="profile-box">
        <div class="section-title">👤 Profile</div>
        <div class="profile-info">
            <strong>SALHI Reda</strong><br>
            Engineering student at <strong>Centrale Méditerranée</strong><br>
            Passionate about <strong>mathematics</strong>, <strong>financial markets</strong>, and <strong>quantitative finance</strong><br><br>
            
            🎯 <strong>Specialization:</strong> Financial Engineering & Derivatives Pricing<br>
            📊 <strong>Interests:</strong> Risk Management, Algorithmic Trading, Financial Modeling<br>
            🔬 <strong>Research Focus:</strong> Options Pricing Models, Monte Carlo Methods, Greek Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Skills & Expertise Section
# ----------------------
st.markdown("""
<div class="success-box">
    <div class="section-title">🚀 Skills & Expertise</div>
    <div class="content-text">
        <strong>Programming Languages:</strong> Python, R, MATLAB, C++<br>
        <strong>Financial Libraries:</strong> QuantLib, pandas, NumPy, SciPy<br>
        <strong>Mathematical Modeling:</strong> Black-Scholes, Monte Carlo, Binomial Trees<br>
        <strong>Data Analysis:</strong> Statistical Analysis, Time Series, Risk Metrics<br>
        <strong>Development:</strong> Streamlit, Flask, Git, Docker
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------
# Contact Section
# ----------------------
st.markdown("""
<div class="contact-box">
    <div class="section-title">📫 Get in Touch</div>
    <div class="content-text">
        I'm always interested in discussing financial markets, quantitative methods, 
        and potential collaboration opportunities. Feel free to reach out!
    </div>
""", unsafe_allow_html=True)

# Contact form with unified styling
formsubmit_email = "salhi.reda47@gmail.com"

form_code = f"""
<div class="contact-form">
    <form action="https://formsubmit.co/{formsubmit_email}" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="hidden" name="_template" value="table">
        <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">
        <input type="hidden" name="_subject" value="New contact from Derivatives Pricing App">
        
        <input 
            type="text" 
            name="name" 
            placeholder="Your Name" 
            class="form-input"
            required
        >
        
        <input 
            type="email" 
            name="email" 
            placeholder="Your Email" 
            class="form-input"
            required
        >
        
        <input 
            type="text" 
            name="subject" 
            placeholder="Subject" 
            class="form-input"
            required
        >
        
        <textarea 
            name="message" 
            placeholder="Your Message" 
            class="form-input"
            rows="5" 
            required
        ></textarea>
        
        <button type="submit" class="form-button">
            Send Message 📧
        </button>
    </form>
</div>
</div>
"""

st.markdown(form_code, unsafe_allow_html=True)

# ----------------------
# Links Section
# ----------------------
st.markdown("""
<div class="links-box">
    <div class="section-title">🔗 Connect & Resources</div>
    
    <div class="link-item">
        📧 <a href="mailto:salhi.reda47@gmail.com">salhi.reda47@gmail.com</a>
    </div>
    
    <div class="link-item">
        💼 <a href="https://linkedin.com/in/reda-salhi" target="_blank">LinkedIn Profile</a>
    </div>
    
    <div class="link-item">
        💻 <a href="https://github.com/reda-salhi" target="_blank">GitHub Repository</a>
    </div>
    
    <div class="link-item">
        📚 <a href="https://scholar.google.com/citations?user=example" target="_blank">Academic Publications</a>
    </div>
    
    <div class="link-item">
        🎓 <a href="https://centrale-med.fr" target="_blank">Centrale Méditerranée</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------
# Download Section
# ----------------------
st.markdown("""
<div class="download-section">
    <div class="section-title">📄 Download Resources</div>
    <div class="content-text">
        Get access to my academic work and professional materials.
    </div>
</div>
""", unsafe_allow_html=True)

# Download buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("📋 Download CV", key="cv_download"):
        st.success("CV download initiated!")
        # Here you would add the actual download logic

with col2:
    if st.button("📊 Research Papers", key="papers_download"):
        st.success("Research papers download initiated!")
        # Here you would add the actual download logic

# ----------------------
# Footer with professional note
# ----------------------
st.markdown("""
<div class="info-box">
    <div class="content-text" style="text-align: center;">
        <strong>💡 About This Application</strong><br>
        This derivatives pricing tool was developed as part of my financial engineering studies. 
        It demonstrates practical implementation of quantitative finance concepts and serves as 
        an educational resource for students and practitioners in finance.
    </div>
</div>
""", unsafe_allow_html=True)
