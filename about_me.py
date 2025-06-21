import streamlit as st
import sys
import os
from styles.app_styles import load_theme


# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

load_theme()

# Main header
st.markdown('<div class="main-header">About Me</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Financial Engineering Student | Quant Researcher</div>', unsafe_allow_html=True)

# ----------------------
# Profile Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="profile-box">
        <div class="section-title">Profile</div>
        <div class="profile-info">
            <strong>SALHI Reda</strong><br>
            Engineering student at <strong>Centrale Méditerranée</strong><br>
            Passionate about <strong>mathematics</strong>, <strong>financial markets</strong>, and <strong>quantitative research</strong><br>
            Specializing in quantitative finance and derivatives pricing<br>
            Currently developing advanced pricing models and risk management tools
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# CV Downloads Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="download-section">
        <div class="section-title">Resume Downloads</div>
        <p style="margin-bottom: 1.5rem; color: #6c757d;">Download my latest resume in your preferred language:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CV download buttons
    col1, col2 = st.columns(2)
    
    cv_en = "assets/Reda_Salhi_CV_EN.pdf"
    cv_fr = "assets/Reda_Salhi_CV_FR.pdf"
    
    with col1:
        if os.path.exists(cv_en):
            with open(cv_en, "rb") as f:
                st.download_button(
                    label="Download CV - English Version",
                    data=f,
                    file_name="Reda_Salhi_CV_EN.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("English CV - Coming Soon")
    
    with col2:
        if os.path.exists(cv_fr):
            with open(cv_fr, "rb") as f:
                st.download_button(
                    label="Download CV - French Version",
                    data=f,
                    file_name="Reda_Salhi_CV_FR.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("French CV - Coming Soon")

# ----------------------
# Links Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="links-box">
        <div class="section-title">🔗 Connect With Me</div>
        <div class="link-item">
            <a href="https://www.linkedin.com/in/reda-salhi-195297290/" target="_blank">
                LinkedIn Profile
            </a>
        </div>
        <div class="link-item">
            <a href="https://github.com/RedaSalhi" target="_blank">
                GitHub Portfolio
            </a>
        </div>
        <div class="link-item">
            <a href="mailto:salhi.reda47@gmail.com" class="email-link">
                salhi.reda47@gmail.com
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Contact Form Section
# ----------------------

with st.container():
    st.markdown("---")
    st.subheader("📬 Contact Me")
    st.markdown("If you'd like to get in touch, just fill out the form below:")

    formsubmit_email = "salhi.reda47@gmail.com"

    # CSS personnalisé pour un look moderne et sobre
    form_css = ""

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


# ----------------------
# Skills & Interests (Additional Section)
# ----------------------
with st.container():
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Technical Skills")
        st.markdown("""
        - **Programming**: Python, SQL, MATLAB, Excel
        - **Finance**: Derivatives Pricing, Risk Management, Portfolio Optimization
        - **Tools**: Streamlit, NumPy, Pandas
        - **Mathematics**: Stochastic Calculus, Statistics
        """)
    
    with col2:
        st.markdown("### Areas of Interest")
        st.markdown("""
        - Quantitative Finance
        - Financial Engineering
        - Risk Management
        - Economic Research
        """)
    
    with col3:
        st.markdown("### Current Focus")
        st.markdown("""
        - Derivatives Pricing
        - Monte Carlo Simulations
        - Interest Rate Models
        - Portfolio Optimization
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-style: italic;'>"
    "Thank you for visiting my profile! Looking forward to connecting with you."
    "</div>", 
    unsafe_allow_html=True
)
