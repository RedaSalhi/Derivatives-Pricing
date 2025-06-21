import streamlit as st
import sys
import os
from styles.app_styles import load_theme


# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

def about_me_tab():
    """Main about me function that can be called from main.py"""
    
    load_theme()

    # Main header with enhanced styling
    st.markdown('<div class="main-header animate-fade-in-about">About Me</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle animate-fade-in-delay-about">Financial Engineering Student | Quantitative Researcher</div>', unsafe_allow_html=True)

    # ----------------------
    # Profile Section with Animation
    # ----------------------
    with st.container():
        st.markdown("""
        <div class="profile-box animate-fade-in-about">
            <div class="section-title">👤 Profile</div>
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
    # CV Downloads Section with Enhanced Styling
    # ----------------------
    with st.container():
        st.markdown("""
        <div class="download-section animate-fade-in-delay-about">
            <div class="section-title">📄 Resume Downloads</div>
            <p style="margin-bottom: 1.5rem; color: var(--gray-600); font-family: var(--font-family);">
                Download my latest resume in your preferred language:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # CV download buttons with better styling
        col1, col2 = st.columns(2)
        
        cv_en = "assets/Reda_Salhi_CV_EN.pdf"
        cv_fr = "assets/Reda_Salhi_CV_FR.pdf"
        
        with col1:
            if os.path.exists(cv_en):
                with open(cv_en, "rb") as f:
                    st.download_button(
                        label="📋 Download CV - English Version",
                        data=f,
                        file_name="Reda_Salhi_CV_EN.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        help="Download my resume in English"
                    )
            else:
                st.info("📋 English CV - Coming Soon")
        
        with col2:
            if os.path.exists(cv_fr):
                with open(cv_fr, "rb") as f:
                    st.download_button(
                        label="📋 Download CV - French Version",
                        data=f,
                        file_name="Reda_Salhi_CV_FR.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        help="Télécharger mon CV en français"
                    )
            else:
                st.info("📋 French CV - Coming Soon")

    # ----------------------
    # Links Section with Enhanced Styling
    # ----------------------
    with st.container():
        st.markdown("""
        <div class="links-box animate-fade-in-about">
            <div class="section-title">🔗 Connect With Me</div>
            <div class="link-item">
                <a href="https://www.linkedin.com/in/reda-salhi-195297290/" target="_blank">
                    💼 LinkedIn Profile
                </a>
            </div>
            <div class="link-item">
                <a href="https://github.com/RedaSalhi" target="_blank">
                    🐙 GitHub Portfolio
                </a>
            </div>
            <div class="link-item">
                <a href="mailto:salhi.reda47@gmail.com" class="email-link">
                    📧 salhi.reda47@gmail.com
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------------
    # Contact Form Section with Enhanced Styling
    # ----------------------
    with st.container():
        st.markdown("---")
        st.markdown('<div class="sub-header animate-fade-in-delay-about">📬 Contact Me</div>', unsafe_allow_html=True)
        st.markdown('<p style="color: var(--gray-600); font-family: var(--font-family); margin-bottom: var(--space-6);">If you\'d like to get in touch, just fill out the form below:</p>', unsafe_allow_html=True)

        formsubmit_email = "salhi.reda47@gmail.com"

        form_code = f"""
        <div class="contact-form animate-fade-in-about">
            <form action="https://formsubmit.co/{formsubmit_email}" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="hidden" name="_template" value="table">
                <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">
                <input type="text" name="name" placeholder="Your Name" required>
                <input type="email" name="email" placeholder="Your Email" required>
                <textarea name="message" placeholder="Your Message" rows="5" required></textarea>
                <button type="submit">Send Message 📨</button>
            </form>
        </div>
        """

        st.markdown(form_code, unsafe_allow_html=True)

    # ----------------------
    # Skills & Interests Section with Enhanced Styling
    # ----------------------
    with st.container():
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="skills-section animate-fade-in-about">
                <h3>🛠️ Technical Skills</h3>
                <ul>
                    <li><strong>Programming:</strong> Python, SQL, MATLAB, Excel</li>
                    <li><strong>Finance:</strong> Derivatives Pricing, Risk Management, Portfolio Optimization</li>
                    <li><strong>Tools:</strong> Streamlit, NumPy, Pandas</li>
                    <li><strong>Mathematics:</strong> Stochastic Calculus, Statistics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="skills-section animate-fade-in-delay-about">
                <h3>🎯 Areas of Interest</h3>
                <ul>
                    <li>Quantitative Finance</li>
                    <li>Financial Engineering</li>
                    <li>Risk Management</li>
                    <li>Economic Research</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="skills-section animate-fade-in-delay-about">
                <h3>🔍 Current Focus</h3>
                <ul>
                    <li>Derivatives Pricing</li>
                    <li>Monte Carlo Simulations</li>
                    <li>Interest Rate Models</li>
                    <li>Portfolio Optimization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Footer with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div class="footer-section animate-fade-in-delay-about">
        <div>
            Thank you for visiting my profile! Looking forward to connecting with you. 🚀
        </div>
    </div>
    """, unsafe_allow_html=True)


# If the file is run directly (for testing)
if __name__ == "__main__":
    st.set_page_config(
        page_title="About Me - Reda SALHI", 
        layout="centered",
        page_icon="👤"
    )
    about_me_tab()
