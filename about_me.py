import streamlit as st
import sys
import os
from styles.app_styles import load_theme

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

def about_me_tab():
    """About Me Tab Content"""
    
    load_theme()
    
    # ----------------------
    # Page Header
    # ----------------------
    st.markdown('<div class="main-header">About Me</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Financial Engineering Student | Quantitative Research Enthusiast</div>', unsafe_allow_html=True)
    st.markdown("---")

# ----------------------
# Hero Profile Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="profile-box animate-fade-in">
        <div class="section-title">👨‍💼 Professional Profile</div>
        <div class="profile-info">
            <strong>SALHI Reda</strong><br><br>
            🎓 <strong>Engineering Student</strong> at <strong>Centrale Méditerranée</strong><br>
            📊 Passionate about <strong>quantitative finance</strong> and <strong>financial markets</strong><br>
            🔬 Specializing in <strong>derivatives pricing</strong> and <strong>risk management</strong><br>
            💻 Currently developing advanced <strong>pricing models</strong> and <strong>financial tools</strong><br>
            📈 Focused on bridging <strong>mathematical theory</strong> with <strong>practical applications</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Academic & Professional Focus
# ----------------------
with st.container():
    st.markdown("""
    <div class="info-box">
        <div class="section-title">🎯 Current Focus Areas</div>
        <div class="content-text">
            My academic journey at Centrale Méditerranée has equipped me with a solid foundation in 
            <strong>mathematics</strong> and <strong>engineering principles</strong>, which I now apply to the 
            fascinating world of quantitative finance. I'm particularly interested in the intersection of 
            <strong>stochastic calculus</strong>, <strong>computational methods</strong>, and 
            <strong>financial modeling</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Skills & Expertise Grid
# ----------------------
with st.container():
    st.markdown('<div class="sub-header">🛠️ Technical Expertise</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--primary-blue); margin-bottom: 1rem;">💻 Programming & Tools</h4>
            <ul style="line-height: 1.8;">
                <li><strong>Python</strong> - NumPy, Pandas, SciPy</li>
                <li><strong>MATLAB</strong> - Financial modeling</li>
                <li><strong>SQL</strong> - Data analysis</li>
                <li><strong>Excel</strong> - Advanced financial functions</li>
                <li><strong>Streamlit</strong> - Web applications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--success-green); margin-bottom: 1rem;">📊 Financial Knowledge</h4>
            <ul style="line-height: 1.8;">
                <li><strong>Derivatives Pricing</strong></li>
                <li><strong>Risk Management</strong></li>
                <li><strong>Portfolio Optimization</strong></li>
                <li><strong>Monte Carlo Methods</strong></li>
                <li><strong>Interest Rate Models</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: var(--info-cyan); margin-bottom: 1rem;">🧮 Mathematical Foundation</h4>
            <ul style="line-height: 1.8;">
                <li><strong>Stochastic Calculus</strong></li>
                <li><strong>Statistics & Probability</strong></li>
                <li><strong>Numerical Methods</strong></li>
                <li><strong>Optimization Theory</strong></li>
                <li><strong>Differential Equations</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ----------------------
# Projects & Achievements
# ----------------------
with st.container():
    st.markdown('<div class="sub-header">🚀 Projects & Achievements</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="portfolio-box">
        <div class="section-title">📈 Financial Engineering Projects</div>
        <div class="content-text">
            <strong>Derivatives Pricing Application:</strong> Developed a comprehensive Streamlit application 
            featuring Black-Scholes, Binomial, and Monte Carlo pricing models with interactive visualizations 
            and real-time Greeks calculations.
        </div>
        <div class="content-text">
            <strong>Risk Management Tools:</strong> Created sophisticated risk assessment modules including 
            VaR calculations, stress testing scenarios, and portfolio optimization algorithms.
        </div>
        <div class="content-text">
            <strong>Academic Excellence:</strong> Consistently maintaining high academic performance while 
            pursuing advanced coursework in financial mathematics and engineering principles.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# CV Downloads Section
# ----------------------
with st.container():
    st.markdown('<div class="sub-header">📄 Resume Downloads</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="download-section">
        <div class="section-title">📋 Professional Documents</div>
        <p style="margin-bottom: 1.5rem; color: #6c757d; font-size: 1.1rem;">
            Download my comprehensive resume showcasing my academic background, technical skills, 
            and project experience in quantitative finance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CV download buttons with enhanced styling
    col1, col2 = st.columns(2)
    
    cv_en = "assets/Reda_Salhi_CV_EN.pdf"
    cv_fr = "assets/Reda_Salhi_CV_FR.pdf"
    
    with col1:
        if os.path.exists(cv_en):
            with open(cv_en, "rb") as f:
                st.download_button(
                    label="📥 Download CV - English Version",
                    data=f,
                    file_name="Reda_Salhi_CV_EN.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); 
                        padding: 1rem; border-radius: 0.75rem; text-align: center; 
                        border: 2px dashed #d1d5db;">
                📄 English CV - Coming Soon
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if os.path.exists(cv_fr):
            with open(cv_fr, "rb") as f:
                st.download_button(
                    label="📥 Download CV - Version Française",
                    data=f,
                    file_name="Reda_Salhi_CV_FR.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); 
                        padding: 1rem; border-radius: 0.75rem; text-align: center; 
                        border: 2px dashed #d1d5db;">
                📄 French CV - Bientôt Disponible
            </div>
            """, unsafe_allow_html=True)

# ----------------------
# Professional Network
# ----------------------
with st.container():
    st.markdown('<div class="sub-header">🌐 Professional Network</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="links-box">
        <div class="section-title">🔗 Connect & Collaborate</div>
        <p style="margin-bottom: 1.5rem; color: #6c757d; font-size: 1.1rem;">
            I'm always open to discussing quantitative finance, sharing knowledge, and exploring 
            collaboration opportunities in financial engineering and research.
        </p>
        
        <div class="link-item">
            <a href="https://www.linkedin.com/in/reda-salhi-195297290/" target="_blank">
                🔗 LinkedIn Profile - Professional Network & Updates
            </a>
        </div>
        
        <div class="link-item">
            <a href="https://github.com/RedaSalhi" target="_blank">
                💻 GitHub Portfolio - Code & Projects Repository
            </a>
        </div>
        
        <div class="link-item">
            <a href="mailto:salhi.reda47@gmail.com" class="email-link">
                ✉️ Direct Email - salhi.reda47@gmail.com
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Contact Form Section
# ----------------------
with st.container():
    st.markdown('<div class="sub-header">📬 Get In Touch</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                padding: 2rem; border-radius: 1rem; margin: 1.5rem 0;">
        <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 1.5rem; text-align: center;">
            Whether you're interested in discussing quantitative finance, exploring collaboration opportunities, 
            or have questions about my projects, I'd love to hear from you!
        </p>
    </div>
    """, unsafe_allow_html=True)

    formsubmit_email = "salhi.reda47@gmail.com"

    form_code = f"""
    <div class="contact-form">
        <form action="https://formsubmit.co/{formsubmit_email}" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="hidden" name="_template" value="table">
            <input type="hidden" name="_subject" value="New Contact from Finance Portfolio">
            <input type="hidden" name="_autoresponse" value="Thank you for reaching out! I appreciate your interest and will respond as soon as possible. - Reda">
            
            <input type="text" name="name" placeholder="Your Full Name" required>
            <input type="email" name="email" placeholder="Your Email Address" required>
            <input type="text" name="subject" placeholder="Subject (Optional)">
            <textarea name="message" placeholder="Your Message - Feel free to discuss quantitative finance, collaboration opportunities, or any questions about my work!" rows="6" required></textarea>
            
            <button type="submit">
                📤 Send Message
            </button>
        </form>
    </div>
    """

    st.markdown(form_code, unsafe_allow_html=True)

# ----------------------
# Philosophy & Vision
# ----------------------
with st.container():
    st.markdown('<div class="sub-header">💭 Professional Philosophy</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
        <div class="section-title">🎯 My Vision</div>
        <div class="content-text">
            <strong>"Mathematics is the language of finance, and technology is its voice."</strong>
        </div>
        <div class="content-text">
            I believe in the power of combining rigorous mathematical foundations with practical 
            technological solutions to solve complex financial problems. My goal is to contribute 
            to the evolution of quantitative finance by developing innovative models and tools 
            that bridge theoretical concepts with real-world applications.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Footer with Call to Action
# ----------------------
with st.container():
    st.markdown("---")
    
    st.markdown("""
    <div class="footer-section">
        <div class="section-title">🚀 Let's Build the Future of Finance</div>
        <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 1.5rem;">
            Interested in quantitative finance, derivatives pricing, or financial engineering? 
            Let's connect and explore how we can push the boundaries of financial innovation together.
        </p>
        <p style="font-style: italic; color: #94a3b8;">
            "The future belongs to those who can bridge mathematics, technology, and finance."
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Additional Styling for Enhanced UX
# ----------------------
st.markdown("""
<style>
/* Additional smooth scroll behavior */
html {
    scroll-behavior: smooth;
}

/* Enhanced list styling within metric cards */
.metric-card ul {
    list-style: none;
    padding-left: 0;
}

.metric-card li {
    padding: 0.5rem 0;
    border-bottom: 1px solid #e5e7eb;
    transition: all 0.2s ease;
}

.metric-card li:last-child {
    border-bottom: none;
}

.metric-card li:hover {
    padding-left: 0.5rem;
    color: var(--primary-blue);
}

/* Enhanced form accessibility */
.contact-form input:invalid,
.contact-form textarea:invalid {
    border-color: #ef4444;
}

.contact-form input:valid,
.contact-form textarea:valid {
    border-color: #10b981;
}
</style>
""", unsafe_allow_html=True)
