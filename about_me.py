import streamlit as st
import sys
import os


# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #ff7f0e;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .profile-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .contact-box {
        background: linear-gradient(135deg, #fff3cd 0%, #fef9e7 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .links-box {
        background: linear-gradient(135deg, #d4edda 0%, #e8f5e8 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .download-section {
        background: linear-gradient(135deg, #f8d7da 0%, #fdeaea 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #dc3545;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        color: #1f77b4;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .profile-info {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2c3e50;
    }
    .profile-info strong {
        color: #1f77b4;
    }
    .contact-form {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .form-input {
        width: 100%;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    .form-input:focus {
        border-color: #1f77b4;
        outline: none;
    }
    .form-button {
        background: linear-gradient(135deg, #1f77b4 0%, #2e86de 100%);
        color: white;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .form-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .link-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.5rem 0;
        font-size: 1.1rem;
    }
    .link-item a {
        color: #1f77b4;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    .link-item a:hover {
        color: #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">About Me</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Financial Engineering Student | Quant Researcher</div>', unsafe_allow_html=True)

# ----------------------
# Profile Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="profile-box">
        <div class="section-title">👨‍🎓 Profile</div>
        <div class="profile-info">
            <strong>SALHI Reda</strong><br>
            🎓 Engineering student at <strong>Centrale Méditerranée</strong><br>
            📈 Passionate about <strong>mathematics</strong>, <strong>financial markets</strong>, and <strong>economic research</strong><br>
            💡 Specializing in quantitative finance and derivatives pricing<br>
            🔬 Currently developing advanced pricing models and risk management tools
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# CV Downloads Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="download-section">
        <div class="section-title">📄 Resume Downloads</div>
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
                    label="🇺🇸 Download CV - English Version",
                    data=f,
                    file_name="Reda_Salhi_CV_EN.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("🇺🇸 English CV - Coming Soon")
    
    with col2:
        if os.path.exists(cv_fr):
            with open(cv_fr, "rb") as f:
                st.download_button(
                    label="🇫🇷 Download CV - French Version",
                    data=f,
                    file_name="Reda_Salhi_CV_FR.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("🇫🇷 French CV - Coming Soon")

# ----------------------
# Links Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="links-box">
        <div class="section-title">🔗 Connect With Me</div>
        <div class="link-item">
            💼 <a href="https://www.linkedin.com/in/reda-salhi-195297290/" target="_blank">LinkedIn Profile</a>
        </div>
        <div class="link-item">
            💻 <a href="https://github.com/RedaSalhi" target="_blank">GitHub Portfolio</a>
        </div>
        <div class="link-item">
            📧 Email: salhi.reda47@gmail.com
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------
# Contact Form Section
# ----------------------
with st.container():
    st.markdown("""
    <div class="contact-box">
        <div class="section-title">📬 Get In Touch</div>
        <p style="margin-bottom: 1.5rem; color: #6c757d;">
            I'd love to hear from you! Whether it's about collaboration opportunities, 
            questions about my work, or just to connect, feel free to reach out.
        </p>
    </div>
    """, unsafe_allow_html=True)

formsubmit_email = "salhi.reda47@gmail.com"
form_code = f"""
<style>
    .styled-form {{
        max-width: 500px;
        margin: 2rem auto;
        padding: 1.5rem;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #ddd;
    }}
    .styled-form input,
    .styled-form textarea {{
        width: 100%;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border-radius: 6px;
        border: 1px solid #ccc;
        font-size: 1rem;
    }}
    .styled-form button {{
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        border-radius: 6px;
        cursor: pointer;
    }}
    .styled-form button:hover {{
        background-color: #45a049;
    }}
</style>

<form action="https://formsubmit.co/{formsubmit_email}" method="POST" class="styled-form">
    <input type="hidden" name="_captcha" value="false">
    <input type="hidden" name="_template" value="table">
    <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">

    <input type="text" name="name" placeholder="Your Name" required>
    <input type="email" name="email" placeholder="Your Email" required>
    <textarea name="message" placeholder="Your Message" rows="5" required></textarea>

    <button type="submit">Send Message</button>
</form>
"""


st.markdown(form_code, unsafe_allow_html=True)

# ----------------------
# Skills & Interests (Additional Section)
# ----------------------
with st.container():
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔧 Technical Skills")
        st.markdown("""
        - **Programming**: Python, R, MATLAB
        - **Finance**: Derivatives Pricing, Risk Management
        - **Tools**: Streamlit, NumPy, Pandas
        - **Mathematics**: Stochastic Calculus, Statistics
        """)
    
    with col2:
        st.markdown("### 📚 Areas of Interest")
        st.markdown("""
        - Quantitative Finance
        - Financial Engineering
        - Algorithmic Trading
        - Risk Management
        - Economic Research
        """)
    
    with col3:
        st.markdown("### 🎯 Current Focus")
        st.markdown("""
        - Advanced Derivatives Pricing
        - Monte Carlo Simulations
        - Interest Rate Models
        - Portfolio Optimization
        - Financial Data Analysis
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-style: italic;'>"
    "Thank you for visiting my profile! Looking forward to connecting with you."
    "</div>", 
    unsafe_allow_html=True
)
