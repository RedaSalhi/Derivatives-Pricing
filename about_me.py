import streamlit as st
import os

# ----------------------
# Header
# ----------------------
st.title("👨‍💼 About Me")
st.caption("Financial Engineering Student | Quant Researcher")

st.markdown("""
**SALHI Reda**  
🎓 Engineering student at **Centrale Méditerranée**  
📈 Passionate about **mathematics**, **financial markets**, and **economic research**  
🌍 International solo travel experience: France, Germany, Switzerland, Czech Republic, Spain, Malta, Portugal, United Kingdom, etc.   
""")

# ----------------------
# CV Downloads
# ----------------------
cv_en = "assets/Reda_Salhi_CV_EN.pdf"
cv_fr = "assets/Reda_Salhi_CV_FR.pdf"

if os.path.exists(cv_en):
    with open(cv_en, "rb") as f:
        st.download_button(
            label="📄 Download My CV - English Version",
            data=f,
            file_name="Reda_Salhi_CV_EN.pdf",
            mime="application/pdf"
        )

if os.path.exists(cv_fr):
    with open(cv_fr, "rb") as f:
        st.download_button(
            label="📄 Download My CV - French Version",
            data=f,
            file_name="Reda_Salhi_CV_FR.pdf",
            mime="application/pdf"
        )

# ----------------------
# Links
# ----------------------
st.markdown("---")
st.subheader("🔗 Links")

st.markdown("""
- [LinkedIn](https://www.linkedin.com/in/reda-salhi-195297290/)
- [GitHub](https://github.com/RedaSalhi)
""")

# ----------------------
# Current Projects
# ----------------------
st.markdown("---")
st.subheader("📊 Current Interests & Projects")

st.markdown("""
- 📘 Research on **Zipf’s Law** in urban economics, with applications across cities and brain networks  
- 📉 Financial modeling: **Black-Scholes**, **Hull-White**, **Monte Carlo**, **EVT for systemic risk**  
- 🤖 Building a **web-based derivatives pricing app** for students and quants  
- 🧮 Strong foundation in **portfolio theory**, **CAPM regression**, and **macro forecasting with VAR models**  
""")

# ----------------------
# Project Ideas & Aspirations
# ----------------------
st.markdown("---")
st.subheader("🚀 Project Ideas & Aspirations")

st.markdown("""
- 🧠 Applying **machine learning** to volatility surface calibration  
- 🌐 Creating a **real-time market dashboard** with news sentiment + economic indicators  
- 📊 Exploring **agent-based models** to simulate financial crises and contagion  
- 🛰️ Using **remote sensing data** to forecast agricultural commodities (geo-finance research)  
- 🧾 Publishing a research paper on **EVT-based stress-testing** for global banks  
""")

# ----------------------
# Contact Info
# ----------------------
st.markdown("---")
st.subheader("📬 Contact Me")
st.markdown("If you'd like to get in touch, just fill out the form below:")

# Obfuscated email (only used in form action, not visible)
formsubmit_email = "salhi.reda47@gmail.com"

form_code = f"""
<form action="https://formsubmit.co/{formsubmit_email}" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="hidden" name="_template" value="table">
    <input type="hidden" name="_autoresponse" value="Thanks for reaching out! I'll respond as soon as possible.">
    <input type="text" name="name" placeholder="Your Name" required style="width: 100%; padding: 0.5rem; margin-bottom: 1rem;"><br>
    <input type="email" name="email" placeholder="Your Email" required style="width: 100%; padding: 0.5rem; margin-bottom: 1rem;"><br>
    <textarea name="message" placeholder="Your Message" rows="5" required style="width: 100%; padding: 0.5rem; margin-bottom: 1rem;"></textarea><br>
    <button type="submit" style="background-color:#4CAF50;color:white;padding:0.75rem 1.5rem;border:none;cursor:pointer;">
        Send Message
    </button>
</form>
"""

st.markdown(form_code, unsafe_allow_html=True)
