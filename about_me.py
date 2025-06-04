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
- 📅 [Book a Call](https://calendly.com/reda-salhi/30min) — Schedule a meeting
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
st.subheader("📬 Contact")

st.markdown("If you'd like to get in touch, just fill out the form below:")

# Contact form using FormSubmit
form_html = """
<form action="https://formsubmit.co/salhi.reda47@gmail.com" method="POST">
    <!-- Protects from spam -->
    <input type="hidden" name="_captcha" value="false">
    <input type="hidden" name="_template" value="table">
    <input type="hidden" name="_autoresponse" value="Thanks for contacting Reda! He'll reply shortly.">
    <input type="hidden" name="_subject" value="New message from your Streamlit app!">

    <label for="name">Your Name</label><br>
    <input type="text" name="name" required style="width: 100%; padding: 0.5rem;"><br><br>

    <label for="email">Your Email</label><br>
    <input type="email" name="email" required style="width: 100%; padding: 0.5rem;"><br><br>

    <label for="message">Message</label><br>
    <textarea name="message" rows="6" required style="width: 100%; padding: 0.5rem;"></textarea><br><br>

    <button type="submit" style="padding: 0.5rem 1rem; background-color: #4CAF50; color: white; border: none; cursor: pointer;">Send</button>
</form>
"""

# Render form
st.markdown(form_html, unsafe_allow_html=True)
