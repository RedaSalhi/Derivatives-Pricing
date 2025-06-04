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

# Obfuscated email
email_user = "salhi.reda47"
email_domain = "gmail.com"
obfuscated_email = f"{email_user} @ {email_domain.replace('.', ' [dot] ')}"

st.markdown(f"""
📧 Email: `{obfuscated_email}`  
📱 Phone (France): +33 7 58 29 80 19 
""")

# ----------------------
# Contact Form
# ----------------------
with st.form("contact_form"):
    st.markdown("### 📩 Get in Touch")

    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message", height=150)
    submitted = st.form_submit_button("Send Message")

    if submitted:
        if name and email and message:
            st.success(f"Thank you {name}, your message has been received!")
            # Optionally: store to a database, send email, or log the message
        else:
            st.warning("Please complete all fields before submitting.")
