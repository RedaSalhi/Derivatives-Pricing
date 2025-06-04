import streamlit as st
import os

# Header
st.title("👨‍💼 About Me")
st.caption("Engineering student | Finance enthusiast | Quant researcher")

st.markdown("""
**SALHI Reda**  
🎓 Engineering student at **Centrale Méditerranée**  
📈 Passionate about **mathematics**, **financial markets**, and **economic research**  
🌍 International solo travel experience: France, Germany, Switzerland, Czech Republic, Spain, Malta, Portugal, United Kingdom, etc.  
🏅 Solo traveler   
""")

# CV Download
cv_en = "assets/Reda_Salhi_CV_EN.pdf"  # Or just "SALHI_Reda_CV.pdf" if in the same directory

cv_fr = "assets/Reda_Salhi_CV_FR.pdf"  # Or just "SALHI_Reda_CV.pdf" if in the same directory

with open(cv_en, "rb") as f:
    st.download_button(
        label="📄 Download My CV - English Version",
        data=f,
        file_name="Reda_Salhi_CV_EN.pdf",
        mime="application/pdf"
    )

with open(cv_fr, "rb") as f:
    st.download_button(
        label="📄 Download My CV - French Version",
        data=f,
        file_name="Reda_Salhi_CV_FR.pdf",
        mime="application/pdf"
    )

# Links
st.markdown("---")
st.subheader("🔗 Links")
st.markdown("""
- [LinkedIn](https://www.linkedin.com/in/reda-salhi-195297290/)
- [GitHub](https://github.com/RedaSalhi)
""")

# Projects & Interests
st.markdown("---")
st.subheader("📊 Current Interests & Projects")

st.markdown("""
- 📘 Research on **Zipf’s Law** in urban economics, with applications across cities and brain networks  
- 📉 Financial modeling: **Black-Scholes**, **Hull-White**, **Monte Carlo**, **EVT for systemic risk**  
- 🤖 Building a **web-based derivatives pricing app** for students and quants  
- 🧮 Strong foundation in **portfolio theory**, **CAPM regression**, and **macro forecasting with VAR models**  
""")

# Optional: contact section
st.markdown("---")
st.subheader("📬 Contact")
st.markdown("Feel free to reach out via LinkedIn or GitHub for collaboration opportunities or internship inquiries.")
