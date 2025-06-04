import streamlit as st
import os

# Header
st.title("👨‍💼 About Me")
st.caption("Engineering student | Finance enthusiast | Quant researcher")

st.markdown("""
**SALHI Reda**  
🎓 Engineering student at **Centrale Méditerranée**  
📈 Passionate about **quantitative finance**, **stochastic models**, and **economic research**  
🌍 International experience: France, Germany, Switzerland, Czech Republic, Spain, Malta, etc.  
🏅 Merit scholar | Ranked 1st out of 150+ students | Solo traveler and lifelong learner  
""")

# CV Download
cv_path = "assets/Reda_Salhi_CV_EN.pdf"  # Or just "SALHI_Reda_CV.pdf" if in the same directory
if os.path.exists(cv_path):
    with open(cv_path, "rb") as f:
        st.download_button(
            label="📄 Download My CV",
            data=f,
            file_name="Reda_Salhi_CV_EN.pdf",
            mime="application/pdf"
        )
else:
    st.warning("CV not found. Please make sure 'SALHI_Reda_CV.pdf' is in the correct path.")

# Links
st.markdown("---")
st.subheader("🔗 Links")
st.markdown("""
- [LinkedIn](https://www.linkedin.com/in/reda-salhi-195297290/)
- [GitHub](https://github.com/RedaSalhi)
#- [ResearchGate (if any)](https://www.researchgate.net/)
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
