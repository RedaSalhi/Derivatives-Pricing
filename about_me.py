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
# Future Enhancements
# ----------------------
st.markdown("---")
st.subheader("Future Enhancements for Derivatives-Pricing App")

st.markdown("""
This platform is an ongoing project. Future enhancement include:

#### Phase 1 — Core Features
- **Exotic Greeks Engine**: Add closed-form Greeks for digital options and Monte Carlo-based Greeks (e.g., pathwise derivatives) for barrier and lookback payoffs.
- **Strategy Optimizer**: Build a new tab for optimizing multi-leg strategies with objectives like max profit, min premium, or risk-neutral hedging, using numerical solvers (`scipy.optimize`).
- **Futures Pricing Module**: Finalize the placeholder with fair value pricing, arbitrage detection, and visual comparisons against spot prices.

#### Phase 2 — Advanced Analytics & Visualization
- **Real-Time Market Feed**: Pull live data from Yahoo Finance or Finnhub, auto-populating inputs (e.g., spot price, interest rate).
- **3D Surface Viewer**: Visualize price or Greek values across strike and maturity dimensions using 3D plots.
- **Monte Carlo Animations**: Animate sample paths for exotic options like Asian or Barrier, helping users intuit risk profiles.

#### Phase 3 — Quant Models & Research Tools
- **Local & Stochastic Volatility Models**: Add support for Heston and Dupire models with implied surface calibration and dynamic plotting.
- **Stress Testing Suite**: Automate sensitivity analysis across key parameters and export results (PDF/CSV) for audit or report use.

#### Phase 4 — Deployment & Community
- **Public Deployment**: Host the platform on Streamlit Cloud or Render for public and academic use.
- **Full Documentation & Tutorials**: Expand the `README.md` with a roadmap, screenshots, feature map, and auto-generated model docs via `pdoc`.

#### Bonus: Model Benchmarking Dashboard
- Compare execution speed and accuracy across models (Black-Scholes, Binomial, Monte Carlo) under various scenarios. Display errors vs analytical benchmarks and visualize trade-offs.

---

Each enhancement is aligned with my long-term goal: to offer a transparent, educational, and practical platform for derivatives learning, strategy design, and risk management simulation.
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
