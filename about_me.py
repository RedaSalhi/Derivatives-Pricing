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
🌍 International backpacking experience: France, Germany, Switzerland, Czech Republic, Spain, Malta, Portugal, United Kingdom, etc.   
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
st.subheader("📈 Future Enhancements for Derivatives-Pricing Platform")

st.markdown("""
This platform is an ongoing project designed to evolve into a multi-asset, model-rich educational and professional toolkit. Future enhancements include:

#### 🧱 Phase 1 — Core Features
- **Exotic Greeks Engine**: Compute Greeks for digital, barrier, and lookback options using Monte Carlo and analytical methods.
- **Strategy Optimizer**: Build a new tab to construct and optimize option strategies under cost/risk/return constraints.
- **Futures Pricing Module**: Finalize the placeholder with fair value pricing, carry arbitrage logic, and visual comparisons.

#### 📊 Phase 2 — Advanced Analytics & Visualization
- **Monte Carlo Path Animations**: Animate simulations for exotics (Asian, Barrier, Lookback) to visualize risk dynamics.
- **Stress Testing Suite**: Automate multi-parameter stress tests (e.g., spot, rate, vol) with exportable results.

#### 🧮 Phase 3 — Fixed Income & Credit Products
- **Interest Rate Swaps (IRS)**: Price using **discounted cash flows**, with discounting from **OIS curves** and forecasting from **forward LIBOR/SOFR curves**.
- **Currency Swaps**: Implement dual-currency valuation with **FX forward curves** and **cross-currency basis adjustments**.
- **Equity Swaps**: Model synthetic total return swaps via **forward replication** and dividend assumptions.
- **Caps & Floors**: Price using **Black’s model** (caplets/floorlets) and **Hull-White** for term structure consistency.
- **Swaptions**: Add models including **Black lognormal**, **Hull-White**, and the **LIBOR Market Model (LMM)** for more realistic term volatility.

#### 🧷 Phase 4 — Credit Derivatives & Structured Products
- **Credit Default Swaps (CDS)**: Implement pricing with **reduced-form (Jarrow-Turnbull)** and **structural models (Merton)**.
- **Collateralized Debt Obligations (CDOs)**: Integrate **copula models** (e.g., Gaussian copula) for tranche-level pricing and correlation stress.

#### 🔁 Phase 5 — Complex & Embedded Options
- **Convertible Bonds**: Decompose into bond + option; price via **binomial trees**, **Monte Carlo**, or **finite difference PDE solvers**.
- **Swaptions with Smile Adjustment**: Add SABR or LMM-based volatility surfaces for realistic implied smiles.

#### 📈 Phase 6 — Volatility & Real Options
- **Variance & Volatility Swaps**: Use replication via portfolios of options and model-free estimators.
- **Real Options Analysis**: Extend the pricing engine to include corporate finance applications (e.g., delay, abandon, expand projects) via binomial/MC models.

#### 🚀 Deployment & Community
- **Public Deployment**: Host on **Streamlit Cloud** or **Render**, enabling public demos and academic access.
- **Documentation & Tutorials**: Include video walkthroughs, model comparisons, and GitHub-backed docs.

#### 🧪 Bonus: Model Benchmarking Dashboard
- Benchmark models across pricing speed, memory, and error vs analytical targets.
- Visualize accuracy–performance trade-offs (e.g., BS vs Binomial vs MC).

---

Each enhancement supports the broader vision: building a transparent, research-grade platform where students, quants, and educators can learn, simulate, and experiment with derivative pricing across markets.
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
