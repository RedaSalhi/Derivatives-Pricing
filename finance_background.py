import streamlit as st



st.header("🎯 Objective")
st.markdown("""
The **Derivatives Pricing App** is built to help users understand and visualize how various financial derivatives are priced and analyzed using different mathematical models. Whether you're a student, a quant, or a curious learner, this app aims to bridge theory and implementation.
""")

st.header("🧮 Methodology & Pricing Models")
st.markdown("""
We use multiple models depending on the product type:

### 1. **Black-Scholes Model**
- Closed-form analytical model for European options.
- Assumes constant volatility and no early exercise.
- Also extended to price **digital options**.

### 2. **Binomial Tree Model**
- Discrete-time model for both European and American options.
- Useful for early exercise features.
- Supports vanilla, lookback, and barrier options.

### 3. **Monte Carlo Simulation**
- Numerical technique for path-dependent options like Asian or Lookback.
- Simulates thousands of paths under the risk-neutral measure.
- Extended using **Longstaff-Schwartz** for American-style derivatives.

""")

st.header("📐 Greeks & Sensitivities")
st.markdown("""
To help you assess **risk exposure**, the app computes the **Greeks**:
- **Delta** – Sensitivity to underlying price.
- **Gamma** – Convexity of Delta.
- **Vega** – Sensitivity to volatility.
- **Theta** – Time decay of the option.
- **Rho** – Sensitivity to interest rate changes.

We provide **visual plots** to show how these metrics evolve across strike prices or spot movements.
""")

st.header("📊 Multi-Leg Strategies")
st.markdown("""
In real-world trading, investors often combine multiple options into **strategies**:
- **Spreads**: Vertical, horizontal, diagonal
- **Straddles** and **Strangles**
- **Butterflies** and **Condors**

These strategies can be custom-built and visualized, with **net payoff** and **Greek sensitivity charts**.
""")

st.header("🔧 Engineering & Design")
st.markdown("""
- Built in **Python** using **Streamlit** for an interactive frontend.
- Modular pricing logic organized under `/pricing/`:
    - `models/`: Analytical and numerical pricing engines
    - `utils/`: Greek computation and plotting
    - `option_strategies.py`: Strategy builder and payoff engine

You can **extend** this app with new instruments, exotic derivatives, or even real-time data feeds.
""")

st.header("🔍 Risk-Neutral Framework")
st.markdown("""
All models assume the **risk-neutral measure**:
> Expected discounted payoff under risk-neutral probability = Current Price

This fundamental idea allows us to price any derivative whose payoff is known and depends on future asset paths.
""")

st.info("Want to dive deeper? Explore each pricing method through the Pricer tab!")

st.markdown("---")
st.caption("Built with ❤️ for financial learning.")
