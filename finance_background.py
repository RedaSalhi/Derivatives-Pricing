import streamlit as st

st.title("Finance Background & Methodology")
st.markdown("---")

st.header("Objective")
st.markdown("""
The **Derivatives Pricing App** is built to help users understand and visualize how various financial derivatives are priced and analyzed using different mathematical models. Whether you're a student, a quant, or a curious learner, this app aims to bridge theory and implementation.
""")

st.header("Methodology & Pricing Models")

st.markdown(r"""
We implement core pricing methodologies under the **risk-neutral measure** ($ \mathbb{Q} $), where the present value of any derivative is the discounted expected payoff:

$$
V(t) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r(T - t)} \cdot \text{Payoff}(S_T) \mid \mathcal{F}_t \right]
$$

---

### 1. **Black–Scholes Model**

Used for **European options**, assumes:
- Geometric Brownian Motion (GBM): $ dS_t = \mu S_t dt + \sigma S_t dW_t $
- Constant volatility $ \sigma $
- No arbitrage, no dividends

Closed-form price for a European **Call**:

$$
C = S_0 N(d_1) - K e^{-rT} N(d_2)
$$

with:

$$
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left( r + \frac{1}{2}\sigma^2 \right) T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
$$

We also use a modified version for **digital options** where the payoff is binary.

---

### 2. **Binomial Tree Model**

Used for **European and American options**:
- Builds a discrete recombining tree with $ N $ steps
- Up/down factors: $ u = e^{\sigma \sqrt{\Delta t}}, \quad d = \frac{1}{u} $
- Risk-neutral probability:

$$
p = \frac{e^{r \Delta t} - d}{u - d}
$$

Backward induction yields the fair price. Useful for early-exercise features and American options. Adapted for **barrier** and **lookback** options.

---

### 3. **Monte Carlo Simulation**

Used for **path-dependent options** (e.g., Asian, Lookback). Simulates $ M $ sample paths of the underlying:

$$
S_{t+\Delta t} = S_t \cdot \exp\left( \left( r - \frac{1}{2}\sigma^2 \right)\Delta t + \sigma \sqrt{\Delta t} \cdot Z \right), \quad Z \sim \mathcal{N}(0,1)
$$

Option value estimated as:

$$
V_0 \approx e^{-rT} \cdot \frac{1}{M} \sum_{i=1}^M \text{Payoff}^{(i)}
$$

Extended with **Longstaff-Schwartz** regression for American options.

---
""")

st.header("📐 Greeks & Sensitivities")

st.markdown("""
The app computes standard **Greeks** — partial derivatives of the option price with respect to key parameters — for **vanilla options** under the **Black-Scholes** model.

---

### 🔹 Delta ($ \Delta $)
Represents sensitivity of the option price $ V $ to changes in the underlying asset price $ S $:

$$
\Delta = \\frac{\\partial V}{\\partial S}
$$

---

### 🔹 Gamma ($ \Gamma  $)
Second derivative of the option price with respect to $ S $. Measures the curvature of $ V(S) $:

$$
\Gamma = \\frac{\\partial^2 V}{\\partial S^2}
$$

---

### 🔹 Vega ($ \\nu $)
Sensitivity of the option price to volatility $ \\sigma $:

$$
\\nu = \\frac{\\partial V}{\\partial \\sigma}
$$

---

### 🔹 Theta ($ \Theta $)
Rate of change of the option price with respect to time $ t $:

$$
\Theta = \\frac{\\partial V}{\\partial t}
$$

---

### 🔹 Rho ($ \\rho $)
Sensitivity of the option price to the risk-free interest rate $ r $:

$$
\\rho = \\frac{\\partial V}{\\partial r}
$$

---

For each option, these sensitivities are computed numerically and visualized to show how risk exposure evolves across strike prices and underlying values.
""")


st.header("Multi-Leg Strategies")
st.markdown("""
In real-world trading, investors often combine multiple options into **strategies**:
- **Spreads**: Vertical, horizontal, diagonal
- **Straddles** and **Strangles**
- **Butterflies** and **Condors**

These strategies can be custom-built and visualized, with **net payoff** and **Greek sensitivity charts**.
""")


st.header("Risk-Neutral Framework")

st.markdown(r"""
All pricing models in this app are based on the **risk-neutral measure** $ \mathbb{Q} $, under which the discounted price of a traded asset is a **martingale**. This leads to the fundamental pricing formula:

$$
V(t) = \mathbb{E}^{\mathbb{Q}}\left[ e^{-r(T - t)} \cdot \text{Payoff}(S_T) \mid \mathcal{F}_t \right]
$$

Where:
- $ V(t) $ is the value of the derivative at time $ t $
- $ r $ is the risk-free interest rate
- $ T $ is the maturity
- $ S_T $ is the underlying asset price at time $ T $
- $ \mathcal{F}_t $ is the information available at time $ t $

---

### Application in Models

- In **Black-Scholes**, we transform real-world drift $ \mu $ to $ r $, and simulate:

$$
dS_t = r S_t dt + \sigma S_t dW_t^{\mathbb{Q}}
$$

- In **Binomial models**, we construct a **risk-neutral probability**:

$$
p = \frac{e^{r \Delta t} - d}{u - d}
$$

- In **Monte Carlo**, each simulated payoff is discounted:

$$
V_0 \approx e^{-rT} \cdot \frac{1}{M} \sum_{i=1}^M \text{Payoff}^{(i)}
$$

This unified framework enables the consistent pricing of a wide range of derivatives.
""")


st.header("Engineering & Design")
st.markdown("""
All the code is written in **Python** and available on **GitHub**. The app is fully modular and easy to extend.
""")



st.markdown("---")
st.caption("Built for financial learning.")
