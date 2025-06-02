import streamlit as st
import numpy as np
from scipy.stats import norm
import yfinance as yf

st.set_page_config(page_title="Derivative Pricing", layout="centered")
st.title("📈 Derivative Pricing App - Tier 1")

def inject_css():
    st.markdown("""
        <style>
            /* Make headings bolder and modern */
            h1, h2, h3 {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-weight: 700;
            }

            /* Modern button style */
            .stButton>button {
                background-color: #1E88E5;
                color: white;
                font-size: 16px;
                padding: 10px 24px;
                border-radius: 8px;
                border: none;
            }

            /* Center all Streamlit containers */
            .css-1v0mbdj, .block-container {
                max-width: 900px;
                margin: auto;
            }

            /* Modern input fields */
            .stTextInput>div>input {
                border: 1px solid #dcdcdc;
                padding: 10px;
                border-radius: 6px;
            }

            /* Light card-style background */
            .stApp {
                background-color: #f9f9f9;
            }
        </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Black-Scholes Pricing
# -----------------------------
def black_scholes_price(S, K, T, r, sigma, option_type="Call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# -----------------------------
# Greeks
# -----------------------------
def black_scholes_greeks(S, K, T, r, sigma, option_type="Call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "Call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "Call" else -d2)
    )
    rho = (
        K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == "Call"
        else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    )
    return delta, gamma, theta, vega, rho

# -----------------------------
# Implied Volatility Estimation
# -----------------------------
def implied_volatility(S, K, T, r, market_price, option_type="Call", tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = black_scholes_greeks(S, K, T, r, sigma, option_type)[3]
        if vega == 0:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
    return None

def binomial_tree_price(S, K, T, r, sigma, steps, option_type="Call", american=False):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Initialize asset prices at maturity
    asset_prices = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])

    # Option payoffs at maturity
    if option_type == "Call":
        values = np.maximum(asset_prices - K, 0)
    else:
        values = np.maximum(K - asset_prices, 0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        values = discount * (q * values[1:] + (1 - q) * values[:-1])
        if american:
            asset_prices = asset_prices[:i+1] * u
            if option_type == "Call":
                values = np.maximum(values, asset_prices - K)
            else:
                values = np.maximum(values, K - asset_prices)

    return values[0]


# -----------------------------
# Real-Time Market Data
# -----------------------------
model = st.sidebar.selectbox("Choose a model", ["Black-Scholes", "Binomial Tree", "Monte Carlo"])
with st.sidebar:
    st.header("🔍 Market Data")
    ticker = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
    data = yf.Ticker(ticker).history(period="1y")
    spot_price = data["Close"].iloc[-1] if not data.empty else 100
    st.write(f"Last price: ${spot_price:.2f}")

st.subheader("🧮 Black-Scholes Option Pricing")

S = st.number_input("Spot Price", value=float(spot_price))
K = st.number_input("Strike Price", value=100.0)
T = st.number_input("Time to maturity (in years)", value=1.0, min_value=0.01)
r = st.number_input("Risk-free rate (e.g., 0.05 = 5%)", value=0.05)
sigma = st.number_input("Volatility (e.g., 0.2 = 20%)", value=0.2)
option_type = st.selectbox("Option Type", ["Call", "Put"])
use_iv = st.checkbox("🔍 Estimate Implied Volatility")

if st.button("Calculate"):
    if use_iv:
        market_price = st.number_input("Observed Market Option Price", value=10.0)
        implied_vol = implied_volatility(S, K, T, r, market_price, option_type)
        if implied_vol:
            st.success(f"Estimated Implied Volatility: {implied_vol:.4f}")
        else:
            st.error("Could not converge to a solution.")
    else:
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        st.success(f"{option_type} Option Price: ${price:.2f}")
        delta, gamma, theta, vega, rho = black_scholes_greeks(S, K, T, r, sigma, option_type)

        with st.expander("📉 Option Greeks"):
            st.write(f"Delta: {delta:.4f}")
            st.write(f"Gamma: {gamma:.4f}")
            st.write(f"Theta: {theta:.4f}")
            st.write(f"Vega:  {vega:.4f}")
            st.write(f"Rho:   {rho:.4f}")

elif model == "Binomial Tree":
    st.subheader("🌲 Binomial Tree Option Pricing")
    S = st.number_input("Spot Price", value=float(spot_price))
    K = st.number_input("Strike Price", value=100.0)
    T = st.number_input("Time to maturity (in years)", value=1.0)
    r = st.number_input("Risk-free rate", value=0.05)
    sigma = st.number_input("Volatility", value=0.2)
    steps = st.slider("Number of steps", 1, 500, 50)
    option_type = st.selectbox("Option Type", ["Call", "Put"])
    style = st.selectbox("Option Style", ["American", "European"])

    if st.button("Price Option (Binomial)"):
        price = binomial_tree_price(S, K, T, r, sigma, steps, option_type, style)
        st.success(f"{'American' if american else 'European'} {option_type} Price: ${price:.4f}")


