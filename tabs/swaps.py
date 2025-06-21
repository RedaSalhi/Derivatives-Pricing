# tabs/swaps.py
# Swaps Tab - Tab 5

import streamlit as st
import numpy as np
import pandas as pd
from styles.app_styles import load_theme

# Import your pricing functions
from pricing.swaps import *
from pricing.models.swaps.ois_fx import (
    build_flat_discount_curve,
    build_flat_fx_forward_curve
)


def swaps_tab():
    """Swaps Tab Content"""

    load_theme()
    
    st.markdown('<div class="main-header">Swap Pricer (In Progress)</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Introduction
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### **Welcome to the Swaps Pricing Module**
    
    This module provides pricing and analysis for various types of swaps:
    - **Interest Rate Swaps (IRS)**: Exchange fixed for floating interest rate payments
    - **Currency Swaps**: Exchange cash flows in different currencies
    - **Equity Swaps**: Exchange equity returns for fixed/floating rates
    
    **Models Available:**
    - **DCF (Discounted Cash Flow)**: Standard present value approach
    - **LMM (LIBOR Market Model)**: Advanced stochastic rate modeling
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main swap configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="sub-header">Swap Configuration</div>', unsafe_allow_html=True)
        
        swap_type = st.selectbox("Swap Type", ["IRS", "Currency", "Equity"])
        
        # Dynamic model selection based on swap type
        available_models = {
            "IRS": ["DCF", "LMM"],
            "Currency": ["DCF"],
            "Equity": ["DCF"]
        }
        
        model = st.selectbox("Model", available_models[swap_type])
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.write(f"**Selected:** {swap_type} Swap using {model} Model")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Dynamic parameters based on selection
        st.markdown('<div class="sub-header">Swap Parameters</div>', unsafe_allow_html=True)
        
        swap_type_lower = swap_type.lower()
        model_lower = model.lower()
        
        fixed_params = {}
        payment_times = []
        
        if swap_type_lower == "irs":
            _configure_irs_parameters(model_lower, fixed_params, payment_times)
        elif swap_type_lower == "currency":
            _configure_currency_parameters(fixed_params, payment_times)
        elif swap_type_lower == "equity":
            _configure_equity_parameters(fixed_params, payment_times)
    
    # Pricing section
    st.markdown("---")
    st.markdown('<div class="sub-header">Pricing Results</div>', unsafe_allow_html=True)
    
    col_calc1, col_calc2 = st.columns([1, 2])
    
    with col_calc1:
        calculate_btn = st.button("🔢 Calculate Swap Price", type="primary", use_container_width=True)
        
        if calculate_btn:
            _calculate_swap_price(swap_type_lower, model_lower, fixed_params)
    
    with col_calc2:
        # Display calculation details
        if calculate_btn and fixed_params:
            _display_swap_details(swap_type_lower, model_lower, fixed_params, payment_times)
    
    # Educational content
    st.markdown("---")
    _display_educational_content()


def _configure_irs_parameters(model_lower, fixed_params, payment_times):
    """Configure Interest Rate Swap parameters"""
    
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        notional = st.number_input("Notional ($)", value=100, min_value=1, step=10)
        fixed_rate = st.slider("Fixed Rate (%)", 1.0, 10.0, 3.0, 0.1) / 100
        
    with col_param2:
        tenor_years = st.selectbox("Swap Tenor", [1, 2, 3, 5, 7, 10], index=4)
        payment_freq = st.selectbox("Payment Frequency", ["Annual", "Semi-Annual", "Quarterly"], index=1)
    
    # Create payment schedule
    freq_map = {"Annual": 1.0, "Semi-Annual": 0.5, "Quarterly": 0.25}
    dt = freq_map[payment_freq]
    payment_times.extend([dt * i for i in range(1, int(tenor_years / dt) + 1)])
    
    if model_lower == "dcf":
        floating_rate = st.slider("Current Floating Rate (%)", 1.0, 10.0, 2.5, 0.1) / 100
        discount_rate = st.slider("Discount Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
        discount_curve = build_flat_discount_curve(discount_rate)

        fixed_params.update({
            "notional": notional,
            "fixed_rate": fixed_rate,
            "floating_rates": [floating_rate] * len(payment_times),
            "payment_times": payment_times,
            "discount_curve": discount_curve
        })

    elif model_lower == "lmm":
        L0 = st.slider("Initial Forward Rate (L0) (%)", 1.0, 10.0, 2.5, 0.1) / 100
        vol = st.slider("Volatility (%)", 1.0, 50.0, 15.0, 1.0) / 100
        discount_rate = st.slider("Discount Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
        discount_curve = build_flat_discount_curve(discount_rate)

        fixed_params.update({
            "notional": notional,
            "fixed_rate": fixed_rate,
            "L0": L0,
            "vol": vol,
            "payment_times": payment_times,
            "discount_curve": discount_curve,
            "n_paths": 5000
        })


def _configure_currency_parameters(fixed_params, payment_times):
    """Configure Currency Swap parameters"""
    
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        notional = st.number_input("Notional (Domestic)", value=100, min_value=1, step=10)
        r_dom = st.slider("Domestic Rate (%)", 0.0, 10.0, 3.0, 0.1) / 100
        r_for = st.slider("Foreign Rate (%)", 0.0, 10.0, 1.5, 0.1) / 100
    
    with col_param2:
        fx_spot = st.number_input("Spot FX Rate", value=1.10, min_value=0.1, step=0.01)
        tenor_years = st.selectbox("Swap Tenor", [1, 2, 3, 5, 7, 10], index=4, key="curr_tenor")
    
    # Create payment schedule
    payment_times.extend([0.5 * i for i in range(1, tenor_years * 2 + 1)])  # Semi-annual
    
    rate_domestic = [r_dom] * len(payment_times)
    rate_foreign = [r_for] * len(payment_times)

    discount_dom = build_flat_discount_curve(r_dom)
    discount_for = build_flat_discount_curve(r_for)
    fx_curve = build_flat_fx_forward_curve(fx_spot, r_dom, r_for)

    fixed_params.update({
        "notional_domestic": notional,
        "rate_domestic": rate_domestic,
        "rate_foreign": rate_foreign,
        "payment_times": payment_times,
        "discount_domestic": discount_dom,
        "discount_foreign": discount_for,
        "fx_forward_curve": fx_curve
    })


def _configure_equity_parameters(fixed_params, payment_times):
    """Configure Equity Swap parameters"""
    
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        notional = st.number_input("Notional ($)", value=100, min_value=1, step=10, key="eq_notional")
        S0 = st.number_input("Equity Start Price ($)", value=100.0, min_value=0.1, step=1.0)
        ST = st.number_input("Equity End Price ($)", value=110.0, min_value=0.1, step=1.0)
    
    with col_param2:
        K = st.number_input("Fixed Strike (K) ($)", value=105.0, min_value=0.1, step=1.0)
        r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0, 0.1, key="eq_rate") / 100
        q = st.slider("Dividend Yield (%)", 0.0, 10.0, 1.0, 0.1, key="eq_div") / 100
        fixed_rate = st.slider("Fixed Leg Rate (%)", 1.0, 10.0, 3.0, 0.1, key="eq_fixed") / 100
    
    T = 5
    payment_times.extend([i for i in range(1, T + 1)])  # Annual payments
    discount_curve = build_flat_discount_curve(r)

    fixed_params.update({
        "notional": notional,
        "equity_start": S0,
        "equity_end": ST,
        "fixed_rate": fixed_rate,
        "payment_times": payment_times,
        "discount_curve": discount_curve
    })


def _calculate_swap_price(swap_type_lower, model_lower, fixed_params):
    """Calculate and display swap price"""
    
    try:
        result = price_swap(swap_type=swap_type_lower, model=model_lower, **fixed_params)
        
        # Display result with appropriate styling
        if result > 0:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.success(f"✅ **Swap Price: ${result:.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        elif result < 0:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.error(f"❌ **Swap Price: ${result:.2f}** (Negative Value)")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.info(f"⚖️ **Swap Price: ${result:.2f}** (Fair Value)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Store result for further analysis
        st.session_state.swap_result = result
        st.session_state.swap_params = fixed_params.copy()
        
    except Exception as e:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.error(f"❌ **Error during pricing:** {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)


def _display_swap_details(swap_type_lower, model_lower, fixed_params, payment_times):
    """Display detailed swap information"""
    
    st.markdown('<div class="sub-header">Swap Details</div>', unsafe_allow_html=True)
    
    # Create details dataframe based on swap type
    if swap_type_lower == "irs":
        details_data = {
            "Parameter": [
                "Swap Type", "Model", "Notional", "Fixed Rate", 
                "Payment Frequency", "Number of Payments"
            ],
            "Value": [
                "Interest Rate Swap", 
                model_lower.upper(),
                f"${fixed_params.get('notional', 0):,.0f}",
                f"{fixed_params.get('fixed_rate', 0)*100:.2f}%",
                f"Every {payment_times[1]:.1f} years" if len(payment_times) > 1 else "N/A",
                len(payment_times)
            ]
        }
        
    elif swap_type_lower == "currency":
        details_data = {
            "Parameter": [
                "Swap Type", "Notional (Domestic)", "Domestic Rate", 
                "Foreign Rate", "FX Spot", "Number of Payments"
            ],
            "Value": [
                "Currency Swap",
                f"${fixed_params.get('notional_domestic', 0):,.0f}",
                f"{fixed_params.get('rate_domestic', [0])[0]*100:.2f}%" if fixed_params.get('rate_domestic') else "N/A",
                f"{fixed_params.get('rate_foreign', [0])[0]*100:.2f}%" if fixed_params.get('rate_foreign') else "N/A",
                f"{fixed_params.get('fx_forward_curve', {}).get('spot', 0):.4f}" if fixed_params.get('fx_forward_curve') else "N/A",
                len(payment_times)
            ]
        }
        
    elif swap_type_lower == "equity":
        details_data = {
            "Parameter": [
                "Swap Type", "Notional", "Equity Start", "Equity End",
                "Fixed Rate", "Number of Payments"
            ],
            "Value": [
                "Equity Swap",
                f"${fixed_params.get('notional', 0):,.0f}",
                f"${fixed_params.get('equity_start', 0):.2f}",
                f"${fixed_params.get('equity_end', 0):.2f}",
                f"{fixed_params.get('fixed_rate', 0)*100:.2f}%",
                len(payment_times)
            ]
        }
    
    details_df = pd.DataFrame(details_data)
    st.table(details_df)
    
    # Payment schedule
    if payment_times:
        st.markdown("**Payment Schedule:**")
        schedule_df = pd.DataFrame({
            "Payment #": range(1, len(payment_times) + 1),
            "Payment Time (Years)": payment_times
        })
        st.dataframe(schedule_df, use_container_width=True)


def _display_educational_content():
    """Display educational content about swaps"""
    
    st.markdown('<div class="sub-header">Educational Resources</div>', unsafe_allow_html=True)
    
    # Swaps overview
    with st.expander("📚 Understanding Swaps"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### What are Swaps?
        
        **Swaps** are derivative contracts where two parties exchange cash flows based on different underlying rates, currencies, or assets.
        
        ### Types of Swaps:
        
        **Interest Rate Swaps (IRS):**
        - Exchange fixed interest payments for floating rate payments
        - Most common type of swap
        - Used for hedging interest rate risk or speculation
        
        **Currency Swaps:**
        - Exchange cash flows in different currencies
        - Includes both principal and interest exchanges
        - Used for hedging foreign exchange risk
        
        **Equity Swaps:**
        - Exchange equity returns for fixed or floating interest payments
        - One leg pays equity performance, other pays fixed/floating rate
        - Used for gaining equity exposure without owning the underlying
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("🔧 Pricing Models"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Pricing Methodologies:
        
        **Discounted Cash Flow (DCF):**
        - Present value of expected future cash flows
        - Uses discount curves for each currency/rate
        - Standard approach for most swap types
        - **Formula:** PV = Σ(CF_t / (1 + r_t)^t)
        
        **LIBOR Market Model (LMM):**
        - Stochastic model for forward interest rates
        - Captures volatility and correlation of rates
        - More sophisticated for interest rate derivatives
        - Used for complex IRS structures
        
        ### Key Concepts:
        - **Par Rate:** Fixed rate that makes swap value zero at inception
        - **Mark-to-Market:** Current market value of existing swap
        - **DV01:** Dollar value of a 1 basis point change in rates
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("⚠️ Risk Factors"):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Key Risks in Swap Trading:
        
        **Interest Rate Risk:**
        - Changes in interest rate curves affect swap values
        - Duration risk increases with swap maturity
        - Basis risk between different rate indices
        
        **Credit Risk:**
        - Counterparty may default on obligations
        - Bilateral exposure changes as rates move
        - Collateral agreements help mitigate this risk
        
        **Foreign Exchange Risk (Currency Swaps):**
        - Exchange rate movements affect cash flows
        - Cross-currency basis risk
        - Funding risk in different currencies
        
        **Model Risk:**
        - Pricing models may not capture all market dynamics
        - Parameter estimation uncertainty
        - Model assumptions may break down in stressed markets
        
        **Operational Risk:**
        - Payment processing errors
        - Settlement failures
        - Documentation and legal risks
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("💼 Market Applications"):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Common Use Cases:
        
        **Hedging Applications:**
        - **Corporations:** Convert floating debt to fixed, or vice versa
        - **Banks:** Manage interest rate risk on loan portfolios
        - **International Companies:** Hedge foreign currency exposure
        
        **Speculation:**
        - Take directional bets on interest rate movements
        - Arbitrage between different rate curves
        - Express views on currency movements
        
        **Asset-Liability Management:**
        - Match duration of assets and liabilities
        - Optimize funding costs
        - Regulatory capital management
        
        **Portfolio Management:**
        - Gain exposure to different markets without direct investment
        - Enhance returns through carry strategies
        - Diversify risk across multiple currencies/rates
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Market data section
    st.markdown('<div class="sub-header">Market Insights</div>', unsafe_allow_html=True)
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("""
        **📊 Market Size:**
        - Global derivatives market: >$600 trillion notional
        - Interest rate swaps: ~80% of all derivatives
        - Daily trading volume: $2-3 trillion
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_insight2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("""
        **🏦 Key Players:**
        - Major banks (JP Morgan, Goldman Sachs, etc.)
        - Corporations hedging debt
        - Asset managers and hedge funds
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p><strong>Swaps Pricing Module</strong> | Built with Streamlit & Python</p>
        <p>⚠️ For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)
