# tabs/forward_contracts.py
# Forward Contracts Tab - Tab 2

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import your pricing functions
from pricing.forward import *


def forward_contracts_tab():
    """Forward Contracts Tab Content"""
    
    st.markdown('<div class="main-header">Forward Contract Pricing & Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Parameter input section
    st.markdown('<div class="sub-header">Contract Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spot_price = st.number_input(
            "Spot Price ($)", 
            min_value=0.01, 
            value=100.0, 
            step=1.0,
            help="Current price of the underlying asset"
        )
        
        strike_price = st.number_input(
            "Strike Price ($)", 
            min_value=0.01, 
            value=105.0, 
            step=1.0,
            help="Agreed delivery price at maturity"
        )
        
    
    with col2:
        interest_rate = st.number_input(
            "Risk-free Rate (%)", 
            min_value=0.0, 
            value=5.0, 
            step=0.1
        ) / 100
        
        time_input_method = st.selectbox(
            "Time Input Method",
            ["Years", "Days", "Calendar Date"]
        )
    
    with col3:
        storage_cost = st.number_input(
            "Storage Cost Rate (%)", 
            min_value=0.0, 
            value=0.0, 
            step=0.1
        ) / 100
        
        dividend_yield = st.number_input(
            "Dividend Yield (%)", 
            min_value=0.0, 
            value=0.0, 
            step=0.1
        ) / 100
    
    # Time to maturity calculation
    if time_input_method == "Years":
        time_to_maturity = st.number_input(
            "Time to Maturity (years)", 
            min_value=0.01, 
            value=1.0, 
            step=0.1
        )
    elif time_input_method == "Days":
        days_to_maturity = st.number_input(
            "Days to Maturity", 
            min_value=1, 
            value=365, 
            step=1
        )
        time_to_maturity = days_to_maturity / 365.25
    else:  # Calendar Date
        maturity_date = st.date_input(
            "Maturity Date",
            value=datetime.now().date() + timedelta(days=365)
        )
        days_to_maturity = (maturity_date - datetime.now().date()).days
        time_to_maturity = max(days_to_maturity / 365.25, 0.01)
    
    position = st.selectbox("Position", ["Long", "Short"])
    
    # Calculate forward price
    forward_price = price_forward_contract(
        spot_price, interest_rate, time_to_maturity, storage_cost, dividend_yield
    )
    
    # Main tabs
    ta1, ta2, ta3, ta4 = st.tabs(["Pricing Results", "Mark-to-Market", "Payout Analysis", "Sensitivity"])
    
    with ta1:
        st.markdown('<div class="sub-header">Forward Contract Pricing Results</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Forward Price", f"${forward_price:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            basis = forward_price - spot_price
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Basis", f"${basis:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            carry_cost = (interest_rate + storage_cost - dividend_yield) * time_to_maturity
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Carry Cost", f"{carry_cost*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            total_return = forward_price / spot_price - 1
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Return", f"{total_return*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Pricing formula
        st.markdown('<div class="sub-header">Cost of Carry Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.latex(r"F = S_0 \cdot e^{(r + c - q) \cdot T}")
        st.markdown("""
        **Where:**
        - **F** = Forward Price = ${:.2f}
        - **S₀** = Current Spot Price = ${:.2f}
        - **r** = Risk-free Interest Rate = {:.2f}%
        - **c** = Storage Cost Rate = {:.2f}%
        - **q** = Dividend Yield = {:.2f}%
        - **T** = Time to Maturity = {:.1f} years
        """.format(forward_price, spot_price, interest_rate*100, storage_cost*100, dividend_yield*100, time_to_maturity))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed breakdown
        with st.expander("Calculation Breakdown"):
            net_carry = interest_rate + storage_cost - dividend_yield
            exp_factor = np.exp(net_carry * time_to_maturity)
            forward_price_calc = spot_price * exp_factor
            
            st.write(f"**Net Carry Rate:** {interest_rate*100:.2f}% + {storage_cost*100:.2f}% - {dividend_yield*100:.2f}% = {net_carry*100:.2f}%")
            st.latex(r"\text{Exponential Factor: } e^{(r + c - q) \cdot T}")
            st.latex(fr"e^{{({net_carry*100:.2f}\% \cdot {time_to_maturity:.4f})}} = {exp_factor:.6f}")
            st.latex(r"\text{Forward Price: } F = S \cdot e^{(r + c - q) \cdot T}")
            st.latex(fr"F = {spot_price:.2f} \cdot {exp_factor:.6f} = {forward_price_calc:.2f}")
    
    with ta2:
        st.markdown('<div class="sub-header">Mark-to-Market Analysis</div>', unsafe_allow_html=True)
        st.markdown("*Contract value before maturity (t < T)*")
        
        # Interactive Plotly chart
        fig_mtm = create_plotly_mtm_chart(
            strike_price, time_to_maturity, interest_rate, 
            storage_cost, dividend_yield, position.lower()
        )
        st.plotly_chart(fig_mtm, use_container_width=True)
        
        # Current contract value
        current_value = spot_price * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity) - strike_price * np.exp(-interest_rate * time_to_maturity)
        if position.lower() == "short":
            current_value = -current_value
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Current Contract Value", f"${current_value:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            breakeven = strike_price * np.exp(-interest_rate * time_to_maturity) / np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity)
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Breakeven Spot Price", f"${breakeven:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Time to Maturity", f"{time_to_maturity:.3f} years")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Original matplotlib chart
        st.markdown('<div class="sub-header">Original Chart (Matplotlib)</div>', unsafe_allow_html=True)
        plot_forward_mark_to_market(
            strike_price, time_to_maturity, interest_rate, 
            storage_cost, dividend_yield, position.lower()
        )
    
    with ta3:
        st.markdown('<div class="sub-header">Payout Analysis at Maturity</div>', unsafe_allow_html=True)
        st.markdown("*Profit/Loss when contract expires (t = T)*")
        
        # Interactive Plotly payout chart
        fig_payout = create_plotly_payout_chart(strike_price, position.lower())
        st.plotly_chart(fig_payout, use_container_width=True)
        
        # Scenario analysis
        st.markdown('<div class="sub-header">Payout Scenarios</div>', unsafe_allow_html=True)
        spot_scenarios = [strike_price * mult for mult in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
        
        scenarios_data = []
        for spot in spot_scenarios:
            long_payout = spot - strike_price
            short_payout = strike_price - spot
            scenarios_data.append({
                'Spot Price at Maturity': f"${spot:.2f}",
                'Long Position Payout': f"${long_payout:.2f}",
                'Short Position Payout': f"${short_payout:.2f}",
                'Current Position Payout': f"${long_payout if position.lower() == 'long' else short_payout:.2f}"
            })
        
        scenarios_df = pd.DataFrame(scenarios_data)
        st.dataframe(scenarios_df, use_container_width=True)
        
        # Original matplotlib chart
        st.markdown('<div class="sub-header">Original Chart (Matplotlib)</div>', unsafe_allow_html=True)
        plot_forward_payout_and_value(strike_price, position.lower())
    
    with ta4:
        st.markdown('<div class="sub-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
        st.markdown("*How forward prices respond to parameter changes*")
        
        # Sensitivity analysis
        base_params = {
            'spot_price': spot_price,
            'interest_rate': interest_rate,
            'time_to_maturity': time_to_maturity,
            'storage_cost': storage_cost,
            'dividend_yield': dividend_yield,
            'base_forward': forward_price
        }
        
        fig_sensitivity = create_sensitivity_analysis(base_params)
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Risk metrics
        st.markdown('<div class="sub-header">Risk Sensitivities</div>', unsafe_allow_html=True)
        
        # Calculate numerical derivatives
        delta_s = 0.01 * spot_price
        delta_r = 0.0001
        delta_t = 0.01
        
        # Spot sensitivity (Delta equivalent)
        spot_up = price_forward_contract(spot_price + delta_s, interest_rate, time_to_maturity, storage_cost, dividend_yield)
        spot_sensitivity = (spot_up - forward_price) / delta_s
        
        # Rate sensitivity (Rho equivalent)
        rate_up = price_forward_contract(spot_price, interest_rate + delta_r, time_to_maturity, storage_cost, dividend_yield)
        rate_sensitivity = (rate_up - forward_price) / delta_r
        
        # Time sensitivity (Theta equivalent)
        time_up = price_forward_contract(spot_price, interest_rate, time_to_maturity + delta_t, storage_cost, dividend_yield)
        time_sensitivity = (time_up - forward_price) / delta_t
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Spot Sensitivity", f"{spot_sensitivity:.4f}", 
                     help="Forward price change per $1 change in spot price")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Rate Sensitivity", f"{rate_sensitivity:.2f}", 
                     help="Forward price change per 1bp change in interest rate")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Time Sensitivity", f"{time_sensitivity:.4f}", 
                     help="Forward price change per 1% change in time to maturity")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*Forward Contract Pricing & Analysis • Built with Streamlit*")
