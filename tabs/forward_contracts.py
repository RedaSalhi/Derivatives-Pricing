# tabs/forward_contracts.py
# Forward Contracts Tab - Tab 2

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from styles.app_styles import load_theme

# Import your pricing functions
from pricing.forward import *


def forward_contracts_tab():
    """Forward Contracts Tab Content"""

    load_theme()
    
    
    st.markdown('<div class="main-header">Forward Contract Pricing & Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Parameter input section
    st.markdown('<div class="sub-header">Contract Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>Market Prices</h4>
        </div>
        """, unsafe_allow_html=True)
        
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
            value=105.00, 
            step=1.00,
            help="Agreed delivery price at maturity"
        )
        
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Time & Rates</h4>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown("""
        <div class="info-box">
            <h4>Cost Parameters</h4>
        </div>
        """, unsafe_allow_html=True)
        
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
    st.markdown('<div class="sub-header">Maturity Configuration</div>', unsafe_allow_html=True)
    
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
    
    position = st.selectbox("Position", ["Long", "Short"], help="Long = Buy forward, Short = Sell forward")
    
    # Calculate forward price
    try:
        forward_price = price_forward_contract(
            spot_price, interest_rate, time_to_maturity, storage_cost, dividend_yield
        )
    except Exception as e:
        st.markdown(f"""
        <div class="warning-box">
            <h4>❌ Calculation Error</h4>
            <p>Error calculating forward price: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        forward_price = 0.0
    
    st.divider()
    
    # Main tabs
    ta1, ta2, ta3, ta4 = st.tabs(["Pricing Results", "Mark-to-Market", "Payout Analysis", "Sensitivity"])
    
    with ta1:
        st.markdown('<div class="sub-header">Forward Contract Pricing Results</div>', unsafe_allow_html=True)
        
        # Key metrics in professional table format
        basis = forward_price - spot_price
        carry_cost = (interest_rate + storage_cost - dividend_yield) * time_to_maturity
        total_return = forward_price / spot_price - 1 if spot_price > 0 else 0
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>Key Metrics Summary</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                    <td style="padding: 12px; font-weight: bold;">Value</td>
                    <td style="padding: 12px; font-weight: bold;">Description</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Forward Price</td>
                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold; font-size: 1.1em;">${forward_price:.2f}</td>
                    <td style="padding: 10px; font-style: italic;">Theoretical fair value</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Basis (F - S)</td>
                    <td style="padding: 10px; font-family: monospace; color: {"#2E8B57" if basis >= 0 else "#dc3545"}; font-weight: bold;">${basis:.2f}</td>
                    <td style="padding: 10px; font-style: italic;">Forward premium/discount</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Carry Cost</td>
                    <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">{carry_cost*100:.2f}%</td>
                    <td style="padding: 10px; font-style: italic;">Net cost of holding asset</td>
                </tr>
                <tr>
                    <td style="padding: 10px; font-weight: bold;">Total Return</td>
                    <td style="padding: 10px; font-family: monospace; color: {"#2E8B57" if total_return >= 0 else "#dc3545"}; font-weight: bold;">{total_return*100:.2f}%</td>
                    <td style="padding: 10px; font-style: italic;">Forward vs spot return</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Pricing formula with enhanced formatting
        st.markdown('<div class="sub-header">Cost of Carry Model</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="formula">
            <strong>Forward Pricing Formula:</strong><br><br>
            F = S₀ × e^((r + c - q) × T)
        </div>
        """, unsafe_allow_html=True)
        
        # Parameter breakdown
        st.markdown(f"""
        <div class="parameter-box">
            <h4 style="color: #1f77b4; margin-top: 0;">Parameter Breakdown</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 8px; font-weight: bold;">Parameter</td>
                    <td style="padding: 8px; font-weight: bold;">Symbol</td>
                    <td style="padding: 8px; font-weight: bold;">Value</td>
                    <td style="padding: 8px; font-weight: bold;">Description</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 6px;">Forward Price</td>
                    <td style="padding: 6px; font-weight: bold; color: #1f77b4;">F</td>
                    <td style="padding: 6px; font-family: monospace;">${forward_price:.2f}</td>
                    <td style="padding: 6px; font-size: 0.9em;">Calculated fair value</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 6px;">Spot Price</td>
                    <td style="padding: 6px; font-weight: bold; color: #1f77b4;">S₀</td>
                    <td style="padding: 6px; font-family: monospace;">${spot_price:.2f}</td>
                    <td style="padding: 6px; font-size: 0.9em;">Current market price</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 6px;">Interest Rate</td>
                    <td style="padding: 6px; font-weight: bold; color: #1f77b4;">r</td>
                    <td style="padding: 6px; font-family: monospace;">{interest_rate*100:.2f}%</td>
                    <td style="padding: 6px; font-size: 0.9em;">Risk-free rate</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 6px;">Storage Cost</td>
                    <td style="padding: 6px; font-weight: bold; color: #1f77b4;">c</td>
                    <td style="padding: 6px; font-family: monospace;">{storage_cost*100:.2f}%</td>
                    <td style="padding: 6px; font-size: 0.9em;">Storage and insurance</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 6px;">Dividend Yield</td>
                    <td style="padding: 6px; font-weight: bold; color: #1f77b4;">q</td>
                    <td style="padding: 6px; font-family: monospace;">{dividend_yield*100:.2f}%</td>
                    <td style="padding: 6px; font-size: 0.9em;">Income from asset</td>
                </tr>
                <tr>
                    <td style="padding: 6px;">Time to Maturity</td>
                    <td style="padding: 6px; font-weight: bold; color: #1f77b4;">T</td>
                    <td style="padding: 6px; font-family: monospace;">{time_to_maturity:.2f} years</td>
                    <td style="padding: 6px; font-size: 0.9em;">Contract duration</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Step-by-Step Calculation"):
            net_carry = interest_rate + storage_cost - dividend_yield
            exp_factor = np.exp(net_carry * time_to_maturity)
            forward_price_calc = spot_price * exp_factor
        
            st.markdown("#### Calculation Steps:")
        
            st.latex(rf"""
            \text{{Step 1: Net Carry Rate}} \\
            r + c - q = {interest_rate:.4f} + {storage_cost:.2f} - {dividend_yield:.2f} = {net_carry:.2f}
            """)
        
            st.latex(rf"""
            \text{{Step 2: Exponential Factor}} \\
            e^{{(r + c - q) \cdot T}} = e^{{{net_carry:.2f} \cdot {time_to_maturity:.2f}}} = {exp_factor:.3f}
            """)
        
            st.latex(rf"""
            \text{{Step 3: Forward Price}} \\
            F = S_0 \cdot e^{{(r + c - q) \cdot T}} = {spot_price:.2f} \cdot {exp_factor:.3f} = {forward_price_calc:.2f}
            """)

    
    with ta2:
        st.markdown('<div class="sub-header">Mark-to-Market Analysis</div>', unsafe_allow_html=True)
        st.info("**Mark-to-Market Value:** Current value of the forward contract before maturity (t < T)")
        
        try:
            # Interactive Plotly chart
            fig_mtm = create_plotly_mtm_chart(
                strike_price, time_to_maturity, interest_rate, 
                storage_cost, dividend_yield, position.lower()
            )
            st.plotly_chart(fig_mtm, use_container_width=True)
            
            # Current contract value calculations
            current_value = spot_price * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity) - strike_price * np.exp(-interest_rate * time_to_maturity)
            if position.lower() == "short":
                current_value = -current_value
            
            breakeven = strike_price * np.exp(-interest_rate * time_to_maturity) / np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity)
            
            # Professional metrics display
            st.markdown(f"""
            <div class="metric-container">
                <h4>Current Position Analysis</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                        <td style="padding: 12px; font-weight: bold;">Metric</td>
                        <td style="padding: 12px; font-weight: bold;">Value</td>
                        <td style="padding: 12px; font-weight: bold;">Interpretation</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold;">Contract Value ({position})</td>
                        <td style="padding: 10px; font-family: monospace; color: {"#2E8B57" if current_value >= 0 else "#dc3545"}; font-weight: bold; font-size: 1.1em;">${current_value:.2f}</td>
                        <td style="padding: 10px; font-style: italic;">{"Profit" if current_value >= 0 else "Loss"} if closed now</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold;">Breakeven Spot Price</td>
                        <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">${breakeven:.2f}</td>
                        <td style="padding: 10px; font-style: italic;">Zero P&L spot level</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: bold;">Time Remaining</td>
                        <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">{time_to_maturity:.3f} years ({time_to_maturity*365:.0f} days)</td>
                        <td style="padding: 10px; font-style: italic;">Contract duration left</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Position status
            if current_value > 0:
                st.markdown(f"""
                <div class="success-box">
                    <h4>✅ Position Status: In-the-Money</h4>
                    <p>Your {position.lower()} position is currently profitable by <strong>${current_value:.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            elif current_value < 0:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Position Status: Out-of-the-Money</h4>
                    <p>Your {position.lower()} position is currently losing <strong>${abs(current_value):.2f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <h4>Position Status: At-the-Money</h4>
                    <p>Your position is currently at breakeven</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                <h4>❌ Chart Generation Error</h4>
                <p>Error creating mark-to-market chart: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with ta3:
        st.markdown('<div class="sub-header">Payout Analysis at Maturity</div>', unsafe_allow_html=True)
        st.info("**Payoff Analysis:** Profit/Loss when the forward contract expires (t = T)")
        
        try:
            # Interactive Plotly payout chart
            fig_payout = create_plotly_payout_chart(strike_price, position.lower())
            st.plotly_chart(fig_payout, use_container_width=True)
        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                <h4>❌ Chart Error</h4>
                <p>Error creating payout chart: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced scenario analysis
        st.markdown('<div class="sub-header">Payout Scenarios</div>', unsafe_allow_html=True)
        
        # Create more comprehensive scenarios
        spot_scenarios = [strike_price * mult for mult in [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4]]
        
        scenarios_data = []
        for spot in spot_scenarios:
            long_payout = spot - strike_price
            short_payout = strike_price - spot
            current_payout = long_payout if position.lower() == 'long' else short_payout
            
            scenarios_data.append({
                'Spot at Maturity': spot,
                'Long Payout': long_payout,
                'Short Payout': short_payout,
                f'{position} Position': current_payout,
                'Status': 'Profit' if current_payout > 0 else 'Loss' if current_payout < 0 else 'Breakeven'
            })
        
        scenarios_df = pd.DataFrame(scenarios_data)
        
        # Format the dataframe for better display
        scenarios_df['Spot at Maturity'] = scenarios_df['Spot at Maturity'].apply(lambda x: f"${x:.2f}")
        scenarios_df['Long Payout'] = scenarios_df['Long Payout'].apply(lambda x: f"${x:.2f}")
        scenarios_df['Short Payout'] = scenarios_df['Short Payout'].apply(lambda x: f"${x:.2f}")
        scenarios_df[f'{position} Position'] = scenarios_df[f'{position} Position'].apply(lambda x: f"${x:.2f}")
        
        st.markdown('<div class="results-table">', unsafe_allow_html=True)
        st.dataframe(scenarios_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary statistics
        profit_scenarios = len([s for s in scenarios_data if s[f'{position} Position'] > 0])
        loss_scenarios = len([s for s in scenarios_data if s[f'{position} Position'] < 0])
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📈 Scenario Summary</h4>
            <p><strong>Profitable scenarios:</strong> {profit_scenarios}/{len(scenarios_data)} ({profit_scenarios/len(scenarios_data)*100:.1f}%)</p>
            <p><strong>Loss scenarios:</strong> {loss_scenarios}/{len(scenarios_data)} ({loss_scenarios/len(scenarios_data)*100:.1f}%)</p>
            <p><strong>Breakeven spot price:</strong> ${strike_price:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with ta4:
        st.markdown('<div class="sub-header">Sensitivity Analysis</div>', unsafe_allow_html=True)
        st.info("**Parameter Sensitivity:** How forward prices respond to changes in underlying parameters")
        
        # Sensitivity analysis chart
        try:
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
            
        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                <h4>❌ Sensitivity Chart Error</h4>
                <p>Error creating sensitivity analysis: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk metrics calculations
        st.markdown('<div class="sub-header">Risk Sensitivities (Greeks)</div>', unsafe_allow_html=True)
        
        try:
            # Calculate numerical derivatives
            delta_s = 0.01 * spot_price  # 1% change in spot
            delta_r = 0.0001  # 1 basis point
            delta_t = 0.01    # 1% change in time
            
            # Spot sensitivity (Delta equivalent)
            spot_up = price_forward_contract(spot_price + delta_s, interest_rate, time_to_maturity, storage_cost, dividend_yield)
            spot_sensitivity = (spot_up - forward_price) / delta_s
            
            # Rate sensitivity (Rho equivalent)
            rate_up = price_forward_contract(spot_price, interest_rate + delta_r, time_to_maturity, storage_cost, dividend_yield)
            rate_sensitivity = (rate_up - forward_price) / (100 * delta_r)
            
            # Time sensitivity (Theta equivalent)
            time_up = price_forward_contract(spot_price, interest_rate, time_to_maturity + delta_t, storage_cost, dividend_yield)
            time_sensitivity = (time_up - forward_price) / delta_t
            
            # Storage cost sensitivity
            storage_up = price_forward_contract(spot_price, interest_rate, time_to_maturity, storage_cost + delta_r, dividend_yield)
            storage_sensitivity = (storage_up - forward_price) / (100 * delta_r)
            
            # Dividend sensitivity
            dividend_up = price_forward_contract(spot_price, interest_rate, time_to_maturity, storage_cost, dividend_yield + delta_r)
            dividend_sensitivity = (dividend_up - forward_price) / (100 * delta_r)
            
            # Display sensitivities in professional table
            st.markdown(f"""
            <div class="metric-container">
                <h4>Risk Sensitivities</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                        <td style="padding: 12px; font-weight: bold;">Risk Factor</td>
                        <td style="padding: 12px; font-weight: bold;">Sensitivity</td>
                        <td style="padding: 12px; font-weight: bold;">Unit</td>
                        <td style="padding: 12px; font-weight: bold;">Interpretation</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold;">Spot Price (Delta)</td>
                        <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">{spot_sensitivity:.4f}</td>
                        <td style="padding: 10px;">$/$ spot change</td>
                        <td style="padding: 10px; font-style: italic;">Forward price change per $1 spot move</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold;">Interest Rate (Rho)</td>
                        <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">{rate_sensitivity:.2f}</td>
                        <td style="padding: 10px;">$/bp change</td>
                        <td style="padding: 10px; font-style: italic;">Forward price change per 1bp rate move</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold;">Time Decay (Theta)</td>
                        <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">{time_sensitivity:.4f}</td>
                        <td style="padding: 10px;">$/1% time change</td>
                        <td style="padding: 10px; font-style: italic;">Forward price change per 1% time change</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 10px; font-weight: bold;">Storage Cost</td>
                        <td style="padding: 10px; font-family: monospace; color: #2E8B57; font-weight: bold;">{storage_sensitivity:.2f}</td>
                        <td style="padding: 10px;">$/bp change</td>
                        <td style="padding: 10px; font-style: italic;">Forward price change per 1bp storage cost</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; font-weight: bold;">Dividend Yield</td>
                        <td style="padding: 10px; font-family: monospace; color: #dc3545; font-weight: bold;">{dividend_sensitivity:.2f}</td>
                        <td style="padding: 10px;">$/bp change</td>
                        <td style="padding: 10px; font-style: italic;">Forward price change per 1bp dividend</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk interpretation
            if abs(spot_sensitivity - 1.0) < 0.01:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Spot Sensitivity Analysis</h4>
                    <p>Delta ≈ 1.0: Forward price moves almost 1:1 with spot price, indicating proper hedging characteristics.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if rate_sensitivity > 0:
                st.markdown(f"""
                <div class="info-box">
                    <h4>Interest Rate Risk</h4>
                    <p>Positive rate sensitivity ({rate_sensitivity:.2f}): Forward prices increase when interest rates rise. This is typical for most forward contracts.</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
                <h4>❌ Sensitivity Calculation Error</h4>
                <p>Error calculating sensitivities: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Educational Content Section
    st.markdown("---")
    st.markdown('<div class="sub-header"> Educational Resources</div>', unsafe_allow_html=True)
    
    with st.expander(" Understanding Forward Contracts"):
        st.markdown("#### Forward Contract Fundamentals")
        
        st.markdown(
            '<div style="text-align: center; font-size: 1.2em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">Forward Contract: Agreement to buy/sell an asset at a specified price on a future date</div>', 
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                '<div style="background-color: #d4edda; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #28a745;"><strong>🟢 Long Position (Buyer)</strong></div>', 
                unsafe_allow_html=True
            )
            st.markdown("""
            • **Obligation:** Buy the asset at maturity  
            • **Payoff:** $S_T - K$ (where $S_T =$ spot at maturity, $K =$ forward price)  
            • **Profit when:** $S_T > K$ (spot price rises above forward price)  
            • **Risk:** Unlimited upside, limited downside to $-K$
            """)
        
        with col2:
            st.markdown(
                '<div style="background-color: #f8d7da; padding: 12px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #dc3545;"><strong>🔴 Short Position (Seller)</strong></div>', 
                unsafe_allow_html=True
            )
            st.markdown("""
            • **Obligation:** Sell the asset at maturity  
            • **Payoff:** $K - S_T  $
            • **Profit when:** $S_T < K$ (spot price falls below forward price)  
            • **Risk:** Limited upside to $K$, unlimited downside
            """)
        
        st.markdown("#### Key Characteristics")
        st.markdown("""
        • **No upfront cost:** Unlike options, forwards have zero initial value  
        • **Symmetric payoff:** Equal probability of profit/loss  
        • **Settlement:** Can be cash-settled or physically delivered  
        • **Customizable:** Terms negotiated between counterparties
        """)
    
    with st.expander("Cost of Carry Model"):
        st.markdown("#### Theoretical Foundation")
        
        st.markdown(
            '<div style="text-align: center; font-size: 1.3em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">F = S₀ × e^((r + c - q) × T)</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("#### Component Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Costs (Increase Forward Price)")
            st.markdown("""
            • **Interest Rate (r):** Cost of financing the purchase  
            • **Storage Costs (c):** Physical storage, insurance, maintenance  
            • **Convenience Yield:** Benefit of immediate availability  
            """)
        
        with col2:
            st.markdown("##### Benefits (Decrease Forward Price)")
            st.markdown("""
            • **Dividends (q):** Cash flows received from holding asset  
            • **Commodity Yields:** Benefits from physical ownership  
            • **Lease Rates:** Income from lending the asset  
            """)
        
        st.markdown("#### No-Arbitrage Principle")
        st.markdown("""
        The forward price is determined by the **no-arbitrage condition**:  
        
        **If $F > S_0 × e^{(r+c-q)T}$:** Sell forward, buy spot → Risk-free profit  
        **If $F < S_0 × e^{(r+c-q)T}$:** Buy forward, sell spot → Risk-free profit  
        
        Market forces ensure $F = S_0 × e^{(r+c-q)T}$ in equilibrium.
        """)
    
    with st.expander("Mark-to-Market Valuation"):
        st.markdown("#### Valuation Before Maturity")
        
        st.markdown(
            '<div style="text-align: center; font-size: 1.2em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">V = S × e^((r+c-q)×(T-t)) - K × e^(-r×(T-t))</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("#### Value Components")
        st.markdown("""
        • **$S × e^{(r+c-q)×(T-t)}$:** Present value of receiving the asset  
        • **$K × e^{-r×(T-t)}$:** Present value of paying the strike price  
        • **Net Value:** Difference between these present values
        """)
        
        st.markdown("#### Key Insights")
        st.markdown("""
        • **At inception (t=0):** $V = 0$ (forward is fairly priced)  
        • **During life:** $V$ fluctuates with spot price, rates, and time  
        • **At maturity (t=T):** $V = S_T - K$ (terminal payoff)  
        • **Long position:** Gains when spot rises, loses when spot falls  
        • **Short position:** Opposite of long position
        """)
    
    with st.expander("Risk Management & Applications"):
        st.markdown("#### Common Applications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Hedging Applications")
            st.markdown("""
            • **Export Company:** Sell foreign currency forward  
            • **Import Company:** Buy foreign currency forward  
            • **Farmer:** Sell crop forward to lock in price  
            • **Manufacturer:** Buy raw materials forward  
            • **Investor:** Hedge portfolio against market moves
            """)
        
        with col2:
            st.markdown("##### Speculation & Arbitrage")
            st.markdown("""
            • **Directional Bets:** Take views on future price direction  
            • **Carry Trade:** Exploit interest rate differentials  
            • **Basis Trading:** Trade forward vs spot price relationships  
            • **Calendar Spreads:** Trade different maturity forwards  
            • **Cross-Asset Arbitrage:** Exploit pricing discrepancies
            """)
        
        st.markdown("#### Risk Considerations")
        st.markdown("""
        ##### Market Risks
        • **Price Risk:** Adverse moves in underlying asset price  
        • **Interest Rate Risk:** Changes in risk-free rates  
        • **Basis Risk:** Forward-spot relationship changes  
        • **Volatility Risk:** Impact of price fluctuations
        
        ##### Operational Risks
        • **Counterparty Risk:** Risk of default by other party  
        • **Liquidity Risk:** Difficulty in closing positions  
        • **Settlement Risk:** Issues with physical delivery  
        • **Regulatory Risk:** Changes in rules and regulations
        """)
    
    with st.expander("Advanced Formulas & Calculations"):
        st.markdown("#### Mathematical Framework")
        
        st.markdown("##### Forward Price Formula")
        st.markdown(
            '<div style="text-align: center; font-size: 1.1em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">F = S₀ × e^((r + c - q) × T)</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("##### Mark-to-Market Value")
        st.markdown(
            '<div style="text-align: center; font-size: 1.1em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">V_long = S_t × e^((r+c-q)×(T-t)) - K × e^(-r×(T-t))</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("##### Payoff at Maturity")
        st.markdown(
            '<div style="text-align: center; font-size: 1.1em; font-weight: bold; color: #1f77b4; margin: 15px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px; border: 1px solid #d0e7ff;">Payoff_long = S_T - K<br>Payoff_short = K - S_T</div>', 
            unsafe_allow_html=True
        )
        
        st.markdown("##### Risk Sensitivities")
        st.latex(r"""
            \begin{aligned}
            \text{Delta } \left( \frac{\partial F}{\partial S_0} \right): & \quad F_\Delta = e^{(r + c - q)T} \approx 1 \text{ for short maturities} \\
            \text{Rho } \left( \frac{\partial F}{\partial r} \right): & \quad F_\rho = S_0 \cdot T \cdot e^{(r + c - q)T} \\
            \text{Theta } \left( \frac{\partial F}{\partial T} \right): & \quad F_\Theta = S_0 \cdot (r + c - q) \cdot e^{(r + c - q)T} \\
            \text{Storage Sensitivity } \left( \frac{\partial F}{\partial c} \right): & \quad F_{\text{storage}} = S_0 \cdot T \cdot e^{(r + c - q)T} \\
            \text{Dividend Sensitivity } \left( \frac{\partial F}{\partial q} \right): & \quad F_{\text{dividend}} = -S_0 \cdot T \cdot e^{(r + c - q)T}
            \end{aligned}
            """)
        
        st.markdown("##### Basis and Spreads")
        st.markdown("""
        • **Basis:** $F - S$ (forward premium/discount)  
        • **Calendar Spread:** $ F_1 - F_2 $ (between different maturities)  
        • **Cross-Asset Spread:** $ F_{{asset}_1} - F_{{asset}_2} $ (between different assets) 
        """)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 1px solid #dee2e6;'>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2rem;"></span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">Forward Contract Pricing & Analysis</p>
        <p style="margin: 8px 0; color: #6c757d;">Built with Streamlit & Python</p>
        <p style="margin: 0; color: #dc3545; font-weight: bold;">⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
