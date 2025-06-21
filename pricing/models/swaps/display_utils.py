# Enhanced Display Manager with Educational Content
# File: pricing/models/enhanced_display_utils.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class EducationalSwapDisplayManager:
    """Enhanced display manager with educational explanations and accurate visualizations"""
    
    @staticmethod
    def display_swap_explanation(swap_type: str):
        """Display educational explanation for swap mechanics"""
        
        explanations = {
            "interest_rate": {
                "title": "🔄 Interest Rate Swap Fundamentals",
                "description": """
                An Interest Rate Swap (IRS) is an agreement to exchange interest payments between two parties.
                """,
                "mechanics": [
                    "**Fixed Leg**: One party pays a predetermined fixed rate",
                    "**Floating Leg**: Other party pays a variable rate (usually SOFR/LIBOR + spread)",
                    "**Notional**: Principal amount (not exchanged, used for calculation)",
                    "**Payment Frequency**: Typically quarterly or semi-annual",
                    "**Settlement**: Only net payment is exchanged"
                ],
                "use_cases": [
                    "**Interest Rate Risk Management**: Convert fixed-rate debt to floating or vice versa",
                    "**Speculation**: Bet on interest rate direction",
                    "**Asset-Liability Matching**: Align cash flows with obligations",
                    "**Cost Reduction**: Potentially lower financing costs"
                ]
            },
            "currency": {
                "title": "💱 Cross-Currency Swap Fundamentals", 
                "description": """
                A Currency Swap involves exchanging principal and interest payments in different currencies.
                """,
                "mechanics": [
                    "**Principal Exchange**: Initial and final exchange of notional amounts",
                    "**Interest Payments**: Periodic payments in each currency",
                    "**FX Risk**: Exposure to exchange rate movements",
                    "**Cross-Currency Basis**: Additional spread reflecting funding costs",
                    "**Dual Curve Pricing**: Separate discount curves for each currency"
                ],
                "use_cases": [
                    "**Currency Hedging**: Eliminate FX exposure from foreign investments",
                    "**Funding Arbitrage**: Access cheaper funding in foreign markets",
                    "**Asset-Liability Matching**: Match foreign currency assets with liabilities",
                    "**Synthetic Foreign Investment**: Create foreign exposure without direct investment"
                ]
            },
            "equity": {
                "title": "📈 Equity Swap Fundamentals",
                "description": """
                An Equity Swap exchanges the returns of an equity position for fixed or floating interest payments.
                """,
                "mechanics": [
                    "**Equity Leg**: Returns based on equity performance (price + dividends)",
                    "**Fixed/Floating Leg**: Traditional interest payment (SOFR + spread)",
                    "**Total Return**: Includes capital appreciation and dividend income",
                    "**Reset Periods**: Performance calculated over specific intervals",
                    "**No Physical Ownership**: Synthetic exposure without buying shares"
                ],
                "use_cases": [
                    "**Synthetic Equity Exposure**: Gain equity exposure without buying shares",
                    "**Leverage**: Amplify equity exposure beyond available capital",
                    "**Tax Efficiency**: Potentially favorable tax treatment",
                    "**Regulatory Capital**: Reduce balance sheet impact vs. physical holdings"
                ]
            }
        }
        
        content = explanations.get(swap_type, explanations["interest_rate"])
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 15px; margin-bottom: 25px; color: white;">
            <h3 style="margin: 0 0 15px 0; color: white;">{content['title']}</h3>
            <p style="font-size: 1.1em; margin-bottom: 20px; color: #f8f9fa;">{content['description']}</p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #ffd700; margin-bottom: 10px;">💡 How It Works</h4>
                    {''.join([f'<p style="margin: 5px 0; font-size: 0.9em;">• {item}</p>' for item in content['mechanics']])}
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                    <h4 style="color: #98fb98; margin-bottom: 10px;">🎯 Common Uses</h4>
                    {''.join([f'<p style="margin: 5px 0; font-size: 0.9em;">• {item}</p>' for item in content['use_cases']])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_enhanced_irs_results(result, notional: float, fixed_rate: float, 
                                   tenor_years: float, model: str, market_data: Dict):
        """Display comprehensive IRS results with educational context"""
        
        # Main results display
        npv = result.npv
        par_rate = result.par_rate
        market_rate = market_data.get('market_rate', par_rate)
        
        # Determine swap favorability
        rate_diff_bp = (fixed_rate - market_rate) * 10000
        npv_color = "#28a745" if npv > 0 else "#dc3545"
        npv_status = "Favorable" if npv > 0 else "Unfavorable"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 25px; border-radius: 15px; border-left: 5px solid #007bff; margin-bottom: 20px;">
            <h3 style="color: #007bff; margin-bottom: 20px;">💰 Swap Valuation Results</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">💵 Net Present Value</h5>
                    <div style="font-size: 2em; font-weight: bold; color: {npv_color};">${npv:,.0f}</div>
                    <div style="color: {npv_color}; font-weight: bold;">● {npv_status}</div>
                    <small style="color: #6c757d;">
                        {"You receive more than you pay" if npv > 0 else "You pay more than you receive"}
                    </small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">📊 Par Rate (Fair Value)</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #007bff;">{par_rate*100:.4f}%</div>
                    <div style="color: #6c757d;">vs Your Rate: {fixed_rate*100:.3f}%</div>
                    <small style="color: #6c757d;">
                        Rate that makes NPV = 0
                    </small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">⚡ DV01 (Risk)</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #28a745;">${result.dv01:,.0f}</div>
                    <div style="color: #6c757d;">Per 1bp rate change</div>
                    <small style="color: #6c757d;">
                        Dollar sensitivity to rates
                    </small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">⏰ Duration</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #ffc107;">{result.duration:.2f}</div>
                    <div style="color: #6c757d;">years</div>
                    <small style="color: #6c757d;">
                        Average time to cash flows
                    </small>
                </div>
                
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Educational explanation of results
        st.markdown(f"""
        <div style="background: #d1ecf1; padding: 20px; border-radius: 10px; border-left: 4px solid #17a2b8; margin-bottom: 20px;">
            <h4 style="color: #0c5460;">📚 Understanding Your Results</h4>
            
            <div style="margin: 15px 0;">
                <strong>NPV Interpretation:</strong>
                <p style="margin: 5px 0;">
                    Your swap has an NPV of <strong>${npv:,.0f}</strong>. This means:
                    {'The present value of cash flows you receive exceeds what you pay by this amount.' if npv > 0 else 'The present value of cash flows you pay exceeds what you receive by this amount.'}
                </p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Par Rate vs Your Rate:</strong>
                <p style="margin: 5px 0;">
                    The market fair value (par rate) is <strong>{par_rate*100:.4f}%</strong> vs your fixed rate of <strong>{fixed_rate*100:.3f}%</strong>.
                    You are paying <strong>{rate_diff_bp:+.1f} basis points</strong> {'above' if rate_diff_bp > 0 else 'below'} fair value.
                </p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Risk Metrics:</strong>
                <p style="margin: 5px 0;">
                    <strong>DV01:</strong> If interest rates rise by 1 basis point, your swap value will {'decrease' if npv > 0 else 'increase'} by approximately ${result.dv01:,.0f}.<br>
                    <strong>Duration:</strong> Your swap behaves like a bond with {result.duration:.2f} years to maturity in terms of interest rate sensitivity.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Market context
        if 'curve_data' in market_data:
            EducationalSwapDisplayManager._display_yield_curve_context(market_data['curve_data'], tenor_years, fixed_rate)
    
    @staticmethod
    def _display_yield_curve_context(curve_data: Dict[float, float], tenor: float, fixed_rate: float):
        """Display yield curve context"""
        
        tenors = sorted(curve_data.keys())
        rates = [curve_data[t] * 100 for t in tenors]
        
        fig = go.Figure()
        
        # Add yield curve
        fig.add_trace(go.Scatter(
            x=tenors,
            y=rates,
            mode='lines+markers',
            name='Market Yield Curve',
            line=dict(color='#007bff', width=3),
            marker=dict(size=8, color='#007bff')
        ))
        
        # Highlight swap tenor and rate
        if tenor in curve_data:
            market_rate_at_tenor = curve_data[tenor] * 100
            fig.add_trace(go.Scatter(
                x=[tenor],
                y=[market_rate_at_tenor],
                mode='markers',
                name='Market Rate at Tenor',
                marker=dict(size=15, color='#28a745', symbol='circle')
            ))
        
        # Add user's fixed rate
        fig.add_trace(go.Scatter(
            x=[tenor],
            y=[fixed_rate * 100],
            mode='markers',
            name='Your Fixed Rate',
            marker=dict(size=15, color='#dc3545', symbol='x')
        ))
        
        fig.update_layout(
            title='Your Swap Rate vs Market Yield Curve',
            xaxis_title='Maturity (Years)',
            yaxis_title='Rate (%)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Curve analysis
        if len(rates) >= 2:
            curve_slope = rates[-1] - rates[0]
            curve_shape = "Normal (upward sloping)" if curve_slope > 0 else \
                         "Inverted (downward sloping)" if curve_slope < -0.1 else "Flat"
            
            st.info(f"📈 **Yield Curve Analysis**: {curve_shape} | "
                   f"Slope: {curve_slope:.0f}bp | "
                   f"Your rate vs {tenor}Y market: {(fixed_rate - curve_data.get(tenor, fixed_rate)) * 10000:+.1f}bp")
    
    @staticmethod
    def display_enhanced_currency_swap_results(result, base_currency: str, quote_currency: str, 
                                             fx_spot: float, market_data: Dict):
        """Display comprehensive currency swap results with educational context"""
        
        npv_domestic = result.npv_domestic
        fx_delta = result.fx_delta
        
        # Determine favorability
        color = "#28a745" if npv_domestic > 0 else "#dc3545"
        status = "✅ Favorable" if npv_domestic > 0 else "❌ Unfavorable"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 25px; border-radius: 15px; border-left: 5px solid #17a2b8; margin-bottom: 20px;">
            <h3 style="color: #17a2b8; margin-bottom: 20px;">💱 Cross-Currency Swap Results</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">💰 NPV ({base_currency})</h5>
                    <div style="font-size: 2em; font-weight: bold; color: {color};">{base_currency} {npv_domestic:,.0f}</div>
                    <div style="color: {color}; font-weight: bold;">{status}</div>
                    <small style="color: #6c757d;">Net present value in base currency</small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">💱 Current FX Rate</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #007bff;">{fx_spot:.4f}</div>
                    <div style="color: #6c757d;">{quote_currency}{base_currency}</div>
                    <small style="color: #6c757d;">
                        Change: {market_data.get('fx_change', 0):+.2f}%
                    </small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">⚡ FX Delta</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #ffc107;">{base_currency} {fx_delta:,.0f}</div>
                    <div style="color: #6c757d;">Per 1% FX move</div>
                    <small style="color: #6c757d;">Currency exposure sensitivity</small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">📊 Cross Gamma</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #28a745;">{base_currency} {result.cross_gamma:,.0f}</div>
                    <div style="color: #6c757d;">Second-order risk</div>
                    <small style="color: #6c757d;">FX convexity measure</small>
                </div>
                
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk breakdown
        st.markdown(f"""
        <div style="background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 4px solid #ffc107; margin-bottom: 20px;">
            <h4 style="color: #856404;">⚡ Risk Breakdown</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                <div>
                    <h5 style="color: #856404;">Interest Rate Risk</h5>
                    <p><strong>{base_currency} DV01:</strong> {base_currency} {result.domestic_dv01:,.0f}</p>
                    <p><strong>{quote_currency} DV01:</strong> {base_currency} {result.foreign_dv01:,.0f}</p>
                    <small>Sensitivity to 1bp rate change in each currency</small>
                </div>
                
                <div>
                    <h5 style="color: #856404;">Currency Risk</h5>
                    <p><strong>FX Delta:</strong> {base_currency} {fx_delta:,.0f}</p>
                    <p><strong>Volatility:</strong> {market_data.get('fx_volatility', 10.0):.1f}%</p>
                    <small>FX exposure and market volatility</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Educational explanation
        st.markdown(f"""
        <div style="background: #d1ecf1; padding: 20px; border-radius: 10px; border-left: 4px solid #17a2b8; margin-bottom: 20px;">
            <h4 style="color: #0c5460;">📚 Understanding Currency Swap Risk</h4>
            
            <div style="margin: 15px 0;">
                <strong>NPV Interpretation:</strong>
                <p style="margin: 5px 0;">
                    Your currency swap has an NPV of <strong>{base_currency} {npv_domestic:,.0f}</strong>. 
                    {'This represents the present value advantage of receiving ' + quote_currency + ' payments vs paying ' + base_currency + ' payments.' if npv_domestic > 0 else 'This represents the present value cost of the swap structure.'}
                </p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>FX Risk Management:</strong>
                <p style="margin: 5px 0;">
                    <strong>FX Delta:</strong> If {quote_currency} strengthens by 1% vs {base_currency}, your swap value will {'increase' if fx_delta > 0 else 'decrease'} by approximately {base_currency} {abs(fx_delta * 0.01):,.0f}.<br>
                    <strong>Current Volatility:</strong> {market_data.get('fx_volatility', 10.0):.1f}% indicates {'high' if market_data.get('fx_volatility', 10.0) > 15 else 'moderate' if market_data.get('fx_volatility', 10.0) > 8 else 'low'} FX risk.
                </p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Dual Currency Exposure:</strong>
                <p style="margin: 5px 0;">
                    You have interest rate risk in both currencies. Rate changes in {base_currency} affect your swap by {base_currency} {result.domestic_dv01:,.0f} per basis point, 
                    while {quote_currency} rate changes affect it by {base_currency} {result.foreign_dv01:,.0f} per basis point.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_enhanced_equity_swap_results(result, reference_asset: str, notional: float, 
                                           swap_direction: str, market_data: Dict):
        """Display comprehensive equity swap results with educational context"""
        
        npv = result.npv
        equity_price = market_data.get('price', 100.0)
        volatility = market_data.get('volatility', 25.0)
        dividend_yield = market_data.get('dividend_yield', 2.0)
        
        color = "#28a745" if npv > 0 else "#dc3545"
        status = "✅ Favorable" if npv > 0 else "❌ Unfavorable"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 25px; border-radius: 15px; border-left: 5px solid #28a745; margin-bottom: 20px;">
            <h3 style="color: #28a745; margin-bottom: 20px;">📈 Equity Swap Results</h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">💰 NPV</h5>
                    <div style="font-size: 2em; font-weight: bold; color: {color};">${npv:,.0f}</div>
                    <div style="color: {color}; font-weight: bold;">{status}</div>
                    <small style="color: #6c757d;">Net present value of swap</small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">📊 Current Price</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #007bff;">${equity_price:.2f}</div>
                    <div style="color: #6c757d;">{reference_asset}</div>
                    <small style="color: #6c757d;">
                        Change: {market_data.get('change_pct', 0):+.2f}%
                    </small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">⚡ Volatility</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #ffc107;">{volatility:.1f}%</div>
                    <div style="color: #6c757d;">Annualized</div>
                    <small style="color: #6c757d;">
                        Risk level: {'High' if volatility > 30 else 'Medium' if volatility > 20 else 'Low'}
                    </small>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h5 style="color: #6c757d; margin-bottom: 10px;">💸 Dividend Yield</h5>
                    <div style="font-size: 2em; font-weight: bold; color: #17a2b8;">{dividend_yield:.2f}%</div>
                    <div style="color: #6c757d;">Annual yield</div>
                    <small style="color: #6c757d;">Income component</small>
                </div>
                
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Greeks and risk metrics
        if result.greeks:
            st.markdown(f"""
            <div style="background: #e7f3ff; padding: 20px; border-radius: 10px; border-left: 4px solid #007bff; margin-bottom: 20px;">
                <h4 style="color: #004085;">🔬 Risk Analytics (Greeks)</h4>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <div style="text-align: center;">
                        <h5 style="color: #004085;">Delta</h5>
                        <div style="font-size: 1.5em; font-weight: bold;">{result.greeks.get('equity_delta', 0):,.0f}</div>
                        <small>Shares equivalent exposure</small>
                    </div>
                    
                    <div style="text-align: center;">
                        <h5 style="color: #004085;">Vega</h5>
                        <div style="font-size: 1.5em; font-weight: bold;">${result.greeks.get('equity_vega', 0):,.0f}</div>
                        <small>Volatility sensitivity</small>
                    </div>
                    
                    <div style="text-align: center;">
                        <h5 style="color: #004085;">Beta</h5>
                        <div style="font-size: 1.5em; font-weight: bold;">{market_data.get('beta', 1.0):.2f}</div>
                        <small>Market correlation</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Educational explanation
        st.markdown(f"""
        <div style="background: #d1ecf1; padding: 20px; border-radius: 10px; border-left: 4px solid #17a2b8; margin-bottom: 20px;">
            <h4 style="color: #0c5460;">📚 Understanding Equity Swap Exposure</h4>
            
            <div style="margin: 15px 0;">
                <strong>Position Analysis:</strong>
                <p style="margin: 5px 0;">
                    You are <strong>{swap_direction.lower()}</strong> on {reference_asset}. 
                    {'You benefit if the equity outperforms the fixed rate.' if 'receive equity' in swap_direction.lower() else 'You benefit if the fixed rate outperforms the equity.'}
                </p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Risk Factors:</strong>
                <p style="margin: 5px 0;">
                    <strong>Price Risk:</strong> With {volatility:.1f}% volatility, daily moves could be ±{volatility/np.sqrt(252):.1f}%.<br>
                    <strong>Dividend Risk:</strong> Current yield of {dividend_yield:.2f}% affects total return calculations.<br>
                    <strong>Market Risk:</strong> Beta of {market_data.get('beta', 1.0):.2f} means the asset is {'more volatile than' if market_data.get('beta', 1.0) > 1.0 else 'less volatile than' if market_data.get('beta', 1.0) < 1.0 else 'as volatile as'} the overall market.
                </p>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Synthetic vs Physical:</strong>
                <p style="margin: 5px 0;">
                    This swap gives you synthetic exposure equivalent to owning {abs(result.greeks.get('equity_delta', notional/equity_price)):,.0f} shares without:
                    • Physical settlement • Voting rights • Direct dividend payments • Full regulatory capital requirements
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_comprehensive_risk_analytics(result, swap_type: str, scenario_analysis: pd.DataFrame = None):
        """Display comprehensive risk analytics with scenario analysis"""
        
        st.markdown('<div style="color: #495057; font-size: 1.5em; font-weight: bold; margin: 25px 0 15px 0;">⚡ Advanced Risk Analytics</div>', unsafe_allow_html=True)
        
        # Scenario analysis
        if scenario_analysis is not None and not scenario_analysis.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # P&L scenario chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=scenario_analysis['Rate_Shock_bp'],
                    y=scenario_analysis['PnL'],
                    marker_color=['red' if x < 0 else 'green' for x in scenario_analysis['PnL']],
                    name='P&L by Scenario'
                ))
                
                fig.update_layout(
                    title='P&L Sensitivity to Rate Shocks',
                    xaxis_title='Rate Shock (bp)',
                    yaxis_title='P&L ($)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scenario table
                st.markdown("**Scenario Analysis**")
                scenario_display = scenario_analysis.copy()
                scenario_display['New_NPV'] = scenario_display['New_NPV'].apply(lambda x: f"${x:,.0f}")
                scenario_display['PnL'] = scenario_display['PnL'].apply(lambda x: f"${x:,.0f}")
                scenario_display['PnL_Percent'] = scenario_display['PnL_Percent'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(scenario_display, use_container_width=True, hide_index=True)
        
        # Risk metrics explanation
        st.markdown(f"""
        <div style="background: #f8d7da; padding: 20px; border-radius: 10px; border-left: 4px solid #dc3545; margin: 20px 0;">
            <h4 style="color: #721c24;">⚠️ Risk Management Guidelines</h4>
            
            <div style="margin: 15px 0;">
                <strong>Daily Monitoring:</strong>
                <ul>
                    <li>Track NPV changes and market moves</li>
                    <li>Monitor key risk metrics (DV01, Delta, Vega)</li>
                    <li>Check correlation breakdowns in stress scenarios</li>
                </ul>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Hedging Considerations:</strong>
                <ul>
                    <li>Consider delta hedging for large positions</li>
                    <li>Monitor cross-asset correlations</li>
                    <li>Set appropriate stop-loss levels</li>
                </ul>
            </div>
            
            <div style="margin: 15px 0;">
                <strong>Regulatory & Accounting:</strong>
                <ul>
                    <li>Ensure proper hedge accounting designation</li>
                    <li>Meet collateral and margin requirements</li>
                    <li>Document business purpose and risk management</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_market_intelligence_dashboard():
        """Display comprehensive market intelligence dashboard"""
        
        st.markdown('<div style="color: #495057; font-size: 1.8em; font-weight: bold; margin: 25px 0 15px 0;">📊 Live Market Intelligence</div>', unsafe_allow_html=True)
        
        # Market overview cards
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 15px; margin-bottom: 25px; color: white;">
            <h3 style="margin: 0 0 15px 0; color: white;">🌍 Global Markets Overview</h3>
            <p style="font-size: 1.1em; margin-bottom: 20px; color: #f8f9fa;">
                Real-time market data and analysis for informed swap pricing decisions
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: #ffd700; margin-bottom: 10px;">🏦 Interest Rates</h4>
                    <p style="margin: 5px 0; font-size: 0.9em;">Live yield curves</p>
                    <p style="margin: 5px 0; font-size: 0.9em;">Central bank policy</p>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: #98fb98; margin-bottom: 10px;">💱 FX Markets</h4>
                    <p style="margin: 5px 0; font-size: 0.9em;">Major currency pairs</p>
                    <p style="margin: 5px 0; font-size: 0.9em;">Volatility surfaces</p>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: #87ceeb; margin-bottom: 10px;">📈 Equity Markets</h4>
                    <p style="margin: 5px 0; font-size: 0.9em;">Index performance</p>
                    <p style="margin: 5px 0; font-size: 0.9em;">Sector rotation</p>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; text-align: center;">
                    <h4 style="color: #dda0dd; margin-bottom: 10px;">⚡ Volatility</h4>
                    <p style="margin: 5px 0; font-size: 0.9em;">VIX and term structure</p>
                    <p style="margin: 5px 0; font-size: 0.9em;">Cross-asset vol</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_educational_methodology():
        """Display educational content about swap pricing methodology"""
        
        st.markdown('<div style="color: #495057; font-size: 1.8em; font-weight: bold; margin: 25px 0 15px 0;">📚 Pricing Methodology & Education</div>', unsafe_allow_html=True)
        
        # Tabbed educational content
        method_tab1, method_tab2, method_tab3, method_tab4 = st.tabs([
            "🔢 Pricing Models", "📊 Risk Metrics", "🎯 Market Data", "⚠️ Limitations"
        ])
        
        with method_tab1:
            st.markdown("""
            <div style="background: #e7f3ff; padding: 25px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #004085;">🔢 Swap Pricing Models Explained</h3>
                
                <div style="margin: 20px 0;">
                    <h4 style="color: #004085;">1. Discounted Cash Flow (DCF)</h4>
                    <p><strong>Method:</strong> Calculate present value of all future cash flows</p>
                    <p><strong>Formula:</strong> NPV = Σ(CF_i × DF_i) where CF = cash flow, DF = discount factor</p>
                    <p><strong>Use Case:</strong> Standard pricing for most swaps</p>
                    <p><strong>Advantages:</strong> Transparent, easy to understand, industry standard</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <h4 style="color: #004085;">2. Monte Carlo Simulation</h4>
                    <p><strong>Method:</strong> Simulate thousands of interest rate paths</p>
                    <p><strong>Process:</strong> Generate random paths → Calculate payoffs → Average results</p>
                    <p><strong>Use Case:</strong> Complex swaps with path-dependent features</p>
                    <p><strong>Advantages:</strong> Handles complex payoffs, incorporates volatility</p>
                </div>
                
                <div style="margin: 20px 0;">
                    <h4 style="color: #004085;">3. Hull-White Model</h4>
                    <p><strong>Method:</strong> One-factor interest rate model with mean reversion</p>
                    <p><strong>SDE:</strong> dr = α(θ(t) - r)dt + σdW</p>
                    <p><strong>Use Case:</strong> When mean reversion is important</p>
                    <p><strong>Advantages:</strong> Fits current term structure, analytical solutions available</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with method_tab2:
            st.markdown("""
            <div style="background: #fff3cd; padding: 25px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #856404;">📊 Risk Metrics Explained</h3>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4 style="color: #856404;">First-Order Greeks</h4>
                        <p><strong>DV01 (Dollar Value 01):</strong><br>
                        Change in swap value for 1bp rate move<br>
                        <em>Formula: DV01 = |dNPV/dr| × 0.0001</em></p>
                        
                        <p><strong>Delta (Equity):</strong><br>
                        Change in swap value for 1% equity move<br>
                        <em>Equivalent shares exposure</em></p>
                        
                        <p><strong>FX Delta:</strong><br>
                        Change in swap value for 1% FX move<br>
                        <em>Currency exposure amount</em></p>
                    </div>
                    
                    <div>
                        <h4 style="color: #856404;">Second-Order Greeks</h4>
                        <p><strong>Convexity:</strong><br>
                        Curvature of price-yield relationship<br>
                        <em>Measures gamma risk</em></p>
                        
                        <p><strong>Vega (Equity):</strong><br>
                        Sensitivity to volatility changes<br>
                        <em>Important for equity swaps</em></p>
                        
                        <p><strong>Cross Gamma:</strong><br>
                        Cross-derivative sensitivities<br>
                        <em>Currency-interest rate interactions</em></p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with method_tab3:
            st.markdown("""
            <div style="background: #d1ecf1; padding: 25px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #0c5460;">🎯 Market Data Sources & Quality</h3>

                <div style="margin: 20px 0;">
                    <h4 style="color: #0c5460;">Data Hierarchy</h4>
                    <ol>
                        <li><strong>Primary Sources:</strong> FRED (Federal Reserve), ECB, BoE official rates</li>
                        <li><strong>Financial Data:</strong> Yahoo Finance for real-time quotes</li>
                        <li><strong>Market Vendors:</strong> Bloomberg, Reuters, ICE Data</li>
                        <li><strong>Fallback:</strong> Interpolated or simulated data when live sources fail</li>
                    </ol>
                </div>
                <div style="margin: 20px 0;">
                    <h4 style="color: #0c5460;">Data Quality</h4>
                    <p>Cross-check sources and validate data before use.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with method_tab4:
            st.markdown("""
            <div style="background: #f8d7da; padding: 25px; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #721c24;">⚠️ Important Disclaimers and Limitations</h3>

                <div style="margin: 20px 0;">
                    <h4 style="color: #721c24;">Educational Purpose Only</h4>
                    <p>This platform is for educational and research use only and should not guide actual trading decisions.</p>
                </div>

                <div style="margin: 20px 0;">
                    <h4 style="color: #721c24;">Model Limitations</h4>
                    <ul>
                        <li>Simplified assumptions about market behavior</li>
                        <li>Historical data does not guarantee future results</li>
                        <li>Real conditions may differ from model assumptions</li>
                        <li>Liquidity constraints often not fully captured</li>
                    </ul>
                </div>

                <div style="margin: 20px 0;">
                    <h4 style="color: #721c24;">Risk Warnings</h4>
                    <ul>
                        <li>Derivative instruments carry significant risk of loss</li>
                        <li>Consult qualified professionals for real trading decisions</li>
                        <li>Ensure compliance with all applicable regulations</li>
                        <li>Counterparty credit risk not fully modeled</li>
                    </ul>
                </div>

                <div style="margin: 20px 0;">
                    <h4 style="color: #721c24;">Data Disclaimers</h4>
                    <ul>
                        <li>Data may be delayed, estimated, or simulated</li>
                        <li>No warranty of accuracy or completeness</li>
                        <li>Third-party data subject to provider terms</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)


class PortfolioAnalyticsDisplay:
    """Display a high level summary of portfolio analytics.

    If this class cannot be imported, you may be running a reduced
    version of the project that omits advanced portfolio tools.
    """

    @staticmethod
    def display_portfolio_summary(analytics: Dict[str, float]):
        """Show an overview of the portfolio in Streamlit."""
        if not analytics:
            st.info("No portfolio analytics available.")
            return

        st.markdown(
            "<div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>",
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Notional", f"${analytics.get('total_notional', 0):,.0f}")
        col2.metric("Portfolio NPV", f"${analytics.get('portfolio_npv', 0):,.0f}")
        col3.metric("Number of Swaps", analytics.get('num_swaps', 0))

        col4, col5, col6 = st.columns(3)
        col4.metric("DV01", f"${analytics.get('portfolio_dv01', 0):,.0f}")
        col5.metric("Vega", f"${analytics.get('portfolio_vega', 0):,.0f}")
        col6.metric("Concentration", f"{analytics.get('concentration_risk', 0):.1%}")

        st.markdown("</div>", unsafe_allow_html=True)
