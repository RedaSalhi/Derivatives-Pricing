# pricing/models/display_utils.py
# Clean Display Utilities for Swaps

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from .market_data import market_data_manager


class SwapDisplayManager:
    """Centralized display management for swaps"""
    
    @staticmethod
    def display_irs_results(result, notional, fixed_rate, tenor_years, model, market_rate=None):
        """Display IRS pricing results"""
        
        npv = result.npv
        par_rate = result.par_rate
        
        # Market comparison
        if market_rate:
            rate_diff_bp = (fixed_rate - market_rate) * 10000
        else:
            rate_diff_bp = 0
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>🎯 Swap Pricing Results ({model})</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #007bff;">
                    <h5 style="color: #007bff; margin-bottom: 10px;">💰 Valuation</h5>
                    <p><strong>NPV:</strong> <span style="color: {'green' if npv > 0 else 'red'}; font-size: 1.2em;">${npv:,.0f}</span></p>
                    <p><strong>Par Rate:</strong> {par_rate*100:.4f}%</p>
                    <p><strong>DV01:</strong> ${result.dv01:,.0f}</p>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #28a745;">
                    <h5 style="color: #28a745; margin-bottom: 10px;">📊 Market</h5>
                    <p><strong>Your Rate:</strong> {fixed_rate*100:.3f}%</p>
                    <p><strong>Market Rate:</strong> {market_rate:.3f}% {f'({rate_diff_bp:+.1f}bp)' if market_rate else ''}</p>
                    <p><strong>Duration:</strong> {result.duration:.2f} years</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_currency_swap_results(result, base_currency, quote_currency, fx_spot, tenor_years):
        """Display currency swap results"""
        
        npv_domestic = result.npv_domestic
        fx_delta = result.fx_delta
        
        color = "#2E8B57" if npv_domestic > 0 else "#DC143C"
        status = "✅ Favorable" if npv_domestic > 0 else "❌ Unfavorable"
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>💱 Currency Swap Results</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                    <td style="padding: 12px; font-weight: bold;">Value</td>
                    <td style="padding: 12px; font-weight: bold;">Description</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">NPV ({base_currency})</td>
                    <td style="padding: 10px; font-family: monospace; color: {color}; font-weight: bold; font-size: 1.2em;">{base_currency} {npv_domestic:,.0f}</td>
                    <td style="padding: 10px; font-style: italic;">Net Present Value</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Status</td>
                    <td style="padding: 10px; font-weight: bold; color: {color};">{status}</td>
                    <td style="padding: 10px; font-style: italic;">From {base_currency} perspective</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">FX Delta</td>
                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">{base_currency} {fx_delta:,.0f}</td>
                    <td style="padding: 10px; font-style: italic;">Sensitivity to 1% FX move</td>
                </tr>
                <tr>
                    <td style="padding: 10px; font-weight: bold;">FX Rate</td>
                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">{fx_spot:.4f}</td>
                    <td style="padding: 10px; font-style: italic;">Current market rate</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_equity_swap_results(result, reference_asset, notional, swap_direction, tenor_years):
        """Display equity swap results"""
        
        npv = result.npv
        color = "#2E8B57" if npv > 0 else "#DC143C"
        status = "✅ Favorable" if npv > 0 else "❌ Unfavorable"
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>📈 Equity Swap Results</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                    <td style="padding: 12px; font-weight: bold;">Metric</td>
                    <td style="padding: 12px; font-weight: bold;">Value</td>
                    <td style="padding: 12px; font-weight: bold;">Description</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">NPV</td>
                    <td style="padding: 10px; font-family: monospace; color: {color}; font-weight: bold; font-size: 1.2em;">${npv:,.0f}</td>
                    <td style="padding: 10px; font-style: italic;">Net Present Value</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Status</td>
                    <td style="padding: 10px; font-weight: bold; color: {color};">{status}</td>
                    <td style="padding: 10px; font-style: italic;">{swap_direction}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Equity PV</td>
                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">${result.pv_floating:,.0f}</td>
                    <td style="padding: 10px; font-style: italic;">Equity leg value</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px; font-weight: bold;">Fixed PV</td>
                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">${result.pv_fixed:,.0f}</td>
                    <td style="padding: 10px; font-style: italic;">Fixed leg value</td>
                </tr>
                <tr>
                    <td style="padding: 10px; font-weight: bold;">Reference</td>
                    <td style="padding: 10px; font-family: monospace; font-weight: bold;">{reference_asset}</td>
                    <td style="padding: 10px; font-style: italic;">Underlying asset</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_risk_analytics(result, swap_type="IRS", **kwargs):
        """Display risk analytics"""
        
        st.markdown('<div class="sub-header">⚡ Risk Analytics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        if swap_type == "IRS":
            with col1:
                st.markdown(f"""
                <div class="greeks-delta">
                    <h4>📊 Interest Rate Risk</h4>
                    <p><strong>DV01:</strong> ${result.dv01:,.0f}</p>
                    <p><strong>Duration:</strong> {result.duration:.2f} years</p>
                    <p><strong>Convexity:</strong> {(result.duration ** 2 * 0.5):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if hasattr(result, 'greeks') and result.greeks:
                    model_info = result.greeks.get('model', 'DCF')
                    paths = result.greeks.get('monte_carlo_paths', 'N/A')
                    
                    st.markdown(f"""
                    <div class="greeks-gamma">
                        <h4>🔧 Model Details</h4>
                        <p><strong>Model:</strong> {model_info}</p>
                        <p><strong>Paths:</strong> {paths}</p>
                        <p><strong>Par Rate:</strong> {result.par_rate*100:.4f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif swap_type == "Currency":
            base_currency = kwargs.get('base_currency', 'USD')
            quote_currency = kwargs.get('quote_currency', 'EUR')
            
            with col1:
                st.markdown(f"""
                <div class="greeks-delta">
                    <h4>💱 FX Risk</h4>
                    <p><strong>FX Delta:</strong> {base_currency} {result.fx_delta:,.0f}</p>
                    <p><strong>Cross Gamma:</strong> {base_currency} {result.cross_gamma:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="greeks-gamma">
                    <h4>📊 Rate Risk</h4>
                    <p><strong>{base_currency} DV01:</strong> {base_currency} {result.domestic_dv01:,.0f}</p>
                    <p><strong>{quote_currency} DV01:</strong> {base_currency} {result.foreign_dv01:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif swap_type == "Equity":
            equity_data = kwargs.get('equity_data', {})
            
            with col1:
                st.markdown(f"""
                <div class="greeks-delta">
                    <h4>📊 Market Risk</h4>
                    <p><strong>Equity Delta:</strong> {result.greeks.get('equity_delta', 0):,.0f} shares</p>
                    <p><strong>Equity Vega:</strong> ${result.greeks.get('equity_vega', 0):,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="greeks-gamma">
                    <h4>💰 Live Data</h4>
                    <p><strong>Price:</strong> ${equity_data.get('price', 0):.2f}</p>
                    <p><strong>Volatility:</strong> {equity_data.get('volatility', 0):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def display_market_comparison(result, market_rate, fixed_rate):
        """Display market comparison chart"""
        
        st.markdown('<div class="sub-header">📈 Market Analysis</div>', unsafe_allow_html=True)
        
        metrics = ['Your Rate', 'Market Rate', 'Par Rate']
        values = [fixed_rate * 100, market_rate, result.par_rate * 100]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics,
                y=values,
                marker_color=['blue', 'green', 'orange'],
                text=[f'{v:.3f}%' for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Rate Comparison Analysis',
            yaxis_title='Rate (%)',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def display_live_yield_curve():
        """Display live yield curve"""
        
        st.markdown('<div class="sub-header">📈 Live USD Yield Curve</div>', unsafe_allow_html=True)
        
        treasury_data = market_data_manager.get_treasury_curve_data()
        
        if treasury_data is not None:
            try:
                tenors = [0.25, 2, 10, 30]
                treasury_symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
                yields = [treasury_data.get(symbol, 4.0) for symbol in treasury_symbols]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tenors,
                    y=yields,
                    mode='lines+markers',
                    name='US Treasury Yield Curve',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title='Live US Treasury Yield Curve',
                    xaxis_title='Maturity (Years)',
                    yaxis_title='Yield (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Curve analysis
                curve_slope = yields[-1] - yields[0]
                st.info(f"📊 Curve slope (30Y-3M): {curve_slope:.0f} basis points")
                
            except Exception as e:
                st.info("Error displaying yield curve")
        else:
            st.info("Live yield curve data temporarily unavailable")
    
    @staticmethod
    def display_fx_market_overview():
        """Display FX market overview"""
        
        st.markdown('<div class="sub-header">🌍 FX Market Overview</div>', unsafe_allow_html=True)
        
        indices_data = market_data_manager.get_major_indices_data()
        
        if indices_data is not None:
            try:
                major_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("EUR/USD", f"{1.0850:.4f}")  # Would use real data
                    st.metric("USD/JPY", f"{150.0:.2f}")
                
                with col2:
                    st.metric("GBP/USD", f"{1.2650:.4f}")
                    st.metric("USD/CHF", f"{0.8890:.4f}")
                    
            except Exception as e:
                st.info("FX data temporarily unavailable")
        else:
            st.info("Live FX data temporarily unavailable")
    
    @staticmethod
    def display_equity_market_overview():
        """Display equity market overview"""
        
        st.markdown('<div class="sub-header">📈 Equity Market Overview</div>', unsafe_allow_html=True)
        
        indices_data = market_data_manager.get_major_indices_data()
        
        if indices_data is not None:
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("S&P 500", f"{indices_data.get('^GSPC', 4500):.0f}")
                    st.metric("NASDAQ", f"{indices_data.get('^IXIC', 14000):.0f}")
                
                with col2:
                    st.metric("Dow Jones", f"{indices_data.get('^DJI', 35000):.0f}")
                    st.metric("VIX", f"{indices_data.get('^VIX', 20):.1f}")
                    
            except Exception as e:
                st.info("Equity data temporarily unavailable")
        else:
            st.info("Live equity data temporarily unavailable")
    
    @staticmethod
    def display_portfolio_analytics(analytics):
        """Display portfolio analytics"""
        
        st.markdown('<div class="sub-header">📊 Portfolio Analytics</div>', unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Notional", f"${analytics['total_notional']:,.0f}")
        
        with col2:
            st.metric("Portfolio NPV", f"${analytics['portfolio_npv']:,.0f}")
        
        with col3:
            st.metric("Portfolio DV01", f"${analytics['portfolio_dv01']:,.0f}")
        
        with col4:
            st.metric("Number of Swaps", f"{analytics['num_swaps']}")
        
        # Risk breakdown
        if analytics['total_notional'] > 0:
            st.markdown('<div class="sub-header">🎯 Risk Breakdown</div>', unsafe_allow_html=True)
            
            risk_data = pd.DataFrame({
                'Asset Class': ['Interest Rate', 'FX', 'Equity'],
                'Exposure': [analytics['ir_exposure'], analytics['fx_exposure'], analytics['equity_exposure']]
            })
            
            # Only show chart if there's meaningful data
            if risk_data['Exposure'].sum() > 0:
                fig = px.pie(
                    risk_data,
                    values='Exposure',
                    names='Asset Class',
                    title='Portfolio Exposure by Asset Class'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No portfolio exposure data to display")
        else:
            st.info("No portfolio data to analyze")


class MarketIntelligenceDisplay:
    """Display utilities for market intelligence"""
    
    @staticmethod
    def display_global_rates_dashboard():
        """Display global rates dashboard"""
        
        st.markdown('<div class="sub-header">🌐 Global Interest Rates</div>', unsafe_allow_html=True)
        
        treasury_data = market_data_manager.get_treasury_curve_data()
        
        if treasury_data is not None:
            try:
                tenors = [0.25, 2, 10, 30]
                treasury_symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=tenors,
                    y=[treasury_data.get(symbol, 4.0) for symbol in treasury_symbols],
                    mode='lines+markers',
                    name='US Treasury Yield Curve',
                    line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title='US Treasury Yield Curve',
                    xaxis_title='Maturity (Years)',
                    yaxis_title='Yield (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.info("Live rates data temporarily unavailable")
        
        # Display static rates table
        rates_data = pd.DataFrame({
            'Country': ['United States', 'Germany', 'United Kingdom', 'Japan', 'Canada'],
            '2Y': ['4.20%', '2.85%', '4.45%', '0.15%', '4.10%'],
            '5Y': ['4.15%', '2.65%', '4.25%', '0.35%', '3.95%'],
            '10Y': ['4.10%', '2.45%', '4.15%', '0.75%', '3.85%']
        })
        
        st.dataframe(rates_data, use_container_width=True, hide_index=True)
    
    @staticmethod
    def display_fx_markets_dashboard():
        """Display FX markets dashboard"""
        
        st.markdown('<div class="sub-header">💱 FX Markets Dashboard</div>', unsafe_allow_html=True)
        
        pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
        correlation_data = market_data_manager.get_fx_correlation_data(pairs)
        
        if correlation_data is not None:
            try:
                fig = px.imshow(
                    correlation_data.values,
                    labels=dict(x="Currency Pair", y="Currency Pair", color="Correlation"),
                    x=[pair.replace('=X', '') for pair in correlation_data.columns],
                    y=[pair.replace('=X', '') for pair in correlation_data.index],
                    color_continuous_scale='RdBu_r',
                    title='FX Correlation Matrix (30D)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.info("FX correlation data temporarily unavailable")
        else:
            st.info("FX correlation data temporarily unavailable")
    
    @staticmethod
    def display_volatility_dashboard():
        """Display volatility dashboard"""
        
        st.markdown('<div class="sub-header">📊 Volatility Analysis</div>', unsafe_allow_html=True)
        
        vix_data = market_data_manager.get_vix_data()
        
        if vix_data is not None:
            try:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=vix_data.index,
                    y=vix_data['Close'],
                    mode='lines',
                    name='VIX',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title='VIX - Market Volatility Index (90 Days)',
                    xaxis_title='Date',
                    yaxis_title='VIX Level',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # VIX level analysis
                current_vix = vix_data['Close'].iloc[-1]
                if current_vix > 25:
                    st.warning(f"⚠️ High volatility: VIX at {current_vix:.1f}")
                elif current_vix < 15:
                    st.info(f"📉 Low volatility: VIX at {current_vix:.1f}")
                else:
                    st.success(f"✅ Normal volatility: VIX at {current_vix:.1f}")
                    
            except Exception as e:
                st.info("Volatility data temporarily unavailable")
        else:
            st.info("Volatility data temporarily unavailable")
    
    @staticmethod
    def display_research_insights():
        """Display market research and insights"""
        
        st.markdown('<div class="sub-header">📚 Market Research & Insights</div>', unsafe_allow_html=True)
        
        # Current market themes
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Current Market Themes</h4>
            <ul>
                <li><strong>Central Bank Policy:</strong> Fed pause vs ECB/BoE divergence</li>
                <li><strong>Yield Curves:</strong> Monitoring inversion signals globally</li>
                <li><strong>Credit Markets:</strong> Widening spreads in corporate bonds</li>
                <li><strong>FX Volatility:</strong> Dollar strength pressuring emerging markets</li>
                <li><strong>Equity Valuations:</strong> Growth vs value rotation continues</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Educational content
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
                <h4>📖 Swap Fundamentals</h4>
                <h5>Key Concepts:</h5>
                <ul>
                    <li><strong>Par Rate:</strong> Break-even swap rate</li>
                    <li><strong>DV01:</strong> Dollar value per basis point</li>
                    <li><strong>Cross-Currency Basis:</strong> FX swap premium</li>
                    <li><strong>Equity Risk Premium:</strong> Expected excess return</li>
                </ul>
                
                <h5>Risk Management:</h5>
                <ul>
                    <li><strong>Delta Hedging:</strong> Neutralize price sensitivity</li>
                    <li><strong>Duration Matching:</strong> Asset-liability alignment</li>
                    <li><strong>Stress Testing:</strong> Scenario analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Risk Considerations</h4>
                <h5>Market Risks:</h5>
                <ul>
                    <li><strong>Interest Rate Risk:</strong> Duration exposure</li>
                    <li><strong>FX Risk:</strong> Currency fluctuations</li>
                    <li><strong>Credit Risk:</strong> Counterparty exposure</li>
                    <li><strong>Liquidity Risk:</strong> Market depth concerns</li>
                </ul>
                
                <h5>Model Risks:</h5>
                <ul>
                    <li><strong>Parameter Risk:</strong> Volatility estimates</li>
                    <li><strong>Model Risk:</strong> Pricing assumptions</li>
                    <li><strong>Correlation Risk:</strong> Breakdown scenarios</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Methodology
        with st.expander("📊 Methodology & Data Sources"):
            st.markdown("""
            <div class="info-box">
                <h4>Pricing Methodology</h4>
                
                <h5>Data Sources:</h5>
                <ul>
                    <li><strong>Yahoo Finance:</strong> Real-time market data</li>
                    <li><strong>FRED:</strong> Economic indicators (when available)</li>
                    <li><strong>Live Feeds:</strong> Continuous updates</li>
                </ul>
                
                <h5>Models:</h5>
                <ul>
                    <li><strong>DCF:</strong> Discounted cash flow analysis</li>
                    <li><strong>Monte Carlo:</strong> Stochastic simulation</li>
                    <li><strong>Hull-White:</strong> Mean-reverting rates</li>
                </ul>
                
                <h5>Risk Metrics:</h5>
                <ul>
                    <li><strong>Greeks:</strong> First/second order sensitivities</li>
                    <li><strong>VaR:</strong> Value-at-Risk calculations</li>
                    <li><strong>Stress Tests:</strong> Scenario analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
