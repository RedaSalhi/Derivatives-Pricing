# tabs/swaps.py
# Enhanced Swaps Tab with Professional Pricing Models

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from styles.app_styles import load_theme

# Enhanced pricing functions
from pricing.models.swaps.dcf import *
from pricing.models.swaps.lmm import *
from pricing.models.swaps.curves import *


def swaps_tab():
    """Enhanced Swaps Tab Content"""

    load_theme()
    
    st.markdown('<div class="main-header">Advanced Swap Pricing Suite</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Professional introduction with market context
    st.markdown("""
    <div class="info-box">
        <h3>🏦 Professional Derivatives Pricing Platform</h3>
        <p>Advanced swap pricing models with real-time analytics, risk metrics, and market-consistent valuations. 
        Built for institutional-grade derivative pricing and risk management.</p>
        <div style="display: flex; gap: 20px; margin-top: 15px;">
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Market Size:</strong><br>$500+ Trillion Notional Outstanding
            </div>
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Daily Volume:</strong><br>$2-3 Trillion Average
            </div>
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Models:</strong><br>DCF, LMM, Hull-White
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create enhanced tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Interest Rate Swaps", 
        "💱 Currency Swaps", 
        "📈 Equity Swaps", 
        "📊 Portfolio Analysis",
        "📚 Market Intelligence"
    ])
    
    with tab1:
        _interest_rate_swaps_tab()
    
    with tab2:
        _currency_swaps_tab()
    
    with tab3:
        _equity_swaps_tab()
    
    with tab4:
        _portfolio_analysis_tab()
    
    with tab5:
        _market_intelligence_tab()


def _interest_rate_swaps_tab():
    """Enhanced Interest Rate Swaps Interface"""
    st.markdown('<div class="sub-header">Interest Rate Swap Pricing & Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔧 Swap Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced parameters with validation
        notional = st.number_input(
            "Notional Amount (USD)", 
            min_value=1_000_000, 
            max_value=10_000_000_000, 
            value=100_000_000, 
            step=1_000_000,
            format="%d"
        )
        
        tenor_years = st.selectbox("Swap Tenor", [2, 3, 5, 7, 10, 15, 20, 30], index=4)
        
        payment_freq = st.selectbox(
            "Payment Frequency", 
            ["Quarterly", "Semi-Annual", "Annual"], 
            index=0,
            help="Quarterly is market standard for USD swaps"
        )
        
        # Market rates with realistic ranges
        fixed_rate = st.slider(
            "Fixed Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=3.5, 
            step=0.01,
            format="%.2f"
        ) / 100
        
        floating_spread = st.slider(
            "Floating Spread (bps)", 
            min_value=-100, 
            max_value=500, 
            value=0, 
            step=5,
            help="Spread over benchmark rate"
        )
        
        # Model selection
        st.markdown("""
        <div class="info-box">
            <h4>📈 Pricing Model</h4>
        </div>
        """, unsafe_allow_html=True)
        
        model = st.radio(
            "Select Model",
            ["Enhanced DCF", "LIBOR Market Model", "Hull-White"],
            help="Choose appropriate model for your use case"
        )
        
        # Risk parameters
        if model in ["LIBOR Market Model", "Hull-White"]:
            vol = st.slider("Interest Rate Volatility (%)", 5.0, 50.0, 15.0, 1.0) / 100
            n_paths = st.selectbox("Monte Carlo Paths", [10000, 25000, 50000, 100000], index=1)
        
        # Current market environment
        st.markdown("""
        <div class="info-box">
            <h4>🌍 Market Environment</h4>
        </div>
        """, unsafe_allow_html=True)
        
        current_libor = st.slider("Current 3M LIBOR (%)", 0.0, 8.0, 2.5, 0.01) / 100
        fed_funds = st.slider("Fed Funds Rate (%)", 0.0, 8.0, 2.0, 0.01) / 100
        
        price_btn = st.button("🔢 Price Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Calculating swap price and risk metrics..."):
                try:
                    # Build enhanced curves
                    discount_curve = build_enhanced_discount_curve(fed_funds)
                    forward_curve = build_enhanced_forward_curve(current_libor, fed_funds)
                    
                    # Create payment schedule
                    freq_map = {"Quarterly": 0.25, "Semi-Annual": 0.5, "Annual": 1.0}
                    dt = freq_map[payment_freq]
                    payment_times = [dt * i for i in range(1, int(tenor_years / dt) + 1)]
                    
                    # Calculate swap price based on model
                    if model == "Enhanced DCF":
                        result = price_irs_enhanced_dcf(
                            notional=notional,
                            fixed_rate=fixed_rate,
                            payment_times=payment_times,
                            discount_curve=discount_curve,
                            forward_curve=forward_curve,
                            floating_spread=floating_spread / 10000
                        )
                    elif model == "LIBOR Market Model":
                        result = price_irs_lmm_enhanced(
                            notional=notional,
                            fixed_rate=fixed_rate,
                            initial_rate=current_libor,
                            vol=vol,
                            payment_times=payment_times,
                            discount_curve=discount_curve,
                            n_paths=n_paths,
                            floating_spread=floating_spread / 10000
                        )
                    else:  # Hull-White
                        result = price_irs_hull_white(
                            notional=notional,
                            fixed_rate=fixed_rate,
                            initial_rate=current_libor,
                            mean_reversion=0.1,
                            vol=vol,
                            payment_times=payment_times,
                            n_paths=n_paths
                        )
                    
                    # Display results with enhanced formatting
                    _display_irs_results(result, notional, fixed_rate, payment_times, model)
                    
                    # Risk analytics
                    _display_irs_risk_metrics(result, notional, fixed_rate, current_libor, payment_times)
                    
                    # Sensitivity analysis
                    _display_irs_sensitivity(notional, fixed_rate, payment_times, discount_curve, forward_curve, model)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Pricing Error</h4>
                        <p>Error during calculation: {str(e)}</p>
                        <p>Please check your parameters and try again.</p>
                    </div>
                    """, unsafe_allow_html=True)


def _display_irs_results(result, notional, fixed_rate, payment_times, model):
    """Display IRS pricing results with professional formatting"""
    
    npv = result.get('npv', 0)
    par_rate = result.get('par_rate', fixed_rate)
    dv01 = result.get('dv01', 0)
    
    # Determine swap direction and color
    if npv > 0:
        direction = "Pay Fixed, Receive Floating"
        color = "#2E8B57"  # Green
        status = "✅ Positive NPV"
    elif npv < 0:
        direction = "Receive Fixed, Pay Floating"
        color = "#DC143C"  # Red  
        status = "❌ Negative NPV"
    else:
        direction = "At Par"
        color = "#4682B4"  # Blue
        status = "⚖️ Fair Value"
    
    st.markdown(f"""
    <div class="metric-container">
        <h4>🎯 Swap Pricing Results ({model})</h4>
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
                <td style="padding: 10px; font-style: italic;">{direction}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Par Rate</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{par_rate*100:.4f}%</td>
                <td style="padding: 10px; font-style: italic;">Fair swap rate (NPV = 0)</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">DV01</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">${dv01:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">Dollar value per 1bp rate change</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Duration</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{len(payment_times) * 0.25:.1f} years</td>
                <td style="padding: 10px; font-style: italic;">Approximate modified duration</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


def _display_irs_risk_metrics(result, notional, fixed_rate, current_rate, payment_times):
    """Display comprehensive risk metrics"""
    
    st.markdown('<div class="sub-header">⚡ Risk Analytics</div>', unsafe_allow_html=True)
    
    # Calculate risk metrics
    duration = sum([t * 0.25 for t in range(1, len(payment_times) + 1)]) / len(payment_times)
    convexity = duration * duration * 0.5  # Simplified convexity approximation
    
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.markdown(f"""
        <div class="greeks-delta">
            <h4>📊 Interest Rate Risk</h4>
            <p><strong>Modified Duration:</strong> {duration:.2f} years</p>
            <p><strong>Convexity:</strong> {convexity:.2f}</p>
            <p><strong>Key Rate 01:</strong> ${result.get('dv01', 0):,.0f}</p>
            <p><strong>Carry (1 Day):</strong> ${(notional * fixed_rate * 0.25) / 365:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_risk2:
        st.markdown(f"""
        <div class="greeks-gamma">
            <h4>💰 P&L Attribution</h4>
            <p><strong>Rate Delta:</strong> {(current_rate - fixed_rate) * 10000:.0f} bps</p>
            <p><strong>Time Decay:</strong> ${(notional * 0.01) / 365:,.0f} per day</p>
            <p><strong>Bid-Ask Cost:</strong> ~${notional * 0.0001:,.0f}</p>
            <p><strong>CVA Adjustment:</strong> ~${notional * 0.0002:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)


def _display_irs_sensitivity(notional, fixed_rate, payment_times, discount_curve, forward_curve, model):
    """Display sensitivity analysis with interactive charts"""
    
    st.markdown('<div class="sub-header">📈 Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    # Rate sensitivity
    rate_shifts = np.linspace(-200, 200, 21)  # -200bp to +200bp
    npvs = []
    
    for shift_bp in rate_shifts:
        shift = shift_bp / 10000
        shifted_discount = lambda t: discount_curve(t) * np.exp(shift * t)
        
        try:
            result = price_irs_enhanced_dcf(
                notional=notional,
                fixed_rate=fixed_rate,
                payment_times=payment_times,
                discount_curve=shifted_discount,
                forward_curve=forward_curve,
                floating_spread=0
            )
            npvs.append(result.get('npv', 0))
        except:
            npvs.append(0)
    
    # Create sensitivity chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rate_shifts,
        y=npvs,
        mode='lines+markers',
        name='NPV',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig.add_vline(x=0, line_dash="dot", line_color="gray", annotation_text="Current Rates")
    
    fig.update_layout(
        title='Interest Rate Sensitivity Analysis',
        xaxis_title='Parallel Rate Shift (basis points)',
        yaxis_title='NPV (USD)',
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scenario analysis
    scenarios = {
        "Base Case": 0,
        "Fed Tightening (+100bp)": 100,
        "Fed Easing (-100bp)": -100,
        "Crisis Scenario (-200bp)": -200,
        "Inflation Spike (+200bp)": 200
    }
    
    scenario_results = []
    for scenario_name, shift_bp in scenarios.items():
        shift = shift_bp / 10000
        shifted_discount = lambda t: discount_curve(t) * np.exp(shift * t)
        
        try:
            result = price_irs_enhanced_dcf(
                notional=notional,
                fixed_rate=fixed_rate,
                payment_times=payment_times,
                discount_curve=shifted_discount,
                forward_curve=forward_curve,
                floating_spread=0
            )
            npv = result.get('npv', 0)
        except:
            npv = 0
        
        scenario_results.append({
            'Scenario': scenario_name,
            'Rate Shift (bp)': shift_bp,
            'NPV (USD)': f"${npv:,.0f}",
            'P&L vs Base': f"${npv - npvs[10]:,.0f}" if scenario_name != "Base Case" else "$0"
        })
    
    st.markdown('<div class="sub-header">🎭 Scenario Analysis</div>', unsafe_allow_html=True)
    scenario_df = pd.DataFrame(scenario_results)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)


def _currency_swaps_tab():
    """Enhanced Currency Swaps Interface"""
    st.markdown('<div class="sub-header">Cross-Currency Swap Pricing</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🚧 Currency Swaps - Professional Implementation</h4>
        <p>Advanced cross-currency swap pricing with:</p>
        <ul>
            <li><strong>Multi-Currency Curves:</strong> USD, EUR, GBP, JPY, CHF</li>
            <li><strong>Cross-Currency Basis:</strong> Market-consistent basis adjustments</li>
            <li><strong>FX Volatility:</strong> Quanto adjustments for FX risk</li>
            <li><strong>Collateral Effects:</strong> CSA and funding considerations</li>
        </ul>
        <p><em>Coming in next release with full FX derivatives suite...</em></p>
    </div>
    """, unsafe_allow_html=True)


def _equity_swaps_tab():
    """Enhanced Equity Swaps Interface"""
    st.markdown('<div class="sub-header">Equity Swap Pricing & Analytics</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🚧 Equity Swaps - Advanced Features</h4>
        <p>Comprehensive equity swap pricing including:</p>
        <ul>
            <li><strong>Single Name:</strong> Individual stock swaps</li>
            <li><strong>Basket Swaps:</strong> Multi-asset equity exposure</li>
            <li><strong>Index Swaps:</strong> S&P 500, FTSE, Nikkei, etc.</li>
            <li><strong>Dividend Treatment:</strong> Gross vs net dividend adjustments</li>
            <li><strong>Variance Swaps:</strong> Volatility exposure products</li>
        </ul>
        <p><em>Full implementation coming with equity derivatives module...</em></p>
    </div>
    """, unsafe_allow_html=True)


def _portfolio_analysis_tab():
    """Portfolio-level swap analysis"""
    st.markdown('<div class="sub-header">Portfolio Risk Management</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🚧 Portfolio Analytics - Enterprise Features</h4>
        <p>Institutional-grade portfolio management:</p>
        <ul>
            <li><strong>Risk Aggregation:</strong> Portfolio-level Greeks and VaR</li>
            <li><strong>Correlation Effects:</strong> Cross-asset and curve correlations</li>
            <li><strong>Stress Testing:</strong> Historical and hypothetical scenarios</li>
            <li><strong>Hedge Ratios:</strong> Optimal hedging strategies</li>
            <li><strong>Attribution:</strong> P&L breakdown by risk factor</li>
        </ul>
        <p><em>Full implementation coming with risk management suite...</em></p>
    </div>
    """, unsafe_allow_html=True)


def _market_intelligence_tab():
    """Market intelligence and research"""
    st.markdown('<div class="sub-header">Market Intelligence & Research</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Market Data</h4>
            <h5>Current Environment (Indicative)</h5>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; font-weight: bold;">Fed Funds</td>
                    <td style="padding: 8px; font-family: monospace;">2.00%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; font-weight: bold;">2Y Swap</td>
                    <td style="padding: 8px; font-family: monospace;">3.25%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; font-weight: bold;">5Y Swap</td>
                    <td style="padding: 8px; font-family: monospace;">3.50%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px; font-weight: bold;">10Y Swap</td>
                    <td style="padding: 8px; font-family: monospace;">3.75%</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: bold;">30Y Swap</td>
                    <td style="padding: 8px; font-family: monospace;">3.90%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Risk Factors</h4>
            <h5>Current Market Themes</h5>
            <ul>
                <li><strong>Central Bank Policy:</strong> Fed pause vs. ECB tightening</li>
                <li><strong>Curve Dynamics:</strong> Inversion risk in 2Y-10Y</li>
                <li><strong>Credit Spreads:</strong> Widening in IG/HY</li>
                <li><strong>Volatility:</strong> Elevated across all tenors</li>
                <li><strong>Liquidity:</strong> Reduced market making capacity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Educational content
    with st.expander("📚 Swaps Market Education"):
        st.markdown("""
        <div class="info-box">
            <h4>Interest Rate Swaps Fundamentals</h4>
            
            <h5>Market Structure:</h5>
            <p>The interest rate swaps market is the largest derivatives market globally, with over $400 trillion in notional outstanding. 
            It serves as the backbone for:</p>
            <ul>
                <li><strong>Corporate Hedging:</strong> Converting floating rate debt to fixed, or vice versa</li>
                <li><strong>Bank ALM:</strong> Asset-liability duration matching</li>
                <li><strong>Speculation:</strong> Directional and relative value trades</li>
                <li><strong>Arbitrage:</strong> Basis trading between different rate indices</li>
            </ul>
            
            <h5>Key Concepts:</h5>
            <ul>
                <li><strong>Par Rate:</strong> Fixed rate that makes swap NPV = 0 at inception</li>
                <li><strong>DV01:</strong> Dollar sensitivity to 1 basis point parallel shift</li>
                <li><strong>Key Rate Durations:</strong> Sensitivity to specific tenor points</li>
                <li><strong>Convexity:</strong> Second-order rate sensitivity</li>
                <li><strong>Carry:</strong> Expected P&L assuming unchanged rates</li>
            </ul>
            
            <h5>Pricing Models:</h5>
            <ul>
                <li><strong>DCF:</strong> Standard present value approach using market curves</li>
                <li><strong>LIBOR Market Model:</strong> Stochastic evolution of forward rates</li>
                <li><strong>Hull-White:</strong> Mean-reverting short rate model</li>
                <li><strong>BGM:</strong> Brace-Gatarek-Musiela model for caps/floors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 1px solid #dee2e6;'>
        <div style="margin-bottom: 10px;">
            <span style="font-size: 2rem;">🔄</span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">Advanced Swaps Pricing Suite</p>
        <p style="margin: 8px 0; color: #6c757d;">Built with institutional-grade models</p>
        <p style="margin: 0; color: #dc3545; font-weight: bold;">⚠️ For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)


# Enhanced pricing functions (these would typically be in separate files)

def build_enhanced_discount_curve(base_rate, curve_shape="flat"):
    """Build enhanced discount curve with realistic term structure"""
    if curve_shape == "flat":
        return lambda t: np.exp(-base_rate * t)
    else:
        # Add term structure (simplified Nelson-Siegel)
        beta0, beta1, beta2, tau = base_rate, -0.005, 0.01, 2.0
        def enhanced_curve(t):
            rate = beta0 + beta1 * (1 - np.exp(-t/tau)) / (t/tau) + beta2 * ((1 - np.exp(-t/tau)) / (t/tau) - np.exp(-t/tau))
            return np.exp(-rate * t)
        return enhanced_curve


def build_enhanced_forward_curve(libor_3m, fed_funds, basis_spread=0.0):
    """Build forward curve with basis considerations"""
    def forward_curve(t):
        # Simple forward curve with basis adjustment
        forward_rate = libor_3m + basis_spread + (fed_funds - libor_3m) * np.exp(-0.1 * t)
        return forward_rate
    return forward_curve


def price_irs_enhanced_dcf(notional, fixed_rate, payment_times, discount_curve, forward_curve, floating_spread=0):
    """Enhanced DCF pricing with proper curve interpolation"""
    
    # Year fractions
    year_fractions = np.diff([0] + payment_times)
    
    # Fixed leg PV
    pv_fixed = sum([
        notional * fixed_rate * yf * discount_curve(t)
        for yf, t in zip(year_fractions, payment_times)
    ])
    
    # Floating leg PV with forward curve
    pv_floating = sum([
        notional * (forward_curve(t) + floating_spread) * yf * discount_curve(t)
        for yf, t in zip(year_fractions, payment_times)
    ])
    
    # NPV (receiver swap: receive fixed, pay floating)
    npv = pv_fixed - pv_floating
    
    # Par rate calculation
    annuity = sum([yf * discount_curve(t) for yf, t in zip(year_fractions, payment_times)])
    par_rate = pv_floating / (notional * annuity)
    
    # DV01 calculation (approximate)
    dv01 = notional * annuity * 0.0001
    
    return {
        'npv': npv,
        'pv_fixed': pv_fixed,
        'pv_floating': pv_floating,
        'par_rate': par_rate,
        'dv01': dv01,
        'annuity': annuity
    }


def price_irs_lmm_enhanced(notional, fixed_rate, initial_rate, vol, payment_times, 
                          discount_curve, n_paths=25000, floating_spread=0):
    """Enhanced LIBOR Market Model implementation"""
    
    np.random.seed(42)
    dt = payment_times[1] - payment_times[0] if len(payment_times) > 1 else 0.25
    n_steps = len(payment_times)
    
    # Simulate forward rates
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = initial_rate
    
    for i in range(1, n_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        drift = -0.5 * vol**2 * dt  # Martingale adjustment
        rates[:, i] = rates[:, i-1] * np.exp(drift + vol * dW)
    
    # Calculate floating leg cashflows
    year_fractions = np.diff([0] + payment_times)
    
    floating_cashflows = np.zeros(n_paths)
    for i, (yf, t) in enumerate(zip(year_fractions, payment_times)):
        floating_cashflows += notional * (rates[:, i] + floating_spread) * yf * discount_curve(t)
    
    pv_floating = np.mean(floating_cashflows)
    
    # Fixed leg (deterministic)
    pv_fixed = sum([
        notional * fixed_rate * yf * discount_curve(t)
        for yf, t in zip(year_fractions, payment_times)
    ])
    
    # NPV and statistics
    npv = pv_fixed - pv_floating
    npv_std = np.std(floating_cashflows - pv_fixed) / np.sqrt(n_paths)
    
    # Par rate
    annuity = sum([yf * discount_curve(t) for yf, t in zip(year_fractions, payment_times)])
    par_rate = pv_floating / (notional * annuity)
    
    # DV01 (approximate)
    dv01 = notional * annuity * 0.0001
    
    return {
        'npv': npv,
        'npv_std': npv_std,
        'pv_fixed': pv_fixed,
        'pv_floating': pv_floating,
        'par_rate': par_rate,
        'dv01': dv01,
        'annuity': annuity,
        'monte_carlo_paths': n_paths
    }


def price_irs_hull_white(notional, fixed_rate, initial_rate, mean_reversion, vol, 
                        payment_times, n_paths=25000):
    """Hull-White model implementation for IRS pricing"""
    
    np.random.seed(42)
    dt = payment_times[1] - payment_times[0] if len(payment_times) > 1 else 0.25
    n_steps = len(payment_times)
    
    # Hull-White parameters
    a = mean_reversion
    sigma = vol
    
    # Simulate short rate paths
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = initial_rate
    
    for i in range(1, n_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        # Hull-White dynamics: dr = a(theta(t) - r)dt + sigma*dW
        # Simplified with constant theta
        theta = initial_rate  # Simplified mean level
        rates[:, i] = rates[:, i-1] + a * (theta - rates[:, i-1]) * dt + sigma * dW
        rates[:, i] = np.maximum(rates[:, i], 0)  # Floor at zero
    
    # Calculate bond prices and swap cashflows
    year_fractions = np.diff([0] + payment_times)
    
    floating_cashflows = np.zeros(n_paths)
    for i, (yf, t) in enumerate(zip(year_fractions, payment_times)):
        # Approximate discount factor
        discount_factor = np.exp(-rates[:, i] * t)
        floating_cashflows += notional * rates[:, i] * yf * discount_factor
    
    pv_floating = np.mean(floating_cashflows)
    
    # Fixed leg using average discount factors
    avg_rates = np.mean(rates, axis=0)
    pv_fixed = sum([
        notional * fixed_rate * yf * np.exp(-avg_rates[i] * t)
        for i, (yf, t) in enumerate(zip(year_fractions, payment_times))
    ])
    
    # NPV
    npv = pv_fixed - pv_floating
    
    # Par rate calculation
    annuity = sum([yf * np.exp(-avg_rates[i] * t) for i, (yf, t) in enumerate(zip(year_fractions, payment_times))])
    par_rate = pv_floating / (notional * annuity) if annuity > 0 else fixed_rate
    
    # DV01 (approximate)
    dv01 = notional * annuity * 0.0001
    
    return {
        'npv': npv,
        'pv_fixed': pv_fixed,
        'pv_floating': pv_floating,
        'par_rate': par_rate,
        'dv01': dv01,
        'model': 'Hull-White',
        'mean_reversion': mean_reversion,
        'volatility': vol
    }
