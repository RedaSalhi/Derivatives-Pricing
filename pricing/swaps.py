# tabs/swaps.py
# Clean, Modular Swaps Tab Implementation

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# Import styling
from styles.app_styles import load_theme

# Import modular components
try:
    from pricing.models.market_data import market_data_manager
    from pricing.models.swap_pricing import (
        InterestRateSwapPricer, CurrencySwapPricer, EquitySwapPricer, 
        CurveBuilder, PortfolioAnalyzer
    )
    from pricing.models.display_utils import SwapDisplayManager, MarketIntelligenceDisplay
    PRICING_AVAILABLE = True
except ImportError as e:
    st.error(f"Pricing modules not available: {e}")
    PRICING_AVAILABLE = False


def swaps_tab():
    """Main Swaps Tab Entry Point"""
    
    load_theme()
    
    st.markdown('<div class="main-header">Advanced Swap Pricing Suite</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Live market overview
    if PRICING_AVAILABLE:
        _display_market_header()
    
    # Platform introduction
    st.markdown("""
    <div class="info-box">
        <h3>🏦 Professional Derivatives Pricing Platform</h3>
        <p>Advanced swap pricing models with real-time market data integration and institutional-grade analytics.</p>
        <div style="display: flex; gap: 20px; margin-top: 15px;">
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Market Size:</strong><br>$500+ Trillion Outstanding
            </div>
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Daily Volume:</strong><br>$2-3 Trillion Average
            </div>
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Real-Time Data:</strong><br>Live Market Integration
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Interest Rate Swaps", 
        "💱 Currency Swaps", 
        "📈 Equity Swaps", 
        "📊 Portfolio Analysis",
        "📚 Market Intelligence"
    ])
    
    with tab1:
        interest_rate_swaps_interface()
    
    with tab2:
        currency_swaps_interface()
    
    with tab3:
        equity_swaps_interface()
    
    with tab4:
        portfolio_analysis_interface()
    
    with tab5:
        market_intelligence_interface()


def _display_market_header():
    """Display live market overview header"""
    
    market_overview = market_data_manager.get_live_market_overview()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "10Y USD Treasury", 
            f"{market_overview['usd_10y']:.2f}%",
            f"{market_overview['usd_10y_change']:+.2f}bp"
        )
    
    with col2:
        st.metric(
            "EUR/USD", 
            f"{market_overview['eurusd']:.4f}",
            f"{market_overview['eurusd_change']:+.1f}%"
        )
    
    with col3:
        st.metric(
            "VIX", 
            f"{market_overview['vix']:.1f}",
            f"{market_overview['vix_change']:+.1f}%"
        )


def interest_rate_swaps_interface():
    """Interest Rate Swaps Interface"""
    
    st.markdown('<div class="sub-header">Interest Rate Swap Pricing & Analytics</div>', 
                unsafe_allow_html=True)
    
    if not PRICING_AVAILABLE:
        st.error("Pricing modules not available")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔧 Swap Configuration</h4>
            <p>Enhanced with live market data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get live rates
        live_rates = market_data_manager.get_live_interest_rates("USD")
        
        # Swap parameters
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
            index=0
        )
        
        # Live market rate
        market_rate = live_rates.get(f'{tenor_years}Y', 3.5)
        st.info(f"💡 Live {tenor_years}Y rate: {market_rate:.2f}%")
        
        fixed_rate = st.slider(
            "Fixed Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=market_rate, 
            step=0.01,
            format="%.2f"
        ) / 100
        
        floating_spread = st.slider(
            "Floating Spread (bps)", 
            min_value=-100, 
            max_value=500, 
            value=0, 
            step=5
        )
        
        # Model selection
        model = st.radio(
            "Pricing Model",
            ["Enhanced DCF", "Monte Carlo", "Hull-White"]
        )
        
        if model != "Enhanced DCF":
            vol = st.slider("IR Volatility (%)", 5.0, 50.0, 15.0, 1.0) / 100
            n_paths = st.selectbox("MC Paths", [10000, 25000, 50000], index=1)
        
        price_btn = st.button("🔢 Price Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Pricing swap with live data..."):
                try:
                    # Build curves
                    base_rate = live_rates.get(f'{int(tenor_years)}Y', 0.04)
                    discount_curve = CurveBuilder.build_discount_curve(base_rate)
                    forward_curve = CurveBuilder.build_forward_curve(base_rate, floating_spread / 10000)
                    
                    # Payment schedule
                    freq_map = {"Quarterly": 0.25, "Semi-Annual": 0.5, "Annual": 1.0}
                    dt = freq_map[payment_freq]
                    payment_times = [dt * i for i in range(1, int(tenor_years / dt) + 1)]
                    
                    # Price based on model
                    if model == "Enhanced DCF":
                        result = InterestRateSwapPricer.price_enhanced_dcf(
                            notional, fixed_rate, payment_times, 
                            discount_curve, forward_curve, floating_spread / 10000
                        )
                    elif model == "Monte Carlo":
                        result = InterestRateSwapPricer.price_monte_carlo(
                            notional, fixed_rate, base_rate, vol, payment_times, 
                            discount_curve, n_paths, floating_spread / 10000
                        )
                    else:  # Hull-White
                        result = InterestRateSwapPricer.price_hull_white(
                            notional, fixed_rate, base_rate, vol, payment_times, n_paths
                        )
                    
                    # Display results
                    SwapDisplayManager.display_irs_results(
                        result, notional, fixed_rate, tenor_years, model, market_rate
                    )
                    
                    SwapDisplayManager.display_risk_analytics(result, "IRS")
                    SwapDisplayManager.display_market_comparison(result, market_rate, fixed_rate)
                    
                except Exception as e:
                    st.error(f"Pricing error: {str(e)}")
        else:
            SwapDisplayManager.display_live_yield_curve()


def currency_swaps_interface():
    """Currency Swaps Interface"""
    
    st.markdown('<div class="sub-header">Cross-Currency Swap Pricing</div>', 
                unsafe_allow_html=True)
    
    if not PRICING_AVAILABLE:
        st.error("Pricing modules not available")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>💱 Currency Swap Setup</h4>
            <p>Live FX and rate data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Currency selection
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]
        
        base_currency = st.selectbox("Base Currency (Pay)", currencies, index=0)
        quote_currency = st.selectbox("Quote Currency (Receive)", currencies, index=1)
        
        if base_currency == quote_currency:
            st.error("Currencies must be different")
            return
        
        # Live FX data
        fx_data = market_data_manager.get_live_fx_rate(quote_currency, base_currency)
        
        st.markdown(f"""
        <div class="success-box">
            <h5>📊 Live FX Data</h5>
            <p><strong>Rate:</strong> {fx_data['rate']:.4f}</p>
            <p><strong>Change:</strong> {fx_data['change']:+.2f}%</p>
            <p><strong>Volatility:</strong> {fx_data['volatility']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Notionals
        base_notional = st.number_input(
            f"Base Notional ({base_currency})", 
            min_value=1_000_000, 
            max_value=10_000_000_000, 
            value=100_000_000, 
            step=1_000_000
        )
        
        fx_spot = st.number_input(
            f"FX Rate ({quote_currency}{base_currency})", 
            min_value=0.0001, 
            max_value=1000.0, 
            value=fx_data['rate'], 
            step=0.0001,
            format="%.4f"
        )
        
        quote_notional = base_notional / fx_spot
        st.metric(f"{quote_currency} Notional", f"{quote_notional:,.0f}")
        
        # Swap parameters
        tenor_years = st.selectbox("Tenor", [1, 2, 3, 5, 7, 10], index=3)
        
        # Live rates
        base_rates = market_data_manager.get_live_interest_rates(base_currency)
        quote_rates = market_data_manager.get_live_interest_rates(quote_currency)
        
        st.markdown(f"""
        <div class="info-box">
            <h5>📈 Live Rates</h5>
            <p><strong>{base_currency}:</strong> {base_rates.get(f'{tenor_years}Y', 3.5):.2f}%</p>
            <p><strong>{quote_currency}:</strong> {quote_rates.get(f'{tenor_years}Y', 2.5):.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        base_rate = st.slider(
            f"{base_currency} Rate (%)", 
            0.0, 10.0, 
            base_rates.get(f'{tenor_years}Y', 3.5), 
            0.01
        ) / 100
        
        quote_rate = st.slider(
            f"{quote_currency} Rate (%)", 
            0.0, 10.0, 
            quote_rates.get(f'{tenor_years}Y', 2.5), 
            0.01
        ) / 100
        
        # Advanced options
        with st.expander("🔬 Advanced"):
            include_principal = st.checkbox("Principal Exchange", value=True)
            xccy_basis = st.slider("Cross-Currency Basis (bps)", -100, 100, 0, 5)
        
        price_btn = st.button("💱 Price Currency Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Pricing currency swap..."):
                try:
                    result = CurrencySwapPricer.price_simple(
                        base_notional, quote_notional, base_rate, quote_rate,
                        tenor_years, fx_spot, include_principal, xccy_basis / 10000
                    )
                    
                    SwapDisplayManager.display_currency_swap_results(
                        result, base_currency, quote_currency, fx_spot, tenor_years
                    )
                    
                    SwapDisplayManager.display_risk_analytics(
                        result, "Currency", 
                        base_currency=base_currency, quote_currency=quote_currency
                    )
                    
                except Exception as e:
                    st.error(f"Pricing error: {str(e)}")
        else:
            SwapDisplayManager.display_fx_market_overview()


def equity_swaps_interface():
    """Equity Swaps Interface"""
    
    st.markdown('<div class="sub-header">Equity Swap Pricing</div>', 
                unsafe_allow_html=True)
    
    if not PRICING_AVAILABLE:
        st.error("Pricing modules not available")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📈 Equity Swap Setup</h4>
            <p>Live equity market data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Asset selection
        equity_type = st.radio("Asset Type", ["Index ETF", "Single Stock"], index=0)
        
        if equity_type == "Single Stock":
            reference_asset = st.text_input("Stock Symbol", value="AAPL")
        else:
            reference_asset = st.selectbox(
                "Index ETF", 
                ["SPY", "QQQ", "IWM", "EFA", "EEM", "VTI"], 
                index=0
            )
        
        # Live equity data
        equity_data = market_data_manager.get_live_equity_data(reference_asset)
        
        st.markdown(f"""
        <div class="success-box">
            <h5>📊 Live Data</h5>
            <p><strong>Price:</strong> ${equity_data['price']:.2f}</p>
            <p><strong>Change:</strong> {equity_data['change']:+.2f}%</p>
            <p><strong>Volatility:</strong> {equity_data['volatility']:.1f}%</p>
            <p><strong>Dividend:</strong> {equity_data['dividend_yield']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Swap parameters
        notional = st.number_input(
            "Notional (USD)", 
            min_value=1_000_000, 
            max_value=1_000_000_000, 
            value=50_000_000, 
            step=1_000_000
        )
        
        swap_direction = st.radio(
            "Direction", 
            ["Pay Equity, Receive Fixed", "Pay Fixed, Receive Equity"]
        )
        
        tenor_years = st.selectbox("Tenor", [0.25, 0.5, 1, 2, 3, 5], index=2)
        
        # Fixed rate
        current_rates = market_data_manager.get_live_interest_rates("USD")
        
        fixed_rate = st.slider(
            "Fixed Rate (%)", 
            0.0, 8.0, 
            current_rates.get("1Y", 4.0), 
            0.01
        ) / 100
        
        # Advanced options
        with st.expander("🔬 Advanced"):
            financing_spread = st.slider("Financing Spread (bps)", -100, 500, 50, 10)
            dividend_treatment = st.radio("Dividend Treatment", ["Gross", "Net"])
        
        price_btn = st.button("📈 Price Equity Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Pricing equity swap..."):
                try:
                    result = EquitySwapPricer.price_enhanced(
                        reference_asset, notional, fixed_rate, tenor_years,
                        equity_data, swap_direction, financing_spread / 10000,
                        dividend_treatment
                    )
                    
                    SwapDisplayManager.display_equity_swap_results(
                        result, reference_asset, notional, swap_direction, tenor_years
                    )
                    
                    SwapDisplayManager.display_risk_analytics(
                        result, "Equity", equity_data=equity_data
                    )
                    
                except Exception as e:
                    st.error(f"Pricing error: {str(e)}")
        else:
            SwapDisplayManager.display_equity_market_overview()


def portfolio_analysis_interface():
    """Portfolio Analysis Interface"""
    
    st.markdown('<div class="sub-header">Portfolio Risk Management</div>', 
                unsafe_allow_html=True)
    
    if not PRICING_AVAILABLE:
        st.error("Pricing modules not available")
        return
    
    # Initialize portfolio
    if 'swap_portfolio' not in st.session_state:
        st.session_state.swap_portfolio = []
    
    st.markdown("""
    <div class="info-box">
        <h4>🏗️ Portfolio Construction</h4>
        <p>Build and analyze multi-asset swap portfolios</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio builder
    with st.expander("➕ Add Swap to Portfolio", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            swap_type = st.selectbox("Type", ["Interest Rate", "Currency", "Equity"])
            notional = st.number_input("Notional", 1_000_000, 1_000_000_000, 100_000_000, 1_000_000)
        
        with col2:
            if swap_type == "Interest Rate":
                tenor = st.selectbox("Tenor", ["2Y", "5Y", "10Y"], index=1)
                direction = st.radio("Direction", ["Pay Fixed", "Receive Fixed"])
                extra_params = {}
                
            elif swap_type == "Currency":
                currency_pair = st.selectbox("Pair", ["EURUSD", "GBPUSD", "USDJPY"])
                tenor = st.selectbox("Tenor", ["1Y", "3Y", "5Y"], index=1)
                direction = st.radio("Direction", ["Pay USD", "Receive USD"])
                extra_params = {'currency_pair': currency_pair}
                
            else:  # Equity
                underlying = st.selectbox("Underlying", ["SPY", "QQQ", "IWM"])
                tenor = st.selectbox("Tenor", ["1Y", "2Y", "3Y"], index=0)
                direction = st.radio("Direction", ["Pay Equity", "Receive Equity"])
                extra_params = {'underlying': underlying}
        
        with col3:
            if st.button("Add to Portfolio"):
                swap_obj = PortfolioAnalyzer.create_swap_object(
                    swap_type, notional, tenor, direction, **extra_params
                )
                st.session_state.swap_portfolio.append(swap_obj)
                st.success(f"Added {swap_type} swap")
    
    # Display portfolio
    if st.session_state.swap_portfolio:
        st.markdown('<div class="sub-header">📊 Current Portfolio</div>', 
                    unsafe_allow_html=True)
        
        portfolio_df = pd.DataFrame(st.session_state.swap_portfolio)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        
        with col2:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.swap_portfolio = []
                st.rerun()
            
            if st.button("💰 Analyze Portfolio"):
                analytics = PortfolioAnalyzer.analyze_portfolio(st.session_state.swap_portfolio)
                SwapDisplayManager.display_portfolio_analytics(analytics)
    else:
        st.info("👆 Add swaps to begin portfolio analysis")


def market_intelligence_interface():
    """Market Intelligence Interface"""
    
    st.markdown('<div class="sub-header">Market Intelligence & Research</div>', 
                unsafe_allow_html=True)
    
    if not PRICING_AVAILABLE:
        st.error("Pricing modules not available")
        return
    
    st.markdown("""
    <div class="info-box">
        <h4>📈 Live Market Dashboard</h4>
        <p>Real-time market analysis and research insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market intelligence tabs
    intel_tab1, intel_tab2, intel_tab3, intel_tab4 = st.tabs([
        "🌐 Global Rates", "💱 FX Markets", "📊 Volatility", "📚 Research"
    ])
    
    with intel_tab1:
        MarketIntelligenceDisplay.display_global_rates_dashboard()
    
    with intel_tab2:
        MarketIntelligenceDisplay.display_fx_markets_dashboard()
    
    with intel_tab3:
        MarketIntelligenceDisplay.display_volatility_dashboard()
    
    with intel_tab4:
        MarketIntelligenceDisplay.display_research_insights()


# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; 
     padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
     border-radius: 15px; border: 1px solid #dee2e6;'>
    <div style="margin-bottom: 15px;">
        <span style="font-size: 2rem;">🔄</span>
    </div>
    <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">
        Professional Swaps Pricing Suite
    </p>
    <p style="margin: 8px 0; color: #6c757d;">
        Powered by Live Market Data • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
         gap: 15px; margin: 15px 0;">
        <div style="background: rgba(0,123,255,0.1); padding: 10px; border-radius: 8px;">
            <strong>🏦 Interest Rate</strong><br>
            <small>DCF • Monte Carlo • Hull-White</small>
        </div>
        <div style="background: rgba(40,167,69,0.1); padding: 10px; border-radius: 8px;">
            <strong>💱 Currency</strong><br>
            <small>Live FX • Cross-Currency Basis</small>
        </div>
        <div style="background: rgba(255,193,7,0.1); padding: 10px; border-radius: 8px;">
            <strong>📈 Equity</strong><br>
            <small>Single Name • Index ETFs</small>
        </div>
        <div style="background: rgba(220,53,69,0.1); padding: 10px; border-radius: 8px;">
            <strong>📊 Portfolio</strong><br>
            <small>Risk Aggregation • Analytics</small>
        </div>
    </div>
    <p style="margin: 15px 0 0 0; color: #dc3545; font-weight: bold;">
        ⚠️ Educational and research purposes only
    </p>
</div>
""", unsafe_allow_html=True)
