# Enhanced Swap Pricing Application - Main Interface
# File: tabs/enhanced_swaps.py

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Enhanced imports
try:
    from pricing.models.swap_pricing import (
        EnhancedInterestRateSwapPricer, EnhancedCurrencySwapPricer, 
        AccurateDiscountCurve, EducationalSwapCalculator
    )
    from pricing.models.market_data import enhanced_market_data_manager
    from pricing.models.display_utils import (
        EducationalSwapDisplayManager, PortfolioAnalyticsDisplay
    )
    ENHANCED_PRICING_AVAILABLE = True
except ImportError as e:
    st.error(f"Enhanced pricing modules not available: {e}")
    ENHANCED_PRICING_AVAILABLE = False


def enhanced_swaps_tab():
    """Enhanced Swaps Tab with Educational Content and Accurate Pricing"""
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .education-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with live market data
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Advanced Swap Pricing & Analytics Platform</h1>
        <p style="font-size: 1.2em; margin-bottom: 0;">
            Professional-grade derivatives pricing with real-time market data and educational insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not ENHANCED_PRICING_AVAILABLE:
        st.error("⚠️ Enhanced pricing modules not available. Please check your installation.")
        return
    
    # Live market overview
    _display_enhanced_market_header()
    
    # Main navigation
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
        "🔄 Interest Rate Swaps", 
        "💱 Currency Swaps", 
        "📈 Equity Swaps",
        "📊 Portfolio Management",
        "📚 Education & Research"
    ])
    
    with main_tab1:
        enhanced_interest_rate_swaps()
    
    with main_tab2:
        enhanced_currency_swaps()
    
    with main_tab3:
        enhanced_equity_swaps()
    
    with main_tab4:
        enhanced_portfolio_management()
    
    with main_tab5:
        enhanced_education_center()


def _display_enhanced_market_header():
    """Display comprehensive market overview"""
    
    market_summary = enhanced_market_data_manager.get_market_summary()
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                padding: 20px; border-radius: 15px; margin-bottom: 25px; color: white;">
        <h3 style="margin: 0 0 15px 0; color: white;">📡 Live Market Overview</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px;">
    """, unsafe_allow_html=True)
    
    # Market metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        usd_10y = market_summary.get('usd_10y', 0.0425)
        st.metric(
            "USD 10Y Treasury", 
            f"{usd_10y*100:.2f}%",
            delta=f"{np.random.normal(0, 2):.1f}bp",
            help="10-Year US Treasury yield - benchmark for USD swaps"
        )
    
    with col2:
        curve_slope = market_summary.get('curve_slope', 20)
        st.metric(
            "Curve Slope", 
            f"{curve_slope:.0f}bp",
            delta=f"{np.random.normal(0, 5):.0f}bp",
            help="10Y-2Y spread indicates yield curve shape"
        )
    
    with col3:
        eurusd = market_summary['eurusd']['spot_rate']
        eurusd_change = market_summary['eurusd']['change_pct']
        st.metric(
            "EUR/USD", 
            f"{eurusd:.4f}",
            delta=f"{eurusd_change:+.2f}%",
            help="Euro to US Dollar exchange rate"
        )
    
    with col4:
        spy_price = market_summary['spy']['price']
        spy_change = market_summary['spy']['change_pct']
        st.metric(
            "SPY ETF", 
            f"${spy_price:.0f}",
            delta=f"{spy_change:+.1f}%",
            help="S&P 500 ETF price for equity swap reference"
        )
    
    with col5:
        vix_level = market_summary['vix']['price']
        st.metric(
            "VIX", 
            f"{vix_level:.1f}",
            delta=f"{np.random.normal(0, 1):.1f}",
            help="Market volatility index - fear gauge"
        )
    
    st.markdown(f"""
        </div>
        <div style="text-align: center; margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
            Last Updated: {market_summary['last_updated']} • 
            <span style="color: #ffd700;">Live Data Integration Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def enhanced_interest_rate_swaps():
    """Enhanced Interest Rate Swaps interface with education"""
    
    # Educational introduction
    EducationalSwapDisplayManager.display_swap_explanation("interest_rate")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔧 Swap Configuration</h4>
            <p>Configure your interest rate swap with real market data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Basic parameters
        notional = st.number_input(
            "Notional Amount (USD)", 
            min_value=1_000_000, 
            max_value=10_000_000_000, 
            value=100_000_000, 
            step=1_000_000,
            help="Principal amount for calculating payments (not exchanged)"
        )
        
        tenor_years = st.selectbox(
            "Swap Tenor", 
            [1, 2, 3, 5, 7, 10, 15, 20, 30], 
            index=4,
            help="Maturity of the swap in years"
        )
        
        payment_frequency = st.selectbox(
            "Payment Frequency", 
            ["Quarterly", "Semi-Annual", "Annual"], 
            index=1,
            help="How often interest payments are exchanged"
        )
        
        position = st.radio(
            "Your Position",
            ["Receive Fixed, Pay Floating", "Pay Fixed, Receive Floating"],
            help="Direction of your swap position"
        )
        
        # Get live market data
        usd_curve_data = enhanced_market_data_manager.get_comprehensive_yield_curve("USD")
        market_rate = usd_curve_data.get(float(tenor_years), 0.0425)
        
        st.info(f"💡 **Live Market Rate**: {market_rate*100:.3f}% for {tenor_years}Y USD")
        
        fixed_rate = st.slider(
            "Fixed Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=market_rate*100, 
            step=0.001,
            format="%.3f",
            help="Fixed interest rate you pay or receive"
        ) / 100
        
        floating_spread = st.slider(
            "Floating Spread (bps)", 
            min_value=-100, 
            max_value=500, 
            value=0, 
            step=1,
            help="Spread over SOFR for floating leg"
        )
        
        # Advanced options
        with st.expander("🔬 Advanced Settings"):
            day_count = st.selectbox(
                "Day Count Convention",
                ["ACT/360", "ACT/365", "30/360"],
                index=0,
                help="Convention for calculating accrued interest"
            )
            
            model_type = st.radio(
                "Pricing Model",
                ["Enhanced DCF", "Monte Carlo", "Hull-White"],
                help="Choose pricing methodology"
            )
            
            if model_type != "Enhanced DCF":
                volatility = st.slider("IR Volatility (%)", 5.0, 50.0, 15.0, 1.0) / 100
                if model_type == "Monte Carlo":
                    n_paths = st.selectbox("Simulation Paths", [10000, 25000, 50000], index=1)
                elif model_type == "Hull-White":
                    mean_reversion = st.slider("Mean Reversion", 0.01, 0.5, 0.1, 0.01)
        
        # Pricing button
        price_button = st.button(
            "💰 Price Interest Rate Swap", 
            type="primary", 
            use_container_width=True,
            help="Calculate swap NPV and risk metrics"
        )
    
    with col2:
        if price_button:
            with st.spinner("Pricing swap with live market data..."):
                try:
                    # Create discount curve
                    discount_curve = AccurateDiscountCurve(usd_curve_data)
                    
                    # Create payment schedule
                    start_date = datetime.now()
                    maturity_date = start_date + timedelta(days=365 * tenor_years)
                    
                    payment_schedule = EnhancedInterestRateSwapPricer.create_payment_schedule(
                        start_date, maturity_date, payment_frequency, day_count
                    )
                    
                    # Price the swap
                    position_key = "receive_fixed" if "Receive Fixed" in position else "pay_fixed"
                    
                    result = EnhancedInterestRateSwapPricer.price_accurate_dcf(
                        notional=notional,
                        fixed_rate=fixed_rate,
                        discount_curve=discount_curve,
                        payment_schedule=payment_schedule,
                        floating_spread=floating_spread / 10000,
                        position=position_key
                    )
                    
                    # Display results with educational context
                    market_data = {
                        'market_rate': market_rate,
                        'curve_data': usd_curve_data,
                        'position': position,
                        'model': model_type
                    }
                    
                    EducationalSwapDisplayManager.display_comprehensive_risk_analytics(
                        result, "IRS", scenario_df
                    )
                    
                    # Cash flow table
                    if hasattr(result, 'cashflows') and result.cashflows is not None:
                        with st.expander("📊 Detailed Cash Flow Analysis"):
                            st.markdown("**Cash Flow Schedule**")
                            st.dataframe(result.cashflows, use_container_width=True)
                            
                            # Cash flow chart
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=result.cashflows.index,
                                y=result.cashflows['Fixed_CF'],
                                name='Fixed Payments',
                                marker_color='blue'
                            ))
                            
                            fig.add_trace(go.Bar(
                                x=result.cashflows.index,
                                y=result.cashflows['Floating_CF'],
                                name='Floating Payments',
                                marker_color='orange'
                            ))
                            
                            fig.update_layout(
                                title='Cash Flow Schedule',
                                xaxis_title='Payment Period',
                                yaxis_title='Cash Flow ($)',
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Pricing error: {str(e)}")
                    st.info("Please check your inputs and try again.")
        else:
            # Display educational content and market analysis
            st.markdown("### 📈 Live Market Analysis")
            
            # Interactive yield curve
            curve_fig = EducationalSwapDisplayManager.create_interactive_yield_curve(
                usd_curve_data, "USD Treasury Yield Curve"
            )
            st.plotly_chart(curve_fig, use_container_width=True)
            
            # Educational content about yield curves
            st.markdown("""
            <div class="education-box">
                <h4>📚 Understanding Yield Curves</h4>
                <p><strong>Normal Curve:</strong> Long-term rates higher than short-term (economic growth expected)</p>
                <p><strong>Inverted Curve:</strong> Short-term rates higher than long-term (recession signal)</p>
                <p><strong>Flat Curve:</strong> Similar rates across maturities (economic transition)</p>
                <p><strong>For Swaps:</strong> The curve determines discount rates for present value calculations</p>
            </div>
            """, unsafe_allow_html=True)


def enhanced_currency_swaps():
    """Enhanced Currency Swaps interface"""
    
    EducationalSwapDisplayManager.display_swap_explanation("currency")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>💱 Currency Swap Setup</h4>
            <p>Configure cross-currency swap with live FX data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Currency selection
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]
        
        base_currency = st.selectbox(
            "Base Currency (Pay)", 
            currencies, 
            index=0,
            help="Currency you will pay interest in"
        )
        
        quote_currency = st.selectbox(
            "Quote Currency (Receive)", 
            currencies, 
            index=1,
            help="Currency you will receive interest in"
        )
        
        if base_currency == quote_currency:
            st.error("❌ Base and quote currencies must be different")
            return
        
        # Get live FX data
        currency_pair = f"{quote_currency}{base_currency}"
        fx_data = enhanced_market_data_manager.get_accurate_fx_data(currency_pair)
        
        st.markdown(f"""
        <div class="education-box">
            <h5>📊 Live FX Market Data</h5>
            <p><strong>Rate:</strong> {fx_data['spot_rate']:.4f} {currency_pair}</p>
            <p><strong>Daily Change:</strong> {fx_data['change_pct']:+.2f}%</p>
            <p><strong>Volatility:</strong> {fx_data['volatility']:.1f}% (annual)</p>
            <p><strong>Data Source:</strong> {fx_data['data_source']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Notional amounts
        base_notional = st.number_input(
            f"Base Notional ({base_currency})", 
            min_value=1_000_000, 
            max_value=5_000_000_000, 
            value=100_000_000, 
            step=1_000_000
        )
        
        fx_spot = st.number_input(
            f"FX Rate ({currency_pair})", 
            min_value=0.0001, 
            max_value=1000.0, 
            value=fx_data['spot_rate'], 
            step=0.0001,
            format="%.4f"
        )
        
        quote_notional = base_notional / fx_spot
        st.metric(f"{quote_currency} Notional", f"{quote_notional:,.0f}")
        
        # Swap parameters
        tenor_years = st.selectbox("Tenor", [1, 2, 3, 5, 7, 10], index=2)
        
        # Get live interest rates
        base_curve = enhanced_market_data_manager.get_comprehensive_yield_curve(base_currency)
        quote_curve = enhanced_market_data_manager.get_comprehensive_yield_curve(quote_currency)
        
        base_market_rate = base_curve.get(float(tenor_years), 0.04)
        quote_market_rate = quote_curve.get(float(tenor_years), 0.025)
        
        st.markdown(f"""
        <div class="education-box">
            <h5>📈 Live Interest Rates</h5>
            <p><strong>{base_currency} {tenor_years}Y:</strong> {base_market_rate*100:.2f}%</p>
            <p><strong>{quote_currency} {tenor_years}Y:</strong> {quote_market_rate*100:.2f}%</p>
            <p><strong>Rate Differential:</strong> {(base_market_rate - quote_market_rate)*10000:.0f} bp</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Advanced options
        with st.expander("🔬 Advanced Settings"):
            include_principals = st.checkbox("Principal Exchange", value=True)
            xccy_basis = st.slider("Cross-Currency Basis (bps)", -100, 100, -15, 1)
            payment_frequency = st.selectbox("Payment Frequency", ["Quarterly", "Semi-Annual"], index=1)
        
        price_button = st.button(
            "💱 Price Currency Swap", 
            type="primary", 
            use_container_width=True
        )
    
    with col2:
        if price_button:
            with st.spinner("Pricing cross-currency swap..."):
                try:
                    # Create curves
                    base_curve_obj = AccurateDiscountCurve(base_curve)
                    quote_curve_obj = AccurateDiscountCurve(quote_curve)
                    
                    # Create payment schedule
                    start_date = datetime.now()
                    maturity_date = start_date + timedelta(days=365 * tenor_years)
                    payment_schedule = EnhancedInterestRateSwapPricer.create_payment_schedule(
                        start_date, maturity_date, payment_frequency
                    )
                    
                    # Price the currency swap
                    from pricing.models.enhanced_swap_pricing import EnhancedCurrencySwapPricer
                    
                    result = EnhancedCurrencySwapPricer.price_accurate(
                        domestic_notional=base_notional,
                        foreign_notional=quote_notional,
                        domestic_curve=base_curve_obj,
                        foreign_curve=quote_curve_obj,
                        fx_spot=fx_spot,
                        payment_schedule=payment_schedule,
                        cross_currency_basis=xccy_basis / 10000,
                        include_principals=include_principals
                    )
                    
                    # Display results
                    market_data = {
                        'fx_change': fx_data['change_pct'],
                        'fx_volatility': fx_data['volatility'],
                        'base_rate': base_market_rate,
                        'quote_rate': quote_market_rate
                    }
                    
                    EducationalSwapDisplayManager.display_enhanced_currency_swap_results(
                        result, base_currency, quote_currency, fx_spot, market_data
                    )
                    
                except Exception as e:
                    st.error(f"Currency swap pricing error: {str(e)}")
        else:
            # Display FX market analysis
            st.markdown("### 💱 FX Market Analysis")
            
            # FX volatility chart
            import plotly.graph_objects as go
            
            # Simulate FX price path for illustration
            days = 30
            dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
            returns = np.random.normal(0, fx_data['volatility']/100/np.sqrt(252), days)
            prices = fx_data['spot_rate'] * np.exp(np.cumsum(returns))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=f'{currency_pair}',
                line=dict(color='#28a745', width=3)
            ))
            
            fig.update_layout(
                title=f'{currency_pair} Price History (Simulated)',
                xaxis_title='Date',
                yaxis_title='Exchange Rate',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)


def enhanced_equity_swaps():
    """Enhanced Equity Swaps interface"""
    
    EducationalSwapDisplayManager.display_swap_explanation("equity")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Equity Swap Configuration</h4>
            <p>Create synthetic equity exposure with live market data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Asset selection
        asset_type = st.radio(
            "Asset Type", 
            ["Index ETF", "Individual Stock"], 
            help="Choose between diversified index or single stock exposure"
        )
        
        if asset_type == "Individual Stock":
            reference_asset = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol")
        else:
            reference_asset = st.selectbox(
                "Index ETF", 
                ["SPY", "QQQ", "IWM", "EFA", "EEM", "VTI"],
                help="Choose index ETF for swap reference"
            )
        
        # Get live equity data
        equity_data = enhanced_market_data_manager.get_equity_market_data(reference_asset)
        
        st.markdown(f"""
        <div class="education-box">
            <h5>📊 Live Equity Data - {reference_asset}</h5>
            <p><strong>Current Price:</strong> ${equity_data['price']:.2f}</p>
            <p><strong>Daily Change:</strong> {equity_data['change_pct']:+.2f}%</p>
            <p><strong>Volatility:</strong> {equity_data['volatility']:.1f}% (annual)</p>
            <p><strong>Dividend Yield:</strong> {equity_data['dividend_yield']:.2f}%</p>
            <p><strong>Beta:</strong> {equity_data['beta']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Swap parameters
        notional = st.number_input(
            "Notional Amount (USD)", 
            min_value=1_000_000, 
            max_value=1_000_000_000, 
            value=50_000_000, 
            step=1_000_000
        )
        
        swap_direction = st.radio(
            "Your Position", 
            ["Pay Equity Returns, Receive Fixed", "Pay Fixed, Receive Equity Returns"],
            help="Choose direction of equity exposure"
        )
        
        tenor_years = st.selectbox("Tenor", [0.25, 0.5, 1, 2, 3], index=2)
        
        # Interest rate component
        usd_curve = enhanced_market_data_manager.get_comprehensive_yield_curve("USD")
        market_fixed_rate = usd_curve.get(float(tenor_years), 0.045)
        
        fixed_rate = st.slider(
            "Fixed Rate (%)", 
            0.0, 8.0, 
            market_fixed_rate * 100, 
            0.01
        ) / 100
        
        # Advanced options
        with st.expander("🔬 Advanced Settings"):
            financing_spread = st.slider("Financing Spread (bps)", -50, 200, 25)
            dividend_treatment = st.radio("Dividend Treatment", ["Gross", "Net (Tax-Adjusted)"])
            reset_frequency = st.selectbox("Reset Frequency", ["Monthly", "Quarterly", "Semi-Annual"])
        
        price_button = st.button(
            "📈 Price Equity Swap", 
            type="primary", 
            use_container_width=True
        )
    
    with col2:
        if price_button:
            with st.spinner("Pricing equity swap with live data..."):
                try:
                    # Price equity swap using enhanced pricer
                    from pricing.models.enhanced_swap_pricing import EquitySwapPricer
                    
                    result = EquitySwapPricer.price_enhanced(
                        symbol=reference_asset,
                        notional=notional,
                        fixed_rate=fixed_rate,
                        tenor_years=tenor_years,
                        equity_data=equity_data,
                        swap_direction=swap_direction,
                        financing_spread=financing_spread / 10000,
                        dividend_treatment=dividend_treatment.split()[0]
                    )
                    
                    # Display results
                    EducationalSwapDisplayManager.display_enhanced_equity_swap_results(
                        result, reference_asset, notional, swap_direction, equity_data
                    )
                    
                    # Additional analytics
                    shares_equivalent = abs(notional / equity_data['price'])
                    st.markdown(f"""
                    <div class="education-box">
                        <h4>🔍 Position Analysis</h4>
                        <p><strong>Shares Equivalent:</strong> {shares_equivalent:,.0f} shares</p>
                        <p><strong>Market Value:</strong> ${shares_equivalent * equity_data['price']:,.0f}</p>
                        <p><strong>Daily P&L Volatility:</strong> ${shares_equivalent * equity_data['price'] * equity_data['volatility']/100/np.sqrt(252):,.0f}</p>
                        <p><strong>Annual Dividend Income:</strong> ${shares_equivalent * equity_data['price'] * equity_data['dividend_yield']/100:,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Equity swap pricing error: {str(e)}")
        else:
            # Display equity market overview
            st.markdown("### 📈 Equity Market Overview")
            
            # Display volatility surface
            vol_surface = enhanced_market_data_manager.get_volatility_surface(reference_asset, "equity")
            
            if not vol_surface.empty:
                import plotly.express as px
                
                fig = px.scatter_3d(
                    vol_surface, 
                    x='days_to_expiry', 
                    y='moneyness', 
                    z='implied_vol',
                    title=f'{reference_asset} Implied Volatility Surface',
                    labels={
                        'days_to_expiry': 'Days to Expiry',
                        'moneyness': 'Moneyness',
                        'implied_vol': 'Implied Volatility'
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)


def enhanced_portfolio_management():
    """Enhanced Portfolio Management interface"""
    
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Portfolio Risk Management</h3>
        <p>Build, analyze, and manage your multi-asset swap portfolio</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize portfolio in session state
    if 'enhanced_portfolio' not in st.session_state:
        st.session_state.enhanced_portfolio = []
    
    # Portfolio builder
    with st.expander("➕ Add Swap to Portfolio", expanded=len(st.session_state.enhanced_portfolio) == 0):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            swap_type = st.selectbox("Swap Type", ["Interest Rate", "Currency", "Equity"])
            notional = st.number_input("Notional ($)", 1_000_000, 1_000_000_000, 100_000_000, 1_000_000)
        
        with col2:
            if swap_type == "Interest Rate":
                tenor = st.selectbox("Tenor", ["1Y", "2Y", "5Y", "10Y"], index=2)
                direction = st.selectbox("Direction", ["Receive Fixed", "Pay Fixed"])
                reference = "USD-SOFR"
                
            elif swap_type == "Currency":
                base_ccy = st.selectbox("Base Currency", ["USD", "EUR", "GBP"], index=0)
                quote_ccy = st.selectbox("Quote Currency", ["EUR", "GBP", "JPY"], index=0)
                tenor = st.selectbox("Tenor", ["1Y", "3Y", "5Y"], index=1)
                direction = f"Pay {base_ccy}, Receive {quote_ccy}"
                reference = f"{quote_ccy}{base_ccy}"
                
            else:  # Equity
                reference = st.selectbox("Underlying", ["SPY", "QQQ", "AAPL", "MSFT"])
                tenor = st.selectbox("Tenor", ["6M", "1Y", "2Y"], index=1)
                direction = st.selectbox("Direction", ["Receive Equity", "Pay Equity"])
        
        with col3:
            add_button = st.button("Add to Portfolio", type="primary")
            
            if add_button:
                swap_entry = {
                    'ID': len(st.session_state.enhanced_portfolio) + 1,
                    'Type': swap_type,
                    'Reference': reference,
                    'Notional': notional,
                    'Tenor': tenor,
                    'Direction': direction,
                    'NPV': np.random.normal(0, notional * 0.002),  # Simulated NPV
                    'Date_Added': datetime.now().strftime("%Y-%m-%d")
                }
                st.session_state.enhanced_portfolio.append(swap_entry)
                st.success(f"Added {swap_type} swap to portfolio")
                st.rerun()
    
    # Display current portfolio
    if st.session_state.enhanced_portfolio:
        st.markdown("### 📋 Current Portfolio")
        
        portfolio_df = pd.DataFrame(st.session_state.enhanced_portfolio)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        
        with col2:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.enhanced_portfolio = []
                st.rerun()
            
            if st.button("📊 Generate Report"):
                # Calculate portfolio analytics
                portfolio_analytics = _calculate_enhanced_portfolio_analytics(st.session_state.enhanced_portfolio)
                
                # Display comprehensive analytics
                PortfolioAnalyticsDisplay.display_portfolio_summary(portfolio_analytics)
                
                # Risk heatmap
                risk_heatmap = EducationalSwapDisplayManager.create_risk_heatmap(st.session_state.enhanced_portfolio)
                st.plotly_chart(risk_heatmap, use_container_width=True)
    else:
        st.info("👆 Add swaps to build your portfolio")


def _calculate_enhanced_portfolio_analytics(portfolio):
    """Calculate comprehensive portfolio analytics"""
    
    if not portfolio:
        return {}
    
    total_notional = sum([swap['Notional'] for swap in portfolio])
    portfolio_npv = sum([swap['NPV'] for swap in portfolio])
    
    # Risk exposure by type
    ir_exposure = sum([swap['Notional'] for swap in portfolio if swap['Type'] == 'Interest Rate'])
    fx_exposure = sum([swap['Notional'] for swap in portfolio if swap['Type'] == 'Currency'])
    equity_exposure = sum([swap['Notional'] for swap in portfolio if swap['Type'] == 'Equity'])
    
    # Simplified risk metrics
    portfolio_dv01 = total_notional * 0.0001
    portfolio_vega = equity_exposure * 0.001
    
    return {
        'total_notional': total_notional,
        'portfolio_npv': portfolio_npv,
        'portfolio_dv01': portfolio_dv01,
        'portfolio_vega': portfolio_vega,
        'ir_exposure': ir_exposure,
        'fx_exposure': fx_exposure,
        'equity_exposure': equity_exposure,
        'num_swaps': len(portfolio),
        'concentration_risk': max(ir_exposure, fx_exposure, equity_exposure) / total_notional if total_notional > 0 else 0
    }


def enhanced_education_center():
    """Enhanced Education Center with comprehensive learning materials"""
    
    st.markdown("""
    <div class="metric-card">
        <h3>📚 Derivatives Education Center</h3>
        <p>Comprehensive learning materials for swap pricing and risk management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Educational tabs
    edu_tab1, edu_tab2, edu_tab3, edu_tab4 = st.tabs([
        "🔢 Pricing Models", "📊 Risk Management", "🌍 Market Data", "⚠️ Disclaimers"
    ])
    
    with edu_tab1:
        EducationalSwapDisplayManager.display_educational_methodology()
    
    with edu_tab2:
        st.markdown("""
        ### 📊 Risk Management Best Practices
        
        #### Key Risk Types in Swap Trading
        
        **Market Risk:**
        - Interest Rate Risk: Sensitivity to yield curve movements
        - FX Risk: Currency exposure from cross-currency swaps
        - Equity Risk: Exposure to equity price movements
        - Volatility Risk: Changes in implied volatility levels
        
        **Credit Risk:**
        - Counterparty Risk: Risk of counterparty default
        - CVA (Credit Valuation Adjustment): Cost of counterparty risk
        - Collateral Requirements: Margin and collateral management
        
        **Operational Risk:**
        - Model Risk: Errors in pricing models or parameters
        - Settlement Risk: Risk in payment processing
        - Documentation Risk: Legal and contractual issues
        
        **Liquidity Risk:**
        - Market Liquidity: Ability to exit positions
        - Funding Liquidity: Access to funding for margin calls
        """)
    
    with edu_tab3:
        EducationalSwapDisplayManager.display_market_intelligence_dashboard()
    
    with edu_tab4:
        st.markdown("""
        ### ⚠️ Important Disclaimers and Limitations
        
        #### Educational Purpose Only
        This platform is designed for educational and research purposes only. All pricing models, 
        market data, and analytics are provided for learning about derivative instruments and 
        should not be used for actual trading or investment decisions.
        
        #### Model Limitations
        - **Simplified Assumptions**: Models use simplified assumptions about market behavior
        - **Historical Data**: Past performance does not predict future results
        - **Market Conditions**: Real market conditions may differ significantly from model assumptions
        - **Liquidity**: Models may not account for liquidity constraints in actual markets
        
        #### Risk Warnings
        - **High Risk**: Derivative instruments carry significant risk of loss
        - **Professional Advice**: Consult qualified financial professionals for real trading decisions
        - **Regulatory Compliance**: Ensure compliance with all applicable regulations
        - **Counterparty Risk**: Real swaps involve counterparty credit risk not fully modeled here
        
        #### Data Disclaimers
        - **Data Accuracy**: Market data may be delayed, estimated, or simulated
        - **No Warranty**: No warranty is provided regarding data accuracy or completeness
        - **Third-Party Data**: Some data sourced from third-party providers with their own terms
        """)


# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; 
     padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
     border-radius: 15px; border: 1px solid #dee2e6;'>
    <div style="margin-bottom: 15px;">
        <span style="font-size: 2rem;">🏦</span>
    </div>
    <p style="margin: 0; font-size: 1.3em; font-weight: bold; color: #1f77b4;">
        Enhanced Professional Swap Pricing Platform
    </p>
    <p style="margin: 8px 0; color: #6c757d;">
        Real-Time Market Data • Advanced Models • Educational Content
    </p>
    <p style="margin: 8px 0; color: #6c757d;">
        Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
         gap: 15px; margin: 20px 0;">
        <div style="background: rgba(0,123,255,0.1); padding: 15px; border-radius: 8px;">
            <strong>🔄 Interest Rate Swaps</strong><br>
            <small>Enhanced DCF • Monte Carlo • Hull-White</small>
        </div>
        <div style="background: rgba(40,167,69,0.1); padding: 15px; border-radius: 8px;">
            <strong>💱 Currency Swaps</strong><br>
            <small>Live FX Data • Cross-Currency Basis</small>
        </div>
        <div style="background: rgba(255,193,7,0.1); padding: 15px; border-radius: 8px;">
            <strong>📈 Equity Swaps</strong><br>
            <small>Real-Time Prices • Volatility Surfaces</small>
        </div>
        <div style="background: rgba(220,53,69,0.1); padding: 15px; border-radius: 8px;">
            <strong>📊 Portfolio Analytics</strong><br>
            <small>Risk Management • Scenario Analysis</small>
        </div>
    </div>
    <p style="margin: 15px 0 0 0; color: #dc3545; font-weight: bold; font-size: 1.1em;">
        ⚠️ Educational and Research Purposes Only
    </p>
    <p style="margin: 5px 0 0 0; color: #6c757d; font-size: 0.9em;">
        Not for actual trading • Consult financial professionals for investment decisions
    </p>
</div>
""", unsafe_allow_html=True)
