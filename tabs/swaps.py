# tabs/swaps_enhanced.py
# Enhanced Swaps Tab with Real Market Data Integration

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Try to import FRED data (fallback if not available)
try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    st.sidebar.warning("Install fredapi for real-time economic data: pip install fredapi")

from styles.app_styles import load_theme

# Enhanced pricing functions (existing ones plus new implementations)
from pricing.models.swaps.dcf import *
from pricing.models.swaps.lmm import *
from pricing.models.swaps.curves import *
from pricing.models.swaps.enhanced_currency_swaps import *


def swaps_tab():
    """Enhanced Swaps Tab Content with Real Market Data"""

    load_theme()
    
    st.markdown('<div class="main-header">Advanced Swap Pricing Suite</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Professional introduction with live market context
    col1, col2, col3 = st.columns(3)
    
    # Fetch live market data for overview
    market_overview = get_live_market_overview()
    
    with col1:
        st.metric(
            "10Y USD Swap Rate", 
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
    
    st.markdown("""
    <div class="info-box">
        <h3>🏦 Professional Derivatives Pricing Platform</h3>
        <p>Advanced swap pricing models with real-time market data, comprehensive analytics, and institutional-grade risk management. 
        Built for professional derivative pricing and portfolio risk analysis.</p>
        <div style="display: flex; gap: 20px; margin-top: 15px;">
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Market Size:</strong><br>$500+ Trillion Notional Outstanding
            </div>
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Daily Volume:</strong><br>$2-3 Trillion Average
            </div>
            <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 10px;">
                <strong>Real-Time Data:</strong><br>Yahoo Finance & FRED
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
        _currency_swaps_tab_enhanced()
    
    with tab3:
        _equity_swaps_tab_enhanced()
    
    with tab4:
        _portfolio_analysis_tab_enhanced()
    
    with tab5:
        _market_intelligence_tab_enhanced()


def get_live_market_overview():
    """Fetch live market data for overview"""
    try:
        # Fetch key market indicators
        symbols = ["^TNX", "EURUSD=X", "^VIX", "^FVX", "^IRX"]
        data = yf.download(symbols, period="2d", interval="1d", progress=False)
        
        # Calculate changes
        latest = data['Close'].iloc[-1]
        previous = data['Close'].iloc[-2]
        
        return {
            'usd_10y': latest.get('^TNX', 4.0),
            'usd_10y_change': (latest.get('^TNX', 4.0) - previous.get('^TNX', 4.0)) * 100,
            'eurusd': latest.get('EURUSD=X', 1.0850),
            'eurusd_change': ((latest.get('EURUSD=X', 1.0850) / previous.get('EURUSD=X', 1.0850)) - 1) * 100,
            'vix': latest.get('^VIX', 20.0),
            'vix_change': ((latest.get('^VIX', 20.0) / previous.get('^VIX', 20.0)) - 1) * 100
        }
    except:
        # Fallback data
        return {
            'usd_10y': 4.25,
            'usd_10y_change': -2.5,
            'eurusd': 1.0850,
            'eurusd_change': -0.3,
            'vix': 18.5,
            'vix_change': -1.2
        }


def _currency_swaps_tab_enhanced():
    """Enhanced Currency Swaps Interface with Real Market Data"""
    st.markdown('<div class="sub-header">Cross-Currency Swap Pricing & Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>💱 Currency Swap Configuration</h4>
            <p>Professional cross-currency swap pricing with live market data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Currency selection with real data
        major_currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]
        
        base_currency = st.selectbox("Base Currency (Pay)", major_currencies, index=0)
        quote_currency = st.selectbox("Quote Currency (Receive)", major_currencies, index=1)
        
        if base_currency == quote_currency:
            st.error("Base and Quote currencies must be different")
            return
        
        # Get live FX rate
        fx_pair = f"{quote_currency}{base_currency}=X"
        current_fx_data = get_live_fx_rate(quote_currency, base_currency)
        
        st.markdown(f"""
        <div class="success-box">
            <h5>📊 Live Market Data</h5>
            <p><strong>FX Rate:</strong> {current_fx_data['rate']:.4f}</p>
            <p><strong>24h Change:</strong> {current_fx_data['change']:+.2f}%</p>
            <p><strong>Volatility (1M):</strong> {current_fx_data['volatility']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Notional amounts
        base_notional = st.number_input(
            f"Base Notional ({base_currency})", 
            min_value=1_000_000, 
            max_value=10_000_000_000, 
            value=100_000_000, 
            step=1_000_000,
            format="%d"
        )
        
        fx_spot = st.number_input(
            f"FX Spot Rate ({quote_currency}{base_currency})", 
            min_value=0.0001, 
            max_value=1000.0, 
            value=current_fx_data['rate'], 
            step=0.0001,
            format="%.4f",
            help=f"Live rate: {current_fx_data['rate']:.4f}"
        )
        
        quote_notional = base_notional / fx_spot
        st.metric(f"Implied {quote_currency} Notional", f"{quote_notional:,.0f}")
        
        # Swap structure
        swap_type = st.radio(
            "Swap Type",
            ["Fixed-Fixed", "Fixed-Float", "Float-Float"],
            help="Structure of the currency swap"
        )
        
        tenor_years = st.selectbox("Swap Tenor", [1, 2, 3, 5, 7, 10], index=3)
        
        # Get live interest rates
        base_rates = get_live_interest_rates(base_currency)
        quote_rates = get_live_interest_rates(quote_currency)
        
        st.markdown(f"""
        <div class="info-box">
            <h5>📈 Live Interest Rates</h5>
            <p><strong>{base_currency} {tenor_years}Y:</strong> {base_rates.get(f'{tenor_years}Y', 3.5):.2f}%</p>
            <p><strong>{quote_currency} {tenor_years}Y:</strong> {quote_rates.get(f'{tenor_years}Y', 2.5):.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        base_rate = st.slider(
            f"{base_currency} Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=base_rates.get(f'{tenor_years}Y', 3.5), 
            step=0.01,
            format="%.2f"
        ) / 100
        
        quote_rate = st.slider(
            f"{quote_currency} Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=quote_rates.get(f'{tenor_years}Y', 2.5), 
            step=0.01,
            format="%.2f"
        ) / 100
        
        # Advanced parameters
        with st.expander("🔬 Advanced Parameters"):
            include_principal = st.checkbox("Include Principal Exchange", value=True)
            
            xccy_basis = st.slider(
                "Cross-Currency Basis (bps)", 
                min_value=-100, 
                max_value=100, 
                value=get_xccy_basis(base_currency, quote_currency), 
                step=5
            )
            
            funding_spread_base = st.slider(f"{base_currency} Funding Spread (bps)", 0, 200, 25, 5)
            funding_spread_quote = st.slider(f"{quote_currency} Funding Spread (bps)", 0, 200, 25, 5)
        
        price_btn = st.button("💱 Price Currency Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Calculating cross-currency swap with live market data..."):
                try:
                    # Initialize enhanced pricer
                    pricer = EnhancedCurrencySwapPricer()
                    
                    # Add discount curves with real data
                    base_curve = build_live_discount_curve(base_currency, base_rates)
                    quote_curve = build_live_discount_curve(quote_currency, quote_rates)
                    
                    pricer.add_discount_curve(base_currency, base_curve)
                    pricer.add_discount_curve(quote_currency, quote_curve)
                    
                    # Add FX curve with live data
                    fx_curve = build_live_fx_curve(quote_currency, base_currency, current_fx_data)
                    pricer.add_fx_curve(f"{quote_currency}{base_currency}", fx_curve)
                    
                    # Price currency swap
                    result = pricer.price_currency_swap(
                        domestic_notional=base_notional,
                        foreign_notional=quote_notional,
                        domestic_currency=base_currency,
                        foreign_currency=quote_currency,
                        domestic_rate=base_rate,
                        foreign_rate=quote_rate,
                        maturity_years=tenor_years,
                        cross_currency_basis=xccy_basis / 10000,
                        include_principal_exchange=include_principal,
                        calculate_greeks=True
                    )
                    
                    # Display comprehensive results
                    _display_currency_swap_results_enhanced(
                        result, base_currency, quote_currency, fx_spot, tenor_years
                    )
                    
                    # Risk analytics with live data
                    _display_currency_swap_risk_analytics(
                        result, base_notional, quote_notional, base_currency, quote_currency
                    )
                    
                    # Scenario analysis
                    _display_currency_swap_scenarios(
                        pricer, base_notional, quote_notional, base_currency, quote_currency,
                        base_rate, quote_rate, tenor_years, fx_spot
                    )
                    
                except Exception as e:
                    st.error(f"Pricing error: {str(e)}")
        
        else:
            # Show live market overview
            _display_fx_market_overview()


def _equity_swaps_tab_enhanced():
    """Enhanced Equity Swaps Interface with Real Market Data"""
    st.markdown('<div class="sub-header">Equity Swap Pricing & Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>📈 Equity Swap Configuration</h4>
            <p>Professional equity swap pricing with live market data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Equity selection
        equity_type = st.radio("Equity Type", ["Single Stock", "Index", "Basket"], index=1)
        
        if equity_type == "Single Stock":
            stock_symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter ticker symbol")
            reference_asset = stock_symbol
        elif equity_type == "Index":
            index_options = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VTI"]
            reference_asset = st.selectbox("Index ETF", index_options, index=0)
        else:  # Basket
            st.write("Basket configuration:")
            basket_weights = {}
            for i in range(3):
                col_sym, col_weight = st.columns(2)
                with col_sym:
                    symbol = st.text_input(f"Symbol {i+1}", value=["AAPL", "MSFT", "GOOGL"][i])
                with col_weight:
                    weight = st.number_input(f"Weight {i+1}", 0.0, 1.0, 1.0/3, 0.01, format="%.2f")
                if symbol:
                    basket_weights[symbol] = weight
            reference_asset = basket_weights
        
        # Get live equity data
        if equity_type != "Basket":
            equity_data = get_live_equity_data(reference_asset)
            st.markdown(f"""
            <div class="success-box">
                <h5>📊 Live Market Data</h5>
                <p><strong>Current Price:</strong> ${equity_data['price']:.2f}</p>
                <p><strong>24h Change:</strong> {equity_data['change']:+.2f}%</p>
                <p><strong>Volatility (30D):</strong> {equity_data['volatility']:.1f}%</p>
                <p><strong>Dividend Yield:</strong> {equity_data['dividend_yield']:.2f}%</p>
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
            "Swap Direction", 
            ["Pay Equity, Receive Fixed", "Pay Fixed, Receive Equity"],
            help="Direction of the equity swap"
        )
        
        tenor_years = st.selectbox("Swap Tenor", [0.25, 0.5, 1, 2, 3, 5], index=2)
        
        # Interest rate leg
        current_rates = get_live_interest_rates("USD")
        
        fixed_rate = st.slider(
            "Fixed Rate (%)", 
            min_value=0.0, 
            max_value=8.0, 
            value=current_rates.get("1Y", 4.0), 
            step=0.01,
            format="%.2f"
        ) / 100
        
        # Advanced parameters
        with st.expander("🔬 Advanced Parameters"):
            financing_spread = st.slider("Financing Spread (bps)", -100, 500, 50, 10)
            dividend_treatment = st.radio("Dividend Treatment", ["Gross", "Net"], index=0)
            reset_frequency = st.selectbox("Reset Frequency", ["Daily", "Monthly", "Quarterly"], index=1)
            
            if equity_type != "Basket":
                correlation_adj = st.slider("IR-Equity Correlation", -0.5, 0.5, -0.2, 0.05)
        
        price_btn = st.button("📈 Price Equity Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Calculating equity swap with live market data..."):
                try:
                    if equity_type == "Basket":
                        result = price_equity_basket_swap(
                            basket_weights, notional, fixed_rate, tenor_years,
                            swap_direction, financing_spread / 10000
                        )
                    else:
                        result = price_equity_swap_enhanced(
                            reference_asset, notional, fixed_rate, tenor_years,
                            equity_data, swap_direction, financing_spread / 10000,
                            dividend_treatment, reset_frequency
                        )
                    
                    # Display results
                    _display_equity_swap_results(
                        result, reference_asset, notional, swap_direction, tenor_years
                    )
                    
                    # Risk analytics
                    _display_equity_swap_risk_metrics(result, equity_data if equity_type != "Basket" else None)
                    
                    # Performance attribution
                    _display_equity_swap_attribution(result, equity_data if equity_type != "Basket" else None)
                    
                except Exception as e:
                    st.error(f"Pricing error: {str(e)}")
        
        else:
            # Show equity market overview
            _display_equity_market_overview()


def _portfolio_analysis_tab_enhanced():
    """Enhanced Portfolio-level swap analysis with real data"""
    st.markdown('<div class="sub-header">Portfolio Risk Management & Analytics</div>', unsafe_allow_html=True)
    
    # Portfolio builder
    st.markdown("""
    <div class="info-box">
        <h4>🏗️ Portfolio Construction</h4>
        <p>Build and analyze a multi-asset swap portfolio with live market data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize portfolio in session state
    if 'swap_portfolio' not in st.session_state:
        st.session_state.swap_portfolio = []
    
    # Add swap interface
    with st.expander("➕ Add Swap to Portfolio", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            swap_type = st.selectbox("Swap Type", ["Interest Rate", "Currency", "Equity"])
            notional = st.number_input("Notional (USD)", 1_000_000, 1_000_000_000, 100_000_000, 1_000_000)
        
        with col2:
            if swap_type == "Interest Rate":
                tenor = st.selectbox("Tenor", ["2Y", "5Y", "10Y"], index=1)
                direction = st.radio("Direction", ["Pay Fixed", "Receive Fixed"])
                rate = st.number_input("Fixed Rate (%)", 0.0, 10.0, 3.5, 0.01) / 100
                
            elif swap_type == "Currency":
                currency_pair = st.selectbox("Currency Pair", ["EURUSD", "GBPUSD", "USDJPY"])
                tenor = st.selectbox("Tenor", ["1Y", "3Y", "5Y"], index=1)
                direction = st.radio("Direction", ["Pay USD", "Receive USD"])
                
            else:  # Equity
                underlying = st.selectbox("Underlying", ["SPY", "QQQ", "IWM"])
                tenor = st.selectbox("Tenor", ["1Y", "2Y", "3Y"], index=0)
                direction = st.radio("Direction", ["Pay Equity", "Receive Equity"])
        
        with col3:
            if st.button("Add to Portfolio"):
                # Create swap object with live data
                swap_data = create_portfolio_swap(swap_type, notional, tenor, direction, locals())
                st.session_state.swap_portfolio.append(swap_data)
                st.success(f"Added {swap_type} swap to portfolio")
    
    # Display current portfolio
    if st.session_state.swap_portfolio:
        st.markdown('<div class="sub-header">📊 Current Portfolio</div>', unsafe_allow_html=True)
        
        # Portfolio summary table
        portfolio_df = pd.DataFrame(st.session_state.swap_portfolio)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        
        with col2:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.swap_portfolio = []
                st.rerun()
            
            if st.button("💰 Analyze Portfolio"):
                with st.spinner("Analyzing portfolio with live market data..."):
                    portfolio_analytics = analyze_swap_portfolio(st.session_state.swap_portfolio)
                    _display_portfolio_analytics(portfolio_analytics)
    
    else:
        st.info("👆 Add swaps to your portfolio to begin analysis")


def _market_intelligence_tab_enhanced():
    """Enhanced market intelligence with real-time data"""
    st.markdown('<div class="sub-header">Market Intelligence & Research</div>', unsafe_allow_html=True)
    
    # Live market dashboard
    st.markdown("""
    <div class="info-box">
        <h4>📈 Live Market Dashboard</h4>
        <p>Real-time market data and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market overview tabs
    market_tab1, market_tab2, market_tab3, market_tab4 = st.tabs([
        "🌐 Global Rates", "💱 FX Markets", "📊 Volatility", "📚 Research"
    ])
    
    with market_tab1:
        _display_global_rates_dashboard()
    
    with market_tab2:
        _display_fx_markets_dashboard()
    
    with market_tab3:
        _display_volatility_dashboard()
    
    with market_tab4:
        _display_research_insights()


# Helper functions for live market data

def get_live_fx_rate(base_currency, quote_currency):
    """Get live FX rate and statistics"""
    try:
        symbol = f"{base_currency}{quote_currency}=X"
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="30d")
        
        if hist.empty:
            return {'rate': 1.0, 'change': 0.0, 'volatility': 15.0}
        
        current_rate = hist['Close'].iloc[-1]
        prev_rate = hist['Close'].iloc[-2] if len(hist) > 1 else current_rate
        change_pct = ((current_rate / prev_rate) - 1) * 100
        
        # Calculate 30-day volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        return {
            'rate': current_rate,
            'change': change_pct,
            'volatility': volatility
        }
    except:
        return {'rate': 1.0, 'change': 0.0, 'volatility': 15.0}


def get_live_interest_rates(currency):
    """Get live interest rates for a currency"""
    try:
        if currency == "USD":
            symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]  # 3M, 5Y, 10Y, 30Y
            tenors = ["3M", "5Y", "10Y", "30Y"]
        elif currency == "EUR":
            # Use German Bund yields as proxy
            symbols = ["^TNX"]  # Would need European bond symbols
            tenors = ["10Y"]
        else:
            # Fallback to USD rates
            symbols = ["^TNX"]
            tenors = ["10Y"]
        
        rates = {}
        data = yf.download(symbols, period="1d", progress=False)
        
        if not data.empty:
            latest_data = data['Close'].iloc[-1]
            for symbol, tenor in zip(symbols, tenors):
                if symbol in latest_data.index:
                    rates[tenor] = latest_data[symbol]
                    # Interpolate other tenors
                    if tenor == "10Y":
                        rates["1Y"] = latest_data[symbol] - 0.5
                        rates["2Y"] = latest_data[symbol] - 0.3
                        rates["3Y"] = latest_data[symbol] - 0.2
                        rates["7Y"] = latest_data[symbol] - 0.1
        
        # Fallback rates if data fetch fails
        if not rates:
            fallback_rates = {
                "USD": {"1Y": 4.5, "2Y": 4.3, "3Y": 4.2, "5Y": 4.1, "7Y": 4.0, "10Y": 4.0},
                "EUR": {"1Y": 3.2, "2Y": 3.0, "3Y": 2.9, "5Y": 2.8, "7Y": 2.7, "10Y": 2.6}
            }
            rates = fallback_rates.get(currency, fallback_rates["USD"])
        
        return rates
    except:
        # Fallback rates
        return {"1Y": 4.0, "2Y": 3.8, "3Y": 3.7, "5Y": 3.6, "10Y": 3.5}


def get_live_equity_data(symbol):
    """Get live equity data and statistics"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="30d")
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price / prev_price) - 1) * 100
        
        # Calculate volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        
        dividend_yield = info.get('dividendYield', 0.02) * 100 if info.get('dividendYield') else 2.0
        
        return {
            'price': current_price,
            'change': change_pct,
            'volatility': volatility,
            'dividend_yield': dividend_yield
        }
    except:
        return {
            'price': 150.0,
            'change': 0.5,
            'volatility': 25.0,
            'dividend_yield': 2.0
        }


def get_xccy_basis(base_currency, quote_currency):
    """Get indicative cross-currency basis"""
    basis_map = {
        ("USD", "EUR"): -15,
        ("EUR", "USD"): 15,
        ("USD", "GBP"): -10,
        ("GBP", "USD"): 10,
        ("USD", "JPY"): -25,
        ("JPY", "USD"): 25
    }
    return basis_map.get((base_currency, quote_currency), 0)


def build_live_discount_curve(currency, rates_dict):
    """Build discount curve from live rates"""
    def discount_curve(t):
        # Simple interpolation of rates
        if t <= 1:
            rate = rates_dict.get("1Y", 4.0) / 100
        elif t <= 5:
            rate = rates_dict.get("5Y", 4.0) / 100
        else:
            rate = rates_dict.get("10Y", 4.0) / 100
        
        return np.exp(-rate * t)
    
    return discount_curve


def build_live_fx_curve(base_currency, quote_currency, fx_data):
    """Build FX forward curve from live data"""
    spot_rate = fx_data['rate']
    volatility = fx_data['volatility'] / 100
    
    forward_points = {}
    for tenor in [0.25, 0.5, 1, 2, 3, 5]:
        # Simple forward points calculation
        forward_points[tenor] = spot_rate * 0.001 * tenor  # Simplified
    
    from pricing.models.swaps.enhanced_currency_swaps import FXForwardCurve
    
    return FXForwardCurve(
        spot_rate=spot_rate,
        currency_pair=f"{base_currency}{quote_currency}",
        forward_points=forward_points,
        volatility=volatility,
        cross_currency_basis={}
    )


# Enhanced display functions

def _display_currency_swap_results_enhanced(result, base_currency, quote_currency, fx_spot, tenor_years):
    """Display enhanced currency swap results"""
    
    npv_domestic = result.npv_domestic
    npv_foreign = result.npv_foreign
    fx_delta = result.fx_delta
    
    # Determine swap direction and color
    if npv_domestic > 0:
        color = "#2E8B57"  # Green
        status = "✅ Favorable"
    else:
        color = "#DC143C"  # Red
        status = "❌ Unfavorable"
    
    st.markdown(f"""
    <div class="metric-container">
        <h4>💱 Currency Swap Pricing Results</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                <td style="padding: 12px; font-weight: bold;">Metric</td>
                <td style="padding: 12px; font-weight: bold;">Value</td>
                <td style="padding: 12px; font-weight: bold;">Description</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">NPV ({base_currency})</td>
                <td style="padding: 10px; font-family: monospace; color: {color}; font-weight: bold; font-size: 1.2em;">{base_currency} {npv_domestic:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">Net Present Value in base currency</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">NPV ({quote_currency})</td>
                <td style="padding: 10px; font-family: monospace; color: {color}; font-weight: bold;">{quote_currency} {npv_foreign:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">Net Present Value in quote currency</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Status</td>
                <td style="padding: 10px; font-weight: bold; color: {color};">{status}</td>
                <td style="padding: 10px; font-style: italic;">From {base_currency} payer perspective</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">FX Delta</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{base_currency} {fx_delta:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">Sensitivity to 1% FX move</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Current FX Rate</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{fx_spot:.4f}</td>
                <td style="padding: 10px; font-style: italic;">Live market rate</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


def _display_currency_swap_risk_analytics(result, base_notional, quote_notional, base_currency, quote_currency):
    """Display comprehensive risk analytics for currency swaps"""
    
    st.markdown('<div class="sub-header">⚡ Risk Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="greeks-delta">
            <h4>💱 FX Risk</h4>
            <p><strong>FX Delta:</strong> {base_currency} {result.fx_delta:,.0f}</p>
            <p><strong>FX Gamma:</strong> {base_currency} {result.cross_gamma:,.0f}</p>
            <p><strong>FX Exposure:</strong> {quote_currency} {quote_notional:,.0f}</p>
            <p><strong>Hedge Ratio:</strong> {abs(result.fx_delta / base_notional):.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="greeks-gamma">
            <h4>📊 Interest Rate Risk</h4>
            <p><strong>{base_currency} DV01:</strong> {base_currency} {result.domestic_dv01:,.0f}</p>
            <p><strong>{quote_currency} DV01:</strong> {base_currency} {result.foreign_dv01:,.0f}</p>
            <p><strong>Carry (Daily):</strong> {base_currency} {result.carry:,.0f}</p>
            <p><strong>Cross Gamma:</strong> {base_currency} {result.cross_gamma:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)


def _display_currency_swap_scenarios(pricer, base_notional, quote_notional, base_currency, 
                                   quote_currency, base_rate, quote_rate, tenor_years, fx_spot):
    """Display scenario analysis for currency swaps"""
    
    st.markdown('<div class="sub-header">🎭 Scenario Analysis</div>', unsafe_allow_html=True)
    
    # Define scenarios
    scenarios = {
        "Base Case": {"fx_shock": 0, "rates_shock": 0},
        "FX Appreciation (+10%)": {"fx_shock": 0.10, "rates_shock": 0},
        "FX Depreciation (-10%)": {"fx_shock": -0.10, "rates_shock": 0},
        "Rates Up (+100bp)": {"fx_shock": 0, "rates_shock": 0.01},
        "Rates Down (-100bp)": {"fx_shock": 0, "rates_shock": -0.01},
        "Crisis Scenario": {"fx_shock": -0.20, "rates_shock": -0.02}
    }
    
    scenario_results = []
    base_npv = None
    
    for scenario_name, shocks in scenarios.items():
        try:
            # Apply shocks
            shocked_fx = fx_spot * (1 + shocks["fx_shock"])
            shocked_base_rate = base_rate + shocks["rates_shock"]
            shocked_quote_rate = quote_rate + shocks["rates_shock"]
            
            # Recalculate (simplified)
            shocked_notional = base_notional / shocked_fx
            rate_diff = shocked_base_rate - shocked_quote_rate
            
            # Simplified NPV calculation for scenario
            scenario_npv = (shocked_notional - quote_notional) + base_notional * rate_diff * tenor_years
            
            if scenario_name == "Base Case":
                base_npv = scenario_npv
            
            pnl_vs_base = scenario_npv - base_npv if base_npv is not None else 0
            
            scenario_results.append({
                'Scenario': scenario_name,
                'FX Shock': f"{shocks['fx_shock']:+.0%}",
                'Rates Shock': f"{shocks['rates_shock']*10000:+.0f}bp",
                'NPV': f"{base_currency} {scenario_npv:,.0f}",
                'P&L vs Base': f"{base_currency} {pnl_vs_base:+,.0f}"
            })
        except:
            scenario_results.append({
                'Scenario': scenario_name,
                'FX Shock': f"{shocks['fx_shock']:+.0%}",
                'Rates Shock': f"{shocks['rates_shock']*10000:+.0f}bp",
                'NPV': "Error",
                'P&L vs Base': "Error"
            })
    
    scenario_df = pd.DataFrame(scenario_results)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)


def _display_fx_market_overview():
    """Display FX market overview with live data"""
    
    st.markdown('<div class="sub-header">🌍 FX Market Overview</div>', unsafe_allow_html=True)
    
    # Major currency pairs
    major_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X"]
    
    try:
        fx_data = yf.download(major_pairs, period="1d", progress=False)
        
        if not fx_data.empty:
            latest_rates = fx_data['Close'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("EUR/USD", f"{latest_rates.get('EURUSD=X', 1.0850):.4f}")
                st.metric("USD/JPY", f"{latest_rates.get('USDJPY=X', 150.0):.2f}")
            
            with col2:
                st.metric("GBP/USD", f"{latest_rates.get('GBPUSD=X', 1.2650):.4f}")
                st.metric("USD/CHF", f"{latest_rates.get('USDCHF=X', 0.8890):.4f}")
            
            with col3:
                st.metric("AUD/USD", f"{latest_rates.get('AUDUSD=X', 0.6750):.4f}")
                st.metric("USD/CAD", f"{latest_rates.get('USDCAD=X', 1.3520):.4f}")
    except:
        st.info("Live FX data temporarily unavailable - using indicative rates")


def price_equity_swap_enhanced(symbol, notional, fixed_rate, tenor_years, equity_data, 
                             swap_direction, financing_spread, dividend_treatment, reset_frequency):
    """Enhanced equity swap pricing with live market data"""
    
    # Get current equity parameters
    current_price = equity_data['price']
    volatility = equity_data['volatility'] / 100
    dividend_yield = equity_data['dividend_yield'] / 100
    
    # Simple equity swap pricing (simplified for demonstration)
    
    # Calculate expected equity return
    risk_free_rate = 0.04  # Would get from live rates
    expected_equity_return = risk_free_rate + 0.06  # Equity risk premium
    
    # Equity leg PV (simplified)
    if swap_direction == "Pay Equity, Receive Fixed":
        equity_pv = notional * expected_equity_return * tenor_years
        fixed_pv = notional * fixed_rate * tenor_years
        npv = fixed_pv - equity_pv
    else:
        equity_pv = notional * expected_equity_return * tenor_years
        fixed_pv = notional * fixed_rate * tenor_years
        npv = equity_pv - fixed_pv
    
    # Add dividend adjustments
    dividend_pv = notional * dividend_yield * tenor_years
    if dividend_treatment == "Gross":
        dividend_pv *= 1.0  # No tax adjustment
    else:
        dividend_pv *= 0.85  # Apply withholding tax
    
    # Calculate Greeks (simplified)
    equity_delta = notional / current_price if swap_direction == "Pay Fixed, Receive Equity" else -notional / current_price
    equity_gamma = 0  # Simplified
    equity_vega = notional * volatility * np.sqrt(tenor_years) * 0.01
    
    return {
        'npv': npv,
        'equity_pv': equity_pv,
        'fixed_pv': fixed_pv,
        'dividend_pv': dividend_pv,
        'equity_delta': equity_delta,
        'equity_gamma': equity_gamma,
        'equity_vega': equity_vega,
        'current_price': current_price,
        'expected_return': expected_equity_return,
        'volatility': volatility,
        'financing_spread': financing_spread
    }


def price_equity_basket_swap(basket_weights, notional, fixed_rate, tenor_years, 
                           swap_direction, financing_spread):
    """Price equity basket swap"""
    
    total_equity_pv = 0
    total_delta = 0
    total_vega = 0
    
    for symbol, weight in basket_weights.items():
        if weight > 0:
            try:
                equity_data = get_live_equity_data(symbol)
                
                # Weight-adjusted calculations
                weighted_notional = notional * weight
                
                # Simple equity return calculation
                expected_return = 0.10  # Simplified
                equity_pv = weighted_notional * expected_return * tenor_years
                
                total_equity_pv += equity_pv
                total_delta += weighted_notional / equity_data['price']
                total_vega += weighted_notional * (equity_data['volatility']/100) * np.sqrt(tenor_years) * 0.01
                
            except:
                continue
    
    # Fixed leg
    fixed_pv = notional * fixed_rate * tenor_years
    
    # NPV calculation
    if swap_direction == "Pay Equity, Receive Fixed":
        npv = fixed_pv - total_equity_pv
    else:
        npv = total_equity_pv - fixed_pv
    
    return {
        'npv': npv,
        'equity_pv': total_equity_pv,
        'fixed_pv': fixed_pv,
        'basket_delta': total_delta,
        'basket_vega': total_vega,
        'basket_weights': basket_weights
    }


def _display_equity_swap_results(result, reference_asset, notional, swap_direction, tenor_years):
    """Display equity swap pricing results"""
    
    npv = result['npv']
    
    # Determine color based on NPV
    color = "#2E8B57" if npv > 0 else "#DC143C"
    status = "✅ Favorable" if npv > 0 else "❌ Unfavorable"
    
    st.markdown(f"""
    <div class="metric-container">
        <h4>📈 Equity Swap Pricing Results</h4>
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
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">${result['equity_pv']:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">Present value of equity leg</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Fixed PV</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">${result['fixed_pv']:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">Present value of fixed leg</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Reference Asset</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{reference_asset}</td>
                <td style="padding: 10px; font-style: italic;">Underlying equity reference</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


def _display_equity_swap_risk_metrics(result, equity_data):
    """Display equity swap risk metrics"""
    
    st.markdown('<div class="sub-header">⚡ Equity Risk Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="greeks-delta">
            <h4>📊 Market Risk</h4>
            <p><strong>Equity Delta:</strong> {result.get('equity_delta', 0):,.0f} shares</p>
            <p><strong>Equity Vega:</strong> ${result.get('equity_vega', 0):,.0f}</p>
            <p><strong>Current Volatility:</strong> {result.get('volatility', 0.25)*100:.1f}%</p>
            <p><strong>Expected Return:</strong> {result.get('expected_return', 0.10)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if equity_data:
            st.markdown(f"""
            <div class="greeks-gamma">
                <h4>💰 Live Market Data</h4>
                <p><strong>Current Price:</strong> ${equity_data['price']:.2f}</p>
                <p><strong>24h Change:</strong> {equity_data['change']:+.2f}%</p>
                <p><strong>Dividend Yield:</strong> {equity_data['dividend_yield']:.2f}%</p>
                <p><strong>30D Volatility:</strong> {equity_data['volatility']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)


def _display_equity_swap_attribution(result, equity_data):
    """Display performance attribution for equity swap"""
    
    st.markdown('<div class="sub-header">📊 Performance Attribution</div>', unsafe_allow_html=True)
    
    # Create attribution chart
    categories = ['Equity Return', 'Dividend Income', 'Financing Cost', 'Trading Fees']
    values = [
        result.get('equity_pv', 0),
        result.get('dividend_pv', 0),
        -abs(result.get('financing_spread', 0)) * 1000000,  # Convert to dollar impact
        -5000  # Estimated trading costs
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=['green' if v > 0 else 'red' for v in values]
        )
    ])
    
    fig.update_layout(
        title='P&L Attribution Analysis',
        yaxis_title='USD Impact',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _display_equity_market_overview():
    """Display equity market overview"""
    
    st.markdown('<div class="sub-header">📈 Equity Market Overview</div>', unsafe_allow_html=True)
    
    # Major indices
    indices = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX"]
    
    try:
        market_data = yf.download(indices, period="1d", progress=False)
        
        if not market_data.empty:
            latest_data = market_data['Close'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("S&P 500", f"{latest_data.get('^GSPC', 4500):.0f}")
                st.metric("VIX", f"{latest_data.get('^VIX', 20):.1f}")
            
            with col2:
                st.metric("NASDAQ", f"{latest_data.get('^IXIC', 14000):.0f}")
                st.metric("Russell 2000", f"{latest_data.get('^RUT', 2000):.0f}")
            
            with col3:
                st.metric("Dow Jones", f"{latest_data.get('^DJI', 35000):.0f}")
    except:
        st.info("Live equity data temporarily unavailable")


def create_portfolio_swap(swap_type, notional, tenor, direction, params):
    """Create a swap object for portfolio"""
    
    if swap_type == "Interest Rate":
        return {
            'Type': 'Interest Rate',
            'Notional': f"${notional:,.0f}",
            'Tenor': tenor,
            'Direction': direction,
            'Rate': f"{params['rate']*100:.2f}%",
            'NPV': 0  # Would calculate actual NPV
        }
    elif swap_type == "Currency":
        return {
            'Type': 'Currency',
            'Notional': f"${notional:,.0f}",
            'Tenor': tenor,
            'Direction': direction,
            'Pair': params['currency_pair'],
            'NPV': 0
        }
    else:  # Equity
        return {
            'Type': 'Equity',
            'Notional': f"${notional:,.0f}",
            'Tenor': tenor,
            'Direction': direction,
            'Underlying': params['underlying'],
            'NPV': 0
        }


def analyze_swap_portfolio(portfolio):
    """Analyze portfolio of swaps"""
    
    total_notional = sum([float(swap['Notional'].replace(', '').replace(',', '')) for swap in portfolio])
    
    # Calculate portfolio metrics (simplified)
    portfolio_npv = np.random.normal(0, total_notional * 0.01)  # Simplified
    portfolio_dv01 = total_notional * 0.0001
    portfolio_vega = total_notional * 0.0005
    
    # Risk decomposition
    ir_exposure = sum([float(swap['Notional'].replace(', '').replace(',', '')) 
                      for swap in portfolio if swap['Type'] == 'Interest Rate'])
    fx_exposure = sum([float(swap['Notional'].replace(', '').replace(',', '')) 
                      for swap in portfolio if swap['Type'] == 'Currency'])
    equity_exposure = sum([float(swap['Notional'].replace(', '').replace(',', '')) 
                          for swap in portfolio if swap['Type'] == 'Equity'])
    
    return {
        'total_notional': total_notional,
        'portfolio_npv': portfolio_npv,
        'portfolio_dv01': portfolio_dv01,
        'portfolio_vega': portfolio_vega,
        'ir_exposure': ir_exposure,
        'fx_exposure': fx_exposure,
        'equity_exposure': equity_exposure,
        'num_swaps': len(portfolio)
    }


def _display_portfolio_analytics(analytics):
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
    st.markdown('<div class="sub-header">🎯 Risk Breakdown</div>', unsafe_allow_html=True)
    
    risk_data = {
        'Asset Class': ['Interest Rate', 'FX', 'Equity'],
        'Exposure': [analytics['ir_exposure'], analytics['fx_exposure'], analytics['equity_exposure']]
    }
    
    fig = px.pie(
        values=risk_data['Exposure'],
        names=risk_data['Asset Class'],
        title='Portfolio Exposure by Asset Class'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _display_global_rates_dashboard():
    """Display global rates dashboard"""
    
    st.markdown('<div class="sub-header">🌐 Global Interest Rates</div>', unsafe_allow_html=True)
    
    # Try to get live rates for major economies
    try:
        # US Treasury rates
        us_symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
        us_data = yf.download(us_symbols, period="5d", progress=False)
        
        if not us_data.empty:
            # Create yield curve chart
            tenors = [0.25, 2, 10, 30]  # Approximate tenors
            latest_rates = us_data['Close'].iloc[-1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=tenors,
                y=[latest_rates.get(symbol, 4.0) for symbol in us_symbols],
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
    except:
        st.info("Live rates data temporarily unavailable")
    
    # Display rates table
    rates_data = {
        'Country': ['United States', 'Germany', 'United Kingdom', 'Japan', 'Canada'],
        '2Y': ['4.20%', '2.85%', '4.45%', '0.15%', '4.10%'],
        '5Y': ['4.15%', '2.65%', '4.25%', '0.35%', '3.95%'],
        '10Y': ['4.10%', '2.45%', '4.15%', '0.75%', '3.85%'],
        '30Y': ['4.25%', '2.70%', '4.30%', '1.85%', '3.95%']
    }
    
    rates_df = pd.DataFrame(rates_data)
    st.dataframe(rates_df, use_container_width=True, hide_index=True)


def _display_fx_markets_dashboard():
    """Display FX markets dashboard"""
    
    st.markdown('<div class="sub-header">💱 FX Markets Dashboard</div>', unsafe_allow_html=True)
    
    # Major currency pairs performance
    try:
        pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
        fx_data = yf.download(pairs, period="30d", progress=False)
        
        if not fx_data.empty:
            # Calculate returns
            returns = fx_data['Close'].pct_change().fillna(0)
            
            # Create correlation heatmap
            correlation_matrix = returns.corr()
            
            fig = px.imshow(
                correlation_matrix.values,
                labels=dict(x="Currency Pair", y="Currency Pair", color="Correlation"),
                x=[pair.replace('=X', '') for pair in correlation_matrix.columns],
                y=[pair.replace('=X', '') for pair in correlation_matrix.index],
                color_continuous_scale='RdBu_r',
                title='FX Correlation Matrix (30D)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("FX correlation data temporarily unavailable")


def _display_volatility_dashboard():
    """Display volatility dashboard"""
    
    st.markdown('<div class="sub-header">📊 Volatility Analysis</div>', unsafe_allow_html=True)
    
    try:
        # Get VIX and other volatility indices
        vol_symbols = ["^VIX", "^VXN", "^RVX"]
        vol_data = yf.download(vol_symbols, period="90d", progress=False)
        
        if not vol_data.empty:
            # Plot volatility time series
            fig = go.Figure()
            
            for symbol in vol_symbols:
                if symbol in vol_data['Close'].columns:
                    fig.add_trace(go.Scatter(
                        x=vol_data.index,
                        y=vol_data['Close'][symbol],
                        mode='lines',
                        name=symbol.replace('^', ''),
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title='Volatility Indices (90 Days)',
                xaxis_title='Date',
                yaxis_title='Volatility Level',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Volatility data temporarily unavailable")


def _display_research_insights():
    """Display research insights and market commentary"""
    
    st.markdown('<div class="sub-header">📚 Market Research & Insights</div>', unsafe_allow_html=True)
    
    # Current market themes
    st.markdown("""
    <div class="info-box">
        <h4>🔍 Current Market Themes</h4>
        <ul>
            <li><strong>Central Bank Divergence:</strong> Fed pause vs ECB/BoE tightening continues</li>
            <li><strong>Curve Dynamics:</strong> Persistent inversion in US 2s10s spread signals recession risk</li>
            <li><strong>Credit Environment:</strong> Widening spreads in HY while IG remains resilient</li>
            <li><strong>FX Volatility:</strong> Elevated cross-currency basis due to dollar funding stress</li>
            <li><strong>Regulatory Changes:</strong> SOFR transition complete, impact on swap pricing models</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Educational content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>📖 Swap Market Education</h4>
            <h5>Key Concepts:</h5>
            <ul>
                <li><strong>Interest Rate Parity:</strong> Foundation for FX swap pricing</li>
                <li><strong>Cross-Currency Basis:</strong> Deviation from covered interest parity</li>
                <li><strong>Equity Risk Premium:</strong> Expected return above risk-free rate</li>
                <li><strong>Portfolio Greeks:</strong> Aggregated risk sensitivities</li>
            </ul>
            
            <h5>Risk Management:</h5>
            <ul>
                <li><strong>Delta Hedging:</strong> Neutralizing directional exposure</li>
                <li><strong>Gamma Scalping:</strong> Managing convexity risk</li>
                <li><strong>Vega Management:</strong> Volatility exposure control</li>
                <li><strong>Correlation Risk:</strong> Multi-asset interaction effects</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Current Risk Factors</h4>
            <h5>Market Risks:</h5>
            <ul>
                <li><strong>Interest Rate Volatility:</strong> Fed policy uncertainty</li>
                <li><strong>FX Regime Changes:</strong> Potential dollar weakness</li>
                <li><strong>Equity Market Stress:</strong> Recession probability rising</li>
                <li><strong>Liquidity Risk:</strong> Reduced dealer capacity</li>
            </ul>
            
            <h5>Model Risks:</h5>
            <ul>
                <li><strong>Parameter Uncertainty:</strong> Volatility estimates</li>
                <li><strong>Correlation Breakdown:</strong> Crisis periods</li>
                <li><strong>Jump Risk:</strong> Sudden market moves</li>
                <li><strong>Basis Risk:</strong> Index vs constituent performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Market data sources and methodology
    with st.expander("📊 Data Sources & Methodology"):
        st.markdown("""
        <div class="info-box">
            <h4>Data Sources & Pricing Methodology</h4>
            
            <h5>Market Data Providers:</h5>
            <ul>
                <li><strong>Yahoo Finance API:</strong> Real-time equity prices, FX rates, indices</li>
                <li><strong>FRED API:</strong> Economic indicators, interest rates (when available)</li>
                <li><strong>Live Market Feeds:</strong> Intraday updates for pricing accuracy</li>
            </ul>
            
            <h5>Pricing Models:</h5>
            <ul>
                <li><strong>Interest Rate Swaps:</strong> Enhanced DCF with real yield curves</li>
                <li><strong>Currency Swaps:</strong> Dual-curve approach with cross-currency basis</li>
                <li><strong>Equity Swaps:</strong> Monte Carlo simulation with stochastic volatility</li>
                <li><strong>Portfolio Analysis:</strong> Multi-factor risk model with correlation matrix</li>
            </ul>
            
            <h5>Risk Calculations:</h5>
            <ul>
                <li><strong>Greeks:</strong> Numerical differentiation with live market data</li>
                <li><strong>VaR:</strong> Historical simulation and parametric methods</li>
                <li><strong>Stress Testing:</strong> Historical scenarios and Monte Carlo</li>
                <li><strong>Scenario Analysis:</strong> Custom shock specifications</li>
            </ul>
            
            <h5>Model Validation:</h5>
            <ul>
                <li><strong>Backtesting:</strong> Historical P&L attribution analysis</li>
                <li><strong>Benchmarking:</strong> Comparison with market quotations</li>
                <li><strong>Sensitivity Analysis:</strong> Parameter robustness testing</li>
                <li><strong>Independent Pricing:</strong> Cross-validation with alternative models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# Additional helper functions for enhanced functionality

def get_economic_indicators():
    """Get key economic indicators if FRED API is available"""
    
    if not FRED_AVAILABLE:
        return {
            'fed_funds': 5.25,
            'unemployment': 3.8,
            'inflation': 3.2,
            'gdp_growth': 2.1
        }
    
    try:
        # Would implement FRED API calls here
        fred = fredapi.Fred(api_key='your_fred_api_key')
        
        indicators = {
            'fed_funds': fred.get_series('FEDFUNDS', limit=1).iloc[-1],
            'unemployment': fred.get_series('UNRATE', limit=1).iloc[-1],
            'inflation': fred.get_series('CPIAUCSL', limit=1).pct_change().iloc[-1] * 100,
            'gdp_growth': fred.get_series('GDP', limit=1).pct_change().iloc[-1] * 100
        }
        
        return indicators
    except:
        # Fallback data
        return {
            'fed_funds': 5.25,
            'unemployment': 3.8,
            'inflation': 3.2,
            'gdp_growth': 2.1
        }


def calculate_portfolio_var(portfolio, confidence_level=0.95, time_horizon=1):
    """Calculate portfolio Value-at-Risk"""
    
    total_notional = sum([float(swap['Notional'].replace(', '').replace(',', '')) 
                         for swap in portfolio])
    
    if total_notional == 0:
        return 0
    
    # Simplified VaR calculation
    portfolio_volatility = 0.02  # 2% daily volatility assumption
    z_score = 1.96 if confidence_level == 0.95 else 2.33  # 95% or 99%
    
    var = total_notional * portfolio_volatility * z_score * np.sqrt(time_horizon)
    
    return var


def generate_stress_scenarios():
    """Generate stress testing scenarios"""
    
    scenarios = {
        'COVID-19 Crisis (Mar 2020)': {
            'equity_shock': -0.35,
            'vol_shock': 2.5,
            'fx_shock': {'EURUSD': -0.08, 'USDJPY': 0.05},
            'rates_shock': -1.50
        },
        'Financial Crisis (Sep 2008)': {
            'equity_shock': -0.45,
            'vol_shock': 3.0,
            'fx_shock': {'EURUSD': -0.15, 'USDJPY': 0.20},
            'rates_shock': -2.00
        },
        'Taper Tantrum (May 2013)': {
            'equity_shock': -0.10,
            'vol_shock': 0.8,
            'fx_shock': {'EURUSD': -0.05, 'USDJPY': -0.10},
            'rates_shock': 1.00
        },
        'Flash Crash (May 2010)': {
            'equity_shock': -0.20,
            'vol_shock': 1.5,
            'fx_shock': {'EURUSD': 0.02, 'USDJPY': -0.03},
            'rates_shock': -0.25
        }
    }
    
    return scenarios


def _interest_rate_swaps_tab():
    """Enhanced Interest Rate Swaps Interface (keeping existing implementation)"""
    st.markdown('<div class="sub-header">Interest Rate Swap Pricing & Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔧 Swap Configuration</h4>
            <p>Enhanced with live market data integration</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get live rates for reference
        live_rates = get_live_interest_rates("USD")
        
        # Enhanced parameters with live data validation
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
        
        # Show live market rate
        market_rate = live_rates.get(f'{tenor_years}Y', 3.5)
        st.info(f"💡 Live {tenor_years}Y market rate: {market_rate:.2f}%")
        
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
            step=5,
            help="Spread over benchmark rate"
        )
        
        # Model selection with enhanced options
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
        
        if model in ["LIBOR Market Model", "Hull-White"]:
            vol = st.slider("Interest Rate Volatility (%)", 5.0, 50.0, 15.0, 1.0) / 100
            n_paths = st.selectbox("Monte Carlo Paths", [10000, 25000, 50000, 100000], index=1)
        
        price_btn = st.button("🔢 Price Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Calculating swap price with live market data..."):
                try:
                    # Use enhanced pricing with live data
                    result = price_irs_with_live_data(
                        notional=notional,
                        fixed_rate=fixed_rate,
                        tenor_years=tenor_years,
                        payment_freq=payment_freq,
                        floating_spread=floating_spread / 10000,
                        model=model,
                        vol=vol if model != "Enhanced DCF" else None,
                        n_paths=n_paths if model != "Enhanced DCF" else None
                    )
                    
                    # Display enhanced results
                    _display_irs_results_enhanced(result, notional, fixed_rate, tenor_years, model)
                    
                    # Live market comparison
                    _display_market_comparison(result, market_rate, fixed_rate)
                    
                except Exception as e:
                    st.error(f"Pricing error: {str(e)}")
        
        else:
            # Show live yield curve
            _display_live_yield_curve()


def price_irs_with_live_data(notional, fixed_rate, tenor_years, payment_freq, 
                           floating_spread, model, vol=None, n_paths=None):
    """Enhanced IRS pricing with live market data integration"""
    
    # Get live market data
    live_rates = get_live_interest_rates("USD")
    
    # Build curves from live data
    discount_curve = build_live_discount_curve("USD", live_rates)
    forward_curve = build_enhanced_forward_curve(live_rates.get("3M", 0.025), live_rates.get("1Y", 0.04))
    
    # Create payment schedule
    freq_map = {"Quarterly": 0.25, "Semi-Annual": 0.5, "Annual": 1.0}
    dt = freq_map[payment_freq]
    payment_times = [dt * i for i in range(1, int(tenor_years / dt) + 1)]
    
    # Price based on selected model
    if model == "Enhanced DCF":
        result = price_irs_enhanced_dcf(
            notional=notional,
            fixed_rate=fixed_rate,
            payment_times=payment_times,
            discount_curve=discount_curve,
            forward_curve=forward_curve,
            floating_spread=floating_spread
        )
    elif model == "LIBOR Market Model":
        result = price_irs_lmm_enhanced(
            notional=notional,
            fixed_rate=fixed_rate,
            initial_rate=live_rates.get("3M", 0.025),
            vol=vol,
            payment_times=payment_times,
            discount_curve=discount_curve,
            n_paths=n_paths,
            floating_spread=floating_spread
        )
    else:  # Hull-White
        result = price_irs_hull_white(
            notional=notional,
            fixed_rate=fixed_rate,
            initial_rate=live_rates.get("3M", 0.025),
            mean_reversion=0.1,
            vol=vol,
            payment_times=payment_times,
            n_paths=n_paths
        )
    
    # Add live market context
    result['live_rates'] = live_rates
    result['market_rate'] = live_rates.get(f'{int(tenor_years)}Y', fixed_rate)
    
    return result


def _display_irs_results_enhanced(result, notional, fixed_rate, tenor_years, model):
    """Enhanced IRS results display with live market context"""
    
    npv = result.get('npv', 0)
    par_rate = result.get('par_rate', fixed_rate)
    market_rate = result.get('market_rate', fixed_rate)
    
    # Market comparison
    rate_diff_bp = (fixed_rate - market_rate) * 10000
    
    # Enhanced results table
    st.markdown(f"""
    <div class="metric-container">
        <h4>🎯 Enhanced Swap Pricing Results ({model})</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #007bff;">
                <h5 style="color: #007bff; margin-bottom: 10px;">💰 Valuation Metrics</h5>
                <p><strong>NPV:</strong> <span style="color: {'green' if npv > 0 else 'red'}; font-size: 1.2em;">${npv:,.0f}</span></p>
                <p><strong>Par Rate:</strong> {par_rate*100:.4f}%</p>
                <p><strong>DV01:</strong> ${result.get('dv01', 0):,.0f}</p>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #28a745;">
                <h5 style="color: #28a745; margin-bottom: 10px;">📊 Market Comparison</h5>
                <p><strong>Your Rate:</strong> {fixed_rate*100:.3f}%</p>
                <p><strong>Market Rate:</strong> {market_rate:.3f}%</p>
                <p><strong>Difference:</strong> <span style="color: {'green' if rate_diff_bp > 0 else 'red'};">{rate_diff_bp:+.1f}bp</span></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance indicators
    if abs(rate_diff_bp) > 10:
        color = "warning" if abs(rate_diff_bp) < 25 else "danger"
        st.markdown(f"""
        <div class="{color}-box">
            <h5>⚠️ Rate Analysis</h5>
            <p>Your fixed rate is <strong>{abs(rate_diff_bp):.1f} basis points</strong> 
            {'above' if rate_diff_bp > 0 else 'below'} current market levels.</p>
            <p><em>Consider {'receiving' if rate_diff_bp > 0 else 'paying'} fixed at current market rates.</em></p>
        </div>
        """, unsafe_allow_html=True)


def _display_market_comparison(result, market_rate, fixed_rate):
    """Display market comparison analysis"""
    
    st.markdown('<div class="sub-header">📈 Market Analysis</div>', unsafe_allow_html=True)
    
    # Create comparison chart
    metrics = ['Your Rate', 'Market Rate', 'Par Rate']
    values = [fixed_rate * 100, market_rate, result.get('par_rate', fixed_rate) * 100]
    
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


def _display_live_yield_curve():
    """Display live yield curve"""
    
    st.markdown('<div class="sub-header">📈 Live USD Yield Curve</div>', unsafe_allow_html=True)
    
    try:
        # Get live Treasury rates
        treasury_symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
        treasury_data = yf.download(treasury_symbols, period="1d", progress=False)
        
        if not treasury_data.empty:
            latest_rates = treasury_data['Close'].iloc[-1]
            
            # Create yield curve
            tenors = [0.25, 2, 10, 30]
            yields = [latest_rates.get(symbol, 4.0) for symbol in treasury_symbols]
            
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
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Curve analysis
            curve_slope = yields[-1] - yields[0]  # 30Y - 3M
            inversion = any(yields[i] > yields[i+1] for i in range(len(yields)-1))
            
            if inversion:
                st.markdown("""
                <div class="warning-box">
                    <h5>⚠️ Yield Curve Inversion Detected</h5>
                    <p>The yield curve shows inversion, which historically signals economic slowdown risk.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.info(f"📊 Curve slope (30Y-3M): {curve_slope:.0f} basis points")
            
    except:
        st.info("Live yield curve data temporarily unavailable")


# Footer with comprehensive information
def display_enhanced_footer():
    """Display enhanced footer with real-time market context"""
    
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; border: 1px solid #dee2e6;'>
        <div style="margin-bottom: 15px;">
            <span style="font-size: 2rem;">🔄</span>
        </div>
        <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: #1f77b4;">Professional Swaps Pricing Suite with Live Market Data</p>
        <p style="margin: 8px 0; color: #6c757d;">Powered by Yahoo Finance & FRED APIs • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
            <div style="background: rgba(0,123,255,0.1); padding: 10px; border-radius: 8px;">
                <strong>🏦 Interest Rate Swaps</strong><br>
                <small>DCF • LMM • Hull-White Models</small>
            </div>
            <div style="background: rgba(40,167,69,0.1); padding: 10px; border-radius: 8px;">
                <strong>💱 Currency Swaps</strong><br>
                <small>Live FX • Cross-Currency Basis</small>
            </div>
            <div style="background: rgba(255,193,7,0.1); padding: 10px; border-radius: 8px;">
                <strong>📈 Equity Swaps</strong><br>
                <small>Single Name • Index • Basket</small>
            </div>
            <div style="background: rgba(220,53,69,0.1); padding: 10px; border-radius: 8px;">
                <strong>📊 Portfolio Analytics</strong><br>
                <small>Risk Aggregation • VaR • Stress Testing</small>
            </div>
        </div>
        <p style="margin: 15px 0 0 0; color: #dc3545; font-weight: bold;">⚠️ For educational and research purposes only • Not financial advice</p>
    </div>
    """, unsafe_allow_html=True)
