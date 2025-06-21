def _get_indicative_fx_rate(quote_ccy: str, base_ccy: str) -> float:
    """Get indicative FX rates for major currency pairs"""
    
    fx_rates = {
        ("EUR", "USD"): 1.0850,
        ("GBP", "USD"): 1.2650,
        ("USD", "JPY"): 150.25,
        ("USD", "CHF"): 0.8890,
        ("USD", "CAD"): 1.3520,
        ("AUD", "USD"): 0.6750,
        ("USD", "SEK"): 10.45,
        ("USD", "NOK"): 10.85,
        ("EUR", "GBP"): 0.8580,
        ("EUR", "JPY"): 163.50,
        ("GBP", "JPY"): 190.60
    }
    
    # Try direct lookup
    if (quote_ccy, base_ccy) in fx_rates:
        return fx_rates[(quote_ccy, base_ccy)]
    
    # Try inverse
    if (base_ccy, quote_ccy) in fx_rates:
        return 1.0 / fx_rates[(base_ccy, quote_ccy)]
    
    # Cross rates via USD
    if quote_ccy != "USD" and base_ccy != "USD":
        quote_usd = _get_indicative_fx_rate(quote_ccy, "USD")
        base_usd = _get_indicative_fx_rate(base_ccy, "USD")
        return quote_usd / base_usd
    
    # Default
    return 1.0


def _get_indicative_rate(currency: str) -> float:
    """Get indicative interest rates by currency"""
    
    rates = {
        "USD": 3.50,
        "EUR": 2.25,
        "GBP": 4.75,
        "JPY": 0.50,
        "CHF": 1.25,
        "CAD": 3.25,
        "AUD": 3.85,
        "SEK": 2.75,
        "NOK": 3.15
    }
    
    return rates.get(currency, 3.00)


def _get_indicative_fx_vol(quote_ccy: str, base_ccy: str) -> float:
    """Get indicative FX volatilities"""
    
    vols = {
        ("EUR", "USD"): 12.0,
        ("GBP", "USD"): 15.0,
        ("USD", "JPY"): 10.5,
        ("USD", "CHF"): 11.0,
        ("USD", "CAD"): 8.5,
        ("AUD", "USD"): 18.0,
        ("USD", "SEK"): 13.5,
        ("USD", "NOK"): 14.0
    }
    
    pair = (quote_ccy, base_ccy)
    if pair in vols:
        return vols[pair]
    
    # Try inverse
    inverse_pair = (base_ccy, quote_ccy)
    if inverse_pair in vols:
        return vols[inverse_pair]
    
    # Default
    return 15.0


def _build_currency_curve(currency: str, base_rate: float, credit_spread: float = 0.0):
    """Build simplified currency discount curve"""
    
    # Add country risk premiums
    risk_premiums = {
        "USD": 0.0,
        "EUR": 0.001,
        "GBP": 0.002,
        "JPY": 0.0005,
        "CHF": -0.0005,
        "CAD": 0.0015,
        "AUD": 0.003,
        "SEK": 0.0025,
        "NOK": 0.0035
    }
    
    risk_premium = risk_premiums.get(currency, 0.002)
    adjusted_rate = base_rate + credit_spread + risk_premium
    
    def curve(t):
        # Simple term structure adjustment
        term_adjustment = 0.001 * np.sqrt(t)  # Slight upward slope
        return np.exp(-(adjusted_rate + term_adjustment) * t)
    
    return curve


def _build_fx_forward_curve(fx_spot: float, base_curve, quote_curve, fx_vol: float, xccy_basis: float):
    """Build FX forward curve using interest rate parity with adjustments"""
    
    def fx_forward(t):
        if t <= 0:
            return fx_spot
        
        # Interest rate parity
        base_df = base_curve(t)
        quote_df = quote_curve(t)
        
        # Basic forward rate
        forward = fx_spot * (base_df / quote_df)
        
        # Cross-currency basis adjustment
        basis_adjustment = np.exp(-xccy_basis * t)
        
        # Volatility adjustment (simplified quanto effect)
        vol_adjustment = np.exp(-0.5 * fx_vol**2 * t)
        
        return forward * basis_adjustment * vol_adjustment
    
    return fx_forward


def price_currency_swap_enhanced(
    base_notional: float,
    quote_notional: float,
    base_rate: float,
    quote_rate: float,
    payment_times: list,
    base_curve,
    quote_curve,
    fx_spot: float,
    fx_forward_curve,
    base_currency: str,
    quote_currency: str,
    swap_type: str = "Fixed-Fixed",
    include_principal: bool = True
):
    """Enhanced currency swap pricing with comprehensive analytics"""
    
    year_fractions = np.diff([0] + payment_times)
    
    # Base currency leg (what we pay)
    pv_base_leg = 0.0
    base_cashflows = []
    
    for i, (yf, t) in enumerate(zip(year_fractions, payment_times)):
        if swap_type in ["Fixed-Fixed", "Fixed-Float"]:
            cashflow = base_notional * base_rate * yf
        else:  # Float-Float
            # Use forward rate for floating
            forward_rate = -np.log(base_curve(t) / base_curve(max(0.001, t - yf))) / yf
            cashflow = base_notional * forward_rate * yf
        
        pv = cashflow * base_curve(t)
        pv_base_leg += pv
        
        base_cashflows.append({
            'payment_date': t,
            'year_fraction': yf,
            'rate': base_rate if swap_type != "Float-Float" else forward_rate,
            'cashflow': cashflow,
            'discount_factor': base_curve(t),
            'pv': pv
        })
    
    # Quote currency leg (what we receive, converted to base currency)
    pv_quote_leg = 0.0
    quote_cashflows = []
    
    for i, (yf, t) in enumerate(zip(year_fractions, payment_times)):
        if swap_type == "Fixed-Fixed":
            rate = quote_rate
        else:  # Fixed-Float or Float-Float
            # Use forward rate for floating
            rate = -np.log(quote_curve(t) / quote_curve(max(0.001, t - yf))) / yf
        
        cashflow_quote = quote_notional * rate * yf
        fx_forward = fx_forward_curve(t)
        cashflow_base = cashflow_quote * fx_forward
        
        pv = cashflow_base * base_curve(t)  # Discount in base currency
        pv_quote_leg += pv
        
        quote_cashflows.append({
            'payment_date': t,
            'year_fraction': yf,
            'rate': rate,
            'cashflow_quote': cashflow_quote,
            'fx_forward': fx_forward,
            'cashflow_base': cashflow_base,
            'discount_factor': base_curve(t),
            'pv': pv
        })
    
    # Principal exchanges
    pv_principal = 0.0
    if include_principal:
        # Initial exchange: receive quote currency, pay base currency
        initial_exchange = quote_notional * fx_spot - base_notional
        
        # Final exchange: pay quote currency, receive base currency
        final_fx = fx_forward_curve(payment_times[-1])
        final_exchange = base_notional - quote_notional * final_fx
        final_exchange_pv = final_exchange * base_curve(payment_times[-1])
        
        pv_principal = initial_exchange + final_exchange_pv
    
    # Total NPV (positive = favorable to receiver of quote currency)
    npv = pv_quote_leg - pv_base_leg + pv_principal
    
    # Calculate par rates
    base_annuity = sum([yf * base_curve(t) for yf, t in zip(year_fractions, payment_times)])
    quote_annuity = sum([yf * quote_curve(t) for yf, t in zip(year_fractions, payment_times)])
    
    par_base_rate = (pv_quote_leg + pv_principal) / (base_notional * base_annuity) if base_annuity > 0 else base_rate
    par_quote_rate = (pv_base_leg - pv_principal) / (quote_notional * quote_annuity * fx_spot) if quote_annuity > 0 else quote_rate
    
    # Risk metrics
    fx_delta = pv_quote_leg / fx_spot  # FX sensitivity
    
    return {
        'npv': npv,
        'pv_base_leg': pv_base_leg,
        'pv_quote_leg': pv_quote_leg,
        'pv_principal': pv_principal,
        'par_base_rate': par_base_rate,
        'par_quote_rate': par_quote_rate,
        'fx_delta': fx_delta,
        'base_cashflows': base_cashflows,
        'quote_cashflows': quote_cashflows,
        'current_fx': fx_spot,
        'final_fx': fx_forward_curve(payment_times[-1]) if payment_times else fx_spot
    }


def _display_currency_swap_results(result: dict, base_ccy: str, quote_ccy: str, fx_spot: float, tenor_years: int):
    """Display currency swap pricing results"""
    
    npv = result['npv']
    
    # Determine swap direction and color
    if npv > 0:
        direction = f"Receive {quote_ccy}, Pay {base_ccy}"
        color = "#2E8B57"  # Green
        status = "✅ Positive NPV"
    elif npv < 0:
        direction = f"Pay {quote_ccy}, Receive {base_ccy}"
        color = "#DC143C"  # Red  
        status = "❌ Negative NPV"
    else:
        direction = "At Par"
        color = "#4682B4"  # Blue
        status = "⚖️ Fair Value"
    
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
                <td style="padding: 10px; font-weight: bold;">NPV ({base_ccy})</td>
                <td style="padding: 10px; font-family: monospace; color: {color}; font-weight: bold; font-size: 1.2em;">{npv:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">Net Present Value</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Status</td>
                <td style="padding: 10px; font-weight: bold; color: {color};">{status}</td>
                <td style="padding: 10px; font-style: italic;">{direction}</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Par {base_ccy} Rate</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{result['par_base_rate']*100:.4f}%</td>
                <td style="padding: 10px; font-style: italic;">Fair {base_ccy} swap rate</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">Par {quote_ccy} Rate</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{result['par_quote_rate']*100:.4f}%</td>
                <td style="padding: 10px; font-style: italic;">Fair {quote_ccy} swap rate</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 10px; font-weight: bold;">FX Delta</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{result['fx_delta']:,.0f}</td>
                <td style="padding: 10px; font-style: italic;">FX sensitivity (1 unit move)</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: bold;">Maturity FX</td>
                <td style="padding: 10px; font-family: monospace; font-weight: bold;">{result['final_fx']:.4f}</td>
                <td style="padding: 10px; font-style: italic;">Forward FX at maturity</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Leg breakdown
    col_leg1, col_leg2, col_leg3 = st.columns(3)
    
    with col_leg1:
        st.markdown(f"""
        <div class="greeks-delta">
            <h4>📤 {base_ccy} Leg (Pay)</h4>
            <p><strong>Present Value:</strong> {result['pv_base_leg']:,.0f}</p>
            <p><strong>Par Rate:</strong> {result['par_base_rate']*100:.3f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_leg2:
        st.markdown(f"""
        <div class="greeks-gamma">
            <h4>📥 {quote_ccy} Leg (Receive)</h4>
            <p><strong>Present Value:</strong> {result['pv_quote_leg']:,.0f}</p>
            <p><strong>Par Rate:</strong> {result['par_quote_rate']*100:.3f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_leg3:
        st.markdown(f"""
        <div class="greeks-theta">
            <h4>💰 Principal Exchange</h4>
            <p><strong>PV Impact:</strong> {result['pv_principal']:,.0f}</p>
            <p><strong>FX Forward:</strong> {result['final_fx']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)


def _display_currency_swap_risk_metrics(result: dict, base_notional: float, quote_notional: float, 
                                       fx_spot: float, base_ccy: str, quote_ccy: str, payment_times: list):
    """Display currency swap risk metrics"""
    
    st.markdown('<div class="sub-header">⚡ Multi-Currency Risk Analytics</div>', unsafe_allow_html=True)
    
    # Calculate additional risk metrics
    duration_years = np.mean(payment_times) if payment_times else 0
    fx_exposure = result['fx_delta']
    
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 Interest Rate Risk</h4>
            <p><strong>{base_ccy} Duration:</strong> {duration_years:.2f} years</p>
            <p><strong>{quote_ccy} Duration:</strong> {duration_years:.2f} years</p>
            <p><strong>{base_ccy} DV01:</strong> {base_notional * duration_years * 0.0001:,.0f}</p>
            <p><strong>{quote_ccy} DV01:</strong> {quote_notional * duration_years * 0.0001:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_risk2:
        st.markdown(f"""
        <div class="warning-box">
            <h4>💱 FX & Credit Risk</h4>
            <p><strong>FX Delta:</strong> {fx_exposure:,.0f} {base_ccy}</p>
            <p><strong>FX Gamma:</strong> {abs(fx_exposure) * 0.01:,.0f} per 1% FX move</p>
            <p><strong>Credit Exposure:</strong> {max(result['npv'], 0):,.0f}</p>
            <p><strong>CVA Estimate:</strong> {max(result['npv'], 0) * 0.001:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)


def _display_fx_sensitivity_analysis(base_notional, quote_notional, base_rate, quote_rate,
                                    payment_times, base_curve, quote_curve, fx_spot,
                                    base_ccy, quote_ccy, swap_type, include_principal):
    """Display FX sensitivity analysis with charts"""
    
    st.markdown('<div class="sub-header">📈 FX Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    # FX sensitivity
    fx_shifts = np.linspace(-20, 20, 21)  # -20% to +20%
    npvs = []
    
    for shift_pct in fx_shifts:
        shifted_fx = fx_spot * (1 + shift_pct / 100)
        
        # Rebuild FX forward curve with shifted spot
        shifted_fx_curve = _build_fx_forward_curve(
            shifted_fx, base_curve, quote_curve, 0.15, 0.0
        )
        
        try:
            result = price_currency_swap_enhanced(
                base_notional=base_notional,
                quote_notional=quote_notional,
                base_rate=base_rate,
                quote_rate=quote_rate,
                payment_times=payment_times,
                base_curve=base_curve,
                quote_curve=quote_curve,
                fx_spot=shifted_fx,
                fx_forward_curve=shifted_fx_curve,
                base_currency=base_ccy,
                quote_currency=quote_ccy,
                swap_type=swap_type,
                include_principal=include_principal
            )
            npvs.append(result['npv'])
        except:
            npvs.append(0)
    
    # Create FX sensitivity chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fx_shifts,
        y=npvs,
        mode='lines+markers',
        name='NPV',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig.add_vline(x=0, line_dash="dot", line_color="gray", annotation_text="Current FX")
    
    fig.update_layout(
        title=f'{quote_ccy}{base_ccy} FX Sensitivity Analysis',
        xaxis_title='FX Rate Shift (%)',
        yaxis_title=f'NPV ({base_ccy})',
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scenario analysis
    scenarios = {
        "Base Case": 0,
        f"{quote_ccy} Strengthens (+10%)": 10,
        f"{quote_ccy} Weakens (-10%)": -10,
        f"Crisis Scenario (-20%)": -20,
        f"Rally Scenario (+20%)": 20
    }
    
    scenario_results = []
    base_npv = npvs[10]  # Base case (0% shift)
    
    for scenario_name, shift_pct in scenarios.items():
        npv_idx = 10 + shift_pct  # Index in npvs array
        if 0 <= npv_idx < len(npvs):
            npv = npvs[npv_idx]
        else:
            npv = base_npv
        
        scenario_results.append({
            'Scenario': scenario_name,
            'FX Shift (%)': shift_pct,
            f'NPV ({base_ccy})': f"{npv:,.0f}",
            'P&L vs Base': f"{npv - base_npv:,.0f}" if scenario_name != "Base Case" else "0"
        })
    
    st.markdown('<div class="sub-header">🎭 FX Scenario Analysis</div>', unsafe_allow_html=True)
    scenario_df = pd.DataFrame(scenario_results)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)


def _display_currency_market_overview():
    """Display currency market overview when not pricing"""
    
    st.markdown('<div class="sub-header">🌍 Currency Markets Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>💱 Major FX Rates (Indicative)</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #1f77b4; background-color: #f0f2f6;">
                    <td style="padding: 10px; font-weight: bold;">Currency Pair</td>
                    <td style="padding: 10px; font-weight: bold;">Rate</td>
                    <td style="padding: 10px; font-weight: bold;">Vol (%)</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">EUR/USD</td>
                    <td style="padding: 8px; font-family: monospace;">1.0850</td>
                    <td style="padding: 8px; font-family: monospace;">12.0</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">GBP/USD</td>
                    <td style="padding: 8px; font-family: monospace;">1.2650</td>
                    <td style="padding: 8px; font-family: monospace;">15.0</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">USD/JPY</td>
                    <td style="padding: 8px; font-family: monospace;">150.25</td>
                    <td style="padding: 8px; font-family: monospace;">10.5</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">USD/CHF</td>
                    <td style="padding: 8px; font-family: monospace;">0.8890</td>
                    <td style="padding: 8px; font-family: monospace;">11.0</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">AUD/USD</td>
                    <td style="padding: 8px; font-family: monospace;">0.6750</td>
                    <td style="padding: 8px; font-family: monospace;">18.0</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>💰 Interest Rate Environment</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #f59e0b; background-color: #fef3c7;">
                    <td style="padding: 10px; font-weight: bold;">Currency</td>
                    <td style="padding: 10px; font-weight: bold;">Policy Rate</td>
                    <td style="padding: 10px; font-weight: bold;">5Y Swap</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">USD</td>
                    <td style="padding: 8px; font-family: monospace;">3.50%</td>
                    <td style="padding: 8px; font-family: monospace;">3.75%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">EUR</td>
                    <td style="padding: 8px; font-family: monospace;">2.25%</td>
                    <td style="padding: 8px; font-family: monospace;">2.50%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">GBP</td>
                    <td style="padding: 8px; font-family: monospace;">4.75%</td>
                    <td style="padding: 8px; font-family: monospace;">4.25%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 8px;">JPY</td>
                    <td style="padding: 8px; font-family: monospace;">0.50%</td>
                    <td style="padding: 8px; font-family: monospace;">0.75%</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">CHF</td>
                    <td style="padding: 8px; font-family: monospace;">1.25%</td>
                    <td style="padding: 8px; font-family: monospace;">1.50%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # Currency swap education
    st.markdown('<div class="sub-header">📚 Currency Swaps Fundamentals</div>', unsafe_allow_html=True)
    
    with st.expander("💱 Understanding Cross-Currency Swaps"):
        st.markdown("""
        <div class="info-box">
            <h4>What are Currency Swaps?</h4>
            
            <p><strong>Currency swaps</strong> are agreements between two parties to exchange cash flows in different currencies. 
            They serve multiple purposes in international finance:</p>
            
            <h5>🔄 Key Features:</h5>
            <ul>
                <li><strong>Principal Exchange:</strong> Initial and final exchange of notional amounts</li>
                <li><strong>Interest Payments:</strong> Periodic exchange of interest in respective currencies</li>
                <li><strong>FX Risk:</strong> Exposure to currency movements over the swap life</li>
                <li><strong>Cross-Currency Basis:</strong> Additional spread reflecting funding costs</li>
            </ul>
            
            <h5>📊 Common Structures:</h5>
            <ul>
                <li><strong>Fixed-Fixed:</strong> Fixed rates in both currencies</li>
                <li><strong>Fixed-Float:</strong> Fixed rate in one currency, floating in other</li>
                <li><strong>Float-Float:</strong> Floating rates in both currencies</li>
            </ul>
            
            <h5>💼 Market Applications:</h5>
            <ul>
                <li><strong>Funding Arbitrage:</strong> Access cheaper funding in foreign markets</li>
                <li><strong>FX Hedging:</strong> Hedge long-term foreign currency exposure</li>
                <li><strong>Asset-Liability Matching:</strong> Match currency of assets and liabilities</li>
                <li><strong>Regulatory Capital:</strong> Optimize regulatory capital requirements</li>
            </ul>
            
            <h5>⚠️ Key Risks:</h5>
            <ul>
                <li><strong>FX Risk:</strong> Currency movements affect swap value</li>
                <li><strong>Interest Rate Risk:</strong> Rate changes in both currencies</li>
                <li><strong>Credit Risk:</strong> Counterparty default risk</li>
                <li><strong>Basis Risk:</strong> Cross-currency basis spread changes</li>
                <li><strong>Liquidity Risk:</strong> Difficulty unwinding positions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📈 Pricing Methodology"):
        st.markdown("""
        <div class="info-box">
            <h4>Currency Swap Valuation</h4>
            
            <h5>💰 Present Value Calculation:</h5>
            <p>The NPV of a currency swap is calculated as:</p>
            <div class="formula">
                NPV = PV(Foreign Leg) - PV(Domestic Leg) + PV(Principal Exchanges)
            </div>
            
            <h5>🔄 Each Leg Valuation:</h5>
            <ul>
                <li><strong>Fixed Leg:</strong> PV = Σ[Coupon × DF(t)]</li>
                <li><strong>Floating Leg:</strong> PV = Σ[Forward Rate × DF(t)]</li>
                <li><strong>FX Conversion:</strong> Foreign cashflows × FX Forward</li>
            </ul>
            
            <h5>💱 FX Forward Calculation:</h5>
            <div class="formula">
                F(t) = S(0) × [DF_foreign(t) / DF_domestic(t)] × Basis_Adjustment(t)
            </div>
            
            <h5>📊 Risk Metrics:</h5>
            <ul>
                <li><strong>FX Delta:</strong> Sensitivity to spot FX movements</li>
                <li><strong>IR DV01:</strong> Interest rate sensitivity in each currency</li>
                <li><strong>Cross Gamma:</strong> Cross-currency second-order effects</li>
                <li><strong>Vega:</strong> Sensitivity to FX volatility (for embedded options)</li>
            </ul>
            
            <h5>🏦 Market Conventions:</h5>
            <ul>
                <li><strong>Day Count:</strong> Usually ACT/360 for USD, ACT/365 for GBP</li>
                <li><strong>Payment Frequency:</strong> Semi-annual is standard</li>
                <li><strong>Business Days:</strong> Modified following convention</li>
                <li><strong>Reset:</strong> 2 business days before payment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🌍 Market Structure & Trends"):
        st.markdown("""
        <div class="info-box">
            <h4>Currency Swap Market Dynamics</h4>
            
            <h5>📊 Market Size & Activity:</h5>
            <ul>
                <li><strong>Outstanding Notional:</strong> ~$25 trillion globally</li>
                <li><strong>Daily Trading:</strong> ~$200-300 billion average</li>
                <li><strong>Major Pairs:</strong> USD/EUR, USD/JPY, USD/GBP dominate</li>
                <li><strong>Tenor Distribution:</strong> 70% under 5 years, 20% 5-10 years</li>
            </ul>
            
            <h5>🏛️ Key Market Participants:</h5>
            <ul>
                <li><strong>Central Banks:</strong> Liquidity provision, policy implementation</li>
                <li><strong>Commercial Banks:</strong> Market making, client flow</li>
                <li><strong>Corporations:</strong> Hedging, funding optimization</li>
                <li><strong>Asset Managers:</strong> Currency hedging, relative value trades</li>
                <li><strong>Hedge Funds:</strong> Speculation, arbitrage strategies</li>
            </ul>
            
            <h5>🔄 Current Market Themes:</h5>
            <ul>
                <li><strong>Central Bank Divergence:</strong> Fed vs ECB vs BOJ policy paths</li>
                <li><strong>USD Strength:</strong> Impact on cross-currency basis</li>
                <li><strong>Emerging Markets:</strong> Increased hedging demand</li>
                <li><strong>Regulatory Changes:</strong> Basel III impact on pricing</li>
                <li><strong>Digital Currencies:</strong> Potential future disruption</li>
            </ul>
            
            <h5>📈 Cross-Currency Basis Trends:</h5>
            <ul>
                <li><strong>USD Premium:</strong> Persistent demand for USD funding</li>
                <li><strong>Quarter-End Effects:</strong> Regulatory capital impacts</li>
                <li><strong>Crisis Periods:</strong> Basis can widen dramatically</li>
                <li><strong>Central Bank Actions:</strong> Swap lines affect basis levels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)# tabs/swaps.py
# Enhanced Swaps Tab with Professional Pricing Models

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from styles.app_styles import load_theme

# Enhanced pricing functions
from pricing.models.swaps.enhanced_dcf import *
from pricing.models.swaps.enhanced_lmm import *
from pricing.models.swaps.enhanced_curves import *


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
    st.markdown('<div class="sub-header">Cross-Currency Swap Pricing & Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>💱 Currency Swap Configuration</h4>
            <p>Professional cross-currency swap pricing with market-standard conventions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Currency selection
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "SEK", "NOK"]
        
        base_currency = st.selectbox("Base Currency (Pay)", currencies, index=0)
        quote_currency = st.selectbox("Quote Currency (Receive)", currencies, index=1)
        
        if base_currency == quote_currency:
            st.error("Base and Quote currencies must be different")
            return
        
        # Notional amounts
        st.markdown("**Notional Amounts:**")
        base_notional = st.number_input(
            f"Base Notional ({base_currency})", 
            min_value=1_000_000, 
            max_value=10_000_000_000, 
            value=100_000_000, 
            step=1_000_000,
            format="%d"
        )
        
        # FX spot rate
        fx_pair = f"{quote_currency}{base_currency}"
        current_fx = _get_indicative_fx_rate(quote_currency, base_currency)
        
        fx_spot = st.number_input(
            f"FX Spot Rate ({fx_pair})", 
            min_value=0.0001, 
            max_value=1000.0, 
            value=current_fx, 
            step=0.0001,
            format="%.4f",
            help=f"How many {base_currency} per 1 {quote_currency}"
        )
        
        quote_notional = base_notional / fx_spot
        st.metric(f"Implied {quote_currency} Notional", f"{quote_notional:,.0f}")
        
        # Swap structure
        st.markdown("""
        <div class="info-box">
            <h4>🔧 Swap Structure</h4>
        </div>
        """, unsafe_allow_html=True)
        
        swap_type = st.radio(
            "Swap Type",
            ["Fixed-Fixed", "Fixed-Float", "Float-Float"],
            help="Structure of the currency swap"
        )
        
        tenor_years = st.selectbox("Swap Tenor", [1, 2, 3, 5, 7, 10, 15, 20], index=4)
        
        payment_freq = st.selectbox(
            "Payment Frequency", 
            ["Quarterly", "Semi-Annual", "Annual"], 
            index=1,
            help="Standard is Semi-Annual for cross-currency swaps"
        )
        
        # Interest rates
        st.markdown("**Interest Rates:**")
        
        base_rate = st.slider(
            f"{base_currency} Rate (%)", 
            min_value=0.0, 
            max_value=10.0, 
            value=_get_indicative_rate(base_currency), 
            step=0.01,
            format="%.2f"
        ) / 100
        
        if swap_type == "Fixed-Fixed":
            quote_rate = st.slider(
                f"{quote_currency} Rate (%)", 
                min_value=0.0, 
                max_value=10.0, 
                value=_get_indicative_rate(quote_currency), 
                step=0.01,
                format="%.2f"
            ) / 100
        else:
            quote_rate = _get_indicative_rate(quote_currency) / 100
            st.info(f"Using indicative {quote_currency} rate: {quote_rate:.2%}")
        
        # Advanced parameters
        with st.expander("🔬 Advanced Parameters"):
            include_principal = st.checkbox("Include Principal Exchange", value=True)
            
            fx_vol = st.slider(
                "FX Volatility (%)", 
                min_value=5.0, 
                max_value=50.0, 
                value=_get_indicative_fx_vol(quote_currency, base_currency), 
                step=1.0
            ) / 100
            
            xccy_basis = st.slider(
                "Cross-Currency Basis (bps)", 
                min_value=-100, 
                max_value=100, 
                value=0, 
                step=5,
                help="Cross-currency basis spread"
            )
            
            credit_spread_base = st.slider(
                f"{base_currency} Credit Spread (bps)", 
                min_value=0, 
                max_value=500, 
                value=50, 
                step=10
            )
            
            credit_spread_quote = st.slider(
                f"{quote_currency} Credit Spread (bps)", 
                min_value=0, 
                max_value=500, 
                value=50, 
                step=10
            )
        
        price_btn = st.button("💱 Price Currency Swap", type="primary", use_container_width=True)
    
    with col2:
        if price_btn:
            with st.spinner("Calculating cross-currency swap price and risk metrics..."):
                try:
                    # Build currency curves
                    base_curve = _build_currency_curve(base_currency, base_rate, credit_spread_base / 10000)
                    quote_curve = _build_currency_curve(quote_currency, quote_rate, credit_spread_quote / 10000)
                    
                    # Create payment schedule
                    freq_map = {"Quarterly": 0.25, "Semi-Annual": 0.5, "Annual": 1.0}
                    dt = freq_map[payment_freq]
                    payment_times = [dt * i for i in range(1, int(tenor_years / dt) + 1)]
                    
                    # FX forward curve
                    fx_forward_curve = _build_fx_forward_curve(
                        fx_spot, base_curve, quote_curve, fx_vol, xccy_basis / 10000
                    )
                    
                    # Price currency swap
                    result = price_currency_swap_enhanced(
                        base_notional=base_notional,
                        quote_notional=quote_notional,
                        base_rate=base_rate,
                        quote_rate=quote_rate,
                        payment_times=payment_times,
                        base_curve=base_curve,
                        quote_curve=quote_curve,
                        fx_spot=fx_spot,
                        fx_forward_curve=fx_forward_curve,
                        base_currency=base_currency,
                        quote_currency=quote_currency,
                        swap_type=swap_type,
                        include_principal=include_principal
                    )
                    
                    # Display results
                    _display_currency_swap_results(result, base_currency, quote_currency, fx_spot, tenor_years)
                    
                    # Risk analytics
                    _display_currency_swap_risk_metrics(
                        result, base_notional, quote_notional, fx_spot, 
                        base_currency, quote_currency, payment_times
                    )
                    
                    # FX sensitivity analysis
                    _display_fx_sensitivity_analysis(
                        base_notional, quote_notional, base_rate, quote_rate,
                        payment_times, base_curve, quote_curve, fx_spot,
                        base_currency, quote_currency, swap_type, include_principal
                    )
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>❌ Pricing Error</h4>
                        <p>Error during calculation: {str(e)}</p>
                        <p>Please check your parameters and try again.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            # Display market overview when not pricing
            _display_currency_market_overview()


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
