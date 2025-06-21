# pricing/models/swap_pricing.py
# Clean Swap Pricing Models

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SwapResult:
    """Container for swap pricing results"""
    npv: float
    pv_fixed: float = 0.0
    pv_floating: float = 0.0
    par_rate: float = 0.0
    dv01: float = 0.0
    duration: float = 0.0
    greeks: Optional[Dict[str, float]] = None
    cashflows: Optional[pd.DataFrame] = None


@dataclass
class CurrencySwapResult:
    """Container for currency swap results"""
    npv_domestic: float
    npv_foreign: float
    pv_domestic_leg: float
    pv_foreign_leg: float
    pv_principal_exchanges: float
    fx_delta: float
    domestic_dv01: float
    foreign_dv01: float
    cross_gamma: float


class InterestRateSwapPricer:
    """Clean Interest Rate Swap Pricing"""
    
    @staticmethod
    def price_enhanced_dcf(notional: float, fixed_rate: float, payment_times: List[float], 
                          discount_curve, forward_curve, floating_spread: float = 0.0) -> SwapResult:
        """Enhanced DCF pricing for IRS"""
        
        # Year fractions
        year_fractions = np.diff([0] + payment_times)
        
        # Fixed leg PV
        pv_fixed = sum([
            notional * fixed_rate * yf * discount_curve(t)
            for yf, t in zip(year_fractions, payment_times)
        ])
        
        # Floating leg PV
        pv_floating = sum([
            notional * (forward_curve(t) + floating_spread) * yf * discount_curve(t)
            for yf, t in zip(year_fractions, payment_times)
        ])
        
        # NPV (receiver swap: receive fixed, pay floating)
        npv = pv_fixed - pv_floating
        
        # Par rate calculation
        annuity = sum([yf * discount_curve(t) for yf, t in zip(year_fractions, payment_times)])
        par_rate = pv_floating / (notional * annuity) if annuity > 0 else fixed_rate
        
        # DV01 calculation
        dv01 = notional * annuity * 0.0001
        
        # Duration (approximate)
        duration = sum([t * yf * discount_curve(t) for yf, t in zip(year_fractions, payment_times)]) / annuity if annuity > 0 else 0
        
        return SwapResult(
            npv=npv,
            pv_fixed=pv_fixed,
            pv_floating=pv_floating,
            par_rate=par_rate,
            dv01=dv01,
            duration=duration
        )
    
    @staticmethod
    def price_monte_carlo(notional: float, fixed_rate: float, initial_rate: float, 
                         vol: float, payment_times: List[float], discount_curve, 
                         n_paths: int = 25000, floating_spread: float = 0.0) -> SwapResult:
        """Monte Carlo IRS pricing"""
        
        np.random.seed(42)
        dt = payment_times[1] - payment_times[0] if len(payment_times) > 1 else 0.25
        n_steps = len(payment_times)
        
        # Simulate forward rates
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = initial_rate
        
        for i in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            rates[:, i] = rates[:, i-1] * np.exp(-0.5 * vol**2 * dt + vol * dW)
        
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
        annuity = sum([yf * discount_curve(t) for yf, t in zip(year_fractions, payment_times)])
        par_rate = pv_floating / (notional * annuity) if annuity > 0 else fixed_rate
        dv01 = notional * annuity * 0.0001
        
        return SwapResult(
            npv=npv,
            pv_fixed=pv_fixed,
            pv_floating=pv_floating,
            par_rate=par_rate,
            dv01=dv01,
            greeks={'monte_carlo_paths': n_paths}
        )
    
    @staticmethod
    def price_hull_white(notional: float, fixed_rate: float, initial_rate: float, 
                        vol: float, payment_times: List[float], n_paths: int = 25000, 
                        mean_reversion: float = 0.1) -> SwapResult:
        """Hull-White model IRS pricing"""
        
        np.random.seed(42)
        dt = payment_times[1] - payment_times[0] if len(payment_times) > 1 else 0.25
        n_steps = len(payment_times)
        
        # Hull-White parameters
        a = mean_reversion
        
        # Simulate short rate paths
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = initial_rate
        
        for i in range(1, n_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), n_paths)
            rates[:, i] = rates[:, i-1] + a * (initial_rate - rates[:, i-1]) * dt + vol * dW
            rates[:, i] = np.maximum(rates[:, i], 0)  # Floor at zero
        
        # Calculate cashflows
        year_fractions = np.diff([0] + payment_times)
        
        floating_cashflows = np.zeros(n_paths)
        for i, (yf, t) in enumerate(zip(year_fractions, payment_times)):
            discount_factor = np.exp(-rates[:, i] * t)
            floating_cashflows += notional * rates[:, i] * yf * discount_factor
        
        pv_floating = np.mean(floating_cashflows)
        
        # Fixed leg using average rates
        avg_rates = np.mean(rates, axis=0)
        pv_fixed = sum([
            notional * fixed_rate * yf * np.exp(-avg_rates[i] * t)
            for i, (yf, t) in enumerate(zip(year_fractions, payment_times))
        ])
        
        npv = pv_fixed - pv_floating
        annuity = sum([yf * np.exp(-avg_rates[i] * t) for i, (yf, t) in enumerate(zip(year_fractions, payment_times))])
        par_rate = pv_floating / (notional * annuity) if annuity > 0 else fixed_rate
        dv01 = notional * annuity * 0.0001
        
        return SwapResult(
            npv=npv,
            pv_fixed=pv_fixed,
            pv_floating=pv_floating,
            par_rate=par_rate,
            dv01=dv01,
            greeks={'model': 'Hull-White', 'mean_reversion': mean_reversion}
        )


class CurrencySwapPricer:
    """Clean Currency Swap Pricing"""
    
    @staticmethod
    def price_simple(base_notional: float, quote_notional: float, base_rate: float, 
                    quote_rate: float, tenor_years: float, fx_spot: float, 
                    include_principal: bool = True, xccy_basis: float = 0.0) -> CurrencySwapResult:
        """Simplified currency swap pricing"""
        
        # Payment frequency (semi-annual standard)
        payment_freq = 0.5
        n_payments = int(tenor_years / payment_freq)
        payment_times = [payment_freq * (i + 1) for i in range(n_payments)]
        
        # Simple discount curves
        base_discount = lambda t: np.exp(-base_rate * t)
        quote_discount = lambda t: np.exp(-quote_rate * t)
        
        # Base currency leg (what we pay)
        base_leg_pv = sum([
            base_notional * base_rate * payment_freq * base_discount(t)
            for t in payment_times
        ])
        
        # Quote currency leg (what we receive, converted to base currency)
        quote_leg_pv = sum([
            quote_notional * quote_rate * payment_freq * fx_spot * quote_discount(t)
            for t in payment_times
        ])
        
        # Principal exchange
        principal_pv = 0
        if include_principal:
            # Initial exchange: receive quote notional, pay base notional
            initial_exchange = quote_notional * fx_spot - base_notional
            
            # Final exchange: pay quote notional, receive base notional
            final_exchange = (base_notional - quote_notional * fx_spot) * base_discount(tenor_years)
            
            principal_pv = initial_exchange + final_exchange
        
        # Apply cross-currency basis
        basis_adjustment = quote_leg_pv * xccy_basis * tenor_years
        
        # NPV (from base currency perspective)
        npv_base = quote_leg_pv - base_leg_pv + principal_pv + basis_adjustment
        npv_quote = npv_base / fx_spot
        
        # Calculate Greeks (simplified)
        fx_delta = quote_notional  # Simplified FX sensitivity
        domestic_dv01 = base_notional * tenor_years * 0.0001
        foreign_dv01 = quote_notional * fx_spot * tenor_years * 0.0001
        
        return CurrencySwapResult(
            npv_domestic=npv_base,
            npv_foreign=npv_quote,
            pv_domestic_leg=base_leg_pv,
            pv_foreign_leg=quote_leg_pv,
            pv_principal_exchanges=principal_pv,
            fx_delta=fx_delta,
            domestic_dv01=domestic_dv01,
            foreign_dv01=foreign_dv01,
            cross_gamma=fx_delta * 0.01
        )


class EquitySwapPricer:
    """Clean Equity Swap Pricing"""
    
    @staticmethod
    def price_enhanced(symbol: str, notional: float, fixed_rate: float, tenor_years: float, 
                      equity_data: Dict, swap_direction: str, financing_spread: float = 0.0, 
                      dividend_treatment: str = "Gross") -> SwapResult:
        """Enhanced equity swap pricing"""
        
        # Get current equity parameters
        current_price = equity_data['price']
        volatility = equity_data['volatility'] / 100
        dividend_yield = equity_data['dividend_yield'] / 100
        
        # Calculate expected equity return
        risk_free_rate = 0.04
        equity_risk_premium = 0.06
        expected_equity_return = risk_free_rate + equity_risk_premium
        
        # Equity leg PV
        equity_pv = notional * expected_equity_return * tenor_years
        
        # Fixed leg PV
        fixed_pv = notional * fixed_rate * tenor_years
        
        # Add dividend adjustments
        dividend_pv = notional * dividend_yield * tenor_years
        if dividend_treatment == "Net":
            dividend_pv *= 0.85  # Apply withholding tax
        
        # NPV calculation
        if swap_direction == "Pay Equity, Receive Fixed":
            npv = fixed_pv - equity_pv + dividend_pv
        else:
            npv = equity_pv - fixed_pv + dividend_pv
        
        # Calculate Greeks
        equity_delta = notional / current_price if swap_direction == "Pay Fixed, Receive Equity" else -notional / current_price
        equity_vega = notional * volatility * np.sqrt(tenor_years) * 0.01
        
        greeks = {
            'equity_delta': equity_delta,
            'equity_vega': equity_vega,
            'current_price': current_price,
            'expected_return': expected_equity_return,
            'volatility': volatility,
            'financing_spread': financing_spread
        }
        
        return SwapResult(
            npv=npv,
            pv_fixed=fixed_pv,
            pv_floating=equity_pv + dividend_pv,
            greeks=greeks
        )


class CurveBuilder:
    """Simple curve building utilities"""
    
    @staticmethod
    def build_discount_curve(base_rate: float, curve_type: str = "flat"):
        """Build simple discount curve"""
        if curve_type == "flat":
            return lambda t: np.exp(-base_rate * t)
        else:
            # Simple Nelson-Siegel approximation
            beta0, beta1, beta2, tau = base_rate, -0.005, 0.01, 2.0
            def enhanced_curve(t):
                if t <= 0:
                    return 1.0
                rate = beta0 + beta1 * (1 - np.exp(-t/tau)) / (t/tau) + beta2 * ((1 - np.exp(-t/tau)) / (t/tau) - np.exp(-t/tau))
                return np.exp(-rate * t)
            return enhanced_curve
    
    @staticmethod
    def build_forward_curve(base_rate: float, spread: float = 0.0):
        """Build simple forward curve"""
        return lambda t: base_rate + spread + 0.001 * t  # Slight upward slope


class PortfolioAnalyzer:
    """Portfolio-level analytics"""
    
    @staticmethod
    def create_swap_object(swap_type: str, notional: float, tenor: str, direction: str, **kwargs):
        """Create standardized swap object"""
        
        base_npv = np.random.normal(0, notional * 0.001)  # Simplified NPV
        
        swap_obj = {
            'Type': swap_type,
            'Notional': f"${notional:,.0f}",
            'Tenor': tenor,
            'Direction': direction,
            'NPV': base_npv
        }
        
        # Add type-specific fields
        if swap_type == "Currency":
            swap_obj['Pair'] = kwargs.get('currency_pair', 'EURUSD')
        elif swap_type == "Equity":
            swap_obj['Underlying'] = kwargs.get('underlying', 'SPY')
        elif swap_type == "Interest Rate":
            swap_obj['Rate'] = kwargs.get('rate', '3.50%')
        
        return swap_obj
    
    @staticmethod
    def analyze_portfolio(portfolio: List[Dict]) -> Dict:
        """Analyze portfolio of swaps"""
        
        if not portfolio:
            return {
                'total_notional': 0,
                'portfolio_npv': 0,
                'portfolio_dv01': 0,
                'portfolio_vega': 0,
                'ir_exposure': 0,
                'fx_exposure': 0,
                'equity_exposure': 0,
                'num_swaps': 0
            }
        
        total_notional = sum([
            float(swap['Notional'].replace('$', '').replace(',', '')) 
            for swap in portfolio
        ])
        
        # Calculate portfolio metrics
        portfolio_npv = sum([swap.get('NPV', 0) for swap in portfolio])
        portfolio_dv01 = total_notional * 0.0001
        portfolio_vega = total_notional * 0.0005
        
        # Risk decomposition
        ir_exposure = sum([
            float(swap['Notional'].replace('$', '').replace(',', '')) 
            for swap in portfolio if swap['Type'] == 'Interest Rate'
        ])
        fx_exposure = sum([
            float(swap['Notional'].replace('$', '').replace(',', '')) 
            for swap in portfolio if swap['Type'] == 'Currency'
        ])
        equity_exposure = sum([
            float(swap['Notional'].replace('$', '').replace(',', '')) 
            for swap in portfolio if swap['Type'] == 'Equity'
        ])
        
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
