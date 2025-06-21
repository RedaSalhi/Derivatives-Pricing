# pricing/models/swaps/dcf.py
# Enhanced Discounted Cash Flow Models for Swaps

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SwapResult:
    """Container for swap pricing results"""
    npv: float
    pv_fixed_leg: float
    pv_floating_leg: float
    par_rate: float
    dv01: float
    duration: float
    convexity: float
    carry: float
    annuity: float
    cashflows: Optional[pd.DataFrame] = None
    greeks: Optional[Dict[str, float]] = None


class EnhancedSwapPricer:
    """Professional-grade swap pricing engine"""
    
    def __init__(self, valuation_date: datetime = None):
        self.valuation_date = valuation_date or datetime.now()
        self.day_count_convention = "ACT/360"
        self.business_day_convention = "Modified Following"
    
    def price_interest_rate_swap(
        self,
        notional: float,
        fixed_rate: float,
        tenor_years: float,
        payment_frequency: str,
        discount_curve: Callable[[float], float],
        forward_curve: Callable[[float], float],
        floating_spread: float = 0.0,
        swap_type: str = "payer",  # payer or receiver
        calculate_greeks: bool = True
    ) -> SwapResult:
        """
        Price an interest rate swap using enhanced DCF methodology
        
        Args:
            notional: Notional amount
            fixed_rate: Fixed leg rate
            tenor_years: Swap maturity in years
            payment_frequency: "Quarterly", "Semi-Annual", "Annual"
            discount_curve: Discount factor function P(0,T)
            forward_curve: Forward rate function F(0,T)
            floating_spread: Spread over floating benchmark
            swap_type: "payer" (pay fixed) or "receiver" (receive fixed)
            calculate_greeks: Whether to calculate risk sensitivities
            
        Returns:
            SwapResult object with comprehensive pricing information
        """
        
        # Generate payment schedule
        payment_schedule = self._generate_payment_schedule(tenor_years, payment_frequency)
        
        # Calculate fixed leg present value
        pv_fixed = self._calculate_fixed_leg_pv(
            notional, fixed_rate, payment_schedule, discount_curve
        )
        
        # Calculate floating leg present value
        pv_floating = self._calculate_floating_leg_pv(
            notional, payment_schedule, discount_curve, forward_curve, floating_spread
        )
        
        # NPV calculation (convention: positive = receive fixed, pay floating)
        if swap_type.lower() == "payer":
            npv = pv_floating - pv_fixed  # Pay fixed, receive floating
        else:
            npv = pv_fixed - pv_floating  # Receive fixed, pay floating
        
        # Calculate par rate
        annuity = sum([cf['year_fraction'] * discount_curve(cf['payment_date']) 
                      for cf in payment_schedule])
        par_rate = pv_floating / (notional * annuity) if annuity > 0 else fixed_rate
        
        # Risk metrics
        dv01 = self._calculate_dv01(notional, payment_schedule, discount_curve)
        duration = self._calculate_duration(payment_schedule, discount_curve, annuity)
        convexity = self._calculate_convexity(payment_schedule, discount_curve, duration)
        carry = self._calculate_carry(notional, fixed_rate, payment_frequency)
        
        # Greeks calculation
        greeks = {}
        if calculate_greeks:
            greeks = self._calculate_greeks(
                notional, fixed_rate, payment_schedule, discount_curve, forward_curve
            )
        
        # Cashflow table
        cashflows_df = self._create_cashflow_table(
            notional, fixed_rate, payment_schedule, discount_curve, forward_curve, floating_spread
        )
        
        return SwapResult(
            npv=npv,
            pv_fixed_leg=pv_fixed,
            pv_floating_leg=pv_floating,
            par_rate=par_rate,
            dv01=dv01,
            duration=duration,
            convexity=convexity,
            carry=carry,
            annuity=annuity,
            cashflows=cashflows_df,
            greeks=greeks
        )
    
    def price_currency_swap(
        self,
        notional_domestic: float,
        notional_foreign: float,
        domestic_rate: float,
        foreign_rate: float,
        tenor_years: float,
        fx_spot: float,
        domestic_discount_curve: Callable[[float], float],
        foreign_discount_curve: Callable[[float], float],
        fx_forward_curve: Callable[[float], float],
        include_principal: bool = True
    ) -> SwapResult:
        """Price a cross-currency swap"""
        
        payment_schedule = self._generate_payment_schedule(tenor_years, "Semi-Annual")
        
        # Domestic leg (receive domestic currency)
        pv_domestic = self._calculate_fixed_leg_pv(
            notional_domestic, domestic_rate, payment_schedule, domestic_discount_curve
        )
        
        # Foreign leg (pay foreign currency, converted to domestic)
        pv_foreign_local = self._calculate_fixed_leg_pv(
            notional_foreign, foreign_rate, payment_schedule, foreign_discount_curve
        )
        
        # Convert foreign PV to domestic currency
        pv_foreign = pv_foreign_local * fx_spot
        
        # Principal exchange if included
        if include_principal:
            # Initial exchange: receive foreign, pay domestic
            initial_exchange = notional_foreign * fx_spot - notional_domestic
            
            # Final exchange: pay foreign, receive domestic
            final_exchange = (notional_domestic - notional_foreign * fx_forward_curve(tenor_years)) * \
                           domestic_discount_curve(tenor_years)
            
            principal_pv = initial_exchange + final_exchange
        else:
            principal_pv = 0
        
        # NPV (receive domestic, pay foreign)
        npv = pv_domestic - pv_foreign + principal_pv
        
        return SwapResult(
            npv=npv,
            pv_fixed_leg=pv_domestic,
            pv_floating_leg=pv_foreign,
            par_rate=domestic_rate,  # Simplified
            dv01=0,  # Would need currency-specific calculation
            duration=tenor_years / 2,  # Simplified
            convexity=0,
            carry=0,
            annuity=0
        )
    
    def price_equity_swap(
        self,
        notional: float,
        equity_return: float,
        fixed_rate: float,
        tenor_years: float,
        payment_frequency: str,
        discount_curve: Callable[[float], float],
        dividend_yield: float = 0.0
    ) -> SwapResult:
        """Price an equity swap (receive equity return, pay fixed)"""
        
        payment_schedule = self._generate_payment_schedule(tenor_years, payment_frequency)
        
        # Fixed leg
        pv_fixed = self._calculate_fixed_leg_pv(
            notional, fixed_rate, payment_schedule, discount_curve
        )
        
        # Equity leg (simplified - total return at maturity)
        pv_equity = notional * equity_return * discount_curve(tenor_years)
        
        # Dividend adjustments
        if dividend_yield > 0:
            dividend_pv = sum([
                notional * dividend_yield * cf['year_fraction'] * discount_curve(cf['payment_date'])
                for cf in payment_schedule
            ])
            pv_equity += dividend_pv
        
        # NPV (receive equity, pay fixed)
        npv = pv_equity - pv_fixed
        
        return SwapResult(
            npv=npv,
            pv_fixed_leg=pv_fixed,
            pv_floating_leg=pv_equity,
            par_rate=equity_return / tenor_years,  # Simplified
            dv01=0,
            duration=tenor_years,
            convexity=0,
            carry=0,
            annuity=0
        )
    
    def _generate_payment_schedule(self, tenor_years: float, frequency: str) -> List[Dict]:
        """Generate payment schedule with proper day count conventions"""
        
        freq_map = {
            "Quarterly": 0.25,
            "Semi-Annual": 0.5,
            "Annual": 1.0,
            "Monthly": 1/12
        }
        
        dt = freq_map.get(frequency, 0.25)
        n_payments = int(tenor_years / dt)
        
        schedule = []
        for i in range(1, n_payments + 1):
            payment_date = i * dt
            year_fraction = dt  # Simplified - should use actual day count
            
            schedule.append({
                'payment_number': i,
                'payment_date': payment_date,
                'year_fraction': year_fraction,
                'days': int(year_fraction * 360)  # ACT/360 approximation
            })
        
        return schedule
    
    def _calculate_fixed_leg_pv(
        self, 
        notional: float, 
        fixed_rate: float, 
        payment_schedule: List[Dict], 
        discount_curve: Callable[[float], float]
    ) -> float:
        """Calculate present value of fixed leg"""
        
        pv = 0.0
        for payment in payment_schedule:
            cashflow = notional * fixed_rate * payment['year_fraction']
            discount_factor = discount_curve(payment['payment_date'])
            pv += cashflow * discount_factor
        
        return pv
    
    def _calculate_floating_leg_pv(
        self,
        notional: float,
        payment_schedule: List[Dict],
        discount_curve: Callable[[float], float],
        forward_curve: Callable[[float], float],
        spread: float = 0.0
    ) -> float:
        """Calculate present value of floating leg"""
        
        pv = 0.0
        for payment in payment_schedule:
            forward_rate = forward_curve(payment['payment_date']) + spread
            cashflow = notional * forward_rate * payment['year_fraction']
            discount_factor = discount_curve(payment['payment_date'])
            pv += cashflow * discount_factor
        
        return pv
    
    def _calculate_dv01(
        self, 
        notional: float, 
        payment_schedule: List[Dict], 
        discount_curve: Callable[[float], float]
    ) -> float:
        """Calculate DV01 (dollar value of 1 basis point)"""
        
        annuity = sum([
            cf['year_fraction'] * discount_curve(cf['payment_date']) 
            for cf in payment_schedule
        ])
        
        return notional * annuity * 0.0001  # 1 basis point = 0.0001
    
    def _calculate_duration(
        self, 
        payment_schedule: List[Dict], 
        discount_curve: Callable[[float], float],
        annuity: float
    ) -> float:
        """Calculate modified duration"""
        
        weighted_time = sum([
            cf['payment_date'] * cf['year_fraction'] * discount_curve(cf['payment_date'])
            for cf in payment_schedule
        ])
        
        return weighted_time / annuity if annuity > 0 else 0
    
    def _calculate_convexity(
        self, 
        payment_schedule: List[Dict], 
        discount_curve: Callable[[float], float],
        duration: float
    ) -> float:
        """Calculate convexity"""
        
        # Simplified convexity calculation
        return duration * duration * 0.5
    
    def _calculate_carry(self, notional: float, fixed_rate: float, frequency: str) -> float:
        """Calculate daily carry (theta)"""
        
        freq_map = {"Quarterly": 4, "Semi-Annual": 2, "Annual": 1}
        payments_per_year = freq_map.get(frequency, 4)
        
        return notional * fixed_rate / (365 * payments_per_year)
    
    def _calculate_greeks(
        self,
        notional: float,
        fixed_rate: float,
        payment_schedule: List[Dict],
        discount_curve: Callable[[float], float],
        forward_curve: Callable[[float], float]
    ) -> Dict[str, float]:
        """Calculate risk sensitivities"""
        
        # This would contain sophisticated Greeks calculations
        # For now, returning simplified versions
        
        annuity = sum([cf['year_fraction'] * discount_curve(cf['payment_date']) 
                      for cf in payment_schedule])
        
        return {
            'delta': notional * annuity * 0.01,  # 100bp sensitivity
            'gamma': notional * annuity * 0.0001,  # Second order
            'vega': 0,  # Not applicable for vanilla swaps
            'theta': self._calculate_carry(notional, fixed_rate, "Quarterly"),
            'rho': notional * annuity * 0.01
        }
    
    def _create_cashflow_table(
        self,
        notional: float,
        fixed_rate: float,
        payment_schedule: List[Dict],
        discount_curve: Callable[[float], float],
        forward_curve: Callable[[float], float],
        floating_spread: float
    ) -> pd.DataFrame:
        """Create detailed cashflow table"""
        
        cashflows = []
        
        for payment in payment_schedule:
            payment_date = payment['payment_date']
            year_fraction = payment['year_fraction']
            
            # Fixed leg
            fixed_cashflow = notional * fixed_rate * year_fraction
            fixed_pv = fixed_cashflow * discount_curve(payment_date)
            
            # Floating leg
            forward_rate = forward_curve(payment_date) + floating_spread
            floating_cashflow = notional * forward_rate * year_fraction
            floating_pv = floating_cashflow * discount_curve(payment_date)
            
            cashflows.append({
                'Payment_Date': payment_date,
                'Year_Fraction': year_fraction,
                'Fixed_Rate': fixed_rate,
                'Forward_Rate': forward_rate,
                'Fixed_Cashflow': fixed_cashflow,
                'Floating_Cashflow': floating_cashflow,
                'Discount_Factor': discount_curve(payment_date),
                'Fixed_PV': fixed_pv,
                'Floating_PV': floating_pv,
                'Net_PV': floating_pv - fixed_pv
            })
        
        return pd.DataFrame(cashflows)


# Market data and curve building utilities
def build_nelson_siegel_curve(beta0: float, beta1: float, beta2: float, tau: float) -> Callable[[float], float]:
    """Build Nelson-Siegel yield curve"""
    
    def yield_curve(t: float) -> float:
        if t <= 0:
            return beta0 + beta1
        
        factor1 = (1 - np.exp(-t / tau)) / (t / tau)
        factor2 = factor1 - np.exp(-t / tau)
        
        yield_rate = beta0 + beta1 * factor1 + beta2 * factor2
        return yield_rate
    
    def discount_curve(t: float) -> float:
        return np.exp(-yield_curve(t) * t)
    
    return discount_curve


def build_cubic_spline_curve(tenors: List[float], rates: List[float]) -> Callable[[float], float]:
    """Build cubic spline interpolated curve"""
    
    from scipy.interpolate import CubicSpline
    
    # Create spline interpolation
    cs = CubicSpline(tenors, rates, extrapolate=True)
    
    def discount_curve(t: float) -> float:
        rate = cs(t)
        return np.exp(-rate * t)
    
    return discount_curve


def build_bootstrap_curve(instruments: List[Dict]) -> Callable[[float], float]:
    """Bootstrap curve from market instruments"""
    
    # This would implement full curve bootstrapping
    # For now, returning a simple flat curve
    avg_rate = np.mean([inst['rate'] for inst in instruments])
    
    return lambda t: np.exp(-avg_rate * t)


# Usage example
if __name__ == "__main__":
    # Example usage
    pricer = EnhancedSwapPricer()
    
    # Simple flat curves for demonstration
    discount_curve = lambda t: np.exp(-0.03 * t)
    forward_curve = lambda t: 0.025 + 0.005 * t  # Upward sloping
    
    result = pricer.price_interest_rate_swap(
        notional=100_000_000,
        fixed_rate=0.035,
        tenor_years=5,
        payment_frequency="Quarterly",
        discount_curve=discount_curve,
        forward_curve=forward_curve,
        floating_spread=0.001,
        swap_type="payer"
    )
    
    print(f"Swap NPV: ${result.npv:,.0f}")
    print(f"Par Rate: {result.par_rate:.4%}")
    print(f"DV01: ${result.dv01:,.0f}")
    print(f"Duration: {result.duration:.2f} years")
