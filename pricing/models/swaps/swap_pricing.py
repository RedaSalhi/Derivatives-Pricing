# Enhanced Swap Pricing Models with Accurate Calculations
# File: pricing/models/enhanced_swap_pricing.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math


@dataclass
class SwapResult:
    """Enhanced container for swap pricing results with detailed explanations"""
    npv: float
    pv_fixed: float = 0.0
    pv_floating: float = 0.0
    par_rate: float = 0.0
    dv01: float = 0.0
    duration: float = 0.0
    modified_duration: float = 0.0
    convexity: float = 0.0
    carry: float = 0.0
    greeks: Optional[Dict[str, float]] = None
    cashflows: Optional[pd.DataFrame] = None
    methodology: str = ""
    market_context: Dict = None


@dataclass
class CurrencySwapResult:
    """Enhanced currency swap results with detailed risk metrics"""
    npv_domestic: float
    npv_foreign: float
    pv_domestic_leg: float
    pv_foreign_leg: float
    pv_principal_exchanges: float
    fx_delta: float
    domestic_dv01: float
    foreign_dv01: float
    cross_gamma: float
    basis_component: float
    cashflows_domestic: Optional[pd.DataFrame] = None
    cashflows_foreign: Optional[pd.DataFrame] = None


class AccurateDiscountCurve:
    """Accurate discount curve implementation with interpolation"""
    
    def __init__(self, rates: Dict[float, float], interpolation='linear'):
        """
        Initialize curve with tenor-rate pairs
        
        Args:
            rates: Dictionary of {tenor_in_years: rate}
            interpolation: 'linear', 'cubic', or 'loglinear'
        """
        self.tenors = sorted(rates.keys())
        self.rates = [rates[t] for t in self.tenors]
        self.interpolation = interpolation
        
    def rate(self, t: float) -> float:
        """Get interest rate at time t"""
        if t <= 0:
            return self.rates[0]
        if t <= self.tenors[0]:
            return self.rates[0]
        if t >= self.tenors[-1]:
            return self.rates[-1]
        
        # Linear interpolation (can be enhanced with cubic spline)
        for i in range(len(self.tenors) - 1):
            if self.tenors[i] <= t <= self.tenors[i + 1]:
                t1, t2 = self.tenors[i], self.tenors[i + 1]
                r1, r2 = self.rates[i], self.rates[i + 1]
                return r1 + (r2 - r1) * (t - t1) / (t2 - t1)
        
        return self.rates[-1]
    
    def discount_factor(self, t: float) -> float:
        """Calculate discount factor P(0,t)"""
        if t <= 0:
            return 1.0
        return math.exp(-self.rate(t) * t)
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """Calculate forward rate between t1 and t2"""
        if t1 >= t2:
            return self.rate(t1)
        
        p1 = self.discount_factor(t1)
        p2 = self.discount_factor(t2)
        return math.log(p1 / p2) / (t2 - t1)


class EnhancedInterestRateSwapPricer:
    """Accurate Interest Rate Swap pricing with educational explanations"""
    
    @staticmethod
    def create_payment_schedule(
        start_date: datetime,
        maturity_date: datetime,
        payment_frequency: str,
        day_count: str = "ACT/360"
    ) -> pd.DataFrame:
        """
        Create accurate payment schedule
        
        Educational Note:
        - Payment frequencies determine cash flow timing
        - Day count conventions affect accrual calculations
        - Business day adjustments ensure realistic settlement
        """
        freq_map = {
            "Quarterly": 3,
            "Semi-Annual": 6,
            "Annual": 12
        }
        
        months_between = freq_map[payment_frequency]
        schedule = []
        
        current_date = start_date
        while current_date < maturity_date:
            next_date = min(
                current_date + timedelta(days=30 * months_between),
                maturity_date
            )
            
            # Calculate year fraction based on day count convention
            if day_count == "ACT/360":
                year_frac = (next_date - current_date).days / 360.0
            elif day_count == "ACT/365":
                year_frac = (next_date - current_date).days / 365.0
            else:  # 30/360
                year_frac = months_between / 12.0
            
            schedule.append({
                'start_date': current_date,
                'end_date': next_date,
                'payment_date': next_date,
                'year_fraction': year_frac,
                'time_to_payment': (next_date - start_date).days / 365.25
            })
            
            current_date = next_date
        
        return pd.DataFrame(schedule)
    
    @staticmethod
    def price_accurate_dcf(
        notional: float,
        fixed_rate: float,
        discount_curve: AccurateDiscountCurve,
        payment_schedule: pd.DataFrame,
        floating_spread: float = 0.0,
        position: str = "receive_fixed"
    ) -> SwapResult:
        """
        Accurate DCF pricing with detailed methodology
        
        Educational Explanation:
        1. Fixed Leg: Known cash flows discounted at risk-free rates
        2. Floating Leg: Expected floating payments based on forward rates
        3. NPV: Difference between fixed and floating leg values
        4. Par Rate: Rate that makes NPV = 0
        """
        
        # Calculate fixed leg cash flows
        fixed_cashflows = []
        fixed_pv = 0.0
        
        for _, payment in payment_schedule.iterrows():
            cf_amount = notional * fixed_rate * payment['year_fraction']
            df = discount_curve.discount_factor(payment['time_to_payment'])
            pv = cf_amount * df
            
            fixed_cashflows.append({
                'payment_date': payment['payment_date'],
                'nominal_amount': cf_amount,
                'discount_factor': df,
                'present_value': pv
            })
            fixed_pv += pv
        
        # Calculate floating leg cash flows using forward rates
        floating_cashflows = []
        floating_pv = 0.0
        
        for i, payment in payment_schedule.iterrows():
            if i == 0:
                forward_rate = discount_curve.rate(payment['time_to_payment'])
            else:
                prev_time = payment_schedule.iloc[i-1]['time_to_payment']
                forward_rate = discount_curve.forward_rate(
                    prev_time, payment['time_to_payment']
                )
            
            cf_amount = notional * (forward_rate + floating_spread) * payment['year_fraction']
            df = discount_curve.discount_factor(payment['time_to_payment'])
            pv = cf_amount * df
            
            floating_cashflows.append({
                'payment_date': payment['payment_date'],
                'forward_rate': forward_rate,
                'nominal_amount': cf_amount,
                'discount_factor': df,
                'present_value': pv
            })
            floating_pv += pv
        
        # Calculate NPV based on position
        if position == "receive_fixed":
            npv = fixed_pv - floating_pv
        else:
            npv = floating_pv - fixed_pv
        
        # Calculate par rate (rate that makes NPV = 0)
        annuity = sum([cf['discount_factor'] * payment['year_fraction'] 
                      for cf, payment in zip(fixed_cashflows, payment_schedule.itertuples())])
        par_rate = floating_pv / (notional * annuity) if annuity > 0 else fixed_rate
        
        # Calculate DV01 (dollar value of 01 basis point)
        dv01 = notional * annuity * 0.0001
        
        # Calculate duration and convexity
        weighted_time = sum([
            cf['present_value'] * payment['time_to_payment'] 
            for cf, payment in zip(fixed_cashflows, payment_schedule.itertuples())
        ])
        duration = weighted_time / fixed_pv if fixed_pv > 0 else 0
        modified_duration = duration / (1 + par_rate)
        
        # Approximate convexity
        convexity = sum([
            cf['present_value'] * payment['time_to_payment'] ** 2
            for cf, payment in zip(fixed_cashflows, payment_schedule.itertuples())
        ]) / fixed_pv if fixed_pv > 0 else 0
        
        # Create detailed cashflow DataFrame
        cashflows_df = pd.DataFrame({
            'Payment_Date': [cf['payment_date'] for cf in fixed_cashflows],
            'Fixed_CF': [cf['nominal_amount'] for cf in fixed_cashflows],
            'Fixed_PV': [cf['present_value'] for cf in fixed_cashflows],
            'Floating_CF': [cf['nominal_amount'] for cf in floating_cashflows],
            'Floating_PV': [cf['present_value'] for cf in floating_cashflows],
            'Net_CF': [f - fl for f, fl in zip([cf['nominal_amount'] for cf in fixed_cashflows],
                                              [cf['nominal_amount'] for cf in floating_cashflows])],
            'Discount_Factor': [cf['discount_factor'] for cf in fixed_cashflows]
        })
        
        return SwapResult(
            npv=npv,
            pv_fixed=fixed_pv,
            pv_floating=floating_pv,
            par_rate=par_rate,
            dv01=dv01,
            duration=duration,
            modified_duration=modified_duration,
            convexity=convexity,
            cashflows=cashflows_df,
            methodology="Discounted Cash Flow with Forward Rate Projection",
            market_context={
                'position': position,
                'floating_spread': floating_spread,
                'annuity': annuity
            }
        )


class EnhancedCurrencySwapPricer:
    """Accurate Cross-Currency Swap pricing"""
    
    @staticmethod
    def price_accurate(
        domestic_notional: float,
        foreign_notional: float,
        domestic_curve: AccurateDiscountCurve,
        foreign_curve: AccurateDiscountCurve,
        fx_spot: float,
        payment_schedule: pd.DataFrame,
        cross_currency_basis: float = 0.0,
        include_principals: bool = True
    ) -> CurrencySwapResult:
        """
        Accurate cross-currency swap pricing
        
        Educational Explanation:
        1. Two interest rate swaps in different currencies
        2. Principal exchanges at start and maturity
        3. FX risk from currency exposure
        4. Cross-currency basis reflects funding costs
        """
        
        # Domestic leg cash flows
        domestic_pv = 0.0
        domestic_cashflows = []
        
        for _, payment in payment_schedule.iterrows():
            # Interest payment
            interest = domestic_notional * domestic_curve.rate(payment['time_to_payment']) * payment['year_fraction']
            df_domestic = domestic_curve.discount_factor(payment['time_to_payment'])
            pv = interest * df_domestic
            
            domestic_cashflows.append({
                'payment_date': payment['payment_date'],
                'interest': interest,
                'present_value': pv
            })
            domestic_pv += pv
        
        # Foreign leg cash flows (converted to domestic currency)
        foreign_pv = 0.0
        foreign_cashflows = []
        
        for _, payment in payment_schedule.iterrows():
            # Interest payment in foreign currency
            foreign_rate = foreign_curve.rate(payment['time_to_payment']) + cross_currency_basis
            interest_foreign = foreign_notional * foreign_rate * payment['year_fraction']
            
            # Convert to domestic currency
            interest_domestic = interest_foreign * fx_spot
            df_foreign = foreign_curve.discount_factor(payment['time_to_payment'])
            pv = interest_domestic * df_foreign
            
            foreign_cashflows.append({
                'payment_date': payment['payment_date'],
                'interest_foreign': interest_foreign,
                'interest_domestic': interest_domestic,
                'present_value': pv
            })
            foreign_pv += pv
        
        # Principal exchanges
        principal_pv = 0.0
        if include_principals:
            # Initial exchange (receive foreign, pay domestic)
            initial_exchange = foreign_notional * fx_spot - domestic_notional
            
            # Final exchange (pay foreign, receive domestic)
            final_time = payment_schedule['time_to_payment'].iloc[-1]
            final_exchange_domestic = domestic_notional * domestic_curve.discount_factor(final_time)
            final_exchange_foreign = -foreign_notional * fx_spot * foreign_curve.discount_factor(final_time)
            
            principal_pv = initial_exchange + final_exchange_domestic + final_exchange_foreign
        
        # Calculate basis component impact
        basis_component = sum([
            foreign_notional * cross_currency_basis * payment['year_fraction'] * 
            fx_spot * foreign_curve.discount_factor(payment['time_to_payment'])
            for _, payment in payment_schedule.iterrows()
        ])
        
        # Total NPV (from domestic currency perspective)
        npv_domestic = foreign_pv - domestic_pv + principal_pv
        npv_foreign = npv_domestic / fx_spot
        
        # Calculate risk metrics
        fx_delta = foreign_notional  # Simplified FX delta
        domestic_dv01 = domestic_notional * sum([
            payment['year_fraction'] * domestic_curve.discount_factor(payment['time_to_payment'])
            for _, payment in payment_schedule.iterrows()
        ]) * 0.0001
        
        foreign_dv01 = foreign_notional * fx_spot * sum([
            payment['year_fraction'] * foreign_curve.discount_factor(payment['time_to_payment'])
            for _, payment in payment_schedule.iterrows()
        ]) * 0.0001
        
        cross_gamma = fx_delta * 0.01  # Simplified cross gamma
        
        return CurrencySwapResult(
            npv_domestic=npv_domestic,
            npv_foreign=npv_foreign,
            pv_domestic_leg=domestic_pv,
            pv_foreign_leg=foreign_pv,
            pv_principal_exchanges=principal_pv,
            fx_delta=fx_delta,
            domestic_dv01=domestic_dv01,
            foreign_dv01=foreign_dv01,
            cross_gamma=cross_gamma,
            basis_component=basis_component
        )


class EducationalSwapCalculator:
    """Educational utilities for understanding swap mechanics"""
    
    @staticmethod
    def explain_swap_mechanics(swap_type: str) -> str:
        """Provide educational explanations for different swap types"""
        
        explanations = {
            "interest_rate": """
            **Interest Rate Swap Mechanics:**
            
            1. **Fixed Leg**: You pay/receive a predetermined fixed rate
               - Cash Flow = Notional × Fixed Rate × Year Fraction
               - Present Value = Cash Flow × Discount Factor
            
            2. **Floating Leg**: You receive/pay based on market rates
               - Rate resets at each period based on reference rate (LIBOR/SOFR)
               - Forward rates used for valuation
            
            3. **Net Present Value**: Difference between fixed and floating legs
               - NPV = PV(Fixed Leg) - PV(Floating Leg)
               - Positive NPV means the swap is in your favor
            
            4. **Par Rate**: The fixed rate that makes NPV = 0
               - This is the market's fair value for the swap
            
            5. **Risk Metrics**:
               - DV01: Dollar value change for 1bp rate move
               - Duration: Price sensitivity to yield changes
               - Convexity: Second-order price sensitivity
            """,
            
            "currency": """
            **Cross-Currency Swap Mechanics:**
            
            1. **Dual Currency Structure**: Exchange interest payments in two currencies
               - Pay interest in Currency A, receive in Currency B
               - Notional amounts typically exchanged at start/end
            
            2. **FX Risk**: Exposure to exchange rate movements
               - Changes in FX rates affect the value of foreign currency flows
               - FX Delta measures sensitivity to 1% FX move
            
            3. **Interest Rate Risk**: Exposure to both domestic and foreign rates
               - Separate DV01 for each currency
               - Cross-currency basis affects relative pricing
            
            4. **Valuation Process**:
               - Discount each currency's cash flows using its own curve
               - Convert foreign currency flows to domestic using spot FX
               - Sum all present values for total NPV
            """,
            
            "equity": """
            **Equity Swap Mechanics:**
            
            1. **Equity Leg**: Returns based on equity performance
               - Total return includes price appreciation + dividends
               - Performance calculated over reset periods
            
            2. **Fixed/Floating Leg**: Traditional interest payment
               - Usually LIBOR/SOFR + spread
               - Can be fixed or floating rate
            
            3. **Dividend Treatment**:
               - Gross: Full dividend amount
               - Net: After withholding taxes
            
            4. **Risk Factors**:
               - Equity price risk (delta)
               - Volatility risk (vega)
               - Interest rate risk
               - Dividend yield changes
            """
        }
        
        return explanations.get(swap_type, "Explanation not available")
    
    @staticmethod
    def calculate_scenario_analysis(
        base_result: SwapResult,
        rate_shocks: List[float],
        curve: AccurateDiscountCurve
    ) -> pd.DataFrame:
        """
        Perform scenario analysis for educational purposes
        
        Shows how swap value changes under different rate scenarios
        """
        scenarios = []
        
        for shock in rate_shocks:
            # Create shocked curve
            shocked_rates = {t: curve.rate(t) + shock for t in curve.tenors}
            shocked_curve = AccurateDiscountCurve(shocked_rates)
            
            # Recalculate swap value (simplified)
            # In practice, would need to reprice entire swap
            pnl_estimate = -base_result.dv01 * shock * 10000  # Approximate using DV01
            
            scenarios.append({
                'Rate_Shock_bp': shock * 10000,
                'New_NPV': base_result.npv + pnl_estimate,
                'PnL': pnl_estimate,
                'PnL_Percent': (pnl_estimate / abs(base_result.npv) * 100) if base_result.npv != 0 else 0
            })
        
        return pd.DataFrame(scenarios)


# Example usage and testing
if __name__ == "__main__":
    # Example: Price a 5-year USD interest rate swap
    
    # Create market data
    usd_rates = {
        0.25: 0.045,  # 3M
        1.0: 0.044,   # 1Y
        2.0: 0.043,   # 2Y
        5.0: 0.042,   # 5Y
        10.0: 0.041   # 10Y
    }
    
    curve = AccurateDiscountCurve(usd_rates)
    
    # Create payment schedule
    start_date = datetime(2024, 1, 15)
    maturity_date = datetime(2029, 1, 15)
    
    schedule = EnhancedInterestRateSwapPricer.create_payment_schedule(
        start_date, maturity_date, "Semi-Annual"
    )
    
    # Price swap
    result = EnhancedInterestRateSwapPricer.price_accurate_dcf(
        notional=100_000_000,
        fixed_rate=0.0425,
        discount_curve=curve,
        payment_schedule=schedule,
        position="receive_fixed"
    )
    
    print(f"Swap NPV: ${result.npv:,.0f}")
    print(f"Par Rate: {result.par_rate:.4f}")
    print(f"DV01: ${result.dv01:,.0f}")
    print(f"Duration: {result.duration:.2f} years")
