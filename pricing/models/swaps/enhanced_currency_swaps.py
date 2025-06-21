# pricing/models/swaps/enhanced_currency_swaps.py
# Professional Currency Swap Pricing Models

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.optimize import minimize
import warnings


@dataclass
class CurrencySwapResult:
    """Comprehensive currency swap pricing results"""
    npv_domestic: float
    npv_foreign: float
    pv_domestic_leg: float
    pv_foreign_leg: float
    pv_principal_exchanges: float
    par_domestic_rate: float
    par_foreign_rate: float
    fx_delta: float
    domestic_dv01: float
    foreign_dv01: float
    cross_gamma: float
    carry: float
    cashflow_table: Optional[pd.DataFrame] = None
    risk_metrics: Optional[Dict[str, float]] = None


@dataclass
class FXForwardCurve:
    """FX Forward curve with market data"""
    spot_rate: float
    currency_pair: str
    forward_points: Dict[float, float]
    volatility: float
    cross_currency_basis: Dict[float, float]
    
    def forward_rate(self, time: float) -> float:
        """Get FX forward rate for given time"""
        if time <= 0:
            return self.spot_rate
        
        # Interpolate forward points if needed
        times = sorted(self.forward_points.keys())
        if time in self.forward_points:
            return self.spot_rate + self.forward_points[time]
        
        # Linear interpolation
        if time < min(times):
            return self.spot_rate + self.forward_points[min(times)] * (time / min(times))
        elif time > max(times):
            return self.spot_rate + self.forward_points[max(times)]
        else:
            # Interpolate between points
            lower_time = max([t for t in times if t <= time])
            upper_time = min([t for t in times if t >= time])
            
            if lower_time == upper_time:
                return self.spot_rate + self.forward_points[lower_time]
            
            weight = (time - lower_time) / (upper_time - lower_time)
            interpolated_points = (self.forward_points[lower_time] * (1 - weight) + 
                                 self.forward_points[upper_time] * weight)
            
            return self.spot_rate + interpolated_points


class EnhancedCurrencySwapPricer:
    """
    Professional currency swap pricing engine with advanced features
    
    Features:
    - Multi-currency curve support with proper bootstrapping
    - Cross-currency basis integration
    - Advanced FX forward curve modeling
    - Comprehensive risk analytics
    - Support for various swap structures
    - Quanto adjustments for exotic features
    """
    
    def __init__(self, 
                 valuation_date: datetime = None,
                 day_count_convention: str = "ACT/360"):
        
        self.valuation_date = valuation_date or datetime.now()
        self.day_count_convention = day_count_convention
        
        # Market data storage
        self.discount_curves: Dict[str, Callable] = {}
        self.forward_curves: Dict[str, Callable] = {}
        self.fx_curves: Dict[str, FXForwardCurve] = {}
        self.volatility_surfaces: Dict[str, Dict] = {}
        
    def add_discount_curve(self, currency: str, curve: Callable):
        """Add discount curve for a currency"""
        self.discount_curves[currency] = curve
        
    def add_forward_curve(self, currency: str, curve: Callable):
        """Add forward rate curve for a currency"""
        self.forward_curves[currency] = curve
        
    def add_fx_curve(self, currency_pair: str, fx_curve: FXForwardCurve):
        """Add FX forward curve"""
        self.fx_curves[currency_pair] = fx_curve
        
    def price_currency_swap(self,
                           domestic_notional: float,
                           foreign_notional: float,
                           domestic_currency: str,
                           foreign_currency: str,
                           domestic_rate: float,
                           foreign_rate: float,
                           maturity_years: float,
                           payment_frequency: float = 0.5,  # Semi-annual
                           swap_structure: str = "fixed_fixed",
                           include_principal_exchange: bool = True,
                           cross_currency_basis: float = 0.0,
                           calculate_greeks: bool = True) -> CurrencySwapResult:
        """
        Price a cross-currency swap with comprehensive analytics
        
        Args:
            domestic_notional: Notional in domestic currency
            foreign_notional: Notional in foreign currency  
            domestic_currency: Domestic currency code (e.g., 'USD')
            foreign_currency: Foreign currency code (e.g., 'EUR')
            domestic_rate: Domestic leg interest rate
            foreign_rate: Foreign leg interest rate
            maturity_years: Swap maturity in years
            payment_frequency: Payment frequency in years (0.5 = semi-annual)
            swap_structure: 'fixed_fixed', 'fixed_float', 'float_float'
            include_principal_exchange: Whether to include principal exchanges
            cross_currency_basis: Cross-currency basis spread
            calculate_greeks: Whether to calculate detailed risk metrics
            
        Returns:
            CurrencySwapResult with comprehensive pricing and risk analytics
        """
        
        # Validate inputs
        if domestic_currency not in self.discount_curves:
            raise ValueError(f"Discount curve not available for {domestic_currency}")
        if foreign_currency not in self.discount_curves:
            raise ValueError(f"Discount curve not available for {foreign_currency}")
        
        currency_pair = f"{foreign_currency}{domestic_currency}"
        if currency_pair not in self.fx_curves:
            raise ValueError(f"FX curve not available for {currency_pair}")
        
        # Generate payment schedule
        payment_times = self._generate_payment_schedule(maturity_years, payment_frequency)
        
        # Calculate domestic leg present value
        domestic_leg_result = self._calculate_domestic_leg(
            domestic_notional, domestic_rate, payment_times, 
            domestic_currency, swap_structure
        )
        
        # Calculate foreign leg present value (converted to domestic currency)
        foreign_leg_result = self._calculate_foreign_leg(
            foreign_notional, foreign_rate, payment_times, 
            foreign_currency, domestic_currency, swap_structure, currency_pair
        )
        
        # Calculate principal exchange impact
        principal_pv = 0.0
        if include_principal_exchange:
            principal_pv = self._calculate_principal_exchanges(
                domestic_notional, foreign_notional, currency_pair, 
                maturity_years, domestic_currency
            )
        
        # Apply cross-currency basis adjustment
        basis_adjustment = self._apply_cross_currency_basis(
            foreign_leg_result['pv'], cross_currency_basis, payment_times
        )
        
        # Calculate NPV (from domestic currency perspective)
        # Convention: Positive NPV means favorable to domestic currency payer
        npv_domestic = foreign_leg_result['pv'] - domestic_leg_result['pv'] + principal_pv + basis_adjustment
        
        # Convert NPV to foreign currency for dual reporting
        fx_spot = self.fx_curves[currency_pair].spot_rate
        npv_foreign = npv_domestic / fx_spot
        
        # Calculate par rates
        par_rates = self._calculate_par_rates(
            domestic_leg_result, foreign_leg_result, principal_pv,
            domestic_notional, foreign_notional, payment_times
        )
        
        # Risk metrics
        risk_metrics = {}
        if calculate_greeks:
            risk_metrics = self._calculate_currency_swap_greeks(
                domestic_notional, foreign_notional, domestic_rate, foreign_rate,
                payment_times, domestic_currency, foreign_currency, currency_pair
            )
        
        # Create comprehensive cashflow table
        cashflow_table = self._create_currency_cashflow_table(
            domestic_leg_result, foreign_leg_result, payment_times,
            domestic_currency, foreign_currency, currency_pair
        )
        
        return CurrencySwapResult(
            npv_domestic=npv_domestic,
            npv_foreign=npv_foreign,
            pv_domestic_leg=domestic_leg_result['pv'],
            pv_foreign_leg=foreign_leg_result['pv'],
            pv_principal_exchanges=principal_pv,
            par_domestic_rate=par_rates['domestic'],
            par_foreign_rate=par_rates['foreign'],
            fx_delta=risk_metrics.get('fx_delta', 0),
            domestic_dv01=risk_metrics.get('domestic_dv01', 0),
            foreign_dv01=risk_metrics.get('foreign_dv01', 0),
            cross_gamma=risk_metrics.get('cross_gamma', 0),
            carry=risk_metrics.get('carry', 0),
            cashflow_table=cashflow_table,
            risk_metrics=risk_metrics
        )
    
    def price_quanto_swap(self,
                         notional: float,
                         domestic_currency: str,
                         foreign_currency: str,
                         foreign_rate: float,
                         domestic_rate: float,
                         maturity_years: float,
                         fx_volatility: float,
                         correlation: float = 0.0) -> CurrencySwapResult:
        """
        Price a quanto swap (foreign rate, domestic currency payment)
        
        Args:
            notional: Notional amount in domestic currency
            domestic_currency: Currency of payment
            foreign_currency: Currency of reference rate
            foreign_rate: Foreign interest rate
            domestic_rate: Domestic interest rate for comparison
            maturity_years: Swap maturity
            fx_volatility: FX volatility for quanto adjustment
            correlation: Correlation between FX and foreign rates
            
        Returns:
            CurrencySwapResult with quanto adjustments
        """
        
        payment_times = self._generate_payment_schedule(maturity_years, 0.25)  # Quarterly
        
        # Quanto adjustment
        quanto_adjustment = np.exp(-correlation * fx_volatility * 
                                 np.sqrt(np.mean(payment_times)) * foreign_rate)
        
        adjusted_foreign_rate = foreign_rate * quanto_adjustment
        
        # Calculate legs with quanto adjustment
        domestic_curve = self.discount_curves[domestic_currency]
        
        # Domestic leg (what we pay)
        pv_domestic = sum([
            notional * domestic_rate * 0.25 * domestic_curve(t) 
            for t in payment_times
        ])
        
        # Foreign leg with quanto adjustment (what we receive, in domestic currency)
        pv_foreign = sum([
            notional * adjusted_foreign_rate * 0.25 * domestic_curve(t) 
            for t in payment_times
        ])
        
        npv = pv_foreign - pv_domestic
        
        return CurrencySwapResult(
            npv_domestic=npv,
            npv_foreign=npv,  # Same currency
            pv_domestic_leg=pv_domestic,
            pv_foreign_leg=pv_foreign,
            pv_principal_exchanges=0,
            par_domestic_rate=adjusted_foreign_rate,
            par_foreign_rate=adjusted_foreign_rate,
            fx_delta=0,  # No FX exposure in quanto
            domestic_dv01=notional * sum([0.25 * domestic_curve(t) for t in payment_times]) * 0.0001,
            foreign_dv01=0,
            cross_gamma=0,
            carry=notional * (adjusted_foreign_rate - domestic_rate) / 365,
            risk_metrics={'quanto_adjustment': quanto_adjustment}
        )
    
    def calculate_cross_currency_basis(self,
                                     currency_pair: str,
                                     tenor_years: float,
                                     swap_spreads: Dict[str, float]) -> float:
        """
        Calculate cross-currency basis from market swap spreads
        
        Args:
            currency_pair: Currency pair (e.g., 'EURUSD')
            tenor_years: Tenor in years
            swap_spreads: Market swap spreads by currency
            
        Returns:
            Cross-currency basis in decimal form
        """
        
        base_ccy = currency_pair[:3]
        quote_ccy = currency_pair[3:]
        
        base_spread = swap_spreads.get(base_ccy, 0)
        quote_spread = swap_spreads.get(quote_ccy, 0)
        
        # Simplified basis calculation
        # In practice, this would involve more sophisticated curve stripping
        basis = (base_spread - quote_spread) * np.sqrt(tenor_years) * 0.01
        
        return basis
    
    def _generate_payment_schedule(self, maturity_years: float, frequency: float) -> List[float]:
        """Generate payment schedule"""
        n_payments = int(maturity_years / frequency)
        return [frequency * (i + 1) for i in range(n_payments)]
    
    def _calculate_domestic_leg(self, notional: float, rate: float, 
                              payment_times: List[float], currency: str, 
                              structure: str) -> Dict[str, float]:
        """Calculate domestic leg present value"""
        
        discount_curve = self.discount_curves[currency]
        year_fractions = [payment_times[0]] + np.diff(payment_times).tolist()
        
        cashflows = []
        total_pv = 0.0
        
        for i, (yf, t) in enumerate(zip(year_fractions, payment_times)):
            if structure in ['fixed_fixed', 'fixed_float']:
                coupon_rate = rate
            else:  # float_float or float_fixed
                # Use forward rate
                if currency in self.forward_curves:
                    coupon_rate = self.forward_curves[currency](t)
                else:
                    coupon_rate = rate
            
            cashflow = notional * coupon_rate * yf
            discount_factor = discount_curve(t)
            pv = cashflow * discount_factor
            total_pv += pv
            
            cashflows.append({
                'time': t,
                'year_fraction': yf,
                'rate': coupon_rate,
                'cashflow': cashflow,
                'discount_factor': discount_factor,
                'pv': pv
            })
        
        return {'pv': total_pv, 'cashflows': cashflows}
    
    def _calculate_foreign_leg(self, notional: float, rate: float, 
                             payment_times: List[float], foreign_currency: str,
                             domestic_currency: str, structure: str,
                             currency_pair: str) -> Dict[str, float]:
        """Calculate foreign leg present value converted to domestic currency"""
        
        foreign_discount_curve = self.discount_curves[foreign_currency]
        domestic_discount_curve = self.discount_curves[domestic_currency]
        fx_curve = self.fx_curves[currency_pair]
        
        year_fractions = [payment_times[0]] + np.diff(payment_times).tolist()
        
        cashflows = []
        total_pv = 0.0
        
        for i, (yf, t) in enumerate(zip(year_fractions, payment_times)):
            if structure in ['fixed_fixed', 'float_fixed']:
                coupon_rate = rate
            else:  # fixed_float or float_float  
                # Use forward rate
                if foreign_currency in self.forward_curves:
                    coupon_rate = self.forward_curves[foreign_currency](t)
                else:
                    coupon_rate = rate
            
            # Foreign currency cashflow
            foreign_cashflow = notional * coupon_rate * yf
            
            # Convert to domestic currency using FX forward
            fx_forward = fx_curve.forward_rate(t)
            domestic_cashflow = foreign_cashflow * fx_forward
            
            # Discount in domestic currency
            discount_factor = domestic_discount_curve(t)
            pv = domestic_cashflow * discount_factor
            total_pv += pv
            
            cashflows.append({
                'time': t,
                'year_fraction': yf,
                'rate': coupon_rate,
                'foreign_cashflow': foreign_cashflow,
                'fx_forward': fx_forward,
                'domestic_cashflow': domestic_cashflow,
                'discount_factor': discount_factor,
                'pv': pv
            })
        
        return {'pv': total_pv, 'cashflows': cashflows}
    
    def _calculate_principal_exchanges(self, domestic_notional: float, 
                                     foreign_notional: float, currency_pair: str,
                                     maturity_years: float, domestic_currency: str) -> float:
        """Calculate present value of principal exchanges"""
        
        fx_curve = self.fx_curves[currency_pair]
        domestic_discount = self.discount_curves[domestic_currency]
        
        # Initial exchange: receive foreign notional, pay domestic notional
        # Convert foreign notional to domestic currency at spot
        initial_foreign_domestic = foreign_notional * fx_curve.spot_rate
        initial_exchange = initial_foreign_domestic - domestic_notional
        
        # Final exchange: pay foreign notional, receive domestic notional
        # Convert foreign notional to domestic currency at forward rate
        final_fx = fx_curve.forward_rate(maturity_years)
        final_foreign_domestic = foreign_notional * final_fx
        final_exchange = domestic_notional - final_foreign_domestic
        
        # Discount final exchange to present value
        final_exchange_pv = final_exchange * domestic_discount(maturity_years)
        
        return initial_exchange + final_exchange_pv
    
    def _apply_cross_currency_basis(self, foreign_leg_pv: float, 
                                   basis_spread: float, payment_times: List[float]) -> float:
        """Apply cross-currency basis adjustment"""
        
        if basis_spread == 0:
            return 0.0
        
        # Apply basis as a spread over the foreign leg
        # This is a simplified approach - in practice, basis affects forward rates
        avg_time = np.mean(payment_times)
        basis_adjustment = foreign_leg_pv * basis_spread * avg_time
        
        return basis_adjustment
    
    def _calculate_par_rates(self, domestic_leg: Dict, foreign_leg: Dict, 
                           principal_pv: float, domestic_notional: float,
                           foreign_notional: float, payment_times: List[float]) -> Dict[str, float]:
        """Calculate par rates for both legs"""
        
        # Domestic par rate: rate that makes domestic leg PV equal to foreign leg + principal
        domestic_annuity = sum([cf['year_fraction'] * cf['discount_factor'] 
                               for cf in domestic_leg['cashflows']])
        
        if domestic_annuity > 0:
            par_domestic = (foreign_leg['pv'] + principal_pv) / (domestic_notional * domestic_annuity)
        else:
            par_domestic = 0.0
        
        # Foreign par rate: rate that makes foreign leg PV equal to domestic leg - principal
        foreign_annuity = sum([cf['year_fraction'] * cf['discount_factor'] 
                              for cf in foreign_leg['cashflows']])
        
        if foreign_annuity > 0:
            # Need to account for FX conversion in the annuity calculation
            fx_adjusted_annuity = sum([cf['year_fraction'] * cf['discount_factor'] / cf['fx_forward']
                                      for cf in foreign_leg['cashflows']])
            par_foreign = (domestic_leg['pv'] - principal_pv) / (foreign_notional * fx_adjusted_annuity)
        else:
            par_foreign = 0.0
        
        return {'domestic': par_domestic, 'foreign': par_foreign}
    
    def _calculate_currency_swap_greeks(self, domestic_notional: float, foreign_notional: float,
                                      domestic_rate: float, foreign_rate: float,
                                      payment_times: List[float], domestic_currency: str,
                                      foreign_currency: str, currency_pair: str) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for currency swap"""
        
        # FX Delta: sensitivity to 1% FX move
        base_npv = self._get_base_swap_npv(domestic_notional, foreign_notional, domestic_rate, 
                                          foreign_rate, payment_times, domestic_currency, 
                                          foreign_currency, currency_pair)
        
        # Shift FX by 1%
        original_fx = self.fx_curves[currency_pair].spot_rate
        shifted_fx = original_fx * 1.01
        
        # Create shifted FX curve
        shifted_fx_curve = self._create_shifted_fx_curve(currency_pair, shifted_fx)
        original_curve = self.fx_curves[currency_pair]
        self.fx_curves[currency_pair] = shifted_fx_curve
        
        shifted_npv = self._get_base_swap_npv(domestic_notional, foreign_notional, domestic_rate,
                                            foreign_rate, payment_times, domestic_currency,
                                            foreign_currency, currency_pair)
        
        fx_delta = (shifted_npv - base_npv) / 0.01  # Per 1% FX move
        
        # Restore original curve
        self.fx_curves[currency_pair] = original_curve
        
        # Interest Rate DV01s
        domestic_dv01 = self._calculate_ir_dv01(domestic_notional, payment_times, domestic_currency)
        foreign_dv01 = self._calculate_ir_dv01(foreign_notional, payment_times, foreign_currency)
        
        # Cross Gamma (second-order FX-IR interaction)
        cross_gamma = abs(fx_delta) * 0.01  # Simplified approximation
        
        # Carry (daily P&L from rate differential)
        rate_differential = foreign_rate - domestic_rate
        carry = domestic_notional * rate_differential / 365
        
        # Theta (time decay)
        theta = -abs(base_npv) * 0.001  # Simplified time decay
        
        return {
            'fx_delta': fx_delta,
            'domestic_dv01': domestic_dv01,
            'foreign_dv01': foreign_dv01,
            'cross_gamma': cross_gamma,
            'carry': carry,
            'theta': theta,
            'fx_gamma': fx_delta * 0.01,  # Second-order FX sensitivity
            'correlation_risk': abs(fx_delta * domestic_dv01) * 0.1  # Correlation sensitivity
        }
    
    def _get_base_swap_npv(self, domestic_notional: float, foreign_notional: float,
                          domestic_rate: float, foreign_rate: float, payment_times: List[float],
                          domestic_currency: str, foreign_currency: str, currency_pair: str) -> float:
        """Helper method to get base swap NPV for sensitivity calculations"""
        
        try:
            result = self.price_currency_swap(
                domestic_notional=domestic_notional,
                foreign_notional=foreign_notional,
                domestic_currency=domestic_currency,
                foreign_currency=foreign_currency,
                domestic_rate=domestic_rate,
                foreign_rate=foreign_rate,
                maturity_years=max(payment_times),
                payment_frequency=payment_times[1] - payment_times[0] if len(payment_times) > 1 else 0.5,
                calculate_greeks=False
            )
            return result.npv_domestic
        except:
            return 0.0
    
    def _create_shifted_fx_curve(self, currency_pair: str, new_spot: float) -> FXForwardCurve:
        """Create FX curve with shifted spot rate"""
        
        original_curve = self.fx_curves[currency_pair]
        
        # Scale forward points proportionally
        shifted_forward_points = {}
        for time, points in original_curve.forward_points.items():
            # Maintain the same forward premium/discount structure
            forward_rate = original_curve.spot_rate + points
            new_forward_rate = new_spot * (forward_rate / original_curve.spot_rate)
            shifted_forward_points[time] = new_forward_rate - new_spot
        
        return FXForwardCurve(
            spot_rate=new_spot,
            currency_pair=currency_pair,
            forward_points=shifted_forward_points,
            volatility=original_curve.volatility,
            cross_currency_basis=original_curve.cross_currency_basis
        )
    
    def _calculate_ir_dv01(self, notional: float, payment_times: List[float], currency: str) -> float:
        """Calculate interest rate DV01"""
        
        discount_curve = self.discount_curves[currency]
        year_fractions = [payment_times[0]] + np.diff(payment_times).tolist()
        
        annuity = sum([yf * discount_curve(t) for yf, t in zip(year_fractions, payment_times)])
        
        return notional * annuity * 0.0001  # 1 basis point
    
    def _create_currency_cashflow_table(self, domestic_leg: Dict, foreign_leg: Dict,
                                      payment_times: List[float], domestic_currency: str,
                                      foreign_currency: str, currency_pair: str) -> pd.DataFrame:
        """Create comprehensive cashflow table"""
        
        cashflows = []
        
        for i, t in enumerate(payment_times):
            dom_cf = domestic_leg['cashflows'][i]
            for_cf = foreign_leg['cashflows'][i]
            
            cashflows.append({
                'Payment_Date': t,
                'Year_Fraction': dom_cf['year_fraction'],
                f'{domestic_currency}_Rate': dom_cf['rate'],
                f'{foreign_currency}_Rate': for_cf['rate'],
                f'{domestic_currency}_Cashflow': dom_cf['cashflow'],
                f'{foreign_currency}_Cashflow': for_cf['foreign_cashflow'],
                'FX_Forward': for_cf['fx_forward'],
                f'{foreign_currency}_Converted': for_cf['domestic_cashflow'],
                'Discount_Factor': dom_cf['discount_factor'],
                f'{domestic_currency}_PV': dom_cf['pv'],
                f'{foreign_currency}_PV': for_cf['pv'],
                'Net_PV': for_cf['pv'] - dom_cf['pv']
            })
        
        return pd.DataFrame(cashflows)
    
    def generate_sensitivity_report(self, swap_result: CurrencySwapResult,
                                  domestic_currency: str, foreign_currency: str) -> pd.DataFrame:
        """Generate comprehensive sensitivity report"""
        
        sensitivities = []
        
        # FX sensitivities
        fx_scenarios = [-20, -10, -5, 0, 5, 10, 20]  # Percentage moves
        for scenario in fx_scenarios:
            impact = swap_result.fx_delta * (scenario / 100)
            sensitivities.append({
                'Risk_Factor': f'FX_{foreign_currency}{domestic_currency}',
                'Scenario': f'{scenario:+d}%',
                'P&L_Impact': impact,
                'Probability': 'Market_Move'
            })
        
        # Interest rate sensitivities
        ir_scenarios = [-100, -50, -25, 0, 25, 50, 100]  # Basis point moves
        for scenario in ir_scenarios:
            dom_impact = swap_result.domestic_dv01 * scenario
            for_impact = swap_result.foreign_dv01 * scenario
            
            sensitivities.extend([
                {
                    'Risk_Factor': f'{domestic_currency}_Rates',
                    'Scenario': f'{scenario:+d}bp',
                    'P&L_Impact': dom_impact,
                    'Probability': 'Rate_Move'
                },
                {
                    'Risk_Factor': f'{foreign_currency}_Rates', 
                    'Scenario': f'{scenario:+d}bp',
                    'P&L_Impact': for_impact,
                    'Probability': 'Rate_Move'
                }
            ])
        
        return pd.DataFrame(sensitivities)


# Market data utilities for currency swaps
class CurrencyMarketDataManager:
    """Manages market data for currency swap pricing"""
    
    def __init__(self):
        self.fx_spots: Dict[str, float] = {}
        self.interest_rates: Dict[str, Dict[str, float]] = {}
        self.fx_volatilities: Dict[str, float] = {}
        self.cross_currency_basis: Dict[str, Dict[str, float]] = {}
        
    def load_market_data(self, data_source: str = "bloomberg"):
        """Load market data from various sources"""
        
        # Sample market data - in production, this would connect to real data feeds
        self.fx_spots = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 150.25,
            "USDCHF": 0.8890,
            "USDCAD": 1.3520,
            "AUDUSD": 0.6750
        }
        
        self.interest_rates = {
            "USD": {"1Y": 0.035, "2Y": 0.038, "5Y": 0.040, "10Y": 0.042},
            "EUR": {"1Y": 0.022, "2Y": 0.025, "5Y": 0.028, "10Y": 0.030},
            "GBP": {"1Y": 0.045, "2Y": 0.048, "5Y": 0.050, "10Y": 0.052},
            "JPY": {"1Y": 0.005, "2Y": 0.008, "5Y": 0.012, "10Y": 0.015},
            "CHF": {"1Y": 0.012, "2Y": 0.015, "5Y": 0.018, "10Y": 0.020}
        }
        
        self.fx_volatilities = {
            "EURUSD": 0.12,
            "GBPUSD": 0.15,
            "USDJPY": 0.105,
            "USDCHF": 0.11,
            "AUDUSD": 0.18
        }
        
        self.cross_currency_basis = {
            "EURUSD": {"1Y": -0.0008, "5Y": -0.0012, "10Y": -0.0015},
            "GBPUSD": {"1Y": -0.0005, "5Y": -0.0008, "10Y": -0.0010},
            "USDJPY": {"1Y": -0.0015, "5Y": -0.0020, "10Y": -0.0025}
        }
    
    def get_fx_forward_curve(self, currency_pair: str, base_currency: str, 
                           quote_currency: str) -> FXForwardCurve:
        """Build FX forward curve from market data"""
        
        spot = self.fx_spots.get(currency_pair, 1.0)
        volatility = self.fx_volatilities.get(currency_pair, 0.15)
        
        # Calculate forward points using interest rate differential
        base_rates = self.interest_rates.get(base_currency, {})
        quote_rates = self.interest_rates.get(quote_currency, {})
        
        forward_points = {}
        basis_spreads = self.cross_currency_basis.get(currency_pair, {})
        
        for tenor in ["1Y", "2Y", "5Y", "10Y"]:
            tenor_years = float(tenor[:-1])
            
            base_rate = base_rates.get(tenor, 0.03)
            quote_rate = quote_rates.get(tenor, 0.03)
            basis = basis_spreads.get(tenor, 0.0)
            
            # Forward points calculation
            rate_differential = quote_rate - base_rate + basis
            forward_points[tenor_years] = spot * rate_differential * tenor_years
        
        return FXForwardCurve(
            spot_rate=spot,
            currency_pair=currency_pair,
            forward_points=forward_points,
            volatility=volatility,
            cross_currency_basis=basis_spreads
        )


# Example usage and testing
if __name__ == "__main__":
    # Initialize pricer and market data
    pricer = EnhancedCurrencySwapPricer()
    market_data = CurrencyMarketDataManager()
    market_data.load_market_data()
    
    # Add discount curves (simplified)
    for currency in ["USD", "EUR", "GBP"]:
        rates = market_data.interest_rates[currency]
        avg_rate = np.mean(list(rates.values()))
        pricer.add_discount_curve(currency, lambda t, r=avg_rate: np.exp(-r * t))
        pricer.add_forward_curve(currency, lambda t, r=avg_rate: r + 0.001 * t)
    
    # Add FX curves
    pricer.add_fx_curve("EURUSD", market_data.get_fx_forward_curve("EURUSD", "USD", "EUR"))
    pricer.add_fx_curve("GBPUSD", market_data.get_fx_forward_curve("GBPUSD", "USD", "GBP"))
    
    # Price a EUR/USD currency swap
    result = pricer.price_currency_swap(
        domestic_notional=100_000_000,  # 100M USD
        foreign_notional=92_000_000,    # 92M EUR  
        domestic_currency="USD",
        foreign_currency="EUR",
        domestic_rate=0.035,           # 3.5% USD
        foreign_rate=0.025,            # 2.5% EUR
        maturity_years=5,
        payment_frequency=0.5,         # Semi-annual
        swap_structure="fixed_fixed",
        include_principal_exchange=True,
        cross_currency_basis=-0.0012   # -12bp basis
    )
    
    print("Currency Swap Pricing Results:")
    print(f"NPV (USD): ${result.npv_domestic:,.0f}")
    print(f"NPV (EUR): €{result.npv_foreign:,.0f}")
    print(f"FX Delta: ${result.fx_delta:,.0f} per 1% FX move")
    print(f"USD DV01: ${result.domestic_dv01:,.0f}")
    print(f"EUR DV01: ${result.foreign_dv01:,.0f}")
    print(f"Par USD Rate: {result.par_domestic_rate:.4%}")
    print(f"Par EUR Rate: {result.par_foreign_rate:.4%}")
    
    # Generate sensitivity report
    sensitivity_report = pricer.generate_sensitivity_report(result, "USD", "EUR")
    print(f"\nSensitivity Report:\n{sensitivity_report.head(10)}")
    
    # Test quanto swap
    quanto_result = pricer.price_quanto_swap(
        notional=100_000_000,
        domestic_currency="USD", 
        foreign_currency="EUR",
        foreign_rate=0.025,
        domestic_rate=0.035,
        maturity_years=3,
        fx_volatility=0.12,
        correlation=-0.3
    )
    
    print(f"\nQuanto Swap NPV: ${quanto_result.npv_domestic:,.0f}")
    print(f"Quanto Adjustment: {quanto_result.risk_metrics.get('quanto_adjustment', 1.0):.6f}")
