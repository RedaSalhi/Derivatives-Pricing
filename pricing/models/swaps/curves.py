# pricing/models/swaps/curves.py
# Enhanced Curve Building and Market Data Management

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.optimize import minimize
import warnings


@dataclass
class MarketInstrument:
    """Represents a market instrument for curve construction"""
    instrument_type: str  # "cash", "swap", "future", "bond"
    maturity: float       # Time to maturity in years
    rate: float          # Market rate/price
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_updated: Optional[datetime] = None
    source: str = "market"


@dataclass
class CurveNode:
    """Individual point on a yield curve"""
    time: float
    discount_factor: float
    zero_rate: float
    forward_rate: float
    
    @property
    def yield_rate(self) -> float:
        """Continuously compounded yield"""
        return -np.log(self.discount_factor) / self.time if self.time > 0 else 0


class EnhancedDiscountCurve:
    """
    Professional-grade discount curve with advanced interpolation and analytics
    
    Features:
    - Multiple interpolation methods (linear, cubic spline, monotonic cubic)
    - Forward rate calculations with proper conventions
    - Risk metrics (DV01, key rate durations)
    - Curve smoothness constraints
    - Multi-currency support
    """
    
    def __init__(self, 
                 curve_name: str,
                 currency: str = "USD",
                 interpolation_method: str = "cubic_spline",
                 extrapolation_method: str = "flat",
                 day_count: str = "ACT/360"):
        
        self.curve_name = curve_name
        self.currency = currency
        self.interpolation_method = interpolation_method
        self.extrapolation_method = extrapolation_method
        self.day_count = day_count
        
        self.nodes: List[CurveNode] = []
        self.market_instruments: List[MarketInstrument] = []
        self._interpolator = None
        self._last_calibration = None
        
    def add_market_instrument(self, instrument: MarketInstrument):
        """Add market instrument for curve construction"""
        self.market_instruments.append(instrument)
        
    def bootstrap_curve(self, 
                       instruments: List[MarketInstrument] = None,
                       smoothing_penalty: float = 0.01) -> bool:
        """
        Bootstrap discount curve from market instruments
        
        Args:
            instruments: List of market instruments (if None, uses self.market_instruments)
            smoothing_penalty: Penalty parameter for curve smoothness
            
        Returns:
            True if successful, False otherwise
        """
        
        if instruments is None:
            instruments = self.market_instruments
        
        if not instruments:
            warnings.warn("No market instruments provided for bootstrapping")
            return False
        
        try:
            # Sort instruments by maturity
            sorted_instruments = sorted(instruments, key=lambda x: x.maturity)
            
            # Initialize curve points
            times = [inst.maturity for inst in sorted_instruments]
            rates = [inst.rate for inst in sorted_instruments]
            
            # Bootstrap discount factors
            discount_factors = self._bootstrap_discount_factors(times, rates, sorted_instruments)
            
            # Create curve nodes
            self.nodes = []
            for i, (t, df, rate) in enumerate(zip(times, discount_factors, rates)):
                zero_rate = -np.log(df) / t if t > 0 else rate
                
                # Calculate forward rate
                if i == 0:
                    forward_rate = zero_rate
                else:
                    dt = t - times[i-1]
                    forward_rate = (zero_rate * t - self.nodes[-1].zero_rate * times[i-1]) / dt
                
                self.nodes.append(CurveNode(
                    time=t,
                    discount_factor=df,
                    zero_rate=zero_rate,
                    forward_rate=forward_rate
                ))
            
            # Build interpolator
            self._build_interpolator()
            self._last_calibration = datetime.now()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Curve bootstrapping failed: {str(e)}")
            return False
    
    def discount_factor(self, time: float) -> float:
        """Get discount factor for given time"""
        if not self._interpolator:
            raise ValueError("Curve not calibrated. Call bootstrap_curve() first.")
        
        if time <= 0:
            return 1.0
        
        # Handle extrapolation
        max_time = max(node.time for node in self.nodes) if self.nodes else 0
        
        if time > max_time:
            if self.extrapolation_method == "flat":
                return self.nodes[-1].discount_factor * np.exp(-self.nodes[-1].forward_rate * (time - max_time))
            elif self.extrapolation_method == "linear":
                # Linear extrapolation in log space
                last_node = self.nodes[-1]
                return last_node.discount_factor * (last_node.discount_factor / self.nodes[-2].discount_factor) ** ((time - last_node.time) / (last_node.time - self.nodes[-2].time))
        
        return float(self._interpolator(time))
    
    def zero_rate(self, time: float) -> float:
        """Get zero (spot) rate for given time"""
        df = self.discount_factor(time)
        return -np.log(df) / time if time > 0 else 0
    
    def forward_rate(self, start_time: float, end_time: float) -> float:
        """Get forward rate between two times"""
        if start_time >= end_time:
            raise ValueError("start_time must be less than end_time")
        
        df_start = self.discount_factor(start_time)
        df_end = self.discount_factor(end_time)
        
        return np.log(df_start / df_end) / (end_time - start_time)
    
    def instantaneous_forward_rate(self, time: float) -> float:
        """Get instantaneous forward rate at given time"""
        epsilon = 1e-6
        if time < epsilon:
            return self.zero_rate(epsilon)
        
        return self.forward_rate(time - epsilon, time + epsilon)
    
    def par_rate(self, maturity: float, payment_frequency: float = 0.5) -> float:
        """Calculate par swap rate for given maturity"""
        
        # Generate payment schedule
        payment_times = np.arange(payment_frequency, maturity + payment_frequency, payment_frequency)
        
        # Calculate annuity
        annuity = sum(payment_frequency * self.discount_factor(t) for t in payment_times)
        
        # Par rate
        if annuity > 0:
            return (1 - self.discount_factor(maturity)) / annuity
        else:
            return self.zero_rate(maturity)
    
    def dv01(self, maturity: float, shift: float = 0.0001) -> float:
        """Calculate DV01 for given maturity"""
        
        # Create shifted curve
        shifted_curve = self._create_shifted_curve(shift)
        
        # Calculate price difference
        base_df = self.discount_factor(maturity)
        shifted_df = shifted_curve.discount_factor(maturity)
        
        return abs(shifted_df - base_df) * 10000  # Convert to basis points
    
    def key_rate_durations(self, maturities: List[float], shift: float = 0.0001) -> Dict[float, float]:
        """Calculate key rate durations for specified maturities"""
        
        durations = {}
        
        for maturity in maturities:
            # Shift only this maturity
            shifted_instruments = []
            for inst in self.market_instruments:
                if abs(inst.maturity - maturity) < 0.1:  # Shift instruments near this maturity
                    shifted_inst = MarketInstrument(
                        instrument_type=inst.instrument_type,
                        maturity=inst.maturity,
                        rate=inst.rate + shift
                    )
                    shifted_instruments.append(shifted_inst)
                else:
                    shifted_instruments.append(inst)
            
            # Create shifted curve
            shifted_curve = EnhancedDiscountCurve(
                curve_name=f"{self.curve_name}_shifted",
                currency=self.currency,
                interpolation_method=self.interpolation_method
            )
            
            if shifted_curve.bootstrap_curve(shifted_instruments):
                # Calculate duration
                base_price = 1.0  # Unit price
                shifted_price = shifted_curve.discount_factor(maturity)
                
                duration = -(shifted_price - base_price) / (shift * base_price)
                durations[maturity] = duration
        
        return durations
    
    def _bootstrap_discount_factors(self, 
                                   times: List[float], 
                                   rates: List[float],
                                   instruments: List[MarketInstrument]) -> List[float]:
        """Bootstrap discount factors from market rates"""
        
        discount_factors = []
        
        for i, (time, rate, inst) in enumerate(zip(times, rates, instruments)):
            if inst.instrument_type == "cash":
                # Cash instruments: DF = 1 / (1 + r * t)
                df = 1.0 / (1.0 + rate * time)
                
            elif inst.instrument_type == "swap":
                # Swap instruments: solve for DF such that swap NPV = 0
                if i == 0:
                    # First swap - approximate
                    df = np.exp(-rate * time)
                else:
                    # Bootstrap using previous discount factors
                    payment_freq = 0.5  # Assume semi-annual
                    payment_times = np.arange(payment_freq, time + payment_freq, payment_freq)
                    
                    # Sum of previous discount factors (annuity)
                    annuity = 0.0
                    for t in payment_times[:-1]:
                        # Interpolate previous discount factors
                        if t <= times[i-1]:
                            idx = min(range(len(times[:i])), key=lambda x: abs(times[x] - t))
                            annuity += payment_freq * discount_factors[idx]
                    
                    # Solve for final discount factor
                    # rate * annuity + rate * payment_freq * df + (1 - df) = 0
                    # df = (1 + rate * annuity) / (1 + rate * payment_freq)
                    df = (1.0 + rate * annuity) / (1.0 + rate * payment_freq)
                    
            elif inst.instrument_type == "future":
                # Future instruments: DF = (100 - future_price) / 100
                df = (100.0 - rate) / 100.0
                
            else:
                # Default: simple exponential discount
                df = np.exp(-rate * time)
            
            discount_factors.append(max(df, 1e-6))  # Floor to avoid negative or zero DFs
        
        return discount_factors
    
    def _build_interpolator(self):
        """Build interpolation function for discount factors"""
        
        if not self.nodes:
            return
        
        times = [node.time for node in self.nodes]
        discount_factors = [node.discount_factor for node in self.nodes]
        
        if self.interpolation_method == "linear":
            self._interpolator = interpolate.interp1d(
                times, discount_factors, kind='linear', 
                bounds_error=False, fill_value='extrapolate'
            )
            
        elif self.interpolation_method == "cubic_spline":
            # Cubic spline in log space for smoothness
            log_dfs = np.log(discount_factors)
            spline = interpolate.CubicSpline(times, log_dfs, bc_type='natural')
            self._interpolator = lambda t: np.exp(spline(t))
            
        elif self.interpolation_method == "monotonic_cubic":
            # Monotonic cubic interpolation
            self._interpolator = interpolate.PchipInterpolator(times, discount_factors)
            
        else:
            # Default to linear
            self._interpolator = interpolate.interp1d(
                times, discount_factors, kind='linear',
                bounds_error=False, fill_value='extrapolate'
            )
    
    def _create_shifted_curve(self, shift: float) -> 'EnhancedDiscountCurve':
        """Create parallel-shifted version of the curve"""
        
        shifted_instruments = []
        for inst in self.market_instruments:
            shifted_inst = MarketInstrument(
                instrument_type=inst.instrument_type,
                maturity=inst.maturity,
                rate=inst.rate + shift,
                bid=inst.bid + shift if inst.bid else None,
                ask=inst.ask + shift if inst.ask else None,
                source=inst.source
            )
            shifted_instruments.append(shifted_inst)
        
        shifted_curve = EnhancedDiscountCurve(
            curve_name=f"{self.curve_name}_shifted",
            currency=self.currency,
            interpolation_method=self.interpolation_method
        )
        shifted_curve.bootstrap_curve(shifted_instruments)
        
        return shifted_curve
    
    def get_curve_summary(self) -> pd.DataFrame:
        """Get summary of curve points and rates"""
        
        if not self.nodes:
            return pd.DataFrame()
        
        data = []
        for node in self.nodes:
            data.append({
                'Time': node.time,
                'Discount_Factor': node.discount_factor,
                'Zero_Rate': node.zero_rate,
                'Forward_Rate': node.forward_rate,
                'Zero_Rate_Pct': node.zero_rate * 100,
                'Forward_Rate_Pct': node.forward_rate * 100
            })
        
        return pd.DataFrame(data)


class MultiCurrencyCurveManager:
    """
    Manages multiple currency curves with cross-currency relationships
    """
    
    def __init__(self):
        self.curves: Dict[str, EnhancedDiscountCurve] = {}
        self.fx_spots: Dict[str, float] = {}  # FX rates vs USD
        self.fx_volatilities: Dict[str, float] = {}
        
    def add_curve(self, currency: str, curve: EnhancedDiscountCurve):
        """Add curve for a currency"""
        self.curves[currency] = curve
        
    def add_fx_spot(self, currency_pair: str, spot_rate: float, volatility: float = 0.15):
        """Add FX spot rate (e.g., 'EURUSD' -> 1.1000)"""
        self.fx_spots[currency_pair] = spot_rate
        self.fx_volatilities[currency_pair] = volatility
        
    def get_fx_forward(self, currency_pair: str, time: float) -> float:
        """Calculate FX forward rate using interest rate parity"""
        
        if currency_pair not in self.fx_spots:
            raise ValueError(f"FX spot rate not available for {currency_pair}")
        
        # Parse currency pair (e.g., 'EURUSD' -> base='EUR', quote='USD')
        base_ccy = currency_pair[:3]
        quote_ccy = currency_pair[3:]
        
        if base_ccy not in self.curves or quote_ccy not in self.curves:
            raise ValueError(f"Discount curves not available for {base_ccy} or {quote_ccy}")
        
        # Interest rate parity: F = S * (DF_quote / DF_base)
        spot = self.fx_spots[currency_pair]
        df_base = self.curves[base_ccy].discount_factor(time)
        df_quote = self.curves[quote_ccy].discount_factor(time)
        
        return spot * (df_quote / df_base)
    
    def get_cross_currency_basis(self, currency_pair: str, tenor: float) -> float:
        """Estimate cross-currency basis (simplified)"""
        
        # This would typically come from market data
        # For now, return a simple function of tenor and volatility
        vol = self.fx_volatilities.get(currency_pair, 0.15)
        basis = vol * np.sqrt(tenor) * 0.01  # Simplified basis calculation
        
        return basis


# Specialized curve builders
class OISCurveBuilder:
    """Builder for Overnight Index Swap (OIS) curves"""
    
    @staticmethod
    def build_fed_funds_curve(market_data: Dict[str, float]) -> EnhancedDiscountCurve:
        """Build Federal Funds OIS curve"""
        
        curve = EnhancedDiscountCurve(
            curve_name="USD_FedFunds_OIS",
            currency="USD",
            interpolation_method="cubic_spline"
        )
        
        # Add market instruments
        for tenor_str, rate in market_data.items():
            tenor_years = _parse_tenor_string(tenor_str)
            
            if tenor_years <= 1:
                inst_type = "cash"
            else:
                inst_type = "swap"
            
            instrument = MarketInstrument(
                instrument_type=inst_type,
                maturity=tenor_years,
                rate=rate,
                source="bloomberg"
            )
            curve.add_market_instrument(instrument)
        
        curve.bootstrap_curve()
        return curve
    
    @staticmethod
    def build_eonia_curve(market_data: Dict[str, float]) -> EnhancedDiscountCurve:
        """Build EONIA OIS curve"""
        
        curve = EnhancedDiscountCurve(
            curve_name="EUR_EONIA_OIS",
            currency="EUR",
            interpolation_method="cubic_spline"
        )
        
        for tenor_str, rate in market_data.items():
            tenor_years = _parse_tenor_string(tenor_str)
            
            instrument = MarketInstrument(
                instrument_type="swap" if tenor_years > 1 else "cash",
                maturity=tenor_years,
                rate=rate,
                source="refinitiv"
            )
            curve.add_market_instrument(instrument)
        
        curve.bootstrap_curve()
        return curve


class LIBORCurveBuilder:
    """Builder for LIBOR-based curves"""
    
    @staticmethod
    def build_usd_libor_curve(
        ois_curve: EnhancedDiscountCurve,
        libor_instruments: Dict[str, float],
        basis_spreads: Dict[str, float] = None
    ) -> EnhancedDiscountCurve:
        """
        Build USD LIBOR curve using dual-curve approach
        
        Args:
            ois_curve: OIS discount curve
            libor_instruments: LIBOR-based instruments (swaps, futures)
            basis_spreads: OIS-LIBOR basis spreads by tenor
        """
        
        curve = EnhancedDiscountCurve(
            curve_name="USD_LIBOR3M",
            currency="USD",
            interpolation_method="cubic_spline"
        )
        
        # Apply basis adjustments if provided
        for tenor_str, rate in libor_instruments.items():
            tenor_years = _parse_tenor_string(tenor_str)
            
            # Adjust for OIS-LIBOR basis
            if basis_spreads and tenor_str in basis_spreads:
                adjusted_rate = rate + basis_spreads[tenor_str]
            else:
                adjusted_rate = rate
            
            instrument = MarketInstrument(
                instrument_type="swap",
                maturity=tenor_years,
                rate=adjusted_rate,
                source="ice"
            )
            curve.add_market_instrument(instrument)
        
        curve.bootstrap_curve()
        return curve


class CreditCurveBuilder:
    """Builder for credit curves (CDS-based)"""
    
    @staticmethod
    def build_cds_curve(
        risk_free_curve: EnhancedDiscountCurve,
        cds_spreads: Dict[str, float],
        recovery_rate: float = 0.4
    ) -> EnhancedDiscountCurve:
        """
        Build credit curve from CDS spreads
        
        Args:
            risk_free_curve: Risk-free discount curve
            cds_spreads: CDS spreads by tenor
            recovery_rate: Assumed recovery rate
        """
        
        curve = EnhancedDiscountCurve(
            curve_name="Credit_Curve",
            currency=risk_free_curve.currency,
            interpolation_method="linear"
        )
        
        # Bootstrap survival probabilities
        for tenor_str, spread in cds_spreads.items():
            tenor_years = _parse_tenor_string(tenor_str)
            
            # Simplified: hazard rate = spread / (1 - recovery)
            hazard_rate = spread / (1 - recovery_rate)
            
            # Survival probability
            survival_prob = np.exp(-hazard_rate * tenor_years)
            
            # Credit-adjusted discount factor
            rf_df = risk_free_curve.discount_factor(tenor_years)
            credit_df = rf_df * survival_prob
            
            # Create pseudo-instrument
            implied_rate = -np.log(credit_df) / tenor_years if tenor_years > 0 else 0
            
            instrument = MarketInstrument(
                instrument_type="cds",
                maturity=tenor_years,
                rate=implied_rate,
                source="markit"
            )
            curve.add_market_instrument(instrument)
        
        curve.bootstrap_curve()
        return curve


# Utility functions
def _parse_tenor_string(tenor_str: str) -> float:
    """Parse tenor string like '1Y', '6M', '3W' to years"""
    
    tenor_str = tenor_str.upper().strip()
    
    if tenor_str.endswith('Y'):
        return float(tenor_str[:-1])
    elif tenor_str.endswith('M'):
        return float(tenor_str[:-1]) / 12.0
    elif tenor_str.endswith('W'):
        return float(tenor_str[:-1]) / 52.0
    elif tenor_str.endswith('D'):
        return float(tenor_str[:-1]) / 365.0
    else:
        # Assume years if no suffix
        try:
            return float(tenor_str)
        except ValueError:
            raise ValueError(f"Cannot parse tenor string: {tenor_str}")


def create_sample_market_data() -> Dict[str, Dict[str, float]]:
    """Create sample market data for testing"""
    
    return {
        "USD_OIS": {
            "1W": 0.020,
            "1M": 0.021,
            "3M": 0.022,
            "6M": 0.024,
            "1Y": 0.026,
            "2Y": 0.028,
            "3Y": 0.030,
            "5Y": 0.032,
            "7Y": 0.034,
            "10Y": 0.035,
            "30Y": 0.036
        },
        "USD_LIBOR": {
            "3M": 0.025,
            "6M": 0.027,
            "1Y": 0.029,
            "2Y": 0.031,
            "3Y": 0.033,
            "5Y": 0.035,
            "7Y": 0.037,
            "10Y": 0.038,
            "30Y": 0.039
        },
        "EUR_OIS": {
            "1W": 0.015,
            "1M": 0.016,
            "3M": 0.017,
            "6M": 0.018,
            "1Y": 0.019,
            "2Y": 0.020,
            "5Y": 0.022,
            "10Y": 0.024,
            "30Y": 0.025
        },
        "FX_VOLS": {
            "EURUSD": 0.12,
            "GBPUSD": 0.15,
            "USDJPY": 0.10
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample market data
    market_data = create_sample_market_data()
    
    # Build USD OIS curve
    usd_ois = OISCurveBuilder.build_fed_funds_curve(market_data["USD_OIS"])
    
    # Build USD LIBOR curve
    usd_libor = LIBORCurveBuilder.build_usd_libor_curve(
        usd_ois, 
        market_data["USD_LIBOR"]
    )
    
    # Test curve functionality
    print("USD OIS Curve:")
    print(f"5Y Zero Rate: {usd_ois.zero_rate(5.0):.4%}")
    print(f"5Y Discount Factor: {usd_ois.discount_factor(5.0):.6f}")
    print(f"5Y Par Rate: {usd_ois.par_rate(5.0):.4%}")
    
    print("\nUSD LIBOR Curve:")
    print(f"5Y Zero Rate: {usd_libor.zero_rate(5.0):.4%}")
    print(f"2Y5Y Forward Rate: {usd_libor.forward_rate(2.0, 7.0):.4%}")
    
    # Risk metrics
    print(f"\n5Y DV01: {usd_ois.dv01(5.0):.2f} bps")
    
    # Key rate durations
    krd = usd_ois.key_rate_durations([2.0, 5.0, 10.0])
    print(f"Key Rate Durations: {krd}")
    
    # Multi-currency manager
    manager = MultiCurrencyCurveManager()
    manager.add_curve("USD", usd_ois)
    manager.add_fx_spot("EURUSD", 1.1000, 0.12)
    
    # Build EUR curve
    eur_ois = OISCurveBuilder.build_eonia_curve(market_data["EUR_OIS"])
    manager.add_curve("EUR", eur_ois)
    
    # FX forward
    print(f"\n1Y EURUSD Forward: {manager.get_fx_forward('EURUSD', 1.0):.4f}")
    
    # Curve summary
    summary = usd_ois.get_curve_summary()
    print(f"\nCurve Summary:\n{summary.head()}")
