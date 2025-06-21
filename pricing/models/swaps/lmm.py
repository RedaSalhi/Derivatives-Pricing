# pricing/models/swaps/lmm.py
# Enhanced LIBOR Market Model Implementation

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class LMMParameters:
    """LIBOR Market Model parameters"""
    initial_forwards: np.ndarray
    volatilities: np.ndarray
    correlations: np.ndarray
    payment_times: np.ndarray
    tenor_structure: np.ndarray


@dataclass
class LMMResult:
    """Results from LMM pricing"""
    npv: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    pv_fixed_leg: float
    pv_floating_leg: float
    par_rate: float
    effective_duration: float
    convexity: float
    paths_used: int
    convergence_stats: Optional[Dict] = None


class EnhancedLIBORMarketModel:
    """
    Enhanced LIBOR Market Model for pricing interest rate derivatives
    
    Features:
    - Multi-factor dynamics with full correlation structure
    - Drift adjustments for martingale property
    - Advanced variance reduction techniques
    - Calibration to market cap/floor volatilities
    """
    
    def __init__(self, 
                 tenor_structure: List[float] = None,
                 n_factors: int = 3,
                 correlation_decay: float = 0.1):
        """
        Initialize LMM with market-standard parameters
        
        Args:
            tenor_structure: List of forward rate tenors (e.g., [0.25, 0.5, 0.75, ...])
            n_factors: Number of driving factors (typically 2-5)
            correlation_decay: Exponential decay parameter for correlation
        """
        
        self.tenor_structure = np.array(tenor_structure or 
                                       [0.25 * i for i in range(1, 41)])  # 10Y quarterly
        self.n_factors = n_factors
        self.correlation_decay = correlation_decay
        
        # Initialize default correlation matrix
        self.correlation_matrix = self._build_default_correlation_matrix()
        
        # Cholesky decomposition for simulation
        self.cholesky_matrix = np.linalg.cholesky(self.correlation_matrix)
        
    def calibrate_to_market_data(self, 
                                market_data: Dict[str, float],
                                initial_guess: Optional[np.ndarray] = None) -> LMMParameters:
        """
        Calibrate LMM parameters to market cap/floor volatilities
        
        Args:
            market_data: Dictionary with cap/floor vols by tenor
            initial_guess: Initial volatility parameters
            
        Returns:
            Calibrated LMM parameters
        """
        
        # Extract market volatilities
        market_tenors = list(market_data.keys())
        market_vols = list(market_data.values())
        
        # Initial forward rates (would typically come from curve)
        initial_forwards = np.array([0.03 + 0.001 * i for i in range(len(self.tenor_structure))])
        
        # Initial volatility guess
        if initial_guess is None:
            initial_guess = np.full(len(self.tenor_structure), 0.15)  # 15% vol
        
        # Calibration objective function
        def objective(vol_params):
            try:
                model_vols = self._calculate_cap_volatilities(vol_params, initial_forwards)
                market_vols_interp = np.interp(self.tenor_structure[:len(model_vols)], 
                                             [float(t) for t in market_tenors], market_vols)
                return np.sum((model_vols - market_vols_interp) ** 2)
            except:
                return 1e6  # Large penalty for failed calculation
        
        # Constraints: volatilities between 1% and 100%
        bounds = [(0.01, 1.0) for _ in range(len(initial_guess))]
        
        # Optimize
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            calibrated_vols = result.x
        else:
            warnings.warn("Calibration failed, using initial guess")
            calibrated_vols = initial_guess
        
        return LMMResult(
            npv=mean_npv,
            standard_error=std_error,
            confidence_interval=(ci_lower, ci_upper),
            pv_fixed_leg=np.mean(fixed_leg_pv),
            pv_floating_leg=np.mean(floating_leg_pv),
            par_rate=par_rate,
            effective_duration=duration,
            convexity=convexity,
            paths_used=len(swap_npvs),
            convergence_stats={
                'std_error': std_error,
                'relative_error': std_error / abs(mean_npv) if mean_npv != 0 else float('inf'),
                'antithetic_used': antithetic,
                'control_variate_used': control_variate
            }
        )
    
    def price_cap_floor(self,
                       notional: float,
                       strike: float,
                       payment_times: List[float],
                       lmm_params: LMMParameters,
                       option_type: str = "cap",
                       n_paths: int = 50000) -> LMMResult:
        """
        Price interest rate cap or floor using LMM
        
        Args:
            notional: Notional amount
            strike: Cap/floor strike rate
            payment_times: Caplet/floorlet payment dates
            lmm_params: LMM parameters
            option_type: "cap" or "floor"
            n_paths: Monte Carlo paths
            
        Returns:
            LMMResult with option pricing
        """
        
        np.random.seed(42)
        
        # Simulate forward paths
        forward_paths = self._simulate_forward_paths(
            lmm_params, payment_times, n_paths // 2, antithetic=True
        )
        
        # Calculate caplet/floorlet payoffs
        option_values = np.zeros(forward_paths.shape[0])
        
        for i, payment_time in enumerate(payment_times):
            if i < forward_paths.shape[1]:
                forward_rates = forward_paths[:, i]
                
                if option_type.lower() == "cap":
                    payoffs = np.maximum(forward_rates - strike, 0)
                else:  # floor
                    payoffs = np.maximum(strike - forward_rates, 0)
                
                # Discount to present value
                discount_factor = np.exp(-np.mean(lmm_params.initial_forwards) * payment_time)
                year_fraction = 0.25  # Simplified
                
                option_values += notional * payoffs * year_fraction * discount_factor
        
        # Statistics
        mean_value = np.mean(option_values)
        std_error = np.std(option_values) / np.sqrt(len(option_values))
        
        return LMMResult(
            npv=mean_value,
            standard_error=std_error,
            confidence_interval=(mean_value - 1.96 * std_error, mean_value + 1.96 * std_error),
            pv_fixed_leg=0,
            pv_floating_leg=mean_value,
            par_rate=strike,
            effective_duration=np.mean(payment_times),
            convexity=0,
            paths_used=len(option_values)
        )
    
    def _simulate_forward_paths(self,
                               lmm_params: LMMParameters,
                               time_grid: np.ndarray,
                               n_paths: int,
                               antithetic: bool = True) -> np.ndarray:
        """
        Simulate forward rate paths using LMM dynamics
        
        Returns:
            Array of shape (total_paths, n_time_steps, n_forwards)
        """
        
        dt = np.diff(np.concatenate([[0], time_grid]))
        n_steps = len(time_grid)
        n_forwards = len(lmm_params.initial_forwards)
        
        # Initialize paths
        if antithetic:
            paths = np.zeros((2 * n_paths, n_steps, n_forwards))
        else:
            paths = np.zeros((n_paths, n_steps, n_forwards))
        
        # Set initial values
        paths[:, 0, :] = lmm_params.initial_forwards[:n_forwards]
        
        # Simulate paths
        for step in range(1, n_steps):
            dt_step = dt[step - 1]
            
            # Generate random increments
            dW = np.random.normal(0, np.sqrt(dt_step), (n_paths, n_forwards))
            
            if antithetic:
                dW_anti = -dW
                dW_combined = np.vstack([dW, dW_anti])
            else:
                dW_combined = dW
            
            # Apply correlation
            if hasattr(self, 'cholesky_matrix') and self.cholesky_matrix.shape[0] >= n_forwards:
                correlated_dW = dW_combined @ self.cholesky_matrix[:n_forwards, :n_forwards].T
            else:
                correlated_dW = dW_combined
            
            # LMM dynamics with drift adjustment
            for k in range(n_forwards):
                if k < len(lmm_params.volatilities):
                    vol_k = lmm_params.volatilities[k]
                    
                    # Drift adjustment (simplified)
                    drift = self._calculate_drift_adjustment(k, step, paths[:, step-1, :], 
                                                           lmm_params, time_grid)
                    
                    # Update forward rate
                    paths[:, step, k] = paths[:, step-1, k] * np.exp(
                        (drift - 0.5 * vol_k**2) * dt_step + vol_k * correlated_dW[:, k]
                    )
                    
                    # Floor at small positive value to avoid negative rates
                    paths[:, step, k] = np.maximum(paths[:, step, k], 1e-6)
        
        return paths
    
    def _calculate_drift_adjustment(self,
                                  k: int,
                                  step: int,
                                  current_forwards: np.ndarray,
                                  lmm_params: LMMParameters,
                                  time_grid: np.ndarray) -> np.ndarray:
        """
        Calculate drift adjustment for martingale property
        
        This is a simplified version - full implementation would be more complex
        """
        
        # Simplified drift calculation
        # In practice, this involves summing over all forward rates
        drift = np.zeros(current_forwards.shape[0])
        
        if k < len(lmm_params.volatilities):
            vol_k = lmm_params.volatilities[k]
            
            # Sum over forward rates (simplified)
            for j in range(k + 1, min(len(lmm_params.volatilities), current_forwards.shape[1])):
                if j < len(lmm_params.volatilities):
                    vol_j = lmm_params.volatilities[j]
                    corr_kj = self.correlation_matrix[k, j] if k < self.correlation_matrix.shape[0] and j < self.correlation_matrix.shape[1] else 0.8
                    
                    tau_j = 0.25  # Simplified tenor
                    drift += (vol_k * vol_j * corr_kj * tau_j * current_forwards[:, j]) / (1 + tau_j * current_forwards[:, j])
        
        return drift
    
    def _calculate_swap_legs_pv(self,
                               notional: float,
                               fixed_rate: float,
                               payment_times: np.ndarray,
                               time_grid: np.ndarray,
                               forward_paths: np.ndarray,
                               lmm_params: LMMParameters) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate present value of fixed and floating legs
        """
        
        n_paths = forward_paths.shape[0]
        fixed_leg_pv = np.zeros(n_paths)
        floating_leg_pv = np.zeros(n_paths)
        
        for i, payment_time in enumerate(payment_times):
            year_fraction = 0.25  # Simplified
            
            # Find closest time in grid
            time_idx = np.argmin(np.abs(time_grid - payment_time))
            
            # Discount factor (simplified)
            avg_rate = np.mean(forward_paths[:, :time_idx+1, :], axis=(1, 2))
            discount_factor = np.exp(-avg_rate * payment_time)
            
            # Fixed leg cashflow
            fixed_cashflow = notional * fixed_rate * year_fraction
            fixed_leg_pv += fixed_cashflow * discount_factor
            
            # Floating leg cashflow
            if time_idx < forward_paths.shape[1] and 0 < forward_paths.shape[2]:
                floating_rate = forward_paths[:, time_idx, 0]  # Use first forward rate
                floating_cashflow = notional * floating_rate * year_fraction
                floating_leg_pv += floating_cashflow * discount_factor
        
        return fixed_leg_pv, floating_leg_pv
    
    def _apply_control_variate(self,
                              swap_npvs: np.ndarray,
                              forward_paths: np.ndarray,
                              lmm_params: LMMParameters) -> np.ndarray:
        """
        Apply control variate technique for variance reduction
        """
        
        # Use average forward rate as control variate
        control_variate = np.mean(forward_paths[:, -1, :], axis=1)
        control_mean = np.mean(lmm_params.initial_forwards)
        
        # Calculate optimal beta coefficient
        covariance = np.cov(swap_npvs, control_variate)[0, 1]
        variance = np.var(control_variate)
        
        if variance > 1e-10:  # Avoid division by zero
            beta = covariance / variance
            adjusted_npvs = swap_npvs - beta * (control_variate - control_mean)
        else:
            adjusted_npvs = swap_npvs
        
        return adjusted_npvs
    
    def _calculate_annuity(self,
                          payment_times: np.ndarray,
                          forward_rates: np.ndarray) -> float:
        """
        Calculate annuity factor for par rate calculation
        """
        
        annuity = 0.0
        avg_rate = np.mean(forward_rates)
        
        for payment_time in payment_times:
            year_fraction = 0.25  # Simplified
            discount_factor = np.exp(-avg_rate * payment_time)
            annuity += year_fraction * discount_factor
        
        return annuity
    
    def _build_default_correlation_matrix(self) -> np.ndarray:
        """
        Build default exponential correlation matrix
        """
        
        n = len(self.tenor_structure)
        correlation_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                time_diff = abs(self.tenor_structure[i] - self.tenor_structure[j])
                correlation_matrix[i, j] = np.exp(-self.correlation_decay * time_diff)
        
        return correlation_matrix
    
    def _calculate_cap_volatilities(self,
                                   vol_params: np.ndarray,
                                   forward_rates: np.ndarray) -> np.ndarray:
        """
        Calculate model-implied cap volatilities for calibration
        """
        
        # Simplified calculation - would be more sophisticated in practice
        n_caplets = min(len(vol_params), len(forward_rates))
        cap_vols = np.zeros(n_caplets)
        
        for i in range(n_caplets):
            # Approximate cap volatility
            cap_vols[i] = vol_params[i] * np.sqrt(self.tenor_structure[i])
        
        return cap_vols
    
    def calculate_vegas(self,
                       notional: float,
                       fixed_rate: float,
                       payment_times: List[float],
                       lmm_params: LMMParameters,
                       vol_shift: float = 0.01) -> Dict[str, float]:
        """
        Calculate volatility sensitivities (Vegas)
        """
        
        base_result = self.price_interest_rate_swap(notional, fixed_rate, payment_times, lmm_params, n_paths=10000)
        
        vegas = {}
        
        for i, vol in enumerate(lmm_params.volatilities):
            # Shift volatility
            shifted_vols = lmm_params.volatilities.copy()
            shifted_vols[i] += vol_shift
            
            shifted_params = LMMParameters(
                initial_forwards=lmm_params.initial_forwards,
                volatilities=shifted_vols,
                correlations=lmm_params.correlations,
                payment_times=lmm_params.payment_times,
                tenor_structure=lmm_params.tenor_structure
            )
            
            shifted_result = self.price_interest_rate_swap(notional, fixed_rate, payment_times, shifted_params, n_paths=10000)
            
            vegas[f'vega_{i}'] = (shifted_result.npv - base_result.npv) / vol_shift
        
        return vegas


# Utility functions for LMM
def create_market_standard_lmm(currency: str = "USD") -> EnhancedLIBORMarketModel:
    """
    Create LMM with market-standard parameters for given currency
    """
    
    if currency.upper() == "USD":
        tenor_structure = [0.25 * i for i in range(1, 41)]  # 10Y quarterly
        n_factors = 3
        correlation_decay = 0.1
    elif currency.upper() == "EUR":
        tenor_structure = [0.5 * i for i in range(1, 21)]   # 10Y semi-annual
        n_factors = 3
        correlation_decay = 0.12
    else:
        # Default
        tenor_structure = [0.25 * i for i in range(1, 41)]
        n_factors = 3
        correlation_decay = 0.1
    
    return EnhancedLIBORMarketModel(tenor_structure, n_factors, correlation_decay)


def bootstrap_lmm_from_swaps(swap_rates: Dict[float, float],
                            volatility_surface: Dict[Tuple[float, float], float]) -> LMMParameters:
    """
    Bootstrap LMM parameters from swap rates and volatility surface
    
    Args:
        swap_rates: Dictionary of {tenor: rate}
        volatility_surface: Dictionary of {(expiry, tenor): volatility}
    
    Returns:
        Calibrated LMM parameters
    """
    
    # Extract tenors and rates
    tenors = sorted(swap_rates.keys())
    rates = [swap_rates[t] for t in tenors]
    
    # Bootstrap forward rates (simplified)
    forward_rates = np.array(rates)  # Would need proper bootstrapping
    
    # Extract volatilities
    vol_tenors = sorted(set(t[1] for t in volatility_surface.keys()))
    volatilities = []
    
    for tenor in vol_tenors:
        # Use ATM volatility for this tenor
        vol_key = (tenor, tenor)  # ATM
        if vol_key in volatility_surface:
            volatilities.append(volatility_surface[vol_key])
        else:
            # Interpolate or use default
            volatilities.append(0.15)
    
    # Build correlation matrix (simplified)
    n = len(vol_tenors)
    correlations = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                correlations[i, j] = 0.8 * np.exp(-0.1 * abs(i - j))
    
    return LMMParameters(
        initial_forwards=forward_rates,
        volatilities=np.array(volatilities),
        correlations=correlations,
        payment_times=np.array(tenors),
        tenor_structure=np.array(vol_tenors)
    )


# Example usage and testing
if __name__ == "__main__":
    # Create LMM instance
    lmm = EnhancedLIBORMarketModel()
    
    # Mock market data for calibration
    market_cap_vols = {
        "1Y": 0.15,
        "2Y": 0.18,
        "3Y": 0.20,
        "5Y": 0.22,
        "7Y": 0.20,
        "10Y": 0.18
    }
    
    # Calibrate model
    params = lmm.calibrate_to_market_data(market_cap_vols)
    
    # Price a 5Y swap
    payment_times = [0.25 * i for i in range(1, 21)]  # 5Y quarterly
    
    result = lmm.price_interest_rate_swap(
        notional=100_000_000,
        fixed_rate=0.035,
        payment_times=payment_times,
        lmm_params=params,
        n_paths=25000
    )
    
    print(f"Swap NPV: ${result.npv:,.0f} ± ${result.standard_error:,.0f}")
    print(f"95% Confidence Interval: [${result.confidence_interval[0]:,.0f}, ${result.confidence_interval[1]:,.0f}]")
    print(f"Par Rate: {result.par_rate:.4%}")
    print(f"Paths Used: {result.paths_used:,}")
    print(f"Relative Error: {result.convergence_stats['relative_error']:.4%}")
    
    Parameters(
            initial_forwards=initial_forwards,
            volatilities=calibrated_vols,
            correlations=self.correlation_matrix,
            payment_times=self.tenor_structure,
            tenor_structure=self.tenor_structure
        )
    
    def price_interest_rate_swap(self,
                                notional: float,
                                fixed_rate: float,
                                payment_times: List[float],
                                lmm_params: LMMParameters,
                                n_paths: int = 50000,
                                n_steps: int = None,
                                antithetic: bool = True,
                                control_variate: bool = True,
                                random_seed: int = 42) -> LMMResult:
        """
        Price interest rate swap using Monte Carlo simulation under LMM
        
        Args:
            notional: Swap notional amount
            fixed_rate: Fixed leg rate
            payment_times: Swap payment schedule
            lmm_params: Calibrated LMM parameters
            n_paths: Number of Monte Carlo paths
            n_steps: Time steps (if None, uses payment_times)
            antithetic: Use antithetic variance reduction
            control_variate: Use control variate technique
            random_seed: Random seed for reproducibility
            
        Returns:
            LMMResult with pricing and risk metrics
        """
        
        np.random.seed(random_seed)
        
        # Setup simulation grid
        payment_times = np.array(payment_times)
        if n_steps is None:
            time_grid = payment_times
        else:
            time_grid = np.linspace(0, payment_times[-1], n_steps + 1)[1:]
        
        dt = np.diff(np.concatenate([[0], time_grid]))
        
        # Effective paths (account for antithetic)
        effective_paths = n_paths // 2 if antithetic else n_paths
        
        # Simulate forward rate paths
        forward_paths = self._simulate_forward_paths(
            lmm_params, time_grid, effective_paths, antithetic
        )
        
        # Calculate cashflows
        fixed_leg_pv, floating_leg_pv = self._calculate_swap_legs_pv(
            notional, fixed_rate, payment_times, time_grid, forward_paths, lmm_params
        )
        
        # Swap NPV (pay fixed, receive floating convention)
        swap_npvs = floating_leg_pv - fixed_leg_pv
        
        # Control variate adjustment
        if control_variate:
            swap_npvs = self._apply_control_variate(swap_npvs, forward_paths, lmm_params)
        
        # Statistics
        mean_npv = np.mean(swap_npvs)
        std_error = np.std(swap_npvs) / np.sqrt(len(swap_npvs))
        
        # Confidence interval (95%)
        ci_lower = mean_npv - 1.96 * std_error
        ci_upper = mean_npv + 1.96 * std_error
        
        # Par rate calculation
        fixed_annuity = self._calculate_annuity(payment_times, lmm_params.initial_forwards)
        par_rate = np.mean(floating_leg_pv) / (notional * fixed_annuity)
        
        # Risk metrics (simplified)
        duration = np.mean(payment_times)  # Approximate
        convexity = duration ** 2 * 0.5    # Simplified
        
        return LMM
