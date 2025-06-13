"""
Pricing Utilities
pricing/utils/option_strategies_utils.py

This module contains utility functions for option pricing calculations,
including Greeks calculation and strategy analysis.
"""

import math
import numpy as np
from scipy.stats import norm


def calculate_greeks(option_type, S, K, T, r, sigma, q=0):
    """
    Calculate Black-Scholes Greeks for a single option
    
    Parameters:
    -----------
    option_type : str
        'call' or 'put'
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility (annual)
    q : float, optional
        Dividend yield (default=0)
    
    Returns:
    --------
    dict
        Dictionary containing all Greeks
    """
    try:
        if T <= 0:
            return {
                'delta': 0, 'gamma': 0, 'theta': 0, 
                'vega': 0, 'rho': 0
            }
        
        # Calculate d1 and d2
        d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        # Common terms
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        npd1 = norm.pdf(d1)
        
        # Delta
        if option_type == 'call':
            delta = math.exp(-q*T) * nd1
        else:  # put
            delta = math.exp(-q*T) * (nd1 - 1)
        
        # Gamma (same for calls and puts)
        gamma = math.exp(-q*T) * npd1 / (S * sigma * math.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * npd1 * sigma * math.exp(-q*T) / (2 * math.sqrt(T)) 
                    - r * K * math.exp(-r*T) * nd2
                    + q * S * math.exp(-q*T) * nd1) / 365  # Per day
        else:  # put
            theta = (-S * npd1 * sigma * math.exp(-q*T) / (2 * math.sqrt(T)) 
                    + r * K * math.exp(-r*T) * norm.cdf(-d2)
                    - q * S * math.exp(-q*T) * norm.cdf(-d1)) / 365  # Per day
        
        # Vega (same for calls and puts)
        vega = S * npd1 * math.sqrt(T) * math.exp(-q*T) / 100  # Per 1% vol change
        
        # Rho
        if option_type == 'call':
            rho = K * T * math.exp(-r*T) * nd2 / 100  # Per 1% rate change
        else:  # put
            rho = -K * T * math.exp(-r*T) * norm.cdf(-d2) / 100  # Per 1% rate change
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    except Exception as e:
        return {
            'delta': 0, 'gamma': 0, 'theta': 0, 
            'vega': 0, 'rho': 0
        }


def calculate_strategy_greeks_range(legs, spot_range, greek_name, T, r, sigma, q=0):
    """
    Calculate a specific Greek for a strategy across a range of spot prices
    
    Parameters:
    -----------
    legs : list
        List of strategy legs, each with 'type', 'strike', 'qty'
    spot_range : array-like
        Range of spot prices to calculate Greeks for
    greek_name : str
        Name of the Greek to calculate ('delta', 'gamma', 'theta', 'vega', 'rho')
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility (annual)
    q : float, optional
        Dividend yield (default=0)
    
    Returns:
    --------
    numpy.array
        Array of Greek values corresponding to spot_range
    """
    greek_values = np.zeros(len(spot_range))
    
    for i, S in enumerate(spot_range):
        strategy_greek = 0
        
        for leg in legs:
            try:
                greeks = calculate_greeks(
                    option_type=leg['type'],
                    S=S,
                    K=leg['strike'],
                    T=T,
                    r=r,
                    sigma=sigma,
                    q=q
                )
                
                leg_greek = greeks.get(greek_name, 0)
                strategy_greek += leg_greek * leg['qty']
                
            except Exception:
                # Handle edge cases gracefully
                strategy_greek += 0
        
        greek_values[i] = strategy_greek
    
    return greek_values


def compute_strategy_payoff(legs, spot_range):
    """
    Compute strategy payoff at expiration for a range of spot prices
    
    Parameters:
    -----------
    legs : list
        List of strategy legs, each with 'type', 'strike', 'qty'
    spot_range : array-like
        Range of spot prices to calculate payoffs for
    
    Returns:
    --------
    numpy.array
        Array of payoff values corresponding to spot_range
    """
    payoffs = np.zeros(len(spot_range))
    
    for leg in legs:
        leg_payoffs = np.zeros(len(spot_range))
        
        for i, spot in enumerate(spot_range):
            if leg['type'] == 'call':
                intrinsic = max(spot - leg['strike'], 0)
            elif leg['type'] == 'put':
                intrinsic = max(leg['strike'] - spot, 0)
            else:
                intrinsic = 0
            
            leg_payoffs[i] = intrinsic * leg['qty']
        
        payoffs += leg_payoffs
    
    return payoffs


def find_breakeven_points(legs, strategy_cost, spot_min, spot_max, n_points=1000):
    """
    Find breakeven points for a strategy
    
    Parameters:
    -----------
    legs : list
        List of strategy legs
    strategy_cost : float
        Net cost of the strategy (negative for credit strategies)
    spot_min : float
        Minimum spot price to search
    spot_max : float
        Maximum spot price to search
    n_points : int
        Number of points to evaluate
    
    Returns:
    --------
    list
        List of breakeven spot prices
    """
    spot_range = np.linspace(spot_min, spot_max, n_points)
    payoffs = compute_strategy_payoff(legs, spot_range)
    pnl = payoffs - strategy_cost
    
    breakeven_points = []
    
    # Find sign changes (zero crossings)
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] < 0:  # Sign change
            # Linear interpolation to find exact crossing
            x_cross = spot_range[i] + (spot_range[i + 1] - spot_range[i]) * (-pnl[i] / (pnl[i + 1] - pnl[i]))
            breakeven_points.append(x_cross)
    
    return breakeven_points


def validate_strategy_legs(legs):
    """
    Validate strategy legs for common errors
    
    Parameters:
    -----------
    legs : list
        List of strategy legs
    
    Returns:
    --------
    dict
        Validation result with 'valid' boolean and 'errors' list
    """
    errors = []
    
    if not legs:
        errors.append("Strategy must have at least one leg")
        return {'valid': False, 'errors': errors}
    
    for i, leg in enumerate(legs):
        # Check required fields
        required_fields = ['type', 'strike', 'qty']
        for field in required_fields:
            if field not in leg:
                errors.append(f"Leg {i+1}: Missing required field '{field}'")
        
        # Check option type
        if 'type' in leg and leg['type'] not in ['call', 'put']:
            errors.append(f"Leg {i+1}: Invalid option type '{leg['type']}'. Use 'call' or 'put'")
        
        # Check strike price
        if 'strike' in leg and leg['strike'] <= 0:
            errors.append(f"Leg {i+1}: Strike price must be positive")
        
        # Check quantity
        if 'qty' in leg and leg['qty'] == 0:
            errors.append(f"Leg {i+1}: Quantity cannot be zero")
    
    return {'valid': len(errors) == 0, 'errors': errors}


def calculate_strategy_metrics(legs, strategy_cost, spot_range=None):
    """
    Calculate comprehensive strategy metrics
    
    Parameters:
    -----------
    legs : list
        List of strategy legs
    strategy_cost : float
        Net cost of the strategy
    spot_range : array-like, optional
        Range of spot prices for analysis
    
    Returns:
    --------
    dict
        Dictionary of strategy metrics
    """
    if spot_range is None:
        strikes = [leg['strike'] for leg in legs]
        min_strike, max_strike = min(strikes), max(strikes)
        spot_range = np.linspace(min_strike * 0.5, max_strike * 1.5, 1000)
    
    payoffs = compute_strategy_payoff(legs, spot_range)
    pnl = payoffs - strategy_cost
    
    metrics = {
        'max_profit': np.max(pnl),
        'max_loss': np.min(pnl),
        'max_payoff': np.max(payoffs),
        'min_payoff': np.min(payoffs),
        'strategy_cost': strategy_cost,
        'strategy_type': 'Credit' if strategy_cost < 0 else 'Debit',
        'breakeven_points': find_breakeven_points(legs, strategy_cost, 
                                                spot_range[0], spot_range[-1])
    }
    
    # Risk/Reward ratio
    if metrics['max_loss'] < 0:
        metrics['risk_reward_ratio'] = abs(metrics['max_profit'] / metrics['max_loss'])
    else:
        metrics['risk_reward_ratio'] = float('inf')
    
    # Profitable range
    profitable_indices = np.where(pnl > 0)[0]
    if len(profitable_indices) > 0:
        metrics['profitable_range'] = {
            'min': spot_range[profitable_indices[0]],
            'max': spot_range[profitable_indices[-1]]
        }
    else:
        metrics['profitable_range'] = None
    
    return metrics

