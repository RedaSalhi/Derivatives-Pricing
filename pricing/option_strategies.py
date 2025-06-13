"""
Main Pricing Interface
pricing/option_strategies.py

This module provides the main interface for option pricing and strategy analysis.
It combines the models and utilities into easy-to-use functions.
"""

from .models.option_strategies import black_scholes_price, binomial_tree_price, monte_carlo_price, get_pricing_model
from .utils.option_strategies_utils import (
    calculate_greeks, 
    calculate_strategy_greeks_range, 
    compute_strategy_payoff,
    find_breakeven_points,
    validate_strategy_legs,
    calculate_strategy_metrics
)


def price_vanilla_option(option_type, exercise_style, model, S, K, T, r, sigma, q=0, N=100, n_simulations=10000):
    """
    Price a single vanilla option using the specified model
    
    Parameters:
    -----------
    option_type : str
        'call' or 'put'
    exercise_style : str
        'european' or 'american'
    model : str
        'black-scholes', 'binomial', or 'monte-carlo'
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
    N : int, optional
        Number of steps for binomial tree (default=100)
    n_simulations : int, optional
        Number of simulations for Monte Carlo (default=10000)
    
    Returns:
    --------
    float
        Option price
    
    Raises:
    -------
    ValueError
        If invalid parameters are provided
    """
    # Validate inputs
    if option_type not in ['call', 'put']:
        raise ValueError(f"Invalid option_type: {option_type}. Use 'call' or 'put'.")
    
    if exercise_style not in ['european', 'american']:
        raise ValueError(f"Invalid exercise_style: {exercise_style}. Use 'european' or 'american'.")
    
    if model not in ['black-scholes', 'binomial', 'monte-carlo']:
        raise ValueError(f"Invalid model: {model}. Use 'black-scholes', 'binomial', or 'monte-carlo'.")
    
    # Basic parameter validation
    if S <= 0:
        raise ValueError("Stock price must be positive")
    if K <= 0:
        raise ValueError("Strike price must be positive")
    if T < 0:
        raise ValueError("Time to expiration cannot be negative")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    
    # Route to appropriate pricing model
    try:
        if model == 'black-scholes':
            # Black-Scholes only supports European exercise
            if exercise_style == 'american':
                # Fall back to binomial for American options
                return binomial_tree_price(option_type, S, K, T, r, sigma, N, q, exercise_style)
            else:
                return black_scholes_price(option_type, S, K, T, r, sigma, q)
        
        elif model == 'binomial':
            return binomial_tree_price(option_type, S, K, T, r, sigma, N, q, exercise_style)
        
        elif model == 'monte-carlo':
            # Monte Carlo implementation is basic and only supports European
            if exercise_style == 'american':
                # Fall back to binomial for American options
                return binomial_tree_price(option_type, S, K, T, r, sigma, N, q, exercise_style)
            else:
                return monte_carlo_price(option_type, S, K, T, r, sigma, n_simulations, q, exercise_style)
    
    except Exception as e:
        raise ValueError(f"Error pricing option with {model} model: {str(e)}")


def price_option_strategy(legs, exercise_style, model, S, T, r, sigma, q=0, N=100, n_simulations=10000):
    """
    Price a multi-leg option strategy
    
    Parameters:
    -----------
    legs : list
        List of strategy legs, each containing:
        - 'type': 'call' or 'put'
        - 'strike': strike price
        - 'qty': quantity (positive for long, negative for short)
    exercise_style : str
        'european' or 'american'
    model : str
        'black-scholes', 'binomial', or 'monte-carlo'
    S : float
        Current stock price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate
    sigma : float
        Volatility (annual)
    q : float, optional
        Dividend yield (default=0)
    N : int, optional
        Number of steps for binomial tree (default=100)
    n_simulations : int, optional
        Number of simulations for Monte Carlo (default=10000)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'strategy_price': total strategy cost
        - 'individual_prices': list of individual option prices
        - 'legs_info': detailed information about each leg
    
    Raises:
    -------
    ValueError
        If invalid strategy or parameters are provided
    """
    # Validate strategy
    validation = validate_strategy_legs(legs)
    if not validation['valid']:
        raise ValueError(f"Invalid strategy: {'; '.join(validation['errors'])}")
    
    individual_prices = []
    legs_info = []
    total_price = 0
    
    try:
        for i, leg in enumerate(legs):
            leg_price = price_vanilla_option(
                option_type=leg['type'],
                exercise_style=exercise_style,
                model=model,
                S=S,
                K=leg['strike'],
                T=T,
                r=r,
                sigma=sigma,
                q=q,
                N=N,
                n_simulations=n_simulations
            )
            
            individual_prices.append(leg_price)
            leg_cost = leg_price * leg['qty']
            total_price += leg_cost
            
            # Store detailed leg information
            leg_info = {
                'leg_number': i + 1,
                'type': leg['type'],
                'strike': leg['strike'],
                'quantity': leg['qty'],
                'unit_price': leg_price,
                'total_cost': leg_cost,
                'position': 'Long' if leg['qty'] > 0 else 'Short'
            }
            legs_info.append(leg_info)
        
        return {
            'strategy_price': total_price,
            'individual_prices': individual_prices,
            'legs_info': legs_info
        }
    
    except Exception as e:
        raise ValueError(f"Error pricing strategy: {str(e)}")


def get_predefined_strategy(strategy_name, *strikes):
    """
    Get predefined strategy configurations
    
    Parameters:
    -----------
    strategy_name : str
        Name of the predefined strategy
    *strikes : float
        Strike prices required for the strategy
    
    Returns:
    --------
    list or str
        List of strategy legs or error message
    """
    try:
        if strategy_name == "straddle":
            if len(strikes) != 1:
                return "Error: Straddle requires 1 strike price"
            strike = strikes[0]
            return [
                {"type": "call", "strike": strike, "qty": 1.0},
                {"type": "put", "strike": strike, "qty": 1.0}
            ]
        
        elif strategy_name == "bull call spread":
            if len(strikes) != 2:
                return "Error: Bull call spread requires 2 strike prices"
            strike1, strike2 = strikes[0], strikes[1]
            if strike1 >= strike2:
                return "Error: Lower strike must be less than higher strike for bull call spread"
            return [
                {"type": "call", "strike": strike1, "qty": 1.0},
                {"type": "call", "strike": strike2, "qty": -1.0}
            ]
        
        elif strategy_name == "bear put spread":
            if len(strikes) != 2:
                return "Error: Bear put spread requires 2 strike prices"
            strike1, strike2 = strikes[0], strikes[1]
            if strike1 >= strike2:
                return "Error: Lower strike must be less than higher strike for bear put spread"
            return [
                {"type": "put", "strike": strike2, "qty": 1.0},
                {"type": "put", "strike": strike1, "qty": -1.0}
            ]
        
        elif strategy_name == "butterfly":
            if len(strikes) != 3:
                return "Error: Butterfly requires 3 strike prices"
            strike1, strike2, strike3 = strikes[0], strikes[1], strikes[2]
            if not (strike1 < strike2 < strike3):
                return "Error: Strikes must be in ascending order for butterfly spread"
            return [
                {"type": "call", "strike": strike1, "qty": 1.0},
                {"type": "call", "strike": strike2, "qty": -2.0},
                {"type": "call", "strike": strike3, "qty": 1.0}
            ]
        
        elif strategy_name == "iron condor":
            if len(strikes) != 4:
                return "Error: Iron condor requires 4 strike prices"
            strike1, strike2, strike3, strike4 = strikes[0], strikes[1], strikes[2], strikes[3]
            if not (strike1 < strike2 < strike3 < strike4):
                return "Error: Strikes must be in ascending order for iron condor"
            return [
                {"type": "put", "strike": strike1, "qty": 1.0},
                {"type": "put", "strike": strike2, "qty": -1.0},
                {"type": "call", "strike": strike3, "qty": -1.0},
                {"type": "call", "strike": strike4, "qty": 1.0}
            ]
        
        elif strategy_name == "strangle":
            if len(strikes) != 2:
                return "Error: Strangle requires 2 strike prices"
            strike1, strike2 = strikes[0], strikes[1]
            if strike1 >= strike2:
                return "Error: Put strike must be less than call strike for strangle"
            return [
                {"type": "put", "strike": strike1, "qty": 1.0},
                {"type": "call", "strike": strike2, "qty": 1.0}
            ]
        
        elif strategy_name == "covered call":
            if len(strikes) != 1:
                return "Error: Covered call requires 1 strike price"
            strike = strikes[0]
            return [
                {"type": "call", "strike": strike, "qty": -1.0}
            ]  # Note: This assumes you already own the underlying stock
        
        elif strategy_name == "protective put":
            if len(strikes) != 1:
                return "Error: Protective put requires 1 strike price"
            strike = strikes[0]
            return [
                {"type": "put", "strike": strike, "qty": 1.0}
            ]  # Note: This assumes you already own the underlying stock
        
        elif strategy_name == "collar":
            if len(strikes) != 2:
                return "Error: Collar requires 2 strike prices"
            put_strike, call_strike = strikes[0], strikes[1]
            if put_strike >= call_strike:
                return "Error: Put strike must be less than call strike for collar"
            return [
                {"type": "put", "strike": put_strike, "qty": 1.0},
                {"type": "call", "strike": call_strike, "qty": -1.0}
            ]  # Note: This assumes you already own the underlying stock
        
        else:
            return f"Error: Unknown strategy '{strategy_name}'. Available strategies: straddle, bull call spread, bear put spread, butterfly, iron condor, strangle, covered call, protective put, collar"
    
    except Exception as e:
        return f"Error creating {strategy_name}: {str(e)}"


def analyze_strategy_comprehensive(legs, strategy_cost, current_spot, T, r, sigma, q=0):
    """
    Perform comprehensive analysis of an option strategy
    
    Parameters:
    -----------
    legs : list
        List of strategy legs
    strategy_cost : float
        Net cost of the strategy
    current_spot : float
        Current underlying price
    T : float
        Time to expiration
    r : float
        Risk-free rate
    sigma : float
        Volatility
    q : float, optional
        Dividend yield
    
    Returns:
    --------
    dict
        Comprehensive analysis results
    """
    try:
        # Basic strategy metrics
        strikes = [leg['strike'] for leg in legs]
        min_strike, max_strike = min(strikes), max(strikes)
        spot_range = np.linspace(min_strike * 0.6, max_strike * 1.4, 500)
        
        metrics = calculate_strategy_metrics(legs, strategy_cost, spot_range)
        
        # Greeks analysis at current spot
        current_greeks = {}
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            greek_values = calculate_strategy_greeks_range(
                legs, [current_spot], greek, T, r, sigma, q
            )
            current_greeks[greek] = greek_values[0] if len(greek_values) > 0 else 0
        
        # Payoff analysis
        payoffs = compute_strategy_payoff(legs, spot_range)
        pnl = payoffs - strategy_cost
        
        # Current position analysis
        current_payoff = compute_strategy_payoff(legs, [current_spot])[0]
        current_pnl = current_payoff - strategy_cost
        
        # Risk analysis
        risk_analysis = {
            'max_risk': abs(metrics['max_loss']),
            'max_reward': metrics['max_profit'],
            'risk_reward_ratio': metrics['risk_reward_ratio'],
            'probability_of_profit': len(np.where(pnl > 0)[0]) / len(pnl) * 100,
            'current_pnl': current_pnl,
            'current_status': 'Profitable' if current_pnl > 0 else 'Unprofitable' if current_pnl < 0 else 'Breakeven'
        }
        
        # Strategy classification
        net_delta = current_greeks['delta']
        net_gamma = current_greeks['gamma']
        net_theta = current_greeks['theta']
        net_vega = current_greeks['vega']
        
        strategy_profile = {
            'directional_bias': 'Bullish' if net_delta > 0.1 else 'Bearish' if net_delta < -0.1 else 'Neutral',
            'volatility_bias': 'Long Vol' if net_vega > 0.01 else 'Short Vol' if net_vega < -0.01 else 'Vol Neutral',
            'time_decay_bias': 'Benefits from Decay' if net_theta > 0.01 else 'Hurt by Decay' if net_theta < -0.01 else 'Time Neutral',
            'convexity': 'Positive Gamma' if net_gamma > 0.001 else 'Negative Gamma' if net_gamma < -0.001 else 'Gamma Neutral'
        }
        
        return {
            'basic_metrics': metrics,
            'current_greeks': current_greeks,
            'risk_analysis': risk_analysis,
            'strategy_profile': strategy_profile,
            'spot_range': spot_range,
            'payoffs': payoffs,
            'pnl': pnl
        }
    
    except Exception as e:
        raise ValueError(f"Error in comprehensive analysis: {str(e)}")


# Utility functions for strategy recommendations
def get_strategy_recommendations(market_view, volatility_view, time_horizon):
    """
    Get strategy recommendations based on market outlook
    
    Parameters:
    -----------
    market_view : str
        'bullish', 'bearish', or 'neutral'
    volatility_view : str
        'increasing', 'decreasing', or 'stable'
    time_horizon : str
        'short', 'medium', or 'long'
    
    Returns:
    --------
    list
        List of recommended strategies with descriptions
    """
    recommendations = []
    
    if market_view == 'bullish':
        if volatility_view == 'increasing':
            recommendations.extend([
                {
                    'strategy': 'Long Call',
                    'description': 'Benefits from upward price movement and volatility increase',
                    'risk': 'Limited to premium paid',
                    'reward': 'Unlimited upside potential'
                },
                {
                    'strategy': 'Bull Call Spread',
                    'description': 'Lower cost bullish play with limited upside',
                    'risk': 'Limited to spread width minus premium received',
                    'reward': 'Limited to spread width'
                }
            ])
        elif volatility_view == 'decreasing':
            recommendations.extend([
                {
                    'strategy': 'Bull Put Spread',
                    'description': 'Collect premium while expressing bullish view',
                    'risk': 'Limited to spread width minus premium received',
                    'reward': 'Limited to premium received'
                },
                {
                    'strategy': 'Covered Call',
                    'description': 'Generate income on existing long position',
                    'risk': 'Opportunity cost if stock rises above strike',
                    'reward': 'Premium collected plus potential stock appreciation'
                }
            ])
    
    elif market_view == 'bearish':
        if volatility_view == 'increasing':
            recommendations.extend([
                {
                    'strategy': 'Long Put',
                    'description': 'Benefits from downward price movement and volatility increase',
                    'risk': 'Limited to premium paid',
                    'reward': 'Substantial downside potential'
                },
                {
                    'strategy': 'Bear Put Spread',
                    'description': 'Lower cost bearish play with limited downside capture',
                    'risk': 'Limited to premium paid',
                    'reward': 'Limited to spread width'
                }
            ])
        elif volatility_view == 'decreasing':
            recommendations.extend([
                {
                    'strategy': 'Bear Call Spread',
                    'description': 'Collect premium while expressing bearish view',
                    'risk': 'Limited to spread width minus premium received',
                    'reward': 'Limited to premium received'
                }
            ])
    
    elif market_view == 'neutral':
        if volatility_view == 'increasing':
            recommendations.extend([
                {
                    'strategy': 'Long Straddle',
                    'description': 'Profit from large moves in either direction',
                    'risk': 'Limited to premium paid',
                    'reward': 'Unlimited if large move occurs'
                },
                {
                    'strategy': 'Long Strangle',
                    'description': 'Lower cost volatility play than straddle',
                    'risk': 'Limited to premium paid',
                    'reward': 'Unlimited if large move occurs'
                }
            ])
        elif volatility_view == 'decreasing':
            recommendations.extend([
                {
                    'strategy': 'Iron Condor',
                    'description': 'Collect premium from range-bound movement',
                    'risk': 'Limited to spread width minus premium received',
                    'reward': 'Limited to premium received'
                },
                {
                    'strategy': 'Butterfly Spread',
                    'description': 'Profit from minimal price movement',
                    'risk': 'Limited to premium paid',
                    'reward': 'Limited to spread width'
                }
            ])
    
    return recommendations


# Export main functions for easy import
__all__ = [
    'price_vanilla_option',
    'price_option_strategy', 
    'get_predefined_strategy',
    'analyze_strategy_comprehensive',
    'get_strategy_recommendations',
    'calculate_greeks',
    'calculate_strategy_greeks_range',
    'compute_strategy_payoff',
    'find_breakeven_points',
    'validate_strategy_legs',
    'calculate_strategy_metrics'
]
