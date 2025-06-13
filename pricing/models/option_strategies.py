"""
Option Pricing Models
pricing/models.py

This module contains the core mathematical models for option pricing.
"""

import math
import numpy as np
from scipy.stats import norm


def black_scholes_price(option_type, S, K, T, r, sigma, q=0):
    """
    Black-Scholes option pricing formula
    
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
    float
        Option price
    """
    try:
        if T <= 0:
            # Handle expiration case
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # Calculate d1 and d2
        d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        if option_type == 'call':
            price = S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        elif option_type == 'put':
            price = K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)
        else:
            raise ValueError(f"Invalid option_type: {option_type}. Use 'call' or 'put'.")
        
        return max(price, 0)
    
    except Exception as e:
        raise ValueError(f"Error in Black-Scholes calculation: {str(e)}")


def binomial_tree_price(option_type, S, K, T, r, sigma, N=100, q=0, exercise_style='european'):
    """
    Binomial tree option pricing model
    
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
    N : int
        Number of time steps
    q : float, optional
        Dividend yield (default=0)
    exercise_style : str
        'european' or 'american'
    
    Returns:
    --------
    float
        Option price
    """
    try:
        if T <= 0:
            # Handle expiration case
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # Time step
        dt = T / N
        
        # Up and down factors
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (math.exp((r - q) * dt) - d) / (u - d)
        
        if not (0 <= p <= 1):
            raise ValueError("Invalid risk-neutral probability. Check parameters.")
        
        # Initialize asset prices at maturity
        asset_prices = np.zeros(N + 1)
        for i in range(N + 1):
            asset_prices[i] = S * (u ** (N - i)) * (d ** i)
        
        # Initialize option values at maturity
        option_values = np.zeros(N + 1)
        for i in range(N + 1):
            if option_type == 'call':
                option_values[i] = max(asset_prices[i] - K, 0)
            else:  # put
                option_values[i] = max(K - asset_prices[i], 0)
        
        # Backward induction
        for j in range(N - 1, -1, -1):
            for i in range(j + 1):
                # Risk-neutral valuation
                option_values[i] = math.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                # American exercise condition
                if exercise_style == 'american':
                    asset_price = S * (u ** (j - i)) * (d ** i)
                    if option_type == 'call':
                        intrinsic_value = max(asset_price - K, 0)
                    else:  # put
                        intrinsic_value = max(K - asset_price, 0)
                    
                    option_values[i] = max(option_values[i], intrinsic_value)
        
        return option_values[0]
    
    except Exception as e:
        raise ValueError(f"Error in binomial tree calculation: {str(e)}")


def monte_carlo_price(option_type, S, K, T, r, sigma, n_simulations=10000, q=0, exercise_style='european'):
    """
    Monte Carlo option pricing simulation
    
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
    n_simulations : int
        Number of Monte Carlo simulations
    q : float, optional
        Dividend yield (default=0)
    exercise_style : str
        'european' (american not supported in basic MC)
    
    Returns:
    --------
    float
        Option price
    """
    try:
        if T <= 0:
            # Handle expiration case
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        if exercise_style == 'american':
            # For simplicity, fall back to European for Monte Carlo
            # In practice, you'd need Longstaff-Schwartz or similar methods
            pass
        
        # Generate random price paths
        np.random.seed(42)  # For reproducibility
        Z = np.random.standard_normal(n_simulations)
        
        # Final stock prices using geometric Brownian motion
        ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        elif option_type == 'put':
            payoffs = np.maximum(K - ST, 0)
        else:
            raise ValueError(f"Invalid option_type: {option_type}. Use 'call' or 'put'.")
        
        # Discount back to present value
        option_price = math.exp(-r * T) * np.mean(payoffs)
        
        return option_price
    
    except Exception as e:
        raise ValueError(f"Error in Monte Carlo calculation: {str(e)}")


def get_pricing_model(model_name):
    """
    Factory function to get the appropriate pricing model
    
    Parameters:
    -----------
    model_name : str
        'black-scholes', 'binomial', or 'monte-carlo'
    
    Returns:
    --------
    function
        The pricing function
    """
    models = {
        'black-scholes': black_scholes_price,
        'binomial': binomial_tree_price,
        'monte-carlo': monte_carlo_price
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name]m
