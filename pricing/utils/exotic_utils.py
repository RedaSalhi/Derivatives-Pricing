#pricing/utils/exotic_utils.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import your pricing modules
from pricing.asian_option import price_asian_option, plot_asian_option_payoff, plot_monte_carlo_paths
from pricing.barrier_option import price_barrier_option, plot_barrier_payoff, plot_sample_paths_barrier
from pricing.digital_option import price_digital_option, plot_digital_payoff
from pricing.lookback_option import price_lookback_option, plot_payoff, plot_paths, plot_price_distribution





# Greeks calculation functions
def calculate_greeks_asian(S0, K, T, r, sigma, n_steps, n_paths, option_type, asian_type):
    """Calculate Greeks for Asian options using finite differences"""
    h = 0.01  # small increment for finite differences
    
    # Base price
    base_price = price_asian_option(S0, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    
    # Delta (∂V/∂S)
    price_up = price_asian_option(S0 + h, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    price_down = price_asian_option(S0 - h, K, T, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    delta = (price_up - price_down) / (2 * h)
    
    # Gamma (∂²V/∂S²)
    gamma = (price_up - 2 * base_price + price_down) / (h ** 2)
    
    # Theta (∂V/∂T)
    if T > h:
        price_theta = price_asian_option(S0, K, T - h, r, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
        theta = -(base_price - price_theta) / h
    else:
        theta = 0
    
    # Vega (∂V/∂σ)
    price_vega_up = price_asian_option(S0, K, T, r, sigma + h, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    price_vega_down = price_asian_option(S0, K, T, r, sigma - h, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    vega = (price_vega_up - price_vega_down) / (2 * h)
    
    # Rho (∂V/∂r)
    price_rho_up = price_asian_option(S0, K, T, r + h, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    price_rho_down = price_asian_option(S0, K, T, r - h, sigma, n_steps, n_paths, "monte_carlo", option_type, asian_type)
    rho = (price_rho_up - price_rho_down) / (2 * h)
    
    return {
        'Price': base_price,
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

def calculate_greeks_digital(S, K, T, r, sigma, option_type, style, Q=1.0):
    """Calculate Greeks for Digital options analytically"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    base_price = price_digital_option("black_scholes", option_type, style, S, K, T, r, sigma, Q)
    
    if style == "cash":
        if option_type == "call":
            delta = Q * np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
            gamma = -Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
            theta = -Q * np.exp(-r * T) * (r * norm.cdf(d2) + norm.pdf(d2) * (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma * np.sqrt(T)))
            vega = -Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (sigma**2 * np.sqrt(T))
            rho = Q * T * np.exp(-r * T) * norm.cdf(d2) + Q * np.exp(-r * T) * norm.pdf(d2) / (sigma * np.sqrt(T))
        else:  # put
            delta = -Q * np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
            gamma = Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
            theta = -Q * np.exp(-r * T) * (r * norm.cdf(-d2) - norm.pdf(d2) * (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma * np.sqrt(T)))
            vega = Q * np.exp(-r * T) * norm.pdf(d2) * d1 / (sigma**2 * np.sqrt(T))
            rho = Q * T * np.exp(-r * T) * norm.cdf(-d2) - Q * np.exp(-r * T) * norm.pdf(d2) / (sigma * np.sqrt(T))
    else:  # asset
        if option_type == "call":
            delta = norm.cdf(d1) + S * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            gamma = norm.pdf(d1) * (1 - d2) / (S**2 * sigma * np.sqrt(T))
            theta = -S * norm.pdf(d1) * (r + sigma**2/2 - sigma * d1 / (2 * np.sqrt(T))) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * (np.sqrt(T) - d2 / sigma)
            rho = 0  # Asset-or-nothing doesn't depend on r directly
        else:  # put
            delta = -norm.cdf(-d1) - S * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            gamma = -norm.pdf(d1) * (1 - d2) / (S**2 * sigma * np.sqrt(T))
            theta = S * norm.pdf(d1) * (r + sigma**2/2 - sigma * d1 / (2 * np.sqrt(T))) / (sigma * np.sqrt(T))
            vega = -S * norm.pdf(d1) * (np.sqrt(T) - d2 / sigma)
            rho = 0
    
    return {
        'Price': base_price,
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

def plot_sensitivity_analysis(option_type, base_params, param_name, param_range, pricing_func, **kwargs):
    """Create interactive sensitivity analysis plots"""
    prices = []
    deltas = []
    gammas = []
    
    for param_value in param_range:
        params = base_params.copy()
        params[param_name] = param_value
        
        if pricing_func == "asian":
            result = calculate_greeks_asian(**params, **kwargs)
        elif pricing_func == "digital":
            result = calculate_greeks_digital(**params, **kwargs)
        else:
            result = {'Price': 0, 'Delta': 0, 'Gamma': 0}
        
        prices.append(result['Price'])
        deltas.append(result['Delta'])
        gammas.append(result['Gamma'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Option Price', 'Delta', 'Gamma', 'Price vs Delta'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Price
    fig.add_trace(
        go.Scatter(x=param_range, y=prices, mode='lines+markers', name='Price'),
        row=1, col=1
    )
    
    # Delta
    fig.add_trace(
        go.Scatter(x=param_range, y=deltas, mode='lines+markers', name='Delta', line=dict(color='red')),
        row=1, col=2
    )
    
    # Gamma
    fig.add_trace(
        go.Scatter(x=param_range, y=gammas, mode='lines+markers', name='Gamma', line=dict(color='green')),
        row=2, col=1
    )
    
    # Price vs Delta
    fig.add_trace(
        go.Scatter(x=deltas, y=prices, mode='lines+markers', name='Price vs Delta', line=dict(color='purple')),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text=param_name, row=1, col=1)
    fig.update_xaxes(title_text=param_name, row=1, col=2)
    fig.update_xaxes(title_text=param_name, row=2, col=1)
    fig.update_xaxes(title_text="Delta", row=2, col=2)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Delta", row=1, col=2)
    fig.update_yaxes(title_text="Gamma", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text=f"Sensitivity Analysis - {param_name}")
    
    return fig

def create_3d_surface_plot(option_type, base_params, param1_name, param1_range, param2_name, param2_range, pricing_func, **kwargs):
    """Create 3D surface plot for two parameters"""
    X, Y = np.meshgrid(param1_range, param2_range)
    Z = np.zeros_like(X)
    
    for i, param1_val in enumerate(param1_range):
        for j, param2_val in enumerate(param2_range):
            params = base_params.copy()
            params[param1_name] = param1_val
            params[param2_name] = param2_val
            
            if pricing_func == "asian":
                result = calculate_greeks_asian(**params, **kwargs)
            elif pricing_func == "digital":
                result = calculate_greeks_digital(**params, **kwargs)
            else:
                result = {'Price': 0}
            
            Z[j, i] = result['Price']
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title=f'Option Price Surface - {param1_name} vs {param2_name}',
        scene=dict(
            xaxis_title=param1_name,
            yaxis_title=param2_name,
            zaxis_title='Option Price'
        ),
        height=600
    )
    
    return fig
