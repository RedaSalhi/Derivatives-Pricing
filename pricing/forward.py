import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots




# -----------------------------
# Pricing Function
# -----------------------------

def price_forward_contract(
    spot_price: float,
    interest_rate: float,
    time_to_maturity: float,
    storage_cost: float = 0.0,
    dividend_yield: float = 0.0,
    pricing_model: str = "cost_of_carry",
    sub_type: str = "plain"
) -> float:
    """
    Prices a forward contract using the specified pricing model.
    """
    pricing_model = pricing_model.lower()

    if pricing_model == "cost_of_carry":
        F = spot_price * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity)
        return F
    else:
        raise NotImplementedError(f"Pricing model '{pricing_model}' is not implemented for forward contracts.")

# -----------------------------
# Enhanced Plotting Functions
# -----------------------------

def plot_forward_mark_to_market_plotly(
    strike_price: float,
    time_to_maturity: float,
    interest_rate: float,
    storage_cost: float = 0.0,
    dividend_yield: float = 0.0,
    position: str = "long"
):
    """
    Interactive Plotly version of mark-to-market plotting.
    """
    position = position.lower()
    S_t = np.linspace(0.5 * strike_price, 1.5 * strike_price, 500)

    value_t = S_t * np.exp((interest_rate + storage_cost - dividend_yield) * time_to_maturity) \
              - strike_price * np.exp(-interest_rate * time_to_maturity)

    if position == "short":
        value_t = -value_t
        title = "Short Forward Value (t < T)"
        color = 'red'
    else:
        title = "Long Forward Value (t < T)"
        color = 'blue'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_t, 
        y=value_t,
        mode='lines',
        name=title,
        line=dict(color=color, width=3),
        hovertemplate='Spot Price: $%{x:.2f}<br>Contract Value: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)
    fig.update_layout(
        title=f"Mark-to-Market Value of Forward Contract Before Maturity<br><sub>Time to Maturity: {time_to_maturity:.2f} years</sub>",
        xaxis_title="Spot Price at Time t ($)",
        yaxis_title="Forward Contract Value ($)",
        template="plotly_white",
        height=500
    )
    
    return fig

def plot_forward_payout_plotly(strike_price: float, position: str = "long"):
    """
    Interactive Plotly version of payout plotting.
    """
    position = position.lower()
    S_T = np.linspace(0.5 * strike_price, 1.5 * strike_price, 500)

    if position == "long":
        payout = S_T - strike_price
        title = "Long Forward Payout"
        color = 'green'
    elif position == "short":
        payout = strike_price - S_T
        title = "Short Forward Payout"
        color = 'red'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_T, 
        y=payout,
        mode='lines',
        name=title,
        line=dict(color=color, width=3),
        hovertemplate='Spot Price at Maturity: $%{x:.2f}<br>Payout: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)
    fig.update_layout(
        title=f"Forward Contract Payout at Maturity<br><sub>Strike Price: ${strike_price:.2f}</sub>",
        xaxis_title="Spot Price at Maturity ($)",
        yaxis_title="Payout ($)",
        template="plotly_white",
        height=500
    )
    
    return fig

def create_sensitivity_analysis(base_params):
    """
    Create sensitivity analysis for forward pricing.
    """
    # Parameters to analyze
    spot_range = np.linspace(base_params['spot_price'] * 0.7, base_params['spot_price'] * 1.3, 50)
    rate_range = np.linspace(0.01, 0.10, 50)
    time_range = np.linspace(0.1, 2.0, 50)
    
    # Calculate forward prices for different parameters
    spot_prices = [price_forward_contract(s, base_params['interest_rate'], base_params['time_to_maturity'], 
                                         base_params['storage_cost'], base_params['dividend_yield']) 
                  for s in spot_range]
    
    rate_prices = [price_forward_contract(base_params['spot_price'], r, base_params['time_to_maturity'], 
                                         base_params['storage_cost'], base_params['dividend_yield']) 
                  for r in rate_range]
    
    time_prices = [price_forward_contract(base_params['spot_price'], base_params['interest_rate'], t, 
                                         base_params['storage_cost'], base_params['dividend_yield']) 
                  for t in time_range]
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Spot Price Sensitivity', 'Interest Rate Sensitivity', 
                       'Time to Maturity Sensitivity', 'Parameter Summary'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}], 
               [{"type": "scatter"}, {"type": "table"}]]
    )
    
    # Add traces
    fig.add_trace(go.Scatter(x=spot_range, y=spot_prices, name='Forward vs Spot', 
                            line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=rate_range*100, y=rate_prices, name='Forward vs Rate', 
                            line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(x=time_range, y=time_prices, name='Forward vs Time', 
                            line=dict(color='green')), row=2, col=1)
    
    # Add parameter table
    fig.add_trace(go.Table(
        header=dict(values=['Parameter', 'Value']),
        cells=dict(values=[
            ['Spot Price', 'Interest Rate', 'Time to Maturity', 'Storage Cost', 'Dividend Yield'],
            [f"${base_params['spot_price']:.2f}", f"{base_params['interest_rate']*100:.2f}%", 
             f"{base_params['time_to_maturity']:.2f} years", f"{base_params['storage_cost']*100:.2f}%", 
             f"{base_params['dividend_yield']*100:.2f}%"]
        ])
    ), row=2, col=2)
    
    fig.update_xaxes(title_text="Spot Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Interest Rate (%)", row=1, col=2)
    fig.update_xaxes(title_text="Time to Maturity (years)", row=2, col=1)
    fig.update_yaxes(title_text="Forward Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Forward Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Forward Price ($)", row=2, col=1)
    
    fig.update_layout(height=700, showlegend=False, title_text="Forward Contract Sensitivity Analysis")
    
    return fig
