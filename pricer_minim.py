# main.py
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

import sys
import os
import streamlit as st

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import tab modules
from tabs.vanilla_options import vanilla_options_tab
from tabs.forward_contracts import forward_contracts_tab
from tabs.option_strategies import main_option_strategies_tab
from tabs.exotic_options import exotic_options_tab
from tabs.swaps import swaps_tab
from tabs.interest_rate_instruments import interest_rate_instruments_tab

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .formula {
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        color: #1f77b4;
        margin: 15px 0;
        padding: 15px;
        background-color: #f0f8ff;
        border-radius: 8px;
        border: 1px solid #d0e7ff;
    }
    .section-title {
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 1.2em;
    }
    .info-box h4, .warning-box h4 {
        color: #1f77b4;
        margin-bottom: 15px;
        margin-top: 0;
    }
    .warning-box h4 {
        color: #856404;
    }
    .greeks-delta {
        background-color: #e8f4f8;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .greeks-rho {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .greeks-vega {
        background-color: #d1ecf1;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #17a2b8;
    }
    .greeks-theta {
        background-color: #f8d7da;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #dc3545;
    }
    .greeks-gamma {
        background-color: #d4edda;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 4px;
        padding-left: 10px;
        padding-right: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)




# Main header
st.header("Derivatives Pricer")

# Create tabs layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vanilla Options", 
    "Forward Contracts", 
    "Option Strategies", 
    "Exotic Options",
    "Swaps",
    "Interest Rate Instruments"
])

# Execute each tab
with tab1:
    vanilla_options_tab()

with tab2:
    forward_contracts_tab()

with tab3:
    main_option_strategies_tab()

with tab4:
    exotic_options_tab()

with tab5:
    swaps_tab()

with tab6:
    interest_rate_instruments_tab()
