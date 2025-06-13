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
from tabs.option_strategies import option_strategies_tab
from tabs.exotic_options import exotic_options_tab
from tabs.swaps import swaps_tab
from tabs.interest_rate_instruments import interest_rate_instruments_tab

# Custom CSS for enhanced styling
# --- Custom Styling ---
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
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .formula {
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        color: #1f77b4;
        margin: 10px 0;
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
    option_strategies_tab()

with tab4:
    exotic_options_tab()

with tab5:
    swaps_tab()

with tab6:
    interest_rate_instruments_tab()
