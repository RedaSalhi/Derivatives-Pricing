# pricer_minim.py
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

import sys
import os
import streamlit as st
from styles.app_styles import load_theme

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

load_theme()

# Import tab modules
from tabs.vanilla_options import vanilla_options_tab
from tabs.forward_contracts import forward_contracts_tab
from tabs.option_strategies import main_option_strategies_tab
from tabs.exotic_options import exotic_options_tab
from tabs.swaps import swaps_tab
from tabs.interest_rate_instruments import interest_rate_instruments_tab





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
