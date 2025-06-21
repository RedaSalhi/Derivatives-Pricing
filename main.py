import sys
import os
import runpy
import streamlit as st
import streamlit.components.v1 as components
from styles.app_styles import load_theme

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

st.set_page_config(
    page_title="Derivatives Pricing App", 
    layout="centered",
    page_icon=""
)


load_theme()


# Beautiful header
st.markdown('<h1 class="main-title">Derivatives Pricing App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Built for students, quants, and finance enthusiasts</p>', unsafe_allow_html=True)

# Sidebar with clean styling
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">Navigation</h2>', unsafe_allow_html=True)
    
    selected_page = st.radio(
        "",
        ["Pricer", "About Me", "Finance Background"],
    )
    
    # Feature box
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">Professional Tools</div>
        <div class="feature-text">
            Advanced pricing models with educational insights
        </div>
    </div>
    """, unsafe_allow_html=True)

if "Pricer" in selected_page:
    runpy.run_path(os.path.join(os.path.dirname(__file__), "pricer_minim.py"))
    
elif "About Me" in selected_page:
    runpy.run_path(os.path.join(os.path.dirname(__file__), "about_me.py"))
    
elif "Finance Background" in selected_page:
    runpy.run_path(os.path.join(os.path.dirname(__file__), "finance_background.py"))
