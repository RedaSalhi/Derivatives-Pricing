import sys
import os
import runpy
import streamlit as st
import streamlit.components.v1 as components

# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

st.set_page_config(
    page_title="Derivatives Pricing App", 
    layout="centered",
    page_icon="📈"
)

# ✅ Google Analytics (G-KFYG7J821K)
components.html(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-KFYG7J821K"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-KFYG7J821K', {
        'page_path': '/streamlit-app',
        'page_title': 'Derivatives Pricing App',
        'page_location': window.location.href
      });
    </script>
    """,
    height=0
)

# Simple, clean styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        max-width: 800px;
    }
    
    .main-title {
        color: #2563eb;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #64748b;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stRadio > div {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Simple header
st.markdown('<h1 class="main-title">📈 Derivatives Pricing App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Built for students, quants, and finance enthusiasts</p>', unsafe_allow_html=True)

# Clean sidebar
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Select Page",
    ["Pricer", "About Me", "Finance Background"],
)

# Simple page routing
if selected_page == "Pricer":
    runpy.run_path(os.path.join(os.path.dirname(__file__), "pricer_minim.py"))
elif selected_page == "About Me":
    runpy.run_path(os.path.join(os.path.dirname(__file__), "about_me.py"))
elif selected_page == "Finance Background":
    runpy.run_path(os.path.join(os.path.dirname(__file__), "finance_background.py"))
