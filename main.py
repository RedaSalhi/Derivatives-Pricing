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
    page_icon="📈",
    initial_sidebar_state="expanded"
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

# Custom CSS for beautiful financial theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #00d4ff 0%, #00a8ff 50%, #0078ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .css-17eq0hr {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Sidebar title */
    .sidebar-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        text-align: center;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        display: block;
    }
    
    .stRadio > div > label:hover {
        background: rgba(0, 212, 255, 0.1);
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateX(2px);
    }
    
    .stRadio > div > label > div {
        color: #e2e8f0;
        font-weight: 500;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    /* Icons and emojis */
    .nav-icon {
        font-size: 1.2rem;
        margin-right: 0.5rem;
    }
    
    /* Metrics and numbers */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 120, 255, 0.05) 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.2);
        text-align: center;
    }
    
    /* Custom scrollbar */
    .css-1d391kg::-webkit-scrollbar {
        width: 6px;
    }
    
    .css-1d391kg::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    .css-1d391kg::-webkit-scrollbar-thumb {
        background: rgba(0, 212, 255, 0.3);
        border-radius: 3px;
    }
    
    .css-1d391kg::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 212, 255, 0.5);
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Page indicator */
    .page-indicator {
        position: fixed;
        top: 1rem;
        right: 1rem;
        background: rgba(0, 212, 255, 0.1);
        color: #00d4ff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        font-size: 0.9rem;
        font-weight: 500;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Main header with enhanced styling
st.markdown('<h1 class="main-title">📈 Derivatives Pricing App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🎯 Built for students, quants, and finance enthusiasts</p>', unsafe_allow_html=True)

# Enhanced sidebar with icons
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">🧭 Navigation</h2>', unsafe_allow_html=True)
    
    # Create navigation options with icons
    nav_options = {
        "🔢 Pricer": "Pricer",
        "👤 About Me": "About Me", 
        "📚 Finance Background": "Finance Background"
    }
    
    selected_page = st.radio(
        "",
        list(nav_options.keys()),
        key="navigation"
    )
    
    # Add some visual enhancements to sidebar
    st.markdown("---")
    st.markdown("""
    <div class="feature-card">
        <h4 style="color: #00d4ff; margin-bottom: 0.5rem;">🚀 Features</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
        • Real-time pricing models<br>
        • Interactive visualizations<br>
        • Educational content<br>
        • Professional tools
        </p>
    </div>
    """, unsafe_allow_html=True)

# Get the actual page name from the mapping
actual_page = nav_options[selected_page]

# Add page indicator
st.markdown(f'<div class="page-indicator">Current: {actual_page}</div>', unsafe_allow_html=True)

# Enhanced page routing with loading states
if actual_page == "Pricer":
    st.markdown("### 🔢 Derivatives Pricing Engine")
    st.markdown("*Advanced pricing models for financial derivatives*")
    with st.spinner("Loading pricing models..."):
        runpy.run_path(os.path.join(os.path.dirname(__file__), "pricer_minim.py"))
        
elif actual_page == "About Me":
    st.markdown("### 👤 About the Developer")
    st.markdown("*Learn more about the creator of this application*")
    with st.spinner("Loading profile..."):
        runpy.run_path(os.path.join(os.path.dirname(__file__), "about_me.py"))
    
elif actual_page == "Finance Background":
    st.markdown("### 📚 Financial Theory & Background")
    st.markdown("*Educational content on derivatives and financial mathematics*")
    with st.spinner("Loading educational content..."):
        runpy.run_path(os.path.join(os.path.dirname(__file__), "finance_background.py"))

# Footer enhancement
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 2rem;">
    <p>💼 Professional Derivatives Pricing Platform | 🔒 Secure & Reliable</p>
</div>
""", unsafe_allow_html=True)
