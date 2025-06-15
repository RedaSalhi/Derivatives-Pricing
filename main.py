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
    page_icon=""
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

# Beautiful light theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        max-width: 900px;
    }
    
    .main-title {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-17eq0hr {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    .sidebar-title {
        color: #1f2937;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #f9fafb;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .stRadio > div > label {
        background: white;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        cursor: pointer;
        display: block;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    
    .stRadio > div > label:hover {
        background: #f0f9ff;
        border-color: #3b82f6;
        transform: translateX(2px);
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .stRadio > div > label > div {
        color: #374151;
        font-weight: 500;
    }
    
    /* Feature highlight */
    .feature-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid #bfdbfe;
        text-align: center;
    }
    
    .feature-title {
        color: #1e40af;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-text {
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Page headers */
    .page-header {
        color: #1f2937;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #e5e7eb;
    }
    
    .page-subtitle {
        color: #6b7280;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Clean page routing with headers
if "Pricer" in selected_page:
    st.markdown('<h2 class="main-title">Derivatives Pricing Engine</h2>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Advanced pricing models for financial derivatives</p>', unsafe_allow_html=True)
    runpy.run_path(os.path.join(os.path.dirname(__file__), "pricer_minim.py"))
    
elif "About Me" in selected_page:
    st.markdown('<h2 class="main-title">About the Developer</h2>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Learn more about the creator of this application</p>', unsafe_allow_html=True)
    runpy.run_path(os.path.join(os.path.dirname(__file__), "about_me.py"))
    
elif "Finance Background" in selected_page:
    st.markdown('<h2 class="main-title">Financial Theory & Background</h2>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Educational content on derivatives and financial mathematics</p>', unsafe_allow_html=True)
    runpy.run_path(os.path.join(os.path.dirname(__file__), "finance_background.py"))
