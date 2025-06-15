"""
Unified Styling System for Derivatives Pricing App
styles/app_styles.py

This module provides a centralized styling system that ensures consistency
across all pages and components in the application.
"""

import streamlit as st

def apply_global_styles():
    """Apply global styles that should be consistent across all pages"""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ===================
           GLOBAL VARIABLES 
        =================== */
        :root {
            /* Color Palette */
            --primary-blue: #2563eb;
            --primary-blue-light: #3b82f6;
            --primary-blue-dark: #1d4ed8;
            --secondary-orange: #f59e0b;
            --success-green: #10b981;
            --warning-yellow: #f59e0b;
            --danger-red: #ef4444;
            --info-cyan: #06b6d4;
            
            /* Neutral Colors */
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            
            /* Typography */
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --text-xs: 0.75rem;
            --text-sm: 0.875rem;
            --text-base: 1rem;
            --text-lg: 1.125rem;
            --text-xl: 1.25rem;
            --text-2xl: 1.5rem;
            --text-3xl: 1.875rem;
            --text-4xl: 2.25rem;
            --text-5xl: 3rem;
            
            /* Spacing */
            --space-1: 0.25rem;
            --space-2: 0.5rem;
            --space-3: 0.75rem;
            --space-4: 1rem;
            --space-5: 1.25rem;
            --space-6: 1.5rem;
            --space-8: 2rem;
            --space-10: 2.5rem;
            --space-12: 3rem;
            
            /* Border Radius */
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --radius-xl: 1rem;
            --radius-2xl: 1.5rem;
            
            /* Shadows */
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        
        /* ===================
           GLOBAL RESETS 
        =================== */
        * {
            font-family: var(--font-family);
        }
        
        .stApp {
            background: linear-gradient(135deg, #ffffff 0%, var(--gray-50) 100%);
        }
        
        .main .block-container {
            padding-top: var(--space-8);
            background: white;
            border-radius: var(--radius-2xl);
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--gray-200);
            max-width: 1000px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* ===================
           TYPOGRAPHY SYSTEM 
        =================== */
        .app-title {
            background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: var(--text-5xl);
            font-weight: 700;
            text-align: center;
            margin-bottom: var(--space-2);
            letter-spacing: -0.02em;
            line-height: 1.1;
        }
        
        .app-subtitle {
            color: var(--gray-500);
            font-size: var(--text-xl);
            text-align: center;
            margin-bottom: var(--space-8);
            font-weight: 400;
        }
        
        .page-title {
            color: var(--gray-800);
            font-size: var(--text-4xl);
            font-weight: 600;
            text-align: center;
            margin-bottom: var(--space-2);
            letter-spacing: -0.01em;
        }
        
        .page-subtitle {
            color: var(--gray-600);
            font-size: var(--text-lg);
            text-align: center;
            margin-bottom: var(--space-8);
            font-weight: 400;
        }
        
        .section-title {
            color: var(--primary-blue);
            font-size: var(--text-2xl);
            font-weight: 600;
            margin-bottom: var(--space-4);
            display: flex;
            align-items: center;
            gap: var(--space-2);
        }
        
        .subsection-title {
            color: var(--gray-700);
            font-size: var(--text-xl);
            font-weight: 500;
            margin: var(--space-6) 0 var(--space-4) 0;
            padding-bottom: var(--space-2);
            border-bottom: 2px solid var(--gray-200);
            display: inline-block;
        }
        
        /* ===================
           SIDEBAR STYLING 
        =================== */
        .css-1d391kg, .css-17eq0hr {
            background: linear-gradient(180deg, #ffffff 0%, var(--gray-50) 100%);
            border-right: 1px solid var(--gray-200);
        }
        
        .sidebar-title {
            color: var(--gray-800);
            font-size: var(--text-xl);
            font-weight: 600;
            margin-bottom: var(--space-6);
            text-align: center;
            padding-bottom: var(--space-4);
            border-bottom: 2px solid var(--gray-200);
        }
        
        /* Radio button styling */
        .stRadio > div {
            background: var(--gray-50);
            border-radius: var(--radius-xl);
            padding: var(--space-4);
            border: 1px solid var(--gray-200);
            box-shadow: var(--shadow-sm);
        }
        
        .stRadio > div > label {
            background: white;
            border-radius: var(--radius-lg);
            padding: var(--space-3) var(--space-4);
            margin: var(--space-1) 0;
            border: 1px solid var(--gray-200);
            transition: all 0.2s ease;
            cursor: pointer;
            display: block;
            box-shadow: var(--shadow-sm);
        }
        
        .stRadio > div > label:hover {
            background: #f0f9ff;
            border-color: var(--primary-blue-light);
            transform: translateX(2px);
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.1);
        }
        
        .stRadio > div > label > div {
            color: var(--gray-700);
            font-weight: 500;
        }
        
        /* Feature box in sidebar */
        .feature-box {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-radius: var(--radius-xl);
            padding: var(--space-6);
            margin-top: var(--space-8);
            border: 1px solid #bfdbfe;
            text-align: center;
        }
        
        .feature-title {
            color: var(--primary-blue-dark);
            font-size: var(--text-lg);
            font-weight: 600;
            margin-bottom: var(--space-2);
        }
        
        .feature-text {
            color: var(--gray-600);
            font-size: var(--text-sm);
            line-height: 1.5;
        }
    </style>
    """, unsafe_allow_html=True)

def get_component_styles():
    """Return CSS for reusable components"""
    return """
    <style>
        /* ===================
           COMPONENT STYLES 
        =================== */
        
        /* Information Boxes */
        .info-box {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            padding: var(--space-6);
            border-radius: var(--radius-xl);
            border-left: 4px solid var(--primary-blue);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        .success-box {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            padding: var(--space-6);
            border-radius: var(--radius-xl);
            border-left: 4px solid var(--success-green);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        .warning-box {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            padding: var(--space-6);
            border-radius: var(--radius-xl);
            border-left: 4px solid var(--warning-yellow);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        .danger-box {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
            padding: var(--space-6);
            border-radius: var(--radius-xl);
            border-left: 4px solid var(--danger-red);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        /* Profile/Contact Boxes */
        .profile-box {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            padding: var(--space-8);
            border-radius: var(--radius-xl);
            border-left: 5px solid var(--primary-blue);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        .contact-box {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            padding: var(--space-8);
            border-radius: var(--radius-xl);
            border-left: 5px solid var(--warning-yellow);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        .links-box {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            padding: var(--space-8);
            border-radius: var(--radius-xl);
            border-left: 5px solid var(--success-green);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        .download-section {
            background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
            padding: var(--space-8);
            border-radius: var(--radius-xl);
            border-left: 5px solid var(--danger-red);
            margin: var(--space-6) 0;
            box-shadow: var(--shadow-md);
        }
        
        /* Professional Content Boxes */
        .objective-box {
            background: var(--gray-50);
            border: 2px solid var(--gray-200);
            padding: var(--space-10);
            border-radius: var(--radius-xl);
            margin: var(--space-8) 0;
            box-shadow: var(--shadow-md);
            border-left: 6px solid var(--primary-blue);
        }
        
        .methodology-box {
            background: #fffaf0;
            border: 2px solid #fed7aa;
            padding: var(--space-10);
            border-radius: var(--radius-xl);
            margin: var(--space-8) 0;
            box-shadow: var(--shadow-md);
            border-left: 6px solid var(--warning-yellow);
        }
        
        .model-box {
            background: #f0fff4;
            border: 2px solid #c6f6d5;
            padding: var(--space-10);
            border-radius: var(--radius-xl);
            margin: var(--space-8) 0;
            box-shadow: var(--shadow-md);
            border-left: 6px solid var(--success-green);
        }
        
        .greeks-box {
            background: #fff5f5;
            border: 2px solid #fed7d7;
            padding: var(--space-10);
            border-radius: var(--radius-xl);
            margin: var(--space-8) 0;
            box-shadow: var(--shadow-md);
            border-left: 6px solid var(--danger-red);
        }
        
        .strategy-box {
            background: #f0f8ff;
            border: 2px solid #bee3f8;
            padding: var(--space-10);
            border-radius: var(--radius-xl);
            margin: var(--space-8) 0;
            box-shadow: var(--shadow-md);
            border-left: 6px solid var(--info-cyan);
        }
        
        /* Metric Containers */
        .metric-container {
            background: var(--gray-50);
            padding: var(--space-4);
            border-radius: var(--radius-lg);
            margin: var(--space-2) 0;
            border: 1px solid var(--gray-200);
            box-shadow: var(--shadow-sm);
        }
        
        /* Text Styles */
        .content-text {
            font-size: var(--text-lg);
            line-height: 1.8;
            color: var(--gray-700);
            margin-bottom: var(--space-4);
        }
        
        .profile-info {
            font-size: var(--text-lg);
            line-height: 1.8;
            color: var(--gray-700);
        }
        
        .profile-info strong {
            color: var(--primary-blue);
            font-weight: 600;
        }
        
        /* Form Styles */
        .contact-form {
            background: white;
            padding: var(--space-6);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-md);
            border: 1px solid var(--gray-200);
        }
        
        .form-input {
            width: 100%;
            padding: var(--space-3);
            margin-bottom: var(--space-4);
            border: 2px solid var(--gray-200);
            border-radius: var(--radius-lg);
            font-size: var(--text-base);
            transition: border-color 0.3s ease;
            font-family: var(--font-family);
        }
        
        .form-input:focus {
            border-color: var(--primary-blue);
            outline: none;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        .form-button {
            background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%);
            color: white;
            padding: var(--space-3) var(--space-8);
            border: none;
            border-radius: var(--radius-lg);
            font-size: var(--text-base);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        }
        
        .form-button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        /* Link Styles */
        .link-item {
            display: flex;
            align-items: center;
            gap: var(--space-2);
            margin: var(--space-2) 0;
            font-size: var(--text-lg);
        }
        
        .link-item a {
            color: var(--primary-blue);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        
        .link-item a:hover {
            color: var(--secondary-orange);
        }
        
        /* Highlight Box */
        .highlight-box {
            background: rgba(37, 99, 235, 0.1);
            border: 1px solid rgba(37, 99, 235, 0.2);
            border-radius: var(--radius-lg);
            padding: var(--space-4);
            margin: var(--space-4) 0;
        }
        
        /* Animation */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .animate-fade-in {
            animation: fadeInUp 1s ease-out;
        }
        
        .animate-fade-in-delay {
            animation: fadeInUp 1s ease-out 0.2s both;
        }
    </style>
    """

# Convenience functions for common styling patterns
def render_page_header(title, subtitle=None):
    """Render a standardized page header"""
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)

def render_app_header(title, subtitle=None):
    """Render the main app header"""
    st.markdown(f'<h1 class="app-title">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="app-subtitle">{subtitle}</p>', unsafe_allow_html=True)

def render_section_title(title):
    """Render a section title"""
    st.markdown(f'<h2 class="section-title">{title}</h2>', unsafe_allow_html=True)

def render_info_box(content, box_type="info"):
    """Render an information box with specified type"""
    box_class = f"{box_type}-box"
    st.markdown(f'<div class="{box_class}">{content}</div>', unsafe_allow_html=True)

# Color palette for easy reference
COLORS = {
    'primary': '#2563eb',
    'primary_light': '#3b82f6', 
    'primary_dark': '#1d4ed8',
    'secondary': '#f59e0b',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'gray_50': '#f9fafb',
    'gray_100': '#f3f4f6',
    'gray_200': '#e5e7eb',
    'gray_500': '#6b7280',
    'gray_700': '#374151',
    'gray_800': '#1f2937'
}
