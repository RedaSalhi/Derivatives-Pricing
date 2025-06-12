import sys
import os
import runpy
import streamlit as st
# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st
import streamlit.components.v1 as components

# Injecter Google Analytics (GA4)
import streamlit as st
import streamlit.components.v1 as components

# Inject Google Analytics (GA4)
components.html(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-D7MTVEVGH0"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-D7MTVEVGH0', {
        'page_title': 'Derivatives Pricing App',
        'page_path': '/streamlit-app',
        'page_location': 'https://derivatives-pricing.streamlit.app'
      });
    </script>
    """,
    height=0,
)

    




# Allow importing from the pricing directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

st.set_page_config(page_title="Derivatives Pricing App", layout="centered")
st.title("Derivatives Pricing App")
st.caption("Built for students, quants, and finance enthusiasts")

st.sidebar.title("Navigation")
selected_page = st.sidebar.radio(
    "Select Page",
    ["Pricer", "About Me", "Finance Background"],
)

if selected_page == "Pricer":
    runpy.run_path(os.path.join(os.path.dirname(__file__), "pricer_page.py"))

elif selected_page == "About Me":
    runpy.run_path(os.path.join(os.path.dirname(__file__), "about_me.py"))
    
elif selected_page == "Finance Background":
    runpy.run_path(os.path.join(os.path.dirname(__file__), "finance_background.py"))
