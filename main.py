import sys
import os
import runpy
import streamlit as st
from pricer_page import render_pricer_page

    


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
    #runpy.run_path(os.path.join(os.path.dirname(__file__), "pricer_page.py"))
    render_pricer_page()
elif selected_page == "About Me":
    st.header("About Me")
    st.markdown(
        "This section can include your CV, a link to your "
        "[LinkedIn](https://www.linkedin.com/) profile, and showcase other projects."
    )
elif selected_page == "Finance Background":
    st.header("Finance Background & Methodology")
    st.markdown(
        "Provide information about your finance background and the methodology used in the pricer."
    )
