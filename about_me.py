import streamlit as st
import streamlit.components.v1 as components
import os
import sys

# Add the current directory to Python path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from styles.app_styles import load_theme
except ImportError:
    # Fallback if import fails
    def load_theme():
        pass

def about_me_tab():
    """Professional About Me page with integrated styling"""
    
    # Load the theme to ensure consistent styling
    try:
        load_theme()
    except:
        # Fallback CSS if theme loading fails
        st.markdown("""
        <style>
        .profile-box {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 2rem;
            border-radius: 1rem;
            border-left: 6px solid #2563eb;
            margin: 1.5rem 0;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            font-family: 'Inter', sans-serif;
        }
        .section-title {
            color: #2563eb;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .profile-info {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #374151;
        }
        .skills-section ul {
            list-style: none;
            padding: 0;
        }
        .skills-section li {
            padding: 0.5rem 0;
            color: #374151;
            position: relative;
            padding-left: 1.5rem;
        }
        .skills-section li::before {
            content: '▸';
            color: #2563eb;
            font-weight: bold;
            position: absolute;
            left: 0;
            top: 0.5rem;
        }
        .download-section {
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            padding: 2rem;
            border-radius: 1rem;
            border-left: 6px solid #0891b2;
            margin: 1.5rem 0;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .links-box {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            padding: 2rem;
            border-radius: 1rem;
            border-left: 6px solid #059669;
            margin: 1.5rem 0;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .link-item {
            margin: 1rem 0;
            padding: 0.75rem;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 0.5rem;
        }
        .contact-form {
            background: linear-gradient(135deg, #fafafa 0%, #f4f4f5 100%);
            padding: 2rem;
            border-radius: 1rem;
            border: 2px solid #e5e7eb;
            margin: 1.5rem 0;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .info-box {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 4px solid #2563eb;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .footer-section {
            background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
            border: 2px solid #e5e7eb;
            padding: 2rem;
            border-radius: 1rem;
            margin: 2rem 0;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Page header with animation
    st.markdown("""
    <div class="animate-fade-in-about">
        <h1 class="section-title">👋 About Me</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Profile section
    st.markdown("""
    <div class="profile-box animate-fade-in-about">
        <h2 class="section-title">🎯 Professional Profile</h2>
        <div class="profile-info">
            <strong>Welcome to my Derivatives Pricing Application!</strong> I'm passionate about quantitative finance and building tools that make complex financial concepts accessible to students, practitioners, and enthusiasts alike.
            <br><br>
            This application demonstrates <strong>advanced derivatives pricing models</strong> including Black-Scholes for European options, binomial trees for American options, and Monte Carlo simulations for exotic derivatives. Each model includes educational insights to help users understand the underlying mathematics and financial theory.
            <br><br>
            I believe in making <strong>quantitative finance education</strong> more interactive and practical, bridging the gap between academic theory and real-world application through clean, professional tools.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Skills and Interests section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="skills-section animate-fade-in-delay-about">
            <h3>🔧 Technical Skills</h3>
            <ul>
                <li><strong>Programming:</strong> Python, R, SQL, MATLAB</li>
                <li><strong>Finance:</strong> Derivatives Pricing, Risk Management</li>
                <li><strong>Analytics:</strong> Monte Carlo Simulation, Statistical Modeling</li>
                <li><strong>Tools:</strong> Streamlit, NumPy, SciPy, Pandas</li>
                <li><strong>Visualization:</strong> Plotly, Matplotlib, Financial Charts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="skills-section animate-fade-in-delay-about">
            <h3>💡 Interests & Focus</h3>
            <ul>
                <li><strong>Quantitative Finance:</strong> Options pricing and risk metrics</li>
                <li><strong>Financial Engineering:</strong> Model development and validation</li>
                <li><strong>Education:</strong> Making finance accessible through technology</li>
                <li><strong>Open Source:</strong> Building tools for the finance community</li>
                <li><strong>Research:</strong> Market volatility and pricing accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Downloads section
    st.markdown("""
    <div class="download-section animate-fade-in-delay-about">
        <h2 class="section-title">📄 Download Resources</h2>
        <p>Get my CV and project documentation to learn more about my background and this application.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create download buttons - with error handling
    col1, col2 = st.columns(2)
    
    with col1:
        # CV Download - create a simple text file as placeholder if no CV exists
        try:
            if os.path.exists("CV_Reda_Salhi.pdf"):
                with open("CV_Reda_Salhi.pdf", "rb") as file:
                    st.download_button(
                        label="📋 Download CV",
                        data=file,
                        file_name="CV_Reda_Salhi.pdf",
                        mime="application/pdf",
                        help="Download my professional CV"
                    )
            elif os.path.exists("README.md"):
                with open("README.md", "rb") as file:
                    st.download_button(
                        label="📋 Download CV",
                        data=file,
                        file_name="CV_Reda_Salhi.pdf",
                        mime="application/pdf",
                        help="Download my professional CV (placeholder)"
                    )
            else:
                st.info("CV download will be available soon")
        except Exception as e:
            st.info("CV download temporarily unavailable")
    
    with col2:
        # Project documentation
        try:
            if os.path.exists("README.md"):
                with open("README.md", "rb") as file:
                    st.download_button(
                        label="📖 Project Documentation",
                        data=file,
                        file_name="Derivatives_Pricing_Guide.md",
                        mime="text/markdown",
                        help="Comprehensive guide to this derivatives pricing tool"
                    )
            else:
                st.info("Documentation download will be available soon")
        except Exception as e:
            st.info("Documentation download temporarily unavailable")
    
    # Links section
    st.markdown("""
    <div class="links-box animate-fade-in-delay-about">
        <h2 class="section-title">🔗 Connect & Explore</h2>
        <div class="link-item">
            <a href="https://github.com/your-username" target="_blank">🐙 GitHub Portfolio</a>
        </div>
        <div class="link-item">
            <a href="https://linkedin.com/in/your-profile" target="_blank">💼 LinkedIn Profile</a>
        </div>
        <div class="link-item">
            <a href="https://derivatives-pricing.streamlit.app" target="_blank">🚀 Live Application</a>
        </div>
        <div class="link-item">
            <a href="mailto:salhi.reda47@gmail.com" class="email-link">📧 Professional Email</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact form section
    st.markdown("""
    <div class="contact-form animate-fade-in-delay-about">
        <h2 class="section-title">💬 Get In Touch</h2>
        <p>Have questions about derivatives pricing, want to collaborate, or interested in discussing quantitative finance? I'd love to hear from you!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact form using FormSubmit
    contact_form = f"""
    <form action="https://formsubmit.co/salhi.reda47@gmail.com" method="POST" class="contact-form">
        <input type="hidden" name="_captcha" value="false">
        <input type="hidden" name="_subject" value="New Contact from Derivatives Pricing App">
        <input type="hidden" name="_next" value="https://derivatives-pricing.streamlit.app">
        
        <input type="text" name="name" placeholder="Your Name" required class="form-input">
        <input type="email" name="email" placeholder="Your Email" required class="form-input">
        <input type="text" name="subject" placeholder="Subject" required class="form-input">
        <textarea name="message" placeholder="Your Message" rows="5" required class="form-input"></textarea>
        
        <button type="submit" class="form-button">Send Message 📧</button>
    </form>
    """
    
    components.html(contact_form, height=400)
    
    # Application insights
    st.markdown("""
    <div class="info-box animate-fade-in-delay-about">
        <h2 class="section-title">🔍 About This Application</h2>
        <p><strong>Purpose:</strong> This derivatives pricing tool was built to provide accurate, educational, and accessible financial modeling for students, professionals, and enthusiasts.</p>
        
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Multiple pricing models (Black-Scholes, Binomial, Monte Carlo)</li>
            <li>Real-time Greeks calculation and visualization</li>
            <li>Interactive parameter adjustment with instant results</li>
            <li>Educational content explaining each model's assumptions</li>
            <li>Professional-grade accuracy with academic transparency</li>
        </ul>
        
        <p><strong>Technology Stack:</strong> Built with Python, Streamlit, NumPy, and SciPy for robust mathematical computations and an intuitive user interface.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer-section animate-fade-in-delay-about">
        <div>
            🎯 "Making quantitative finance accessible through technology and education"
        </div>
        <br>
        <div style="font-size: 0.9em; color: #666;">
            Built with ❤️ using Streamlit | © 2024 Reda Salhi
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    about_me_tab()
