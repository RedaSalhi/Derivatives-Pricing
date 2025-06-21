

# Quantitative Finance Platform

A comprehensive derivatives pricing and risk management platform built with **Streamlit** and **Python**. This educational tool provides state-of-the-art pricing models, Greeks analysis, and portfolio optimization for various financial instruments.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://derivatives-pricing.streamlit.app/)

## 🚀 Live Demo

[**Access the Platform**](https://derivatives-pricing.streamlit.app)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Pricing Models](#-pricing-models)
- [Supported Instruments](#-supported-instruments)
- [Educational Resources](#-educational-resources)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

## ✨ Features

### 🔬 **Advanced Pricing Models**
- **Black-Scholes**: Analytical solutions for European options
- **Binomial Trees**: American options with early exercise
- **Monte Carlo**: Complex payoffs and path-dependent derivatives
- **Vasicek Model**: Interest rate modeling and bond pricing

### 📈 **Complete Greeks Suite**
- Real-time calculation of Delta, Gamma, Theta, Vega, Rho
- Interactive sensitivity analysis and visualization
- Continuous Greeks plotting with smooth curves
- Multi-parameter sensitivity heatmaps

### 🎯 **Strategy Builder**
- 10+ pre-defined options strategies
- Custom multi-leg strategy constructor
- Interactive P&L visualization
- Breakeven analysis and risk metrics

### 💼 **Professional Risk Management**
- VaR and Expected Shortfall calculations
- Stress testing scenarios
- Portfolio-level analytics
- Model comparison and validation

### 🌐 **Multi-Asset Coverage**
- Vanilla Options (European & American)
- Exotic Options (Asian, Barrier, Digital, Lookback)
- Forward Contracts with cost-of-carry
- Interest Rate Swaps
- Currency Swaps
- Bond Pricing and Options

## 🛠 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/derivatives-pricing.git
   cd derivatives-pricing
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start exploring the platform!

### Alternative Installation (Conda)

```bash
# Create a new conda environment
conda create -n derivatives python=3.9
conda activate derivatives

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## 🎮 Usage

### Quick Start Guide

1. **Navigate to Setup & Parameters** in the Options Strategy tab
2. **Configure market parameters** (spot price, volatility, etc.)
3. **Select your preferred pricing model**
4. **Complete setup** to unlock all analysis tools
5. **Explore different tabs** for various instruments and analyses

### Key Workflows

#### **Pricing a Single Option**
```python
# Example: Price a European call option
from pricing.vanilla_options import price_vanilla_option

price = price_vanilla_option(
    option_type="call",
    exercise_style="european", 
    model="black-scholes",
    S=100, K=105, T=1.0, r=0.05, sigma=0.2
)
```

#### **Building an Options Strategy**
1. Go to **Strategy Builder** tab
2. Choose **Predefined Strategy** (e.g., "straddle", "iron condor")
3. Configure strike prices and parameters
4. View **P&L Analysis** and **Greeks**

#### **Analyzing Interest Rates**
1. Navigate to **Interest Rate Instruments**
2. Upload market data or use default parameters
3. Estimate **Vasicek model parameters**
4. Price bonds and bond options

## 🧮 Pricing Models

### **Black-Scholes Model**
- **Formula**: `C = S₀N(d₁) - Ke^(-rT)N(d₂)`
- **Best for**: European options, quick analysis
- **Limitations**: Constant volatility assumption

### **Binomial Tree Model**  
- **Approach**: Discrete time lattice
- **Best for**: American options, dividend modeling
- **Flexibility**: Adjustable time steps for accuracy

### **Monte Carlo Simulation**
- **Method**: Stochastic path simulation
- **Best for**: Exotic options, complex payoffs
- **Accuracy**: Statistical convergence with more paths

### **Vasicek Interest Rate Model**
- **Equation**: `dr(t) = a(λ - r(t))dt + σ dW(t)`
- **Applications**: Bond pricing, interest rate derivatives
- **Features**: Mean reversion, analytical solutions

## 🎛 Supported Instruments

### **Vanilla Options**
- European & American calls/puts
- Multiple pricing models comparison
- Complete Greeks analysis
- Parameter sensitivity studies

### **Exotic Options**
- **Asian Options**: Average price/strike
- **Barrier Options**: Knock-in/knock-out
- **Digital Options**: Cash-or-nothing, Asset-or-nothing  
- **Lookback Options**: Floating/fixed strike

### **Forward Contracts**
- Cost-of-carry model pricing
- Mark-to-market valuation
- Interactive P&L analysis
- Multi-parameter sensitivity

### **Options Strategies**
- **Directional**: Bull/Bear spreads
- **Volatility**: Straddles, Strangles
- **Complex**: Butterflies, Iron Condors
- **Custom**: Multi-leg builder

### **Interest Rate Instruments**
- Zero-coupon bond pricing
- Coupon bond valuation
- Bond options (European)
- Yield curve construction

### **Swaps**
- Interest Rate Swaps (IRS)
- Currency Swaps
- Equity Swaps
- DCF and LMM models

## Educational Resources

The platform includes comprehensive educational content:

- **Model Explanations**: Theory behind each pricing model
- **Formula References**: Mathematical foundations
- **Trading Strategies**: Real-world applications
- **Risk Management**: Greeks interpretation and usage
- **Market Insights**: Professional trading considerations


## 🤝 Contributing

We welcome contributions! Here's how you can help:

### **Ways to Contribute**
- **Bug Reports**: Found an issue? Open an issue!
- **Feature Requests**: Suggest new instruments or models
- **Documentation**: Improve explanations and examples
- **New Models**: Implement additional pricing models
- **UI/UX**: Enhance the user interface

### **Development Setup**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

### **Code Standards**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 SALHI Reda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Disclaimer

**IMPORTANT: This platform is designed for educational and research purposes only.**

- **Educational Use**: Learn derivatives pricing and risk management
- **Research Tool**: Academic research and model validation  
- **Not Financial Advice**: Do not use for actual trading decisions
- **Model Risk**: All models are theoretical approximations
- **Validation Required**: Independently verify results before any financial use

**The authors and contributors are not responsible for any financial losses that may result from using this software.**

## Acknowledgments

- **Mathematical Models**: Based on foundational work by Black, Scholes, Merton, and others
- **Streamlit Team**: For the excellent web framework
- **Python Community**: For powerful scientific computing libraries
- **Academic Resources**: Various textbooks and research papers in quantitative finance

## Contact & Support

- **Author**: SALHI Reda
- **Email**: salhi.reda47@gmail.com
- **LinkedIn**: [linkedin.com/in/reda-salhi-195297290](https://www.linkedin.com/in/reda-salhi-195297290/)
- **GitHub**: [github.com/RedaSalhi](https://github.com/RedaSalhi)

### **Getting Help**
- 📖 Check the [Wiki](https://github.com/RedaSalhi/derivatives-pricing/wiki) for detailed guides
- 💬 Open an [Issue](https://github.com/RedaSalhi/derivatives-pricing/issues) for bugs or questions
- 📧 Email for academic collaborations or professional inquiries

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star! ⭐**

Made with ❤️ by [SALHI Reda](https://github.com/RedaSalhi)

</div>





## License

This project is licensed under the [Apache License 2.0](./LICENSE).


