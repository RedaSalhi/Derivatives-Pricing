derivatives-pricing/
├── main.py
├── pricing/
│   ├── __init__.py
│   ├── vanilla_option.py         ← Entry point for vanilla option pricing
│   ├── european-options.py       ← Contains payoff and P&L plotting functions
│   └── models/
│       ├── __init__.py
│       ├── black_scholes.py      ← Black-Scholes model (European)
│       ├── binomial_tree.py      ← Binomial Tree model (European & American)
│       ├── monte_carlo.py        ← Standard Monte Carlo (European)
│       ├── longstaff_schwartz.py ← LSM Monte Carlo model (American)
│       └── (more models later)   ← Trinomial Tree, Heston, etc.
├── utils/
│   ├── greeks.py                 ← Greeks calculators (Δ, Γ, Θ, etc.)
│   ├── implied_vol.py            ← Implied volatility solver
│   └── (more utils later)
├── plan.css                      ← App styling (Streamlit layout)
├── requirements.txt              ← Dependencies list
└── README.md                     ← Documentation and usage
