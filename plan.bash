derivatives-pricing/
├── main.py
├── pricing/
│   ├── __init__.py
│   ├── vanilla_option.py            ← Entry point for vanilla options
│   └── models/
│       ├── __init__.py
│       ├── black_scholes.py         ← BSM model here
│       ├── binomial_tree.py         ← Binomial Tree here
│       └── (more models later)      ← Trinomial, Monte Carlo, etc.
├── utils/
│   ├── greeks.py
│   ├── implied_vol.py
├── requirements.txt
└── README.md
