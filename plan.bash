Derivatives-Pricing/
├── main.py                       ← Streamlit application entry
├── pricing/
│   ├── __init__.py
│   ├── vanilla_options.py        ← Vanilla option pricing interface
│   ├── forward.py                ← Forward contract pricing & plots
│   ├── futures.py                ← Futures pricing placeholder
│   ├── option_strategies.py      ← Multi-leg option strategies
│   ├── asian_option.py           ← Asian option helpers
│   ├── barrier_option.py         ← Barrier option helpers
│   ├── digital_option.py         ← Digital option helpers
│   ├── lookback_option.py        ← Lookback option helpers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── black_scholes.py      ← Black–Scholes analytical model
│   │   ├── binomial_tree.py      ← Binomial tree model
│   │   ├── monte_carlo.py        ← Basic Monte Carlo engine
│   │   ├── longstaff_schwartz.py ← LSM Monte Carlo (American)
│   │   ├── digital_black_scholes.py ← Digital option with B–S
│   │   ├── barrier_monte_carlo.py ← Barrier option Monte Carlo
│   │   ├── asian_monte_carlo.py   ← Asian option Monte Carlo
│   │   ├── lookback_binomial.py   ← Lookback option binomial tree
│   │   └── lookback_monte_carlo.py ← Lookback option Monte Carlo
│   └── utils/
│       ├── option_strategies_greeks.py ← Strategy greek calculators
│       └── greeks_vanilla/
│           ├── black_scholes_greeks.py
│           ├── binomial_greeks.py
│           ├── monte_carlo_greeks.py
│           ├── plot_single_greek.py
│           └── greeks_interface.py
├── requirements.txt              ← Python dependencies
├── plan.bash                     ← Repository structure overview
└── README.md                     ← Documentation and usage
