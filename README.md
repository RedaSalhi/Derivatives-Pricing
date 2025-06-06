# Derivatives-Pricing



To open the app click here

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://derivatives-pricing.streamlit.app/#about-me)


| **Derivative Type**        | **Sub-type / Feature**           | **Common Pricing Models**                                                             |
| -------------------------- | -------------------------------- | ------------------------------------------------------------------------------------- |
| **Forward Contract**       | Plain Vanilla                    | Cost of Carry Model (Spot ± Carrying Costs)                                           |
| **Futures Contract**       | Plain Vanilla                    | Cost of Carry + No-Arbitrage Principles                                               |
| **European Option**        | Call / Put                       | Black-Scholes-Merton (BSM)   |                                  
| **American Option**        | Call / Put                       | Binomial Tree, Trinomial Tree, Finite Difference, Longstaff-Schwartz (Monte Carlo)    |
| **Exotic Options**         | Barrier (e.g., Knock-in/out)     | Analytical (if solvable), Monte Carlo, Finite Difference                              |
|                            | Asian (Average Price/Strike)     | Monte Carlo Simulation, PDE methods                                                   |
|                            | Lookback Options                 | Monte Carlo Simulation, PDE Methods                                                   |
|                            | Digital (Binary) Options         | Black-Scholes (with discontinuous payoff adjustment)                                  |
|                            | Compound, Chooser Options        | Specialized closed-form solutions or Monte Carlo                                      |
| **Swaps**                  | Interest Rate Swap (IRS)         | Discounted Cash Flow (DCF), LIBOR Market Model (LMM), OIS Curve for Discounting       |
|                            | Currency Swap                    | DCF in each currency + FX Forward Curve                                               |
|                            | Equity Swap                      | Replication using Forward Contracts, DCF                                              |
| **Caps / Floors**          | Interest Rate Derivatives        | Black's Model (for caplets/floorlets), Hull-White Model                               |
| **Swaptions**              | Option on Interest Rate Swap     | Black's Model (lognormal rates), Hull-White, LMM                                      |
| **Credit Derivatives**     | Credit Default Swap (CDS), CDO   | Reduced-form models (e.g., Jarrow-Turnbull), Structural Models (e.g., Merton), Copula |
| **Convertible Bonds**      | Bond + Embedded Option           | Lattice Models (Binomial Tree), Monte Carlo, PDE                                      |
| **Volatility Derivatives** | Variance Swaps, Volatility Swaps | Replication by static portfolio of options, Monte Carlo                               |
| **Real Options**           | Corporate Finance Applications   | Binomial Trees, Monte Carlo Simulation                                                |




Most Important Quant Models ♥️♥️

📊📊Black-Scholes Family (Lognormal Models)
 
1. Black-Scholes-Merton Model – For European options on non-dividend-paying stocks
 
2. Black Model (Black-76) – For options on futures, interest rates, and caps/floors
 
3. Garman-Kohlhagen Model – For pricing FX options (extension of Black-Scholes for currency pairs)

📊📊 Binomial & Tree-Based Models
 
1. Cox-Ross-Rubinstein (CRR) Binomial Tree – Used for both American and European options
 
2. Trinomial Tree Model – Provides greater accuracy with more branches; suitable for exotic options
 
3. Jarrow-Rudd Model – Similar to CRR, assumes lognormal returns but symmetric binomial tree

📊📊 Monte Carlo Simulation Models
 
1. Basic Monte Carlo Simulation – Simulates multiple price paths for path-dependent options
 
2. Least Squares Monte Carlo (LSM) – Useful for pricing American-style options (e.g., Longstaff-Schwartz)
 
3. Quasi Monte Carlo – Uses low-discrepancy sequences to improve simulation accuracy


📊📊 Stochastic Volatility Models
 
1. Heston Model – Assumes stochastic variance; captures volatility smiles
 
2. SABR Model – Widely used for interest rate derivatives and implied volatility modeling
 
3. CEV Model – Assume volatility depends on level of stock price

📊📊 Interest Rate Models
 
1. Vasicek Model – Mean-reverting normal interest rate model
 
2. Cox-Ingersoll-Ross (CIR) Model – Keeps rates positive; used in fixed income modeling
 
3. Hull-White Model – Extends Vasicek with time-dependent parameters
 
4. Black-Karasinski Model – Log-normal short rate process
 
5. LIBOR Market Model (BGM) – Models forward LIBOR rates; standard in swaption pricing
 
6. Ho-Lee Model – Arbitrage-free, simple time-dependent model
 
7. Kalotay-Williams-Fabozzi Model – Popular for callable bond pricing

📊📊 Jump-Diffusion & Advanced Models
 
1. Merton Jump-Diffusion Model – Adds jumps to Black-Scholes for sudden market movements
 
2. Variance Gamma Model – Accounts for fat tails and skewness
 
3. Normal Inverse Gaussian (NIG) Model – Suitable for modeling heavy-tailed distributions
 
4. CGMY Model – Generalizes Lévy processes with better fit to asset returns

📊📊 Credit Derivative Models
 
1. Reduced Form (Intensity-Based) Models – Use Poisson processes to model credit events
 
2. Structural Models (e.g., Merton Model) – Based on firm value and default barrier
 
3. Copula Models – Capture correlated defaults; widely used in CDO pricing


📊📊 PDE-Based Models
 
1. Finite Difference Methods (FDM) – Solve Black-Scholes PDEs numerically
 
2. Crank-Nicolson Scheme – Stable numerical scheme for pricing European/American options
