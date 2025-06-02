# Derivatives-Pricing

| **Derivative Type**        | **Sub-type / Feature**           | **Common Pricing Models**                                                             |
| -------------------------- | -------------------------------- | ------------------------------------------------------------------------------------- |
| **Forward Contract**       | Plain Vanilla                    | Cost of Carry Model (Spot ± Carrying Costs)                                           |
| **Futures Contract**       | Plain Vanilla                    | Cost of Carry + No-Arbitrage Principles                                               |
| **European Option**        | Call / Put                       | Black-Scholes-Merton (BSM)                                                            |
|                            | Basket, Index Options            | Multivariate BSM, Monte Carlo Simulation                                              |
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
