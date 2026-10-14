# Quantitative Strategy & Exchange Microstructure Core Engine

Production-grade quantitative pipelines focused on high-frequency market microstructure modeling, parametric volatility forecasting, algorithmic asset allocation under strict constraints, and statistical arbitrage execution.

## Core Multi-Asset Projects

### 1. High-Frequency Market Microstructure & Order Flow Imbalance Engine
* **File:** `hft_microstructure_engine.py`
* **Problem Solved:** Non-linear adverse selection, immediate market impact, and short-term price discovery estimation inside fragmented Limit Order Books (LOB).
* **Methodology:** Implements multi-lag Level 1 Order Flow Imbalance (OFI) feature extraction matrices alongside volume-weighted micro-price targets (neutralizing bid-ask bounce). Leverages non-linear tree-based Gradient Boosting ensembles (`scikit-learn`) to map non-linear alpha decay, order book queue depletion, and localized liquidity shocks.

### 2. High-Frequency Market-Making Simulator (Avellaneda-Stoikov Framework)
* **File:** `market_making_simulator.py`
* **Problem Solved:** Microsecond inventory optimization, asymmetric terminal risk exposure, and optimal quoting configurations for market-making desks.
* **Methodology:** Simulates reference mid-prices via a stochastic jump-diffusion process. Computes instantaneous reservation prices (r) and optimal target bid-ask spreads dynamically using terminal inventory penalty thresholds (γ) and Poisson arrival execution intensities (k) to control risk under time-decay properties.

### 3. Constrained Asset Allocation Optimization Engine (Black-Litterman Setup)
* **File:** `black_litterman_optimizer.py`
* **Problem Solved:** Out-of-sample portfolio concentration vulnerability, high parameters sensitivity, and benchmark-relative tracking error under full-investment constraints.
* **Methodology:** Conducts a reverse optimization routine to extract implied market equilibrium prior returns. Blends these priors with subjective market view vectors and explicit confidence metrics (Omega matrix) using Bayesian updating. Solves a non-linear quadratic utility maximization function using `SciPy` Sequential Least Squares Programming (`SLSQP`) to impose boundary constraints and minimize transaction turnovers.

### 4. Custom Parametric GARCH(1,1) Volatility Pipeline
* **File:** `volatility_modeling.py`
* **Problem Solved:** Continuous time-varying variance dynamics, structural heteroskedasticity, and volatility clustering in financial asset returns.
* **Methodology:** Engineered entirely from scratch via maximum likelihood estimation (MLE) using `SciPy` optimization frameworks to preserve raw code integrity and performance transparency. Explicitly maps and minimizes negative log-likelihood state-spaces to construct robust multi-day out-of-sample forward conditional volatility paths.

### 5. Vectorized Cointegrated Pairs-Trading Arbitrage Model
* **File:** `pairs_trading_backtester.py`
* **Problem Solved:** Mean-reversion statistical arbitrage capture, risk-neutral spread modeling, and performance drag minimization under strict execution parameters.
* **Methodology:** Employs Ordinary Least Squares (OLS) residual cointegration tracking to isolate continuous long-term equilibrium relationships between asset pairs. Implements dynamic rolling Z-score parameters and pairs the entire system into a completely vectorized matrix execution layer to eliminate look-ahead data contamination and forward bias.

## Production Tech Stack
* **Languages:** Python 3.10+
* **Data Processing \& Vectorization:** Pandas, NumPy
* **Optimization \& Mathematical Solvers:** SciPy (SLSQP Optimization Engines)
* **Statistical \& Machine Learning Models:** Scikit-Learn (Gradient Boosting Regressors), Statsmodels (Ordinary Least Squares Cointegration Frameworks)

## Workspace Architecture & Project Routing
```text
                          QUANT-TRADING-CORE ENGINE
                                      |
         +----------------------------+----------------------------+

         |                                                         |
 ALPHA GENERATION & ARBITRAGE                               MICROSTRUCTURE & OPTIMIZATION

         |                                                         |
   +-----+-----+                                             +-----+-----+

   |           |                                             |           |
Pairs      GARCH Vol                                       OFI HFT   Avellaneda   Black-Litterman
Trading     Engine                                         Engine    Simulator       Optimizer
```
