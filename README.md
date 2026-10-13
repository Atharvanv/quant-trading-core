# Quantitative Strategy & Market Microstructure Engine

Production-grade pipelines focused on statistical arbitrage backtesting, time-series risk forecasting, and high-frequency order book microstructure modeling.

## Core Projects

### 1. High-Frequency Market Microstructure & Order Flow Imbalance Engine
* **File:** `hft_microstructure_engine.py`
* **Problem Solved:** Non-linear adverse selection and short-term price discovery estimation inside fragmented Limit Order Books (LOB).
* **Methodology:** Implements multi-lag Level 1 Order Flow Imbalance (OFI) trackers and volume-weighted micro-price calculations. Leverages non-linear tree-based gradient boosting ensembles (`scikit-learn`) to predict short-term alpha decay and order-book impact under variance shifts.

### 2. Custom Parametric GARCH Volatility Engine
* **File:** `volatility_modeling.py`
* **Problem Solved:** Time-varying variance dynamics and heteroskedastic risk clusters in financial asset returns.
* **Methodology:** Engineered entirely from scratch using `SciPy` Sequential Least Squares Programming (`SLSQP`) optimization frameworks. Avoids abstract third-party libraries to explicitly construct negative log-likelihood minimization routines for multi-day out-of-sample forward volatility projections.

### 3. Vectorized Pairs-Trading Arbitrage Model
* **File:** `pairs_trading_backtester.py`
* **Problem Solved:** Mean-reversion statistical arbitrage execution under strict transaction costs.
* **Methodology:** Utilizes Ordinary Least Squares (OLS) residual cointegration tracking to capture long-term asset equilibria. Implements dynamic rolling Z-scores and a fully vectorized signal generation engine to eliminate forward data contamination (look-ahead bias).

## Tech Stack
* **Languages:** Python
* **Libraries:** Pandas, NumPy, SciPy, Scikit-Learn, Statsmodels
