# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:22:56 2026

@author: Radhey
"""

import numpy as np
import pandas as pd
from arch import arch_model

def generate_returns_data(seed=42, periods=1200):
    """Generates synthetic returns exhibiting volatility clustering (GARCH properties)."""
    np.random.seed(seed)
    omega = 0.1
    alpha = 0.15
    beta = 0.80
    
    returns = np.zeros(periods)
    volatility = np.zeros(periods)
    
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for t in range(1, periods):
        volatility[t] = np.sqrt(omega + alpha * (returns[t-1]**2) + beta * (volatility[t-1]**2))
        returns[t] = np.random.normal(0, volatility[t])
        
    dates = pd.date_range(start="2024-01-01", periods=periods, freq='B')
    return pd.Series(returns, index=dates, name="Returns")

def estimate_and_forecast_volatility(returns_series):
    """Fits a GARCH(1,1) model to estimate and forecast conditional volatility."""
    # 1. Scale returns to prevent optimization convergence warnings
    scaled_returns = returns_series * 100
    
    # 2. Fit the standard GARCH(1,1) model
    model = arch_model(scaled_returns, p=1, q=1, vol='Garch', dist='normal')
    model_fit = model.fit(disp='off')
    
    print("--- GARCH(1,1) Parameter Estimates ---")
    print(model_fit.params)
    
    # 3. Extract Conditional Volatility
    conditional_vol = model_fit.conditional_volatility / 100
    
    # 4. Out-of-sample Forecast (Next 5 Days)
    forecast = model_fit.forecast(horizon=5)
    forecast_variance = forecast.variance.iloc[-1]
    forecast_volatility = np.sqrt(forecast_variance) / 100
    
    print("\n--- 5-Day Out-of-Sample Volatility Forecast ---")
    for i, vol in enumerate(forecast_volatility, 1):
        print(f"Day T+{i}: {vol:.4f}")
        
    return conditional_vol, forecast_volatility

if __name__ == "__main__":
    returns = generate_returns_data()
    cond_vol, market_forecast = estimate_and_forecast_volatility(returns)
