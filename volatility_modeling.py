# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:22:56 2026

@author: Radhey
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def generate_returns_data(seed=42, periods=1200):
    """Generates synthetic returns exhibiting volatility clustering (GARCH properties)."""
    np.random.seed(seed)
    omega, alpha, beta = 0.1, 0.15, 0.80
    
    returns = np.zeros(periods)
    volatility = np.zeros(periods)
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    
    for t in range(1, periods):
        volatility[t] = np.sqrt(omega + alpha * (returns[t-1]**2) + beta * (volatility[t-1]**2))
        returns[t] = np.random.normal(0, volatility[t])
        
    dates = pd.date_range(start="2024-01-01", periods=periods, freq='B')
    return pd.Series(returns, index=dates, name="Returns")

def garch_likelihood(parameters, returns):
    """Calculates the negative log-likelihood for a GARCH(1,1) model."""
    omega, alpha, beta = parameters
    periods = len(returns)
    variance = np.zeros(periods)
    variance[0] = np.var(returns)
    
    for t in range(1, periods):
        variance[t] = omega + alpha * (returns[t-1]**2) + beta * (variance[t-1])
    
    # Avoid log(0) or negative numbers
    variance = np.where(variance <= 0, 1e-6, variance)
    
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi) + np.log(variance) + (returns**2) / variance)
    return -log_likelihood # Minimize negative log-likelihood

def estimate_and_forecast_volatility(returns_series):
    """Fits a GARCH(1,1) model using MLE via SciPy and forecasts volatility."""
    returns = returns_series.values
    
    # Initial guess [omega, alpha, beta] and strict bounds to ensure stationarity
    initial_guess = [0.05, 0.1, 0.8]
    bounds = ((1e-6, None), (1e-6, 1.0), (1e-6, 1.0))
    
    # Constraint: alpha + beta < 1 (Stationarity constraint)
    constraint = {'type': 'ineq', 'fun': lambda params: 0.999 - (params[1] + params[2])}
    
    # Run optimization
    result = minimize(garch_likelihood, initial_guess, args=(returns,), 
                      bounds=bounds, constraints=constraint, method='SLSQP')
    
    omega_est, alpha_est, beta_est = result.x
    
    print("--- Custom GARCH(1,1) Parameter Estimates ---")
    print(f"Omega (ω): {omega_est:.4f}")
    print(f"Alpha (α): {alpha_est:.4f}")
    print(f"Beta  (β): {beta_est:.4f}\n")
    
    # 2. Out-of-sample Forecast (Next 5 Days)
    forecast_vol = np.zeros(5)
    last_return = returns[-1]
    last_variance = np.var(returns) # Fallback baseline
    
    # Compute the final variance state
    for t in range(1, len(returns)):
        last_variance = omega_est + alpha_est * (returns[t-1]**2) + beta_est * last_variance
        
    # Step forward 5 days into the future
    current_variance = omega_est + alpha_est * (last_return**2) + beta_est * last_variance
    forecast_vol[0] = np.sqrt(current_variance)
    
    for i in range(1, 5):
        # Unconditional expectation for future updates: E[r^2] = variance
        current_variance = omega_est + (alpha_est + beta_est) * current_variance
        forecast_vol[i] = np.sqrt(current_variance)
        
    print("--- 5-Day Out-of-Sample Volatility Forecast ---")
    for i, vol in enumerate(forecast_vol, 1):
        print(f"Day T+{i}: {vol:.4f}")
        
    return forecast_vol

if __name__ == "__main__":
    returns = generate_returns_data()
    market_forecast = estimate_and_forecast_volatility(returns)
