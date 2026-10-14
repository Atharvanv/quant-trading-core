# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:44:04 2026

@author: Radhey
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def run_black_litterman_optimization():
    """Runs a complete Black-Litterman optimization loop, combining market implied

    equilibrium returns with unique investor views under budget constraints.
    """
    # 1. Setup Parameters (4 Asset Universe: Equity US, Equity EU, Bonds, Commodities)
    asset_names = ('US_Eq', 'EU_Eq', 'Global_Bonds', 'Commodities')
    num_assets = len(asset_names)
    
    # Market Capitalization Weights (Baseline Equilibrium)
    market_weights = np.array([0.45, 0.30, 0.15, 0.10])
    
    # Covariance Matrix (Historical Risk Profile)
    covariance = np.array([
        [0.040, 0.025, -0.002, 0.010],
        [0.025, 0.045, -0.003, 0.012],
        [-0.002, -0.003, 0.005, 0.001],
        [0.010, 0.012, 0.001, 0.030]
    ])
    
    # Risk Aversion Coefficient (lambda)
    risk_aversion = 3.0
    # Tau (scalar tuning parameter for uncertainty of prior)
    tau = 0.05
    
    print("=== Black-Litterman Portfolio Optimization Engine ===")
    
    # 2. Reverse Optimization: Calculate Implied Equilibrium Excess Returns (Pi)
    # Formula: Pi = lambda * Sigma * w_mkt
    pi_returns = risk_aversion * np.dot(covariance, market_weights)
    
    print("\n--- Market Implied Equilibrium Returns (Prior) ---")
    for name, ret in zip(asset_names, pi_returns):
        print(f"{name:<13}: {ret*100:.2f}%")
        
    # 3. Inject Subjective Views
    # View 1: US Equities will outperform Global Bonds by 3%
    # View 2: Commodities will absolute outperform by 5%
    num_views = 2
    P_matrix = np.array([
        [1.0, 0.0, -1.0, 0.0],  # Long US_Eq, Short Global_Bonds
        [0.0, 0.0, 0.0, 1.0]   # Long Commodities
    ])
    Q_vector = np.array([0.03, 0.05])
    
    # Omega Matrix: Uncertainty matrix of the views (Diagonal matrix using Tau scaling)
    omega = np.diag(np.diag(tau * np.dot(np.dot(P_matrix, covariance), P_matrix.T)))
    
    # 4. Calculate Combined Expected Returns (Black-Litterman Formula)
    inv_scale_cov = np.linalg.inv(tau * covariance)
    inv_omega = np.linalg.inv(omega)
    
    middle_term = np.linalg.inv(inv_scale_cov + np.dot(np.dot(P_matrix.T, inv_omega), P_matrix))
    right_term = np.dot(inv_scale_cov, pi_returns) + np.dot(np.dot(P_matrix.T, inv_omega), Q_vector)
    
    bl_combined_returns = np.dot(middle_term, right_term)
    
    print("\n--- Black-Litterman Adjusted Returns (Posterior) ---")
    for name, ret in zip(asset_names, bl_combined_returns):
        print(f"{name:<13}: {ret*100:.2f}%")
        
    # 5. Non-linear Portfolio Constrained Optimization (Maximize Utility)
    # Objective: Minimize negative quadratic utility function
    def utility_objective(weights):
        portfolio_return = np.dot(weights, bl_combined_returns)
        portfolio_variance = np.dot(np.dot(weights, covariance), weights)
        utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
        return -utility

    # Constraints: Fully invested (sum of weights = 1), Bounds: Long-only (0 to 1)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = np.array([0.25, 0.25, 0.25, 0.25])
    
    optimization_result = minimize(utility_objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_weights = optimization_result.x
    
    print("\n--- Allocation Optimization Strategy Output ---")
    for name, mkt_w, opt_w in zip(asset_names, market_weights, optimal_weights):
        print(f"{name:<13} | Benchmark Weight: {mkt_w*100:.1f}% | Optimized Target Weight: {opt_w*100:.1f}%")
        
    return optimal_weights

if __name__ == "__main__":
    run_black_litterman_optimization()
