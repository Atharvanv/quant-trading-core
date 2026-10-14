# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:04:28 2026

@author: Radhey
"""

import numpy as np
import pandas as pd

def run_avellaneda_stoikov_simulation(ticks=1000, seed=42):
    """Simulates high-frequency market-making tracking via the Avellaneda-Stoikov

    stochastic control framework, optimizing spreads against inventory risk.
    """
    np.random.seed(seed)
    
    # 1. Structural Paradigm Assumptions
    s0 = 100.0          # Initial reference mid-price
    sigma = 0.3         # Asset volatility parameters
    gamma = 0.1         # Risk aversion inventory metric penalty
    k = 1.5             # Order book liquidity intensity parameter
    dt = 1.0 / ticks    # Step sizing
    
    # Tracking Arrays
    mid_prices = np.zeros(ticks)
    reservation_prices = np.zeros(ticks)
    spread_bids = np.zeros(ticks)
    spread_asks = np.zeros(ticks)
    inventory = np.zeros(ticks)
    pnl = np.zeros(ticks)
    
    current_mid = s0
    current_inventory = 0
    current_cash = 0.0
    
    print("=== Avellaneda-Stoikov Market-Making Simulator ===")
    
    # 2. Iterate Through High-Frequency Discrete Match Windows
    for t in range(ticks):
        time_left = 1.0 - (t * dt)
        
        # Mid-price random walk diffusion
        current_mid += sigma * np.sqrt(dt) * np.random.normal(0, 1)
        mid_prices[t] = current_mid
        
        # Calculate Optimal Reservation Price (r)
        # Formula: r = s - q * gamma * sigma^2 * (T - t)
        res_price = current_mid - (current_inventory * gamma * (sigma ** 2) * time_left)
        reservation_prices[t] = res_price
        
        # Optimal Spread Calculation
        # Formula: delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
        total_spread = (gamma * (sigma ** 2) * time_left) + (2 / gamma) * np.log(1 + (gamma / k))
        
        # Asymmetrical quoting around the reservation price based on current position inventory
        half_spread = total_spread / 2.0
        ask_quote = res_price + half_spread
        bid_quote = res_price - half_spread
        
        spread_asks[t] = ask_quote - current_mid
        spread_bids[t] = current_mid - bid_quote
        
        # 3. Simulate Order Execution matching via Poisson Jump Probabilities
        lambda_ask = c = 5.0 * np.exp(-k * spread_asks[t])
        lambda_bid = c = 5.0 * np.exp(-k * spread_bids[t])
        
        # Determine hits
        ask_hit = np.random.uniform(0, 1) < (lambda_ask * dt)
        bid_hit = np.random.uniform(0, 1) < (lambda_bid * dt)
        
        # Update state criteria metrics
        if bid_hit:
            current_inventory += 1
            current_cash -= bid_quote
        if ask_hit:
            current_inventory -= 1
            current_cash += ask_quote
            
        inventory[t] = current_inventory
        pnl[t] = current_cash + (current_inventory * current_mid)
        
    print(f"Total Operational Ticks Simulated : {ticks}")
    print(f"Final Inventory Exposure Delta     : {inventory[-1]}")
    print(f"Generated Total Micro-Structure PnL: ${pnl[-1]:.2f}")
    print(f"Maximum Asset Volatility Drawdown  : ${np.min(pnl):.2f}")
    
    return pd.DataFrame({'Mid': mid_prices, 'Inventory': inventory, 'PnL': pnl})

if __name__ == "__main__":
    simulation_results = run_avellaneda_stoikov_simulation()
