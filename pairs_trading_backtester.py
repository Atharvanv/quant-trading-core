# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:21:06 2026

@author: Radhey
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

def generate_mock_data(seed=42, periods=1000):
    """Generates synthetic cointegrated asset prices to simulate a real setup."""
    np.random.seed(seed)
    x = np.cumprod(1 + np.random.normal(0, 0.01, periods)) * 100
    # Create y cointegrated with x by adding a stationary mean-reverting spread
    spread = np.zeros(periods)
    for t in range(1, periods):
        spread[t] = 0.8 * spread[t-1] + np.random.normal(0, 0.5)
    y = 1.2 * x + spread + 20
    
    dates = pd.date_range(start="2024-01-01", periods=periods, freq='B')
    return pd.DataFrame({'Asset_A': x, 'Asset_B': y}, index=dates)

def backtest_pairs_trading(df, entry_z=2.0, exit_z=0.0):
    """Runs a fully vectorized pairs trading backtest using OLS spread residuals."""
    # 1. Calculate the spread using Ordinary Least Squares (OLS)
    X = sm.add_constant(df['Asset_A'])
    model = sm.OLS(df['Asset_B'], X).fit()
    beta = model.params['Asset_A']
    alpha = model.params['const']
    
    df['Spread'] = df['Asset_B'] - (beta * df['Asset_A'] + alpha)
    
    # 2. Calculate rolling Z-Score of the spread
    rolling_window = 30
    mean = df['Spread'].rolling(window=rolling_window).mean()
    std = df['Spread'].rolling(window=rolling_window).std()
    df['Z_Score'] = (df['Spread'] - mean) / std
    df.dropna(inplace=True)
    
    # 3. Vectorized Signal Generation
    df['Position_A'] = 0.0
    df['Position_B'] = 0.0
    
    # Long the spread (Buy B, Sell A) when Z-Score is significantly low
    df.loc[df['Z_Score'] < -entry_z, 'Position_B'] = 1.0
    df.loc[df['Z_Score'] < -entry_z, 'Position_A'] = -beta
    
    # Short the spread (Sell B, Buy A) when Z-Score is significantly high
    df.loc[df['Z_Score'] > entry_z, 'Position_B'] = -1.0
    df.loc[df['Z_Score'] > entry_z, 'Position_A'] = beta
    
    # Clear positions when spread returns to the mean
    df.loc[df['Z_Score'].abs() <= exit_z, ['Position_A', 'Position_B']] = 0.0
    
    # Forward-fill positions to hold them over time
    df[['Position_A', 'Position_B']] = df[['Position_A', 'Position_B']].ffill().fillna(0)
    
    # 4. Calculate Returns (Shift positions by 1 day to avoid look-ahead bias)
    df['Ret_A'] = df['Asset_A'].pct_change()
    df['Ret_B'] = df['Asset_B'].pct_change()
    
    df['Strategy_Return'] = (df['Position_A'].shift(1) * df['Ret_A']) + (df['Position_B'].shift(1) * df['Ret_B'])
    
    # 5. Performance Metrics
    sharpe = np.sqrt(252) * (df['Strategy_Return'].mean() / df['Strategy_Return'].std())
    cum_returns = (1 + df['Strategy_Return']).cumprod() - 1
    
    print(f"--- Pairs Trading Strategy Metrics ---")
    print(f"Hedge Ratio (Beta): {beta:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    print(f"Total Cumulative Return: {cum_returns.iloc[-1]*100:.2f}%")
    
    return df

if __name__ == "__main__":
    data = generate_mock_data()
    results = backtest_pairs_trading(data)
