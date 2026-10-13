# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:10:28 2026

@author: Radhey
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def generate_hft_order_book_data(seed=42, ticks=5000):
    """Simulates high-frequency Limit Order Book (LOB) states with microsecond updates.
    Captures Bids, Asks, Volumes, and Order Flow Imbalance (OFI) dynamics.
    """
    np.random.seed(seed)
    
    # Baseline initialization
    mid_price = 100.0
    spread = 0.02
    
    prices_bid = np.zeros(ticks)
    prices_ask = np.zeros(ticks)
    vols_bid = np.zeros(ticks)
    vols_ask = np.zeros(ticks)
    
    current_bid = mid_price - (spread / 2)
    current_ask = mid_price + (spread / 2)
    
    # Using a tuple to prevent text-stripping issues
    spread_choices = (0.00, 0.01, 0.02, 0.03)
    
    for i in range(ticks):
        # Simulate microstructure price jitter and stochastic volume fluctuations
        current_bid += np.random.normal(0, 0.005)
        current_ask = current_bid + spread + float(np.random.choice(spread_choices))
        
        prices_bid[i] = round(current_bid, 2)
        prices_ask[i] = round(current_ask, 2)
        vols_bid[i] = np.random.negative_binomial(20, 0.05) * 100
        vols_ask[i] = np.random.negative_binomial(20, 0.05) * 100
        
    timestamps = pd.date_range(start="2026-10-13 08:00:00", periods=ticks, freq='200ms')
    
    return pd.DataFrame({
        'Bid_Price': prices_bid, 'Bid_Vol': vols_bid,
        'Ask_Price': prices_ask, 'Ask_Vol': vols_ask
    }, index=timestamps)

def engineer_microstructure_features(df):
    """Calculates Order Flow Imbalance (OFI) and Micro-Price targets."""
    # 1. Calculate Micro-Price (Volume-weighted mid-price)
    df['Micro_Price'] = (df['Bid_Price'] * df['Ask_Vol'] + df['Ask_Price'] * df['Bid_Vol']) / (df['Bid_Vol'] + df['Ask_Vol'])
    
    # Target: Predict future 5-tick ahead micro-price return
    df['Target_Return'] = df['Micro_Price'].shift(-5) - df['Micro_Price']
    
    # 2. Compute Level 1 Order Flow Imbalance (OFI)
    dBid_Vol = np.where(df['Bid_Price'] > df['Bid_Price'].shift(1), df['Bid_Vol'],
               np.where(df['Bid_Price'] < df['Bid_Price'].shift(1), 0, 
                        df['Bid_Vol'] - df['Bid_Vol'].shift(1)))
    
    dAsk_Vol = np.where(df['Ask_Price'] < df['Ask_Price'].shift(1), df['Ask_Vol'],
               np.where(df['Ask_Price'] > df['Ask_Price'].shift(1), 0, 
                        df['Ask_Vol'] - df['Ask_Vol'].shift(1)))
    
    df['OFI'] = dBid_Vol - dAsk_Vol
    
    # 3. Multi-lag features using a tuple to avoid formatting bugs
    target_lags = (1, 2, 5)
    for lag in target_lags:
        df[f'OFI_Lag_{lag}'] = df['OFI'].shift(lag)
        df[f'Spread_Lag_{lag}'] = (df['Ask_Price'] - df['Bid_Price']).shift(lag)
        
    df.dropna(inplace=True)
    return df

def train_predictive_engine(df):
    """Trains a Non-linear Gradient Boosted Regressor to predict high-frequency market impact."""
    feature_cols = list(c for c in df.columns if 'Lag' in c or c == 'OFI')
    X = df[feature_cols].values
    y = df['Target_Return'].values
    
    # Temporal train/test split to avoid data leakage
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[0:split_idx], X[split_idx:]
    y_train, y_test = y[0:split_idx], y[split_idx:]
    
    # Model high-frequency interactions via Gradient Boosting
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Evaluation Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("=== HFT Microstructure Research Engine ===")
    print(f"Total High-Frequency LOB States Modeled: {len(df)}")
    print(f"Out-of-Sample Prediction MSE: {mse:.7f}")
    print(f"Out-of-Sample Information Coefficient (R² Score): {r2*100:.4f}%")
    
    # Showcase Feature Importances to prove alpha contribution
    importances = model.feature_importances_
    print("\n--- Alpha Factor Contribution (Microstructure Impact) ---")
    for name, imp in zip(feature_cols, importances):
        print(f"{name:<15}: {imp*100:.2f}%")
        
    return model

if __name__ == "__main__":
    lob_data = generate_hft_order_book_data()
    processed_features = engineer_microstructure_features(lob_data)
    alpha_engine = train_predictive_engine(processed_features)
