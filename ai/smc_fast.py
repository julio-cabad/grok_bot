#!/usr/bin/env python3
"""
Fast SMC Analyzer - Optimized with Numba + Polars
Ultra-fast Smart Money Concepts analysis for institutional trading
"""

import polars as pl
import numba as nb
import numpy as np
import logging
from typing import Dict, Any, Optional


@nb.njit
def _ob_loop(close, open_, high, low):
    """Optimized order block detection with Numba JIT compilation"""
    obs = []
    for i in range(3, len(close)-3):
        if close[i] < open_[i]:                 # bear candle
            if close[i+1] > open_[i+1]:         # next bull
                if high[i+1:i+4].max() > high[i]*1.002:
                    obs.append((float(low[i]), "BULL"))
        else:                                   # bull candle
            if close[i+1] < open_[i+1]:         # next bear
                if low[i+1:i+4].min() < low[i]*0.998:
                    obs.append((float(high[i]), "BEAR"))
    return obs


def extract_smc(df: pl.DataFrame) -> dict:
    """
    Extract Smart Money Concepts from price data
    
    Args:
        df: Polars DataFrame with OHLCV data and timestamp
        
    Returns:
        Dictionary with all SMC components
    """
    try:
        # Sort by timestamp to ensure proper order
        df = df.sort("timestamp")
        
        # Convert to numpy arrays for Numba optimization
        close = df["close"].to_numpy()
        open_ = df["open"].to_numpy() 
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        
        # Get order blocks using optimized function
        obs = _ob_loop(close, open_, high, low)
        
        # Calculate other SMC components
        fvg = _fvg(close, high, low)
        bos = _bos(close, high)
        pd_z = _pd_zone(close)
        atr = float((close[-20:].std() * np.sqrt(252) * close[-1]))
        
        return {
            "symbol": df["symbol"][0] if "symbol" in df.columns else "UNKNOWN",
            "price": float(close[-1]),
            "ob_bull": min([o[0] for o in obs if o[1] == "BULL"], default=None),
            "ob_bear": max([o[0] for o in obs if o[1] == "BEAR"], default=None),
            "fvg_bull": fvg.get("BULL"),
            "fvg_bear": fvg.get("BEAR"),
            "bos_level": bos,
            "liq_s": float(high[-50:].max()),
            "liq_b": float(low[-50:].min()),
            "pd_zone": pd_z["zone"],
            "pd_opt": pd_z["opt"],
            "atr": atr
        }
        
    except Exception as e:
        logging.error(f"SMC extraction failed: {e}")
        return {
            "symbol": "ERROR",
            "price": 0.0,
            "ob_bull": None,
            "ob_bear": None,
            "fvg_bull": None,
            "fvg_bear": None,
            "bos_level": 0.0,
            "liq_s": 0.0,
            "liq_b": 0.0,
            "pd_zone": "EQUILIBRIUM",
            "pd_opt": "WAIT",
            "atr": 0.0
        }


def _fvg(close, high, low):
    """Identify Fair Value Gaps"""
    fvgs = {}
    for i in range(2, len(close)-2):
        # Bullish FVG: gap between previous high and current low
        gap_up = low[i] - high[i-2]
        if gap_up > 0:
            fvgs["BULL"] = (float(high[i-2]), float(low[i]))
            
        # Bearish FVG: gap between previous low and current high  
        gap_dn = low[i-2] - high[i]
        if gap_dn > 0:
            fvgs["BEAR"] = (float(high[i]), float(low[i-2]))
    
    return fvgs


def _bos(close, high):
    """Break of Structure - simple swing high detection"""
    swing_highs = []
    for i in range(5, len(high)-5):
        if high[i] == high[i-5:i+5].max():
            swing_highs.append(high[i])
    
    return float(swing_highs[-1]) if swing_highs else float(close[-1])


def _pd_zone(close):
    """Premium/Discount Zone calculation"""
    p50 = close[-50:]
    r = p50.max() - p50.min()
    mid = p50.min() + r*0.5
    prem = p50.min() + r*0.705  # 70.5% premium threshold
    disc = p50.min() + r*0.295  # 29.5% discount threshold
    
    price = close[-1]
    
    if price > prem:
        return {"zone": "PREMIUM", "opt": "SELL"}
    if price < disc:
        return {"zone": "DISCOUNT", "opt": "BUY"}
    return {"zone": "EQUILIBRIUM", "opt": "WAIT"}


def pandas_to_polars(df_pandas) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars for optimization"""
    try:
        # Convert pandas to polars
        df_polars = pl.from_pandas(df_pandas)
        
        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df_polars.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add symbol if not present
        if "symbol" not in df_polars.columns:
            df_polars = df_polars.with_columns(pl.lit("UNKNOWN").alias("symbol"))
            
        # Add timestamp if not present
        if "timestamp" not in df_polars.columns:
            df_polars = df_polars.with_row_count("timestamp")
            
        return df_polars
        
    except Exception as e:
        logging.error(f"Pandas to Polars conversion failed: {e}")
        raise


def analyze_smc_fast(df_pandas, symbol: str = "UNKNOWN") -> Dict[str, Any]:
    """
    Fast SMC analysis entry point for pandas DataFrames
    
    Args:
        df_pandas: Pandas DataFrame with OHLCV data
        symbol: Trading symbol
        
    Returns:
        SMC analysis results
    """
    try:
        # Convert to Polars for optimization
        df_polars = pandas_to_polars(df_pandas)
        
        # Add symbol if provided
        if symbol != "UNKNOWN":
            df_polars = df_polars.with_columns(pl.lit(symbol).alias("symbol"))
        
        # Extract SMC components
        smc_result = extract_smc(df_polars)
        
        # Add analysis metadata
        smc_result["analysis_type"] = "FAST_SMC"
        smc_result["candles_analyzed"] = len(df_pandas)
        smc_result["data_quality"] = "GOOD" if len(df_pandas) > 100 else "LIMITED"
        
        return smc_result
        
    except Exception as e:
        logging.error(f"Fast SMC analysis failed: {e}")
        return {
            "symbol": symbol,
            "price": 0.0,
            "ob_bull": None,
            "ob_bear": None,
            "fvg_bull": None,
            "fvg_bear": None,
            "bos_level": 0.0,
            "liq_s": 0.0,
            "liq_b": 0.0,
            "pd_zone": "EQUILIBRIUM",
            "pd_opt": "WAIT",
            "atr": 0.0,
            "analysis_type": "FAST_SMC",
            "candles_analyzed": 0,
            "data_quality": "ERROR"
        }


# Example usage and testing
if __name__ == "__main__":
    import pandas as pd
    
    # Create sample data for testing
    np.random.seed(42)
    size = 500
    
    # Generate realistic price data
    close_prices = 50000 + np.cumsum(np.random.randn(size) * 100)
    
    df_test = pd.DataFrame({
        'open': close_prices + np.random.randn(size) * 50,
        'high': close_prices + np.abs(np.random.randn(size) * 100),
        'low': close_prices - np.abs(np.random.randn(size) * 100),
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, size)
    })
    
    print("🚀 Testing Fast SMC Analyzer")
    print("-" * 50)
    
    # Test the fast analyzer
    import time
    start_time = time.time()
    
    result = analyze_smc_fast(df_test, "BTCUSDT")
    
    analysis_time = time.time() - start_time
    
    print(f"✅ Analysis completed in {analysis_time:.4f} seconds")
    print(f"📊 Results:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n🏆 Performance: {len(df_test)/analysis_time:.0f} candles/second")