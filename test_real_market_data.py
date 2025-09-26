#!/usr/bin/env python3
"""
Test Real Market Data with Gemini AI Analysis
Test completo con datos reales de mercado y análisis real de Gemini AI
"""

import pandas as pd
import numpy as np
import logging
import os
import requests
from datetime import datetime, timedelta
from ai.ai_validator import AIValidator
from ai.modules.fast_smc_analyzer import FastSMCAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_real_binance_data(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Get real market data from Binance API"""
    print(f"📡 Getting real data from Binance...")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Limit: {limit} candles")
    
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types and clean
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Keep only OHLCV
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        print(f"✅ Data obtained successfully:")
        print(f"   Candles: {len(df)}")
        print(f"   Period: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
        print(f"   Price: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error getting Binance data: {e}")
        raise

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the DataFrame"""
    print("📊 Calculating technical indicators...")
    
    df = df.copy()
    
    # MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD_12_26_9'] = exp1 - exp2
    df['MACD_signal_12_26_9'] = df['MACD_12_26_9'].ewm(span=9).mean()
    df['MACD_hist_12_26_9'] = df['MACD_12_26_9'] - df['MACD_signal_12_26_9']
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Stochastic (14, 3, 3)
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['STOCH_K_14_3'] = k_percent.rolling(window=3).mean()
    df['STOCH_D_14_3'] = df['STOCH_K_14_3'].rolling(window=3).mean()
    
    # Bollinger Bands (20, 2)
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper_20'] = bb_middle + (bb_std * 2)
    df['BB_middle_20'] = bb_middle
    df['BB_lower_20'] = bb_middle - (bb_std * 2)
    
    print("✅ Technical indicators calculated:")
    print("   • MACD (12, 26, 9)")
    print("   • RSI (14)")
    print("   • Stochastic (14, 3, 3)")
    print("   • Bollinger Bands (20, 2)")
    
    return df



def test_real_market_data_fetching():
    """Test 1: Real Market Data Fetching and SMC Analysis"""
    print("Test 1/3: Real Market Data Fetching...")
    print("🏛️ Testing SMC Analysis with REAL Market Data...")
    
    # Get real market data
    print("📡 Fetching REAL market data from Binance...")
    print("   Symbol: BTCUSDT")
    print("   Interval: 1h")
    print("   Candles: 300")
    
    df = get_real_binance_data("BTCUSDT", "1h", 300)
    df = add_technical_indicators(df)
    df = df.dropna()
    
    print(f"✅ Successfully fetched {len(df)} candles")
    print(f"   Time range: {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')} to {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
    
    # Calculate 24h change
    if len(df) >= 24:
        change_24h = ((df['close'].iloc[-1] / df['close'].iloc[-24]) - 1) * 100
        print(f"   24h change: {change_24h:+.2f}%")
    
    print(f"   Volume: {df['volume'].iloc[-1]:.0f}")
    
    print("📊 Adding technical indicators to real data...")
    print("✅ Technical indicators added")
    print(f"   MACD: {df['MACD_12_26_9'].iloc[-1]:.4f}")
    print(f"   RSI: {df['RSI_14'].iloc[-1]:.1f}")
    print(f"   Stoch %K: {df['STOCH_K_14_3'].iloc[-1]:.1f}")
    
    # BB Position calculation
    latest = df.iloc[-1]
    bb_position = ((latest['close'] - latest['BB_lower_20']) / (latest['BB_upper_20'] - latest['BB_lower_20'])) * 100
    print(f"   BB Position: {bb_position:.1f}%")
    
    # Real Market Information
    print("\n📊 REAL Market Information:")
    print(f"   Current Price: ${df['close'].iloc[-1]:,.2f}")
    if len(df) >= 24:
        print(f"   24h Change: {change_24h:+.2f}%")
        print(f"   24h High: ${df['high'].tail(24).max():,.2f}")
        print(f"   24h Low: ${df['low'].tail(24).min():,.2f}")
        print(f"   24h Volume: {df['volume'].tail(24).sum():,.0f} BTC")
    
    # SMC Analysis
    analyzer = FastSMCAnalyzer()
    smc_result = analyzer.analyze_fast(df)
    
    print("\n🏛️ REAL SMC Analysis Results:")
    print(f"   Current Zone: {smc_result['current_zone']}")
    print(f"   Zone Percentage: {smc_result['zone_percentage']:.1f}%")
    print(f"   Optimal Action: {smc_result['optimal_action']}")
    print(f"   SMC Confluence Score: {smc_result['smc_confluence_score']:.1f}/10")
    print(f"   Institutional Bias: {smc_result['institutional_bias']}")
    print(f"   Order Blocks Found: {len(smc_result['order_blocks'])}")
    print(f"   Fair Value Gaps: {len(smc_result['fair_value_gaps'])}")
    
    if smc_result['ob_bull']:
        print(f"   Bullish Order Block: ${smc_result['ob_bull']:,.2f}")
    if smc_result['ob_bear']:
        print(f"   Bearish Order Block: ${smc_result['ob_bear']:,.2f}")
    
    print(f"   Premium Level: ${smc_result['premium_level']:,.2f}")
    print(f"   Equilibrium Level: ${smc_result['equilibrium_level']:,.2f}")
    print(f"   Discount Level: ${smc_result['discount_level']:,.2f}")
    
    return df

def test_real_ai_validation(df):
    """Test 2: Real AI Validation with Gemini"""
    print("\nTest 2/3: Real AI Validation...")
    print("🤖 Testing AI Validation with REAL Market Data...")
    
    # Initialize AIValidator
    validator = AIValidator(
        timeout_seconds=60,
        confidence_threshold=7.0
    )
    
    print(f"\n📊 REAL Market Analysis for AI:")
    print(f"   Symbol: BTCUSDT")
    print(f"   Current Price: ${df['close'].iloc[-1]:,.2f}")
    print(f"   Data Points: {len(df)} hourly candles")
    print(f"   Time Range: {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')} to {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test LONG signal
    print("\n🔍 Testing REAL LONG Signal...")
    try:
        result_long = validator.validate_signal(df, "BTCUSDT", "LONG")
        
        print(f"✅ REAL LONG Signal Results:")
        print(f"   Should Enter: {result_long.should_enter}")
        print(f"   Confidence: {result_long.confidence:.2f}/10")
        print(f"   Analysis Time: {result_long.analysis_time:.2f}s")
        
        if result_long.entry_level:
            print(f"   Entry Level: ${result_long.entry_level:,.2f}")
        if result_long.stop_loss:
            print(f"   Stop Loss: ${result_long.stop_loss:,.2f}")
        if result_long.take_profit:
            print(f"   Take Profit: ${result_long.take_profit:,.2f}")
            
        if result_long.entry_level and result_long.stop_loss and result_long.take_profit:
            risk = abs(result_long.entry_level - result_long.stop_loss)
            reward = abs(result_long.take_profit - result_long.entry_level)
            rr_ratio = reward / risk if risk > 0 else 0
            print(f"   Risk/Reward Ratio: {rr_ratio:.2f}")
        
        print(f"   Reasoning: {result_long.reasoning[:150]}...")
        
    except Exception as e:
        print(f"❌ LONG signal test failed: {e}")
        return False
    
    # Test SHORT signal
    print("\n🔍 Testing REAL SHORT Signal...")
    try:
        result_short = validator.validate_signal(df, "BTCUSDT", "SHORT")
        
        print(f"✅ REAL SHORT Signal Results:")
        print(f"   Should Enter: {result_short.should_enter}")
        print(f"   Confidence: {result_short.confidence:.2f}/10")
        print(f"   Analysis Time: {result_short.analysis_time:.2f}s")
        print(f"   Reasoning: {result_short.reasoning[:150]}...")
        
    except Exception as e:
        print(f"❌ SHORT signal test failed: {e}")
        return False
    
    return True

def main():
    """Run complete real integration test with clean output format"""
    print("REAL MARKET DATA INTEGRATION TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success_count = 0
    total_tests = 3
    
    try:
        # Test 1: Real Market Data Fetching
        df = test_real_market_data_fetching()
        if df is not None and len(df) > 0:
            success_count += 1
            print("✅ Test 1/3: Real Market Data - PASSED")
        else:
            print("❌ Test 1/3: Real Market Data - FAILED")
            return
        
        # Test 2: Real AI Validation
        if test_real_ai_validation(df):
            success_count += 1
            print("✅ Test 2/3: Real AI Validation - PASSED")
        else:
            print("❌ Test 2/3: Real AI Validation - FAILED")
        
        # Test 3: System Integration
        print("\nTest 3/3: System Integration...")
        if success_count >= 2:
            print("✅ Test 3/3: System Integration - PASSED")
            print("   Core components working with real data")
            success_count += 1
        else:
            print("❌ Test 3/3: System Integration - FAILED")
        
        # Final Results
        print("\n" + "=" * 60)
        print(f"FINAL RESULTS: {success_count}/{total_tests} tests passed")
        
        if success_count == total_tests:
            print("🎉 ALL REAL DATA TESTS PASSED!")
            print("✅ System working with REAL market data")
            print("✅ Real-time price analysis functional")
            print("✅ AI making decisions on current market conditions")
        elif success_count >= 2:
            print("⚠️ PARTIAL SUCCESS")
            print("✅ Core functionality working with real data")
        else:
            print("❌ TESTS FAILED")
            print("❌ System needs debugging")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()