#!/usr/bin/env python3
"""
Test Higher Timeframes Integration
Validates that the bot works correctly with 1H+ timeframes
"""

import sys
import os
import time
from pathlib import Path
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_timeframe_configuration_integration():
    """Test that timeframe configuration is properly integrated"""
    print("🏛️ Testing Timeframe Configuration Integration...")
    
    # Test importing configuration
    from config.settings import time_frame, CHECK_INTERVAL_SECONDS, USE_HIGHER_TIMEFRAMES, get_check_interval
    
    print(f"✅ Current timeframe: {time_frame}")
    print(f"✅ Check interval: {CHECK_INTERVAL_SECONDS} seconds")
    print(f"✅ Higher timeframes enabled: {USE_HIGHER_TIMEFRAMES}")
    
    # Test dynamic interval calculation
    calculated_interval = get_check_interval()
    assert calculated_interval == CHECK_INTERVAL_SECONDS, "Dynamic interval should match configured interval"
    print(f"✅ Dynamic interval calculation: {calculated_interval}s")
    
    # Test that 1H timeframe gives 5 minute interval
    if time_frame == '1h':
        assert CHECK_INTERVAL_SECONDS == 300, "1H timeframe should use 5 minute interval"
        print("✅ 1H timeframe uses optimal 5 minute check interval")
    
    print("✅ Timeframe configuration integration tests passed!\n")


def test_binance_client_with_higher_timeframes():
    """Test that Binance client works with higher timeframes"""
    print("🏛️ Testing Binance Client with Higher Timeframes...")
    
    from bnb.binance import RobotBinance
    from config.settings import time_frame
    
    # Test creating Binance client
    try:
        robot = RobotBinance(pair="BTCUSDT", temporality=time_frame)
        print(f"✅ Binance client created successfully for {time_frame}")
        
        # Test that the timeframe is valid
        assert robot.temporality == time_frame, "Robot should use configured timeframe"
        print(f"✅ Robot configured with timeframe: {robot.temporality}")
        
    except Exception as e:
        print(f"❌ Binance client creation failed: {e}")
        raise
    
    print("✅ Binance client with higher timeframes tests passed!\n")


def test_technical_indicators_with_higher_timeframes():
    """Test that technical indicators work with higher timeframes"""
    print("🏛️ Testing Technical Indicators with Higher Timeframes...")
    
    from indicators.technical_indicators import TechnicalAnalyzer
    from config.settings import time_frame
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create test data for current timeframe
    num_candles = 100
    
    # Adjust frequency based on timeframe
    freq_map = {
        '1m': '1min',
        '5m': '5min', 
        '15m': '15min',
        '1h': '1h',
        '4h': '4h',
        '1d': '1D'
    }
    
    freq = freq_map.get(time_frame, '1h')
    dates = pd.date_range(start=datetime.now() - timedelta(hours=num_candles), 
                         periods=num_candles, freq=freq)
    
    # Generate realistic price data
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, num_candles)
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        volatility = abs(np.random.normal(0, 0.01))
        
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = abs(np.random.normal(1000000, 200000))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Test technical analyzer
    try:
        analyzer = TechnicalAnalyzer(symbol="BTCUSDT", timeframe=time_frame)
        analyzer.df = df
        
        # Test all indicators
        analyzer.trend_magic()
        print(f"✅ Trend Magic works with {time_frame}")
        
        analyzer.squeeze_momentum()
        print(f"✅ Squeeze Momentum works with {time_frame}")
        
        analyzer.calculate_rsi()
        print(f"✅ RSI works with {time_frame}")
        
        analyzer.calculate_bollinger_bands()
        print(f"✅ Bollinger Bands work with {time_frame}")
        
        analyzer.calculate_macd()
        print(f"✅ MACD works with {time_frame}")
        
        analyzer.calculate_stochastic()
        print(f"✅ Stochastic works with {time_frame}")
        
        # Verify indicators have values
        assert not analyzer.df.empty, "DataFrame should not be empty after indicator calculation"
        assert 'MagicTrend_Color' in analyzer.df.columns, "Trend Magic should be calculated"
        assert 'squeeze_color' in analyzer.df.columns, "Squeeze Momentum should be calculated"
        
        print(f"✅ All indicators calculated successfully for {time_frame}")
        
    except Exception as e:
        print(f"❌ Technical indicators failed with {time_frame}: {e}")
        raise
    
    print("✅ Technical indicators with higher timeframes tests passed!\n")


def test_strategy_execution_with_higher_timeframes():
    """Test that strategy execution works with higher timeframes"""
    print("🏛️ Testing Strategy Execution with Higher Timeframes...")
    
    from strategy.strategies import StrategyManager
    from config.settings import time_frame
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create test data with bullish setup
    num_candles = 100
    dates = pd.date_range(start=datetime.now() - timedelta(hours=num_candles), 
                         periods=num_candles, freq='1h')
    
    base_price = 50000
    price_changes = np.random.normal(0.001, 0.02, num_candles)  # Slight upward bias
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))
    
    data = []
    for i, price in enumerate(prices):
        volatility = abs(np.random.normal(0, 0.01))
        
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = abs(np.random.normal(1000000, 200000))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Add bullish technical indicators
    df['squeeze_color'] = 'LIME'  # Bullish squeeze
    df['momentum_color'] = 'LIME'
    df['MagicTrend_Color'] = 'BLUE'  # Bullish trend magic
    df['MagicTrend'] = df['close'] * 0.98
    df['ATR'] = df['close'] * 0.02
    df['RSI_14'] = 60
    df['MACD_12_26_9'] = 0.1
    df['BB_upper_20'] = df['close'] * 1.02
    df['BB_middle_20'] = df['close']
    df['BB_lower_20'] = df['close'] * 0.98
    df['STOCH_K_14_3'] = 60
    df['STOCH_D_14_3'] = 55
    
    # Test strategy execution
    try:
        strategy_manager = StrategyManager()
        signal = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        
        print(f"✅ Strategy executed successfully with {time_frame}")
        print(f"✅ Generated signal: {signal.signal_type.value}")
        print(f"✅ Signal reason: {signal.reason}")
        
        # Should generate a LONG signal with bullish setup
        from strategy.strategies import SignalType
        assert signal.signal_type in [SignalType.LONG, SignalType.WAIT], f"Expected LONG or WAIT, got {signal.signal_type}"
        
        if signal.signal_type == SignalType.LONG:
            assert signal.entry_price is not None, "LONG signal should have entry price"
            assert signal.stop_loss is not None, "LONG signal should have stop loss"
            assert signal.take_profit is not None, "LONG signal should have take profit"
            print("✅ LONG signal has all required levels")
        
    except Exception as e:
        print(f"❌ Strategy execution failed with {time_frame}: {e}")
        raise
    
    print("✅ Strategy execution with higher timeframes tests passed!\n")


def test_performance_with_higher_timeframes():
    """Test performance characteristics with higher timeframes"""
    print("🏛️ Testing Performance with Higher Timeframes...")
    
    from config.settings import CHECK_INTERVAL_SECONDS, time_frame
    
    # Higher timeframes should have longer check intervals for efficiency
    if time_frame in ['1h', '4h', '1d']:
        assert CHECK_INTERVAL_SECONDS >= 300, f"Higher timeframes should use longer intervals, got {CHECK_INTERVAL_SECONDS}s"
        print(f"✅ {time_frame} uses efficient {CHECK_INTERVAL_SECONDS}s check interval")
    
    # Calculate expected API calls per hour
    calls_per_hour = 3600 / CHECK_INTERVAL_SECONDS
    print(f"✅ Expected API calls per hour: {calls_per_hour:.1f}")
    
    # Higher timeframes should make fewer API calls
    if time_frame == '1h':
        assert calls_per_hour <= 12, "1H timeframe should make <= 12 API calls per hour"
        print("✅ 1H timeframe is API efficient")
    elif time_frame == '4h':
        assert calls_per_hour <= 4, "4H timeframe should make <= 4 API calls per hour"
        print("✅ 4H timeframe is very API efficient")
    
    print("✅ Performance with higher timeframes tests passed!\n")


def main():
    """Run all higher timeframes integration tests"""
    print("🚀 HIGHER TIMEFRAMES INTEGRATION TESTS")
    print("=" * 50)
    
    try:
        test_timeframe_configuration_integration()
        test_binance_client_with_higher_timeframes()
        test_technical_indicators_with_higher_timeframes()
        test_strategy_execution_with_higher_timeframes()
        test_performance_with_higher_timeframes()
        
        print("🏛️ ALL HIGHER TIMEFRAMES INTEGRATION TESTS PASSED!")
        print("Higher timeframes integration is working perfectly!")
        print("=" * 50)
        
        # Show current configuration
        from config.settings import time_frame, CHECK_INTERVAL_SECONDS, USE_HIGHER_TIMEFRAMES
        print(f"\n📊 CURRENT CONFIGURATION:")
        print(f"Timeframe: {time_frame}")
        print(f"Check Interval: {CHECK_INTERVAL_SECONDS}s")
        print(f"Higher Timeframes: {USE_HIGHER_TIMEFRAMES}")
        
        return True
        
    except Exception as e:
        print(f"💀 HIGHER TIMEFRAMES INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)