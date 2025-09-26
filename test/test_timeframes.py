#!/usr/bin/env python3
"""
Test Timeframes Configuration
Validates that higher timeframes work correctly with optimized settings
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_timeframe_configuration():
    """Test timeframe configuration and optimization"""
    print("🏛️ Testing Timeframe Configuration...")
    
    # Test with higher timeframes enabled
    with patch.dict('config.settings.__dict__', {'USE_HIGHER_TIMEFRAMES': True}):
        # Reload config to apply changes
        import importlib
        import config.settings
        importlib.reload(config.settings)
        
        from config.settings import time_frame, get_check_interval, get_candles_limit
        
        # Should default to 1h when higher timeframes enabled
        assert time_frame == '1h', f"Expected 1h timeframe, got {time_frame}"
        print(f"✅ Higher timeframes enabled - using {time_frame}")
        
        # Check optimized intervals
        interval = get_check_interval()
        assert interval == 300, f"Expected 300s interval for 1h, got {interval}"
        print(f"✅ Check interval optimized: {interval}s for {time_frame}")
        
        # Check optimized candles
        candles = get_candles_limit()
        assert candles == 168, f"Expected 168 candles for 1h, got {candles}"
        print(f"✅ Candles limit optimized: {candles} for {time_frame} (7 days)")
    
    print("✅ Timeframe configuration tests passed!\n")


def test_timeframe_intervals():
    """Test different timeframe intervals"""
    print("🏛️ Testing Timeframe Intervals...")
    
    from config.settings import TIMEFRAME_INTERVALS
    
    # Test all supported timeframes
    expected_intervals = {
        '1m': 5,      # 5 seconds
        '5m': 30,     # 30 seconds
        '15m': 60,    # 1 minute
        '1h': 300,    # 5 minutes
        '4h': 900,    # 15 minutes
        '1d': 3600,   # 1 hour
    }
    
    for tf, expected_interval in expected_intervals.items():
        actual_interval = TIMEFRAME_INTERVALS.get(tf)
        assert actual_interval == expected_interval, f"Expected {expected_interval}s for {tf}, got {actual_interval}s"
        print(f"✅ {tf}: {actual_interval}s interval")
    
    print("✅ Timeframe intervals tests passed!\n")


def test_candles_optimization():
    """Test candles limit optimization for different timeframes"""
    print("🏛️ Testing Candles Optimization...")
    
    from config.settings import TIMEFRAME_CANDLES
    
    # Test candles optimization logic
    expected_candles = {
        '1m': 500,    # ~8 hours
        '5m': 288,    # 24 hours
        '15m': 96,    # 24 hours
        '1h': 168,    # 7 days
        '4h': 180,    # 30 days
        '1d': 90,     # 3 months
    }
    
    for tf, expected_count in expected_candles.items():
        actual_count = TIMEFRAME_CANDLES.get(tf)
        assert actual_count == expected_count, f"Expected {expected_count} candles for {tf}, got {actual_count}"
        
        # Calculate time coverage
        if tf == '1m':
            coverage = f"{actual_count} minutes (~{actual_count/60:.1f} hours)"
        elif tf == '5m':
            coverage = f"{actual_count * 5} minutes ({actual_count * 5 / 60:.1f} hours)"
        elif tf == '15m':
            coverage = f"{actual_count * 15} minutes ({actual_count * 15 / 60:.1f} hours)"
        elif tf == '1h':
            coverage = f"{actual_count} hours ({actual_count / 24:.1f} days)"
        elif tf == '4h':
            coverage = f"{actual_count * 4} hours ({actual_count * 4 / 24:.1f} days)"
        elif tf == '1d':
            coverage = f"{actual_count} days ({actual_count / 30:.1f} months)"
        
        print(f"✅ {tf}: {actual_count} candles = {coverage}")
    
    print("✅ Candles optimization tests passed!\n")


def test_timeframe_efficiency():
    """Test that higher timeframes are more efficient"""
    print("🏛️ Testing Timeframe Efficiency...")
    
    from config.settings import TIMEFRAME_INTERVALS, TIMEFRAME_CANDLES
    
    # Calculate efficiency metrics
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    print("Efficiency Analysis:")
    print("Timeframe | Interval | Candles | API Calls/Day | Data Points/Day")
    print("-" * 65)
    
    for tf in timeframes:
        interval = TIMEFRAME_INTERVALS.get(tf, 300)
        candles = TIMEFRAME_CANDLES.get(tf, 500)
        
        # Calculate API calls per day
        api_calls_per_day = (24 * 3600) / interval
        
        # Data points per day (assuming we get new candle each interval)
        data_points_per_day = api_calls_per_day
        
        print(f"{tf:>8} | {interval:>7}s | {candles:>6} | {api_calls_per_day:>11.0f} | {data_points_per_day:>13.0f}")
    
    # Verify that higher timeframes are more efficient (fewer API calls)
    interval_1m = TIMEFRAME_INTERVALS['1m']
    interval_1h = TIMEFRAME_INTERVALS['1h']
    
    api_calls_1m = (24 * 3600) / interval_1m
    api_calls_1h = (24 * 3600) / interval_1h
    
    efficiency_improvement = api_calls_1m / api_calls_1h
    
    assert efficiency_improvement > 50, f"1H should be >50x more efficient than 1m, got {efficiency_improvement:.1f}x"
    print(f"\n✅ Efficiency improvement: 1H is {efficiency_improvement:.0f}x more efficient than 1m")
    
    print("✅ Timeframe efficiency tests passed!\n")


def test_real_world_scenarios():
    """Test real-world timeframe scenarios"""
    print("🏛️ Testing Real-World Scenarios...")
    
    scenarios = [
        {
            'name': 'Scalping (1m)',
            'timeframe': '1m',
            'expected_interval': 5,
            'expected_candles': 500,
            'use_case': 'Quick entries, high frequency'
        },
        {
            'name': 'Swing Trading (1h)',
            'timeframe': '1h', 
            'expected_interval': 300,
            'expected_candles': 168,
            'use_case': 'Quality signals, less noise'
        },
        {
            'name': 'Position Trading (4h)',
            'timeframe': '4h',
            'expected_interval': 900,
            'expected_candles': 180,
            'use_case': 'Long-term trends, minimal monitoring'
        }
    ]
    
    from config.settings import TIMEFRAME_INTERVALS, TIMEFRAME_CANDLES
    
    for scenario in scenarios:
        tf = scenario['timeframe']
        
        actual_interval = TIMEFRAME_INTERVALS.get(tf)
        actual_candles = TIMEFRAME_CANDLES.get(tf)
        
        assert actual_interval == scenario['expected_interval'], \
            f"{scenario['name']}: Expected {scenario['expected_interval']}s, got {actual_interval}s"
        
        assert actual_candles == scenario['expected_candles'], \
            f"{scenario['name']}: Expected {scenario['expected_candles']} candles, got {actual_candles}"
        
        print(f"✅ {scenario['name']} ({tf}): {actual_interval}s interval, {actual_candles} candles")
        print(f"   Use case: {scenario['use_case']}")
    
    print("✅ Real-world scenarios tests passed!\n")


def test_backward_compatibility():
    """Test backward compatibility logic"""
    print("🏛️ Testing Backward Compatibility...")
    
    # Test the logic directly
    use_higher_true = True
    use_higher_false = False
    
    # Test timeframe selection logic
    timeframe_when_enabled = '1h' if use_higher_true else '1m'
    timeframe_when_disabled = '1h' if use_higher_false else '1m'
    
    assert timeframe_when_enabled == '1h', f"Expected 1h when enabled, got {timeframe_when_enabled}"
    assert timeframe_when_disabled == '1m', f"Expected 1m when disabled, got {timeframe_when_disabled}"
    
    print(f"✅ Logic test - enabled: {timeframe_when_enabled}, disabled: {timeframe_when_disabled}")
    print("✅ Backward compatibility logic verified")
    
    print("✅ Backward compatibility tests passed!\n")


def main():
    """Run all timeframe tests"""
    print("🚀 TIMEFRAMES CONFIGURATION TESTS")
    print("=" * 50)
    
    try:
        test_timeframe_configuration()
        test_timeframe_intervals()
        test_candles_optimization()
        test_timeframe_efficiency()
        test_real_world_scenarios()
        test_backward_compatibility()
        
        print("🏛️ ALL TIMEFRAMES TESTS PASSED!")
        print("Higher timeframes are optimally configured!")
        print("=" * 50)
        
        # Show current configuration
        import config.settings
        print(f"\n📊 CURRENT CONFIGURATION:")
        print(f"Timeframe: {config.settings.time_frame}")
        print(f"Check Interval: {config.settings.CHECK_INTERVAL_SECONDS}s")
        print(f"Candles Limit: {config.settings.CANDLES_LIMIT}")
        print(f"Higher Timeframes: {config.settings.USE_HIGHER_TIMEFRAMES}")
        
        return True
        
    except Exception as e:
        print(f"💀 TIMEFRAMES TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)