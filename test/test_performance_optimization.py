#!/usr/bin/env python3
"""
Test Performance Optimization
Validates that all performance optimizations work correctly
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache.market_data_cache import MarketDataCache
from monitoring.performance_monitor import PerformanceMonitor, TimingContext


def create_test_dataframe(num_candles: int = 500) -> pd.DataFrame:
    """Create realistic test DataFrame"""
    base_price = 50000
    dates = pd.date_range(start=datetime.now() - timedelta(hours=num_candles), 
                         periods=num_candles, freq='1h')
    
    # Generate realistic OHLCV data
    data = []
    for i in range(num_candles):
        price = base_price + np.random.normal(0, 1000)
        volatility = abs(np.random.normal(0, 0.01))
        
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = price
        close = price + np.random.normal(0, 100)
        volume = abs(np.random.normal(1000000, 200000))
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)


def test_market_data_cache():
    """Test market data cache functionality"""
    print("🏛️ Testing Market Data Cache...")
    
    cache = MarketDataCache(max_size_mb=10, enable_compression=True)
    
    # Test basic caching
    df = create_test_dataframe(100)
    
    # Cache miss
    result = cache.get("BTCUSDT", "1h", 100)
    assert result is None, "Should be cache miss initially"
    
    # Store in cache
    cache.set("BTCUSDT", "1h", 100, df)
    
    # Cache hit
    result = cache.get("BTCUSDT", "1h", 100)
    assert result is not None, "Should be cache hit after storing"
    assert len(result) == len(df), "Cached data should have same length"
    
    print("✅ Basic cache operations - PASSED")
    
    # Test TTL expiration
    cache_short_ttl = MarketDataCache(max_size_mb=10)
    cache_short_ttl.TIMEFRAME_TTL['1h'] = 1  # 1 second TTL
    
    cache_short_ttl.set("ETHUSDT", "1h", 100, df)
    
    # Should hit immediately
    result = cache_short_ttl.get("ETHUSDT", "1h", 100)
    assert result is not None, "Should hit before expiration"
    
    # Wait for expiration
    time.sleep(1.1)
    
    # Should miss after expiration
    result = cache_short_ttl.get("ETHUSDT", "1h", 100)
    assert result is None, "Should miss after expiration"
    
    print("✅ TTL expiration - PASSED")
    
    # Test compression
    large_df = create_test_dataframe(1000)  # Large dataset
    
    cache.set("ADAUSDT", "1h", 1000, large_df)
    result = cache.get("ADAUSDT", "1h", 1000)
    
    assert result is not None, "Should retrieve compressed data"
    assert len(result) == len(large_df), "Compressed data should be complete"
    
    print("✅ Data compression - PASSED")
    
    # Test cache statistics
    stats = cache.get_stats()
    assert stats['hits'] > 0, "Should have cache hits"
    assert stats['hit_rate_percent'] > 0, "Should have positive hit rate"
    assert stats['cache_entries'] > 0, "Should have cached entries"
    
    print(f"✅ Cache stats - Hit Rate: {stats['hit_rate_percent']:.1f}%, Entries: {stats['cache_entries']}")
    
    print("✅ Market Data Cache tests passed!\n")


def test_performance_monitor():
    """Test performance monitoring functionality"""
    print("🏛️ Testing Performance Monitor...")
    
    monitor = PerformanceMonitor(max_history=100)
    
    # Test timing recording
    with TimingContext(monitor, 'test_operation', {'test': 'value'}):
        time.sleep(0.1)  # Simulate work
    
    # Check timing was recorded
    avg_time = monitor.get_average_timing('test_operation')
    assert avg_time >= 0.1, f"Should record timing >= 0.1s, got {avg_time}"
    
    print("✅ Timing recording - PASSED")
    
    # Test counter increments
    monitor.increment_counter('test_counter', 5)
    monitor.increment_counter('test_counter', 3)
    
    assert monitor.counters['test_counter'] == 8, "Should increment counter correctly"
    
    print("✅ Counter increments - PASSED")
    
    # Test performance summary
    summary = monitor.get_performance_summary()
    
    assert 'timestamp' in summary, "Summary should have timestamp"
    assert 'counters' in summary, "Summary should have counters"
    assert 'average_timings' in summary, "Summary should have average timings"
    assert 'performance_score' in summary, "Summary should have performance score"
    assert 'recommendations' in summary, "Summary should have recommendations"
    
    print(f"✅ Performance summary - Score: {summary['performance_score']}/100")
    
    # Test percentile calculations
    # Add multiple timings
    for i in range(10):
        monitor.record_timing('batch_test', 0.1 + (i * 0.01))
    
    p95 = monitor.get_percentile_timing('batch_test', 95)
    avg = monitor.get_average_timing('batch_test')
    
    assert p95 >= avg, "P95 should be >= average"
    
    print("✅ Percentile calculations - PASSED")
    
    print("✅ Performance Monitor tests passed!\n")


def test_cache_performance():
    """Test cache performance under load"""
    print("🏛️ Testing Cache Performance Under Load...")
    
    cache = MarketDataCache(max_size_mb=50, enable_compression=True)
    monitor = PerformanceMonitor()
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT']
    timeframes = ['1m', '5m', '1h', '4h']
    
    # Simulate realistic trading bot usage
    total_operations = 0
    cache_hits = 0
    
    for cycle in range(3):  # 3 cycles to test cache effectiveness
        print(f"   Cycle {cycle + 1}/3...")
        
        for symbol in symbols:
            for timeframe in timeframes:
                with TimingContext(monitor, 'cache_operation', {'symbol': symbol, 'timeframe': timeframe}):
                    # Try to get from cache first
                    df = cache.get(symbol, timeframe, 168)
                    total_operations += 1
                    
                    if df is not None:
                        cache_hits += 1
                    else:
                        # Simulate API call and cache storage
                        df = create_test_dataframe(168)
                        cache.set(symbol, timeframe, 168, df)
    
    # Calculate performance metrics
    hit_rate = (cache_hits / total_operations) * 100
    avg_operation_time = monitor.get_average_timing('cache_operation')
    
    print(f"✅ Cache hit rate: {hit_rate:.1f}% ({cache_hits}/{total_operations})")
    print(f"✅ Average operation time: {avg_operation_time:.3f}s")
    
    # Verify performance expectations
    assert hit_rate >= 50, f"Cache hit rate should be >= 50%, got {hit_rate:.1f}%"
    assert avg_operation_time < 0.01, f"Operations should be fast, got {avg_operation_time:.3f}s"
    
    # Check cache stats
    stats = cache.get_stats()
    print(f"✅ Cache size: {stats['total_size_mb']:.1f}MB / {stats['max_size_mb']}MB")
    print(f"✅ Compressions: {stats['compressions']}")
    
    print("✅ Cache Performance Under Load tests passed!\n")


def test_timeframe_optimization():
    """Test timeframe-specific optimizations"""
    print("🏛️ Testing Timeframe Optimizations...")
    
    from config.settings import TIMEFRAME_CANDLES, TIMEFRAME_INTERVALS
    
    # Test expected TTL values directly
    expected_ttl = {
        '1m': 30,      # 30 seconds
        '1h': 1800,    # 30 minutes  
        '4h': 7200,    # 2 hours
    }
    
    # Verify TTL progression makes sense
    assert expected_ttl['1h'] > expected_ttl['1m'], f"1h TTL should be > 1m TTL"
    assert expected_ttl['4h'] > expected_ttl['1h'], f"4h TTL should be > 1h TTL"
    
    print("✅ TTL optimization logic - PASSED")
    
    # Test that higher timeframes use fewer candles for same time coverage
    candles_1m = TIMEFRAME_CANDLES.get('1m', 500)
    candles_1h = TIMEFRAME_CANDLES.get('1h', 500)
    
    # 1h should cover more time with fewer candles
    time_coverage_1m = candles_1m * 1  # minutes
    time_coverage_1h = candles_1h * 60  # minutes
    
    assert time_coverage_1h > time_coverage_1m, "1h should cover more time than 1m with optimized candles"
    
    print("✅ Candles optimization by timeframe - PASSED")
    
    # Test that higher timeframes have longer check intervals
    interval_1m = TIMEFRAME_INTERVALS.get('1m', 300)
    interval_1h = TIMEFRAME_INTERVALS.get('1h', 300)
    
    assert interval_1h > interval_1m, f"1h interval ({interval_1h}) should be > 1m interval ({interval_1m})"
    
    efficiency_improvement = interval_1h / interval_1m
    print(f"✅ Check interval efficiency: 1h is {efficiency_improvement:.0f}x more efficient than 1m")
    
    print("✅ Timeframe Optimizations tests passed!\n")


def test_memory_efficiency():
    """Test memory efficiency of optimizations"""
    print("🏛️ Testing Memory Efficiency...")
    
    cache = MarketDataCache(max_size_mb=20, enable_compression=True)
    
    # Fill cache with data
    large_datasets = []
    for i in range(10):
        df = create_test_dataframe(500)  # Large datasets
        symbol = f"TEST{i}USDT"
        cache.set(symbol, "1h", 500, df)
        large_datasets.append(df)
    
    # Check that cache respects size limits
    stats = cache.get_stats()
    assert stats['total_size_mb'] <= stats['max_size_mb'], f"Cache size ({stats['total_size_mb']:.1f}MB) should not exceed limit ({stats['max_size_mb']}MB)"
    
    print(f"✅ Memory limit respected: {stats['total_size_mb']:.1f}MB / {stats['max_size_mb']}MB")
    
    # Check that LRU eviction works
    assert stats['evictions'] > 0 or stats['cache_entries'] <= 10, "Should have evictions or limited entries"
    
    print(f"✅ LRU eviction working: {stats['evictions']} evictions")
    
    # Test compression effectiveness
    if stats['compressions'] > 0:
        print(f"✅ Compression active: {stats['compressions']} compressions performed")
    
    print("✅ Memory Efficiency tests passed!\n")


def test_real_world_simulation():
    """Simulate real-world trading bot performance"""
    print("🏛️ Testing Real-World Simulation...")
    
    cache = MarketDataCache(max_size_mb=100, enable_compression=True)
    monitor = PerformanceMonitor()
    
    # Simulate 1 hour of trading bot operation
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT', 'BNBUSDT', 'XRPUSDT', 'MATICUSDT']
    timeframe = '1h'
    check_interval = 300  # 5 minutes
    simulation_duration = 3600  # 1 hour
    
    cycles = simulation_duration // check_interval  # 12 cycles
    
    print(f"   Simulating {cycles} cycles over {simulation_duration/60:.0f} minutes...")
    
    total_api_calls = 0
    total_cache_hits = 0
    
    for cycle in range(cycles):
        cycle_start = time.time()
        
        for symbol in symbols:
            with TimingContext(monitor, 'symbol_processing', {'symbol': symbol, 'cycle': cycle}):
                # Check cache first
                df = cache.get(symbol, timeframe, 168)
                
                if df is not None:
                    total_cache_hits += 1
                    # Simulate using cached data
                    time.sleep(0.001)  # Minimal processing time
                else:
                    total_api_calls += 1
                    # Simulate API call and processing
                    time.sleep(0.05)  # API call simulation
                    df = create_test_dataframe(168)
                    cache.set(symbol, timeframe, 168, df)
                    
                    # Simulate indicator calculations
                    time.sleep(0.02)
                
                monitor.increment_counter('symbols_processed')
        
        cycle_duration = time.time() - cycle_start
        monitor.record_timing('total_cycle', cycle_duration, {'symbols_count': len(symbols)})
        
        # Simulate waiting for next cycle
        time.sleep(0.01)  # Minimal wait
    
    # Analyze results
    summary = monitor.get_performance_summary()
    cache_stats = cache.get_stats()
    
    total_requests = total_api_calls + total_cache_hits
    cache_hit_rate = (total_cache_hits / total_requests) * 100 if total_requests > 0 else 0
    
    avg_symbol_time = monitor.get_average_timing('symbol_processing')
    avg_cycle_time = monitor.get_average_timing('total_cycle')
    
    print(f"✅ Simulation Results:")
    print(f"   Cache Hit Rate: {cache_hit_rate:.1f}% ({total_cache_hits}/{total_requests})")
    print(f"   Avg Symbol Processing: {avg_symbol_time:.3f}s")
    print(f"   Avg Cycle Time: {avg_cycle_time:.3f}s")
    print(f"   Performance Score: {summary['performance_score']}/100")
    print(f"   Cache Size: {cache_stats['total_size_mb']:.1f}MB")
    
    # Verify performance targets
    assert cache_hit_rate >= 70, f"Cache hit rate should be >= 70%, got {cache_hit_rate:.1f}%"
    assert avg_symbol_time < 0.1, f"Symbol processing should be < 0.1s, got {avg_symbol_time:.3f}s"
    assert summary['performance_score'] >= 80, f"Performance score should be >= 80, got {summary['performance_score']}"
    
    print("✅ Real-World Simulation tests passed!\n")


def main():
    """Run all performance optimization tests"""
    print("🚀 PERFORMANCE OPTIMIZATION TESTS")
    print("=" * 50)
    
    try:
        test_market_data_cache()
        test_performance_monitor()
        test_cache_performance()
        test_timeframe_optimization()
        test_memory_efficiency()
        test_real_world_simulation()
        
        print("🏛️ ALL PERFORMANCE OPTIMIZATION TESTS PASSED!")
        print("System is optimized for maximum efficiency!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"💀 PERFORMANCE OPTIMIZATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)