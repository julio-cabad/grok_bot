#!/usr/bin/env python3
"""
Test AI Validator
Validates that AI SMC analysis works correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.ai_validator import AIValidator, ValidationResult, SmartCache


def create_test_dataframe(num_candles: int = 500) -> pd.DataFrame:
    """Create realistic test DataFrame with OHLCV data"""
    
    # Generate realistic price data
    base_price = 50000  # Starting price like BTC
    dates = pd.date_range(start=datetime.now() - timedelta(hours=num_candles), 
                         periods=num_candles, freq='1H')
    
    # Generate price movement with some trend
    price_changes = np.random.normal(0, 0.02, num_candles)  # 2% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum price floor
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))  # 1% intraday volatility
        
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price
        
        # Ensure OHLC logic is correct
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher volume on bigger moves)
        volume = abs(np.random.normal(1000000, 200000))
        if abs(price_changes[i]) > 0.03:  # Big moves get more volume
            volume *= 2
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def test_smart_cache():
    """Test smart cache functionality"""
    print("🏛️ Testing Smart Cache...")
    
    cache = SmartCache(ttl_minutes=1, max_size=5)
    
    # Test basic set/get
    cache.set("test_key", "test_value")
    result = cache.get("test_key")
    assert result == "test_value", "Cache set/get failed"
    print("✅ Basic cache set/get - PASSED")
    
    # Test cache miss
    result = cache.get("nonexistent_key")
    assert result is None, "Cache should return None for missing keys"
    print("✅ Cache miss handling - PASSED")
    
    # Test hit rate calculation
    hit_rate = cache.get_hit_rate()
    assert hit_rate == 50.0, f"Expected 50% hit rate, got {hit_rate}%"  # 1 hit, 1 miss
    print(f"✅ Hit rate calculation - PASSED ({hit_rate}%)")
    
    # Test cache stats
    stats = cache.get_stats()
    assert stats['hits'] == 1, "Expected 1 hit"
    assert stats['misses'] == 1, "Expected 1 miss"
    assert stats['cache_size'] == 1, "Expected 1 item in cache"
    print("✅ Cache statistics - PASSED")
    
    print("✅ Smart Cache tests passed!\n")


def test_ai_validator_creation():
    """Test AI validator creation and configuration"""
    print("🏛️ Testing AI Validator Creation...")
    
    # Test with default settings
    validator = AIValidator()
    assert validator.timeout == 30, "Default timeout should be 30 seconds"
    assert validator.confidence_threshold == 7.5, "Default threshold should be 7.5"
    print("✅ Default configuration - PASSED")
    
    # Test with custom settings
    validator = AIValidator(timeout_seconds=60, confidence_threshold=8.0)
    assert validator.timeout == 60, "Custom timeout not set correctly"
    assert validator.confidence_threshold == 8.0, "Custom threshold not set correctly"
    print("✅ Custom configuration - PASSED")
    
    print("✅ AI Validator creation tests passed!\n")


def test_cache_key_generation():
    """Test cache key generation"""
    print("🏛️ Testing Cache Key Generation...")
    
    validator = AIValidator()
    df = create_test_dataframe(100)
    
    # Test that same data generates same key
    key1 = validator._generate_cache_key(df, "BTCUSDT", "LONG")
    key2 = validator._generate_cache_key(df, "BTCUSDT", "LONG")
    assert key1 == key2, "Same data should generate same cache key"
    print("✅ Consistent cache key generation - PASSED")
    
    # Test that different symbols generate different keys
    key3 = validator._generate_cache_key(df, "ETHUSDT", "LONG")
    assert key1 != key3, "Different symbols should generate different keys"
    print("✅ Different symbols generate different keys - PASSED")
    
    # Test that different signal types generate different keys
    key4 = validator._generate_cache_key(df, "BTCUSDT", "SHORT")
    assert key1 != key4, "Different signal types should generate different keys"
    print("✅ Different signal types generate different keys - PASSED")
    
    print("✅ Cache key generation tests passed!\n")


def test_smc_prompt_building():
    """Test SMC prompt building"""
    print("🏛️ Testing SMC Prompt Building...")
    
    validator = AIValidator()
    df = create_test_dataframe(500)
    
    prompt = validator._build_smc_prompt(df, "BTCUSDT", "LONG")
    
    # Check that prompt contains required elements
    assert "BTCUSDT" in prompt, "Prompt should contain symbol"
    assert "LONG" in prompt, "Prompt should contain signal type"
    assert "SMART MONEY CONCEPTS" in prompt, "Prompt should mention SMC"
    assert "ORDER BLOCKS" in prompt, "Prompt should mention order blocks"
    assert "FAIR VALUE GAPS" in prompt, "Prompt should mention fair value gaps"
    assert "SCORE:" in prompt, "Prompt should request score format"
    assert "ENTER:" in prompt, "Prompt should request enter format"
    
    print("✅ Prompt contains required SMC elements - PASSED")
    print("✅ SMC prompt building tests passed!\n")


def test_ai_response_parsing():
    """Test AI response parsing"""
    print("🏛️ Testing AI Response Parsing...")
    
    validator = AIValidator(confidence_threshold=7.0)
    
    # Test valid response
    mock_response = """
SCORE: 8.5
ENTER: YES
REASONING: Strong bullish order block at support with fair value gap above
ENTRY: 50250.00
SL: 49800.00
TP: 51500.00
"""
    
    result = validator._parse_ai_response(mock_response, "BTCUSDT", "LONG")
    
    assert result.confidence == 8.5, f"Expected confidence 8.5, got {result.confidence}"
    assert result.should_enter == True, "Should enter should be True for score above threshold"
    assert "bullish order block" in result.reasoning.lower(), "Reasoning should be preserved"
    assert result.entry_level == 50250.00, f"Expected entry 50250, got {result.entry_level}"
    assert result.stop_loss == 49800.00, f"Expected SL 49800, got {result.stop_loss}"
    assert result.take_profit == 51500.00, f"Expected TP 51500, got {result.take_profit}"
    
    print("✅ Valid response parsing - PASSED")
    
    # Test response below threshold
    mock_response_low = """
SCORE: 6.0
ENTER: YES
REASONING: Weak setup with mixed signals
"""
    
    result_low = validator._parse_ai_response(mock_response_low, "BTCUSDT", "LONG")
    assert result_low.should_enter == False, "Should not enter when score below threshold"
    assert "below threshold" in result_low.reasoning, "Reasoning should mention threshold"
    
    print("✅ Below threshold handling - PASSED")
    
    # Test malformed response (with threshold 7.0, score 5.0 should be rejected)
    mock_response_bad = "This is not a proper response format"
    
    result_bad = validator._parse_ai_response(mock_response_bad, "BTCUSDT", "LONG")
    assert result_bad.should_enter == False, "Should reject trade when parse fails and score below threshold"
    assert result_bad.confidence == 5.0, "Should default to neutral confidence"
    
    print("✅ Malformed response handling - PASSED")
    print("✅ AI response parsing tests passed!\n")


def test_cache_integration():
    """Test cache integration with AI validator"""
    print("🏛️ Testing Cache Integration...")
    
    validator = AIValidator()
    df = create_test_dataframe(100)
    
    # Mock the AI analysis to avoid actual API calls
    def mock_ai_analysis(prompt):
        return """
SCORE: 8.0
ENTER: YES
REASONING: Mock analysis for testing
ENTRY: 50000.00
SL: 49500.00
TP: 51000.00
"""
    
    # Replace the AI analysis method temporarily
    original_method = validator._get_ai_analysis_with_timeout
    validator._get_ai_analysis_with_timeout = mock_ai_analysis
    
    try:
        # First call should miss cache and call AI
        result1 = validator.validate_signal(df, "BTCUSDT", "LONG")
        cache_stats1 = validator.get_cache_stats()
        
        # Second call with same data should hit cache
        result2 = validator.validate_signal(df, "BTCUSDT", "LONG")
        cache_stats2 = validator.get_cache_stats()
        
        # Verify results are identical
        assert result1.confidence == result2.confidence, "Cached result should be identical"
        assert result1.should_enter == result2.should_enter, "Cached result should be identical"
        assert result1.reasoning == result2.reasoning, "Cached result should be identical"
        
        # Verify cache was used
        assert cache_stats2['hits'] > cache_stats1['hits'], "Cache hits should increase"
        assert cache_stats2['hit_rate'] > 0, "Hit rate should be positive"
        
        print("✅ Cache integration working correctly - PASSED")
        
    finally:
        # Restore original method
        validator._get_ai_analysis_with_timeout = original_method
    
    print("✅ Cache integration tests passed!\n")


def main():
    """Run all AI validator tests"""
    print("🚀 AI VALIDATOR TESTS")
    print("=" * 50)
    
    try:
        test_smart_cache()
        test_ai_validator_creation()
        test_cache_key_generation()
        test_smc_prompt_building()
        test_ai_response_parsing()
        test_cache_integration()
        
        print("🏛️ ALL AI VALIDATOR TESTS PASSED!")
        print("AI Validator is ready for integration!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"💀 AI VALIDATOR TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)