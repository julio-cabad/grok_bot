#!/usr/bin/env python3
"""
Test AI Integration with Strategy Manager
Validates that AI validation works correctly in the trading strategy
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.strategies import StrategyManager, SignalType, SignalStrength
from ai.ai_validator import ValidationResult


def create_test_dataframe_with_indicators(num_candles: int = 500) -> pd.DataFrame:
    """Create test DataFrame with technical indicators"""
    
    # Generate realistic price data
    base_price = 50000
    dates = pd.date_range(start=datetime.now() - timedelta(hours=num_candles), 
                         periods=num_candles, freq='1h')
    
    # Generate price movement
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
    
    # Add technical indicators (mock values for testing)
    df['squeeze_color'] = 'LIME'  # Bullish squeeze
    df['momentum_color'] = 'LIME'
    df['MagicTrend_Color'] = 'BLUE'  # Bullish trend magic
    df['MagicTrend'] = df['close'] * 0.98  # Trend line below price
    df['ATR'] = df['close'] * 0.02  # 2% ATR
    df['RSI_14'] = 60  # Neutral RSI
    df['MACD_12_26_9'] = 0.1  # Positive MACD
    df['BB_upper_20'] = df['close'] * 1.02
    df['BB_middle_20'] = df['close']
    df['BB_lower_20'] = df['close'] * 0.98
    df['STOCH_K_14_3'] = 60
    df['STOCH_D_14_3'] = 55
    
    return df


def test_ai_integration_disabled():
    """Test that strategy works normally when AI is disabled"""
    print("🏛️ Testing AI Integration - DISABLED...")
    
    # Mock the USE_AI_VALIDATION to False
    with patch('strategy.strategies.USE_AI_VALIDATION', False):
        strategy_manager = StrategyManager()
        
        # Verify AI validator is not initialized
        assert strategy_manager.ai_validator is None, "AI validator should be None when disabled"
        
        # Test normal strategy execution
        df = create_test_dataframe_with_indicators(100)
        signal = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        
        # Should generate LONG signal based on technical indicators
        assert signal.signal_type == SignalType.LONG, f"Expected LONG signal, got {signal.signal_type}"
        assert "LONG" in signal.reason, f"Expected LONG reason, got {signal.reason}"
        
        print("✅ AI disabled - normal strategy execution - PASSED")
    
    print("✅ AI Integration disabled tests passed!\n")


def test_ai_integration_enabled_approved():
    """Test AI integration when AI approves the trade"""
    print("🏛️ Testing AI Integration - ENABLED & APPROVED...")
    
    # Mock AI validator that approves trades
    mock_ai_validator = Mock()
    mock_ai_validator.validate_signal.return_value = ValidationResult(
        should_enter=True,
        confidence=8.5,
        reasoning="Strong bullish setup with institutional support",
        entry_level=50000.0,
        stop_loss=49500.0,
        take_profit=51500.0,
        analysis_time=2.5
    )
    
    with patch('strategy.strategies.USE_AI_VALIDATION', True):
        strategy_manager = StrategyManager()
        strategy_manager.ai_validator = mock_ai_validator  # Inject mock
        
        df = create_test_dataframe_with_indicators(100)
        signal = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        
        # Verify AI was called
        mock_ai_validator.validate_signal.assert_called_once()
        call_args = mock_ai_validator.validate_signal.call_args
        assert call_args[0][1] == "BTCUSDT", "AI should be called with correct symbol"
        assert call_args[0][2] == "LONG", "AI should be called with correct signal type"
        
        # Should generate LONG signal since AI approved
        assert signal.signal_type == SignalType.LONG, f"Expected LONG signal, got {signal.signal_type}"
        assert "AI_APPROVED" in signal.reason, f"Expected AI_APPROVED in reason, got {signal.reason}"
        
        print("✅ AI enabled & approved - trade executed - PASSED")
    
    print("✅ AI Integration enabled & approved tests passed!\n")


def test_ai_integration_enabled_rejected():
    """Test AI integration when AI rejects the trade"""
    print("🏛️ Testing AI Integration - ENABLED & REJECTED...")
    
    # Mock AI validator that rejects trades
    mock_ai_validator = Mock()
    mock_ai_validator.validate_signal.return_value = ValidationResult(
        should_enter=False,
        confidence=4.2,
        reasoning="Weak setup with mixed SMC signals, avoid entry",
        analysis_time=1.8
    )
    
    with patch('strategy.strategies.USE_AI_VALIDATION', True):
        strategy_manager = StrategyManager()
        strategy_manager.ai_validator = mock_ai_validator  # Inject mock
        
        df = create_test_dataframe_with_indicators(100)
        signal = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        
        # Verify AI was called
        mock_ai_validator.validate_signal.assert_called_once()
        
        # Should generate WAIT signal since AI rejected
        assert signal.signal_type == SignalType.WAIT, f"Expected WAIT signal, got {signal.signal_type}"
        assert "AI_REJECTED" in signal.reason, f"Expected AI_REJECTED in reason, got {signal.reason}"
        assert signal.entry_price is None, "Entry price should be None for rejected signals"
        
        print("✅ AI enabled & rejected - trade blocked - PASSED")
    
    print("✅ AI Integration enabled & rejected tests passed!\n")


def test_ai_integration_error_handling():
    """Test AI integration error handling"""
    print("🏛️ Testing AI Integration - ERROR HANDLING...")
    
    # Mock AI validator that raises exception
    mock_ai_validator = Mock()
    mock_ai_validator.validate_signal.side_effect = Exception("AI service unavailable")
    
    with patch('strategy.strategies.USE_AI_VALIDATION', True):
        strategy_manager = StrategyManager()
        strategy_manager.ai_validator = mock_ai_validator  # Inject mock
        
        df = create_test_dataframe_with_indicators(100)
        signal = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        
        # Verify AI was called
        mock_ai_validator.validate_signal.assert_called_once()
        
        # Should still generate LONG signal (fallback to technical analysis)
        assert signal.signal_type == SignalType.LONG, f"Expected LONG signal on AI error, got {signal.signal_type}"
        assert "LONG" in signal.reason, f"Expected LONG reason on AI error, got {signal.reason}"
        
        print("✅ AI error handling - fallback to technical analysis - PASSED")
    
    print("✅ AI Integration error handling tests passed!\n")


def test_ai_integration_short_signal():
    """Test AI integration with SHORT signals"""
    print("🏛️ Testing AI Integration - SHORT SIGNAL...")
    
    # Create bearish setup
    df = create_test_dataframe_with_indicators(100)
    df['squeeze_color'] = 'RED'  # Bearish squeeze
    df['MagicTrend_Color'] = 'RED'  # Bearish trend magic
    
    # Mock AI validator that approves SHORT
    mock_ai_validator = Mock()
    mock_ai_validator.validate_signal.return_value = ValidationResult(
        should_enter=True,
        confidence=8.0,
        reasoning="Strong bearish setup with institutional selling",
        analysis_time=2.1
    )
    
    with patch('strategy.strategies.USE_AI_VALIDATION', True):
        strategy_manager = StrategyManager()
        strategy_manager.ai_validator = mock_ai_validator
        
        signal = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        
        # Verify AI was called with SHORT signal
        call_args = mock_ai_validator.validate_signal.call_args
        assert call_args[0][2] == "SHORT", "AI should be called with SHORT signal type"
        
        # Should generate SHORT signal
        assert signal.signal_type == SignalType.SHORT, f"Expected SHORT signal, got {signal.signal_type}"
        assert "AI_APPROVED" in signal.reason, f"Expected AI_APPROVED in reason, got {signal.reason}"
        
        print("✅ AI integration with SHORT signals - PASSED")
    
    print("✅ AI Integration SHORT signal tests passed!\n")


def test_position_management_with_ai():
    """Test that existing position management works with AI integration"""
    print("🏛️ Testing Position Management with AI...")
    
    mock_ai_validator = Mock()
    mock_ai_validator.validate_signal.return_value = ValidationResult(
        should_enter=True,
        confidence=8.5,
        reasoning="Strong setup",
        analysis_time=2.0
    )
    
    with patch('strategy.strategies.USE_AI_VALIDATION', True):
        strategy_manager = StrategyManager()
        strategy_manager.ai_validator = mock_ai_validator
        
        df = create_test_dataframe_with_indicators(100)
        
        # First call should open position
        signal1 = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        assert signal1.signal_type == SignalType.LONG, "First signal should be LONG"
        
        # Verify position was opened
        assert "BTCUSDT" in strategy_manager.open_positions, "Position should be opened"
        position = strategy_manager.open_positions["BTCUSDT"]
        assert position['type'] == SignalType.LONG, "Position type should be LONG"
        
        # Second call with same conditions should return WAIT (position already open)
        signal2 = strategy_manager.squeeze_magic_strategy(df, "BTCUSDT")
        assert signal2.signal_type == SignalType.WAIT, "Second signal should be WAIT (position open)"
        assert signal2.reason == "LONG", "Reason should indicate existing LONG position"
        
        print("✅ Position management with AI integration - PASSED")
    
    print("✅ Position management tests passed!\n")


def main():
    """Run all AI integration tests"""
    print("🚀 AI INTEGRATION TESTS")
    print("=" * 50)
    
    try:
        test_ai_integration_disabled()
        test_ai_integration_enabled_approved()
        test_ai_integration_enabled_rejected()
        test_ai_integration_error_handling()
        test_ai_integration_short_signal()
        test_position_management_with_ai()
        
        print("🏛️ ALL AI INTEGRATION TESTS PASSED!")
        print("AI integration is working correctly!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"💀 AI INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)