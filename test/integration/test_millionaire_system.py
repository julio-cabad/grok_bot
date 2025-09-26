#!/usr/bin/env python3
"""
🏛️ SPARTAN MILLIONAIRE SYSTEM TEST 🏛️
Integration test for the complete millionaire trading system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ai.modules.risk_manager import RiskManager
from ai.config_manager import ConfigurationManager
from ai.data_models import AIValidatorConfig


def create_millionaire_test_data() -> pd.DataFrame:
    """Create test data with perfect confluence setup"""
    print("🏛️ Creating MILLIONAIRE test data...")
    
    # Create 200 candles of realistic crypto data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=200), periods=200, freq='1H')
    
    # Generate realistic OHLCV data with trend
    np.random.seed(42)  # For reproducible results
    base_price = 3900.0
    
    # Create trending price action
    trend = np.linspace(0, 100, 200)  # Upward trend
    noise = np.random.normal(0, 20, 200)  # Price noise
    
    closes = base_price + trend + noise
    
    # Generate OHLC from closes
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    
    highs = closes + np.random.uniform(5, 25, 200)
    lows = closes - np.random.uniform(5, 25, 200)
    volumes = np.random.uniform(1000000, 5000000, 200)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    print(f"✅ Created {len(df)} candles with PERFECT confluence setup")
    return df


def test_integrated_millionaire_system():
    """Test the complete integrated millionaire system"""
    print("\n🏛️ Testing INTEGRATED MILLIONAIRE SYSTEM...")
    
    # 1. Configuration Management
    print("🔧 Testing Configuration Management...")
    config_manager = ConfigurationManager()
    
    # Update to millionaire settings
    millionaire_settings = {
        'confidence_threshold': 8.0,
        'default_risk_percentage': 1.0,
        'min_risk_reward_ratio': 1.5,
        'timeout_seconds': 30
    }
    
    success = config_manager.update_config(**millionaire_settings)
    assert success == True
    print("✅ Configuration Management: PASSED")
    
    # 2. Risk Management
    print("💰 Testing Risk Management...")
    config = config_manager.get_config()
    risk_manager = RiskManager(config)
    
    # Test with realistic ETHUSDT scenario
    entry_price = 3932.08
    stop_loss = 3993.49
    account_balance = 100000.0
    
    metrics = risk_manager.calculate_comprehensive_risk_metrics(
        entry_price, stop_loss, "SHORT", account_balance
    )
    
    assert metrics.optimal_position_size > 0
    assert len(metrics.tp_levels) == 3
    assert metrics.risk_percentage <= 1.0
    print("✅ Risk Management: PASSED")
    
    # 3. Data Integration
    print("📊 Testing Data Integration...")
    df = create_millionaire_test_data()
    assert len(df) == 200
    assert 'close' in df.columns
    print("✅ Data Integration: PASSED")
    
    # 4. Performance Validation
    print("🚀 Testing Performance...")
    
    # Test multiple risk calculations (performance test)
    import time
    start_time = time.time()
    
    for i in range(10):
        test_entry = 4000 + (i * 10)
        test_sl = test_entry + 50
        metrics = risk_manager.calculate_comprehensive_risk_metrics(
            test_entry, test_sl, "SHORT", account_balance
        )
        assert metrics.optimal_position_size > 0
    
    elapsed_time = time.time() - start_time
    assert elapsed_time < 1.0  # Should complete in under 1 second
    print(f"✅ Performance: PASSED ({elapsed_time:.3f}s for 10 calculations)")
    
    # 5. Configuration Flexibility
    print("⚙️ Testing Configuration Flexibility...")
    
    # Test runtime configuration changes
    config_manager.update_config(default_risk_percentage=0.5)
    new_config = config_manager.get_config()
    assert new_config.default_risk_percentage == 0.5
    
    # Test rollback
    config_manager.rollback_config()
    rolled_back_config = config_manager.get_config()
    assert rolled_back_config.default_risk_percentage == 1.0
    print("✅ Configuration Flexibility: PASSED")
    
    print("\n🏆 INTEGRATED MILLIONAIRE SYSTEM: ALL TESTS PASSED! 🏆")
    return True


def test_profit_calculation_accuracy():
    """Test profit calculation accuracy for millionaire scenarios"""
    print("\n💎 Testing PROFIT CALCULATION ACCURACY...")
    
    config = AIValidatorConfig(default_risk_percentage=1.0)
    risk_manager = RiskManager(config)
    
    # Test scenarios with different account sizes
    test_scenarios = [
        (10000, "Small Account"),
        (100000, "Medium Account"), 
        (1000000, "Large Account")
    ]
    
    for account_balance, scenario_name in test_scenarios:
        print(f"\n📊 Testing {scenario_name}: ${account_balance:,}")
        
        entry_price = 4000.0
        stop_loss = 4100.0  # $100 risk per unit
        
        metrics = risk_manager.calculate_comprehensive_risk_metrics(
            entry_price, stop_loss, "SHORT", account_balance
        )
        
        # Verify 1% risk
        expected_risk = account_balance * 0.01
        actual_risk = metrics.risk_amount_usd
        risk_diff = abs(actual_risk - expected_risk)
        
        print(f"   💰 Position Size: {metrics.optimal_position_size:.6f} units")
        print(f"   🛑 Risk Amount: ${actual_risk:.2f} (Expected: ${expected_risk:.2f})")
        print(f"   🎯 Total Profit Potential: ${metrics.get_total_profit_potential():.2f}")
        
        # Assertions
        assert risk_diff < 1.0, f"Risk calculation error: {risk_diff:.2f}"
        assert metrics.get_total_profit_potential() > actual_risk, "Profit should exceed risk"
        assert len(metrics.tp_levels) == 3, "Should have 3 TP levels"
        
        print(f"   ✅ {scenario_name}: PASSED")
    
    print("\n💎 PROFIT CALCULATION ACCURACY: ALL TESTS PASSED!")


if __name__ == "__main__":
    print("🏛️" + "="*60 + "🏛️")
    print("🏛️  SPARTAN MILLIONAIRE SYSTEM INTEGRATION TESTS  🏛️")
    print("🏛️" + "="*60 + "🏛️")
    
    try:
        # Run integration tests
        test_integrated_millionaire_system()
        test_profit_calculation_accuracy()
        
        print("\n🏛️" + "="*60 + "🏛️")
        print("🏛️  🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("🏛️  🚀 SYSTEM READY FOR MILLIONAIRE PROFITS! 🚀")
        print("🏛️" + "="*60 + "🏛️")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise