#!/usr/bin/env python3
"""
Test Score Variety
Verifica si las IAs pueden dar scores diferentes a 5
"""

import logging
import pandas as pd
from ai.ai_validator_ultra import AIValidatorUltra

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_scenario(scenario_name, stoch_k, macd_hist, bb_pos, price_base=50000):
    """Create different market scenarios"""
    
    data = {
        'open': [price_base] * 100,
        'high': [price_base * 1.01] * 100,
        'low': [price_base * 0.99] * 100,
        'close': [price_base] * 100,
        'volume': [1000000] * 100,
        'STOCH_K_14_3': [stoch_k] * 100,
        'STOCH_D_14_3': [stoch_k - 5] * 100,  # D slightly lower
        'MACD_hist_12_26_9': [macd_hist] * 100,
        'MACD_12_26_9': [macd_hist + 0.1] * 100,
        'MACD_signal_12_26_9': [0.1] * 100,
        'BB_upper_20': [price_base * 1.02] * 100,
        'BB_middle_20': [price_base] * 100,
        'BB_lower_20': [price_base * 0.98] * 100,
    }
    
    # Adjust BB position
    if bb_pos > 50:
        data['close'] = [price_base * (1 + (bb_pos - 50) / 5000)] * 100
    else:
        data['close'] = [price_base * (1 - (50 - bb_pos) / 5000)] * 100
    
    return scenario_name, pd.DataFrame(data)

def test_score_variety():
    """Test different market scenarios to see score variety"""
    
    print("🧪 Testing Score Variety...")
    
    # Create different scenarios
    scenarios = [
        # Perfect SHORT setup
        create_test_scenario("PERFECT_SHORT", stoch_k=85, macd_hist=-0.5, bb_pos=95),
        
        # Perfect LONG setup  
        create_test_scenario("PERFECT_LONG", stoch_k=25, macd_hist=0.5, bb_pos=15),
        
        # Terrible SHORT (bullish everything)
        create_test_scenario("TERRIBLE_SHORT", stoch_k=25, macd_hist=0.8, bb_pos=15),
        
        # Terrible LONG (bearish everything)
        create_test_scenario("TERRIBLE_LONG", stoch_k=85, macd_hist=-0.8, bb_pos=95),
        
        # Neutral/Mixed signals
        create_test_scenario("MIXED_SIGNALS", stoch_k=50, macd_hist=0.0, bb_pos=50),
    ]
    
    validator = AIValidatorUltra()
    
    for scenario_name, df in scenarios:
        print(f"\n📊 Testing {scenario_name}:")
        print(f"   Stoch K: {df['STOCH_K_14_3'].iloc[0]}")
        print(f"   MACD Hist: {df['MACD_hist_12_26_9'].iloc[0]}")
        print(f"   BB Pos: {((df['close'].iloc[0] - df['BB_lower_20'].iloc[0]) / (df['BB_upper_20'].iloc[0] - df['BB_lower_20'].iloc[0])) * 100:.1f}%")
        
        try:
            # Test SHORT
            result_short = validator.validate_signal(df, "TESTUSDT", "SHORT")
            print(f"   SHORT Score: {result_short.confidence:.1f} | Enter: {result_short.should_enter}")
            
            # Test LONG  
            result_long = validator.validate_signal(df, "TESTUSDT", "LONG")
            print(f"   LONG Score:  {result_long.confidence:.1f} | Enter: {result_long.should_enter}")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")

if __name__ == "__main__":
    test_score_variety()