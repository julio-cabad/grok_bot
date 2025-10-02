#!/usr/bin/env python3
"""
Debug AI Response - Investigar por qué todos los scores son 5.0
"""

import pandas as pd
import numpy as np
from ai.ai_validator_ultra import AIValidatorUltra
from ai.gemini_client import GeminiClient
import logging

# Configure logging to see everything
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_ai_direct():
    """Test AI directly with a simple prompt"""
    print("🔍 TESTING AI DIRECTLY...")
    
    client = GeminiClient()
    
    simple_prompt = """
BTCUSDT SHORT $67000
SMC: zone=EQUILIBRIUM ob_b=0 ob_s=0 fvg= liq_s=68340 liq_b=65660
Tech: macd_hist=-50 macd_x=NONE stoch_k=75 bb_pos=60% bb_width=2.5%
Weight: SMC 40% MACD 25% STOCH 20% BB 15%

Respond EXACTLY in this format:
SCORE: [0-10 number]
ENTER: [YES/NO]
REASONING: [brief analysis]
ENTRY: [price]
SL: [stop loss]
TP: [take profit]
"""
    
    try:
        response = client.query(simple_prompt)
        print(f"✅ AI Response:")
        print(f"'{response}'")
        print(f"Response type: {type(response)}")
        print(f"Response length: {len(response)}")
        return response
    except Exception as e:
        print(f"❌ AI Direct test failed: {e}")
        return None

def test_validator_parsing():
    """Test the validator parsing logic"""
    print("\n🔍 TESTING VALIDATOR PARSING...")
    
    # Mock AI responses to test parsing
    test_responses = [
        "SCORE: 7.5\nENTER: YES\nREASONING: Good confluence\nENTRY: 67000\nSL: 67500\nTP: 66000",
        "Score: 6.2\nEnter: NO\nReasoning: Weak signal\nEntry: 67000\nSL: 67500\nTP: 66000",
        "The score is 8.1 and I recommend YES to enter",
        "Rating: 4.5 - Not recommended",
        "SCORE: 5.0\nENTER: NO\nREASONING: AI completed"
    ]
    
    validator = AIValidatorUltra()
    
    for i, response in enumerate(test_responses):
        print(f"\n--- Test Response {i+1} ---")
        print(f"Input: '{response}'")
        
        try:
            result = validator._parse_response(response, "TESTUSDT", "SHORT")
            print(f"Parsed Score: {result.confidence}")
            print(f"Parsed Enter: {result.should_enter}")
            print(f"Parsed Reasoning: {result.reasoning}")
        except Exception as e:
            print(f"❌ Parsing failed: {e}")

def test_with_real_data():
    """Test with data similar to what the bot uses"""
    print("\n🔍 TESTING WITH REAL-LIKE DATA...")
    
    # Create realistic data
    data = {
        'close': [67000, 67100, 66900, 66800, 67200],
        'volume': [1000000, 1100000, 900000, 800000, 1200000],
        'MACD_12_26_9': [50, 30, -20, -50, 10],
        'MACD_signal_12_26_9': [20, 40, -10, -30, 20],
        'MACD_hist_12_26_9': [30, -10, -10, -20, -10],
        'STOCH_K_14_3': [60, 65, 45, 30, 55],
        'BB_upper_20': [68000, 68100, 67900, 67800, 68200],
        'BB_middle_20': [67000, 67100, 66900, 66800, 67200],
        'BB_lower_20': [66000, 66100, 65900, 65800, 66200]
    }
    df = pd.DataFrame(data)
    
    validator = AIValidatorUltra()
    
    try:
        print("Testing BTCUSDT SHORT...")
        result = validator.validate_signal(df, 'BTCUSDT', 'SHORT')
        print(f"✅ Final Result:")
        print(f"   Score: {result.confidence}")
        print(f"   Enter: {result.should_enter}")
        print(f"   Reasoning: {result.reasoning}")
        print(f"   Time: {result.analysis_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 AI RESPONSE DEBUG INVESTIGATION")
    print("=" * 50)
    
    # Test 1: Direct AI call
    ai_response = test_ai_direct()
    
    # Test 2: Parsing logic
    test_validator_parsing()
    
    # Test 3: Full validator with realistic data
    test_with_real_data()
    
    print("\n" + "=" * 50)
    print("🎯 DEBUG INVESTIGATION COMPLETE")

if __name__ == "__main__":
    main()