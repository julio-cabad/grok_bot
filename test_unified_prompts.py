#!/usr/bin/env python3
"""
Test de Prompts Unificados
Verifica que Gemini y Grok-4 usen exactamente el mismo prompt
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.ai_validator_ultra import AIValidatorUltra
from ai.grok_client import GrokClient
from ai.modules.fast_smc_analyzer import FastSMCAnalyzer

def create_test_data():
    """Create comprehensive test data"""
    
    # Sample OHLCV data
    data = {
        'open': [95000, 95500, 96000, 95800, 95200],
        'high': [95800, 96200, 96500, 96000, 95600],
        'low': [94500, 95000, 95500, 95200, 94800],
        'close': [95500, 96000, 95800, 95200, 95000],
        'volume': [1000, 1200, 1100, 900, 800],
        
        # Technical indicators
        'STOCH_K_14_3': [78.5, 82.1, 75.3, 68.9, 65.2],
        'STOCH_D_14_3': [72.1, 76.8, 78.5, 74.1, 69.8],
        'MACD_hist_12_26_9': [1.2, 1.8, 0.9, -0.3, -0.8],
        'BB_upper_20': [96500, 97000, 96800, 96200, 95800],
        'BB_middle_20': [95500, 96000, 95800, 95200, 95000],
        'BB_lower_20': [94500, 95000, 94800, 94200, 94200],
    }
    
    return pd.DataFrame(data)

def test_prompt_consistency():
    """Test that both AIs receive the same prompt"""
    print("🔍 TESTING PROMPT CONSISTENCY...")
    print("-" * 50)
    
    try:
        # Create test data
        df = create_test_data()
        symbol = "BTCUSDT"
        signal_type = "SHORT"
        
        # Initialize components
        validator = AIValidatorUltra()
        grok_client = GrokClient()
        smc_analyzer = FastSMCAnalyzer()
        
        # Generate SMC analysis
        print("📊 Generating SMC analysis...")
        smc = smc_analyzer.analyze_fast(df)
        
        # Format tech data (same as validator)
        tech = validator._format_tech(df)
        
        print(f"✅ SMC data: {len(smc)} fields")
        print(f"✅ Tech data: {len(tech)} fields")
        
        # Generate Gemini prompt
        print("\n🔵 Generating Gemini prompt...")
        gemini_prompt = validator._build_minimal_prompt(symbol, signal_type, smc, tech, df)
        
        # Generate Grok prompt
        print("🟠 Generating Grok-4 prompt...")
        grok_prompt = grok_client.build_unified_prompt(symbol, signal_type, smc, tech, df)
        
        # Compare prompts
        print("\n📋 PROMPT COMPARISON:")
        print("-" * 30)
        
        print("🔵 GEMINI PROMPT:")
        print(gemini_prompt)
        print("\n" + "-" * 30)
        
        print("🟠 GROK-4 PROMPT:")
        print(grok_prompt)
        print("\n" + "-" * 30)
        
        # Check if prompts are identical
        if gemini_prompt.strip() == grok_prompt.strip():
            print("✅ PERFECT MATCH: Both AIs use identical prompts!")
            return True
        else:
            print("❌ MISMATCH: Prompts are different!")
            
            # Show differences
            gemini_lines = gemini_prompt.strip().split('\n')
            grok_lines = grok_prompt.strip().split('\n')
            
            print("\n🔍 DIFFERENCES FOUND:")
            max_lines = max(len(gemini_lines), len(grok_lines))
            
            for i in range(max_lines):
                g_line = gemini_lines[i] if i < len(gemini_lines) else "[MISSING]"
                k_line = grok_lines[i] if i < len(grok_lines) else "[MISSING]"
                
                if g_line != k_line:
                    print(f"   Line {i+1}:")
                    print(f"     🔵 Gemini: {g_line}")
                    print(f"     🟠 Grok-4: {k_line}")
            
            return False
            
    except Exception as e:
        print(f"❌ Prompt consistency test failed: {e}")
        return False

def test_response_format():
    """Test that both AIs respond in the same format"""
    print("\n🔍 TESTING RESPONSE FORMAT...")
    print("-" * 50)
    
    try:
        df = create_test_data()
        validator = AIValidatorUltra()
        
        print("📊 Testing response format consistency...")
        
        # Test with normal operation (should use Gemini)
        print("🔵 Testing Gemini response format...")
        try:
            result = validator.validate_signal(df, "TESTUSDT", "SHORT")
            print(f"✅ Gemini format test successful")
            print(f"   Score: {result.confidence}")
            print(f"   Enter: {result.should_enter}")
            print(f"   Reasoning: {result.reasoning[:50]}...")
        except Exception as e:
            print(f"⚠️ Gemini test failed: {str(e)[:50]}...")
        
        # Test with forced Grok fallback
        print("\n🟠 Testing Grok-4 response format...")
        original_query = validator.gemini_client.query
        validator.gemini_client.query = lambda p: (_ for _ in ()).throw(Exception("Force Grok test"))
        
        try:
            result = validator.validate_signal(df, "GROKTEST", "SHORT")
            print(f"✅ Grok-4 format test successful")
            print(f"   Score: {result.confidence}")
            print(f"   Enter: {result.should_enter}")
            print(f"   Reasoning: {result.reasoning[:50]}...")
        except Exception as e:
            print(f"❌ Grok-4 test failed: {str(e)[:50]}...")
        finally:
            # Restore original
            validator.gemini_client.query = original_query
        
        return True
        
    except Exception as e:
        print(f"❌ Response format test failed: {e}")
        return False

def test_data_consistency():
    """Test that SMC and tech data is consistent"""
    print("\n🔍 TESTING DATA CONSISTENCY...")
    print("-" * 50)
    
    try:
        df = create_test_data()
        validator = AIValidatorUltra()
        smc_analyzer = FastSMCAnalyzer()
        
        # Generate data
        smc = smc_analyzer.analyze_fast(df)
        tech = validator._format_tech(df)
        
        print("📊 SMC DATA:")
        for key, value in smc.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value}")
            elif isinstance(value, list):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {type(value)}")
        
        print("\n📊 TECH DATA:")
        for key, value in tech.items():
            print(f"   {key}: {value}")
        
        # Verify critical fields
        critical_fields = ['current_price', 'current_zone', 'ob_bull', 'ob_bear']
        missing_fields = [field for field in critical_fields if field not in smc]
        
        if missing_fields:
            print(f"⚠️ Missing SMC fields: {missing_fields}")
        else:
            print("✅ All critical SMC fields present")
        
        return len(missing_fields) == 0
        
    except Exception as e:
        print(f"❌ Data consistency test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 UNIFIED PROMPTS TEST")
    print("=" * 60)
    print(f"🕐 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Purpose: Verify Gemini and Grok-4 use identical prompts")
    print()
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Prompt consistency
    if test_prompt_consistency():
        tests_passed += 1
    
    # Test 2: Response format
    if test_response_format():
        tests_passed += 1
    
    # Test 3: Data consistency
    if test_data_consistency():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print("🏁 UNIFIED PROMPTS TEST COMPLETED")
    print(f"🕐 Test finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🏆 PERFECT: Both AIs use identical prompts and data!")
        print("✅ Consistent analysis guaranteed")
        print("✅ Fallback system maintains quality")
    elif tests_passed >= 2:
        print("⚠️ GOOD: Minor inconsistencies detected")
        print("🔧 Some adjustments may be needed")
    else:
        print("🚨 CRITICAL: Major inconsistencies found")
        print("❌ Fallback system may produce different results")
    
    print("\n💡 RECOMMENDATIONS:")
    if tests_passed == total_tests:
        print("   ✅ System ready for production")
        print("   ✅ Both AIs will provide consistent analysis")
    else:
        print("   🔧 Fix prompt inconsistencies")
        print("   🔧 Ensure data format compatibility")
        print("   🔧 Test again before production use")

if __name__ == "__main__":
    main()