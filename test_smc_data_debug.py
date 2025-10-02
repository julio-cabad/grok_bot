#!/usr/bin/env python3
"""
Test SMC Data Debug
Verifica qué datos SMC está recibiendo la IA
"""

import logging
import pandas as pd
from ai.ai_validator_ultra import AIValidatorUltra
from ai.modules.fast_smc_analyzer import FastSMCAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_smc_data_debug():
    """Test what SMC data is being sent to AI"""
    
    print("🔍 Testing SMC Data Debug...")
    
    # Create sample AVAX data similar to real market
    data = {
        'open': [30.45, 30.50, 30.55, 30.60] * 25,
        'high': [30.80, 30.85, 30.90, 30.95] * 25,
        'low': [30.20, 30.25, 30.30, 30.35] * 25,
        'close': [30.50, 30.55, 30.60, 30.65] * 25,
        'volume': [1000000, 1100000, 1200000, 1300000] * 25,
        'STOCH_K_14_3': [76.6] * 100,
        'STOCH_D_14_3': [75.0] * 100,
        'MACD_hist_12_26_9': [0.14] * 100,
        'MACD_12_26_9': [0.5] * 100,
        'MACD_signal_12_26_9': [0.36] * 100,
        'BB_upper_20': [31.00] * 100,
        'BB_middle_20': [30.50] * 100,
        'BB_lower_20': [30.00] * 100,
    }
    
    df = pd.DataFrame(data)
    
    print(f"📊 Sample AVAX data created: {df.shape}")
    print(f"💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    # Test SMC analysis directly
    print("\n🏛️ Testing SMC Analysis...")
    smc_analyzer = FastSMCAnalyzer()
    
    try:
        # Try the enhanced analysis first
        try:
            smc = smc_analyzer._analyze_enhanced_pandas(df)
            print("✅ Using enhanced pandas analysis")
        except:
            # Fallback to basic analysis
            smc = smc_analyzer.analyze_fast(df)
            print("✅ Using basic fast analysis")
        
        print("✅ SMC Analysis Results:")
        print(f"   Current Zone: {smc.get('current_zone', 'MISSING')}")
        print(f"   Current Price: ${smc.get('current_price', 0):.2f}")
        print(f"   Order Block Bull: ${smc.get('ob_bull', 0):.2f}")
        print(f"   Order Block Bear: ${smc.get('ob_bear', 0):.2f}")
        print(f"   Liquidity Support: ${smc.get('liq_support', 0):.2f}")
        print(f"   Liquidity Resistance: ${smc.get('liq_resistance', 0):.2f}")
        print(f"   FVG Bull: {len(smc.get('fvg_bull', []))} gaps")
        print(f"   FVG Bear: {len(smc.get('fvg_bear', []))} gaps")
        
        if smc.get('fvg_bull'):
            print(f"   FVG Bull Details: {smc['fvg_bull'][0]}")
        if smc.get('fvg_bear'):
            print(f"   FVG Bear Details: {smc['fvg_bear'][0]}")
            
    except Exception as e:
        print(f"❌ SMC Analysis failed: {e}")
        smc = None
    
    # Test tech indicators
    print("\n📈 Testing Tech Indicators...")
    validator = AIValidatorUltra()
    tech = validator._format_tech(df)
    
    print("✅ Tech Indicators:")
    print(f"   MACD Hist: {tech.get('macd_hist', 'MISSING')}")
    print(f"   MACD Cross: {tech.get('macd_x', 'MISSING')}")
    print(f"   Stoch K: {tech.get('stoch_k', 'MISSING')}")
    print(f"   BB Position: {tech.get('bb_pos', 'MISSING'):.1f}%")
    print(f"   BB Width: {tech.get('bb_width', 'MISSING'):.1f}%")
    print(f"   Close Price: ${tech.get('close_price', 'MISSING'):.2f}")
    
    # Test prompt building
    if smc:
        print("\n📝 Testing Prompt Building...")
        try:
            prompt = validator._build_minimal_prompt("AVAXUSDT", "SHORT", smc, tech, df)
            
            print("✅ Generated Prompt:")
            print("=" * 60)
            print(prompt)
            print("=" * 60)
            
            # Count SMC mentions
            smc_mentions = prompt.count("SMC") + prompt.count("Order Block") + prompt.count("Fair Value") + prompt.count("Liquidity")
            print(f"\n📊 SMC Data Points in Prompt: {smc_mentions}")
            
        except Exception as e:
            print(f"❌ Prompt building failed: {e}")

if __name__ == "__main__":
    test_smc_data_debug()