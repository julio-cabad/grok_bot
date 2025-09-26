#!/usr/bin/env python3
"""
Simple Risk Manager Test - Spartan Trading System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.modules.risk_manager import RiskManager
from ai.data_models import AIValidatorConfig

def test_millionaire_risk_manager():
    print("🏛️ Testing MILLIONAIRE Risk Manager...")
    
    # Setup
    config = AIValidatorConfig(
        default_risk_percentage=1.0,
        min_risk_reward_ratio=1.0  # Adjust for this test
    )
    risk_manager = RiskManager(config)
    account_balance = 100000.0  # $100k account
    
    # ETHUSDT SHORT scenario
    entry_price = 3932.08
    stop_loss = 3993.49
    signal_type = "SHORT"
    
    # Calculate comprehensive metrics
    metrics = risk_manager.calculate_comprehensive_risk_metrics(
        entry_price, stop_loss, signal_type, account_balance, 
        volatility_level="MEDIUM"
    )
    
    print(f"📊 Entry: ${entry_price:.2f} | SL: ${stop_loss:.2f}")
    print(f"💰 Position Size: {metrics.optimal_position_size:.6f} units")
    print(f"💵 Position Value: ${entry_price * metrics.optimal_position_size:.2f}")
    print(f"🛑 Risk Amount: ${metrics.risk_amount_usd:.2f} ({metrics.risk_percentage:.2f}%)")
    
    print("🎯 Take Profits:")
    for i, (tp, rr, profit) in enumerate(zip(metrics.tp_levels, metrics.tp_risk_rewards, metrics.tp_profit_amounts)):
        print(f"   TP{i+1}: ${tp:.2f} (R:R {rr:.1f}) = ${profit:.2f} profit")
    
    total_profit = metrics.get_total_profit_potential()
    print(f"💎 Total Profit Potential: ${total_profit:.2f}")
    
    # Validate the trade
    is_valid, issues = risk_manager.validate_trade_risk(metrics)
    print(f"✅ Trade Valid: {is_valid}")
    
    if issues:
        for issue in issues:
            print(f"⚠️ Issue: {issue}")
    
    # Test assertions
    assert metrics.optimal_position_size > 0, "Position size should be positive"
    assert len(metrics.tp_levels) == 3, "Should have 3 take profit levels"
    assert all(rr >= 1.0 for rr in metrics.tp_risk_rewards), "All R:R ratios should be >= 1.0"
    assert total_profit > metrics.risk_amount_usd, "Total profit should exceed risk"
    assert is_valid == True, "Trade should be valid"
    
    print("🏆 MILLIONAIRE RISK MANAGER: SUCCESS!")
    return True

if __name__ == "__main__":
    test_millionaire_risk_manager()