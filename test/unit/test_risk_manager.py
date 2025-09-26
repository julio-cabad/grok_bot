#!/usr/bin/env python3
"""
Risk Manager Tests - Spartan Trading System
Comprehensive tests for the millionaire risk management system
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.modules.risk_manager import RiskManager
from ai.data_models import RiskMetrics, AIValidatorConfig


class TestRiskManager:
    """Test suite for the Risk Manager - ensuring maximum profitability"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = AIValidatorConfig(
            default_risk_percentage=1.0,
            min_risk_reward_ratio=1.5,
            position_size_limits=(0.1, 10.0)
        )
        self.risk_manager = RiskManager(self.config)
        self.account_balance = 10000.0  # $10k account
    
    def test_position_size_calculation_long(self):
        """Test position sizing for LONG trades"""
        entry_price = 100.0
        stop_loss = 95.0  # 5% risk
        
        position_size = self.risk_manager.calculate_position_size(
            entry_price, stop_loss, self.account_balance, 1.0
        )
        
        # Expected: $100 risk / $5 risk per unit = 20 units
        expected_size = 20.0
        assert abs(position_size - expected_size) < 0.01
        
        # Verify risk amount
        risk_per_unit = abs(entry_price - stop_loss)
        actual_risk = position_size * risk_per_unit
        expected_risk = self.account_balance * 0.01  # 1%
        assert abs(actual_risk - expected_risk) < 0.01
    
    def test_position_size_calculation_short(self):
        """Test position sizing for SHORT trades"""
        entry_price = 100.0
        stop_loss = 105.0  # 5% risk
        
        position_size = self.risk_manager.calculate_position_size(
            entry_price, stop_loss, self.account_balance, 2.0
        )
        
        # Expected: $200 risk / $5 risk per unit = 40 units
        expected_size = 40.0
        assert abs(position_size - expected_size) < 0.01
    
    def test_multiple_take_profits_long(self):
        """Test multiple TP generation for LONG trades"""
        entry_price = 100.0
        stop_loss = 95.0  # $5 risk
        
        tps = self.risk_manager.generate_take_profit_levels(
            entry_price, stop_loss, "LONG"
        )
        
        # Expected TPs: 105 (1:1), 110 (1:2), 115 (1:3)
        expected_tps = [105.0, 110.0, 115.0]
        assert len(tps) == 3
        for actual, expected in zip(tps, expected_tps):
            assert abs(actual - expected) < 0.01
    
    def test_multiple_take_profits_short(self):
        """Test multiple TP generation for SHORT trades"""
        entry_price = 100.0
        stop_loss = 105.0  # $5 risk
        
        tps = self.risk_manager.generate_take_profit_levels(
            entry_price, stop_loss, "SHORT"
        )
        
        # Expected TPs: 95 (1:1), 90 (1:2), 85 (1:3)
        expected_tps = [95.0, 90.0, 85.0]
        assert len(tps) == 3
        for actual, expected in zip(tps, expected_tps):
            assert abs(actual - expected) < 0.01
    
    def test_risk_reward_ratios(self):
        """Test risk/reward ratio calculations"""
        entry_price = 100.0
        stop_loss = 95.0
        take_profits = [105.0, 110.0, 115.0]
        
        ratios = self.risk_manager.calculate_risk_reward_ratios(
            entry_price, stop_loss, take_profits
        )
        
        expected_ratios = [1.0, 2.0, 3.0]
        assert len(ratios) == 3
        for actual, expected in zip(ratios, expected_ratios):
            assert abs(actual - expected) < 0.01
    
    def test_volatility_adjustment_high(self):
        """Test position size reduction in high volatility"""
        base_size = 100.0
        
        adjusted_size = self.risk_manager.adjust_for_volatility(
            base_size, "HIGH"
        )
        
        # Should be reduced by 50% (default config)
        expected_size = base_size * 0.5
        assert abs(adjusted_size - expected_size) < 0.01
    
    def test_volatility_adjustment_low(self):
        """Test position size increase in low volatility"""
        base_size = 100.0
        
        adjusted_size = self.risk_manager.adjust_for_volatility(
            base_size, "LOW"
        )
        
        # Should be increased by 20% (default config)
        expected_size = base_size * 1.2
        assert abs(adjusted_size - expected_size) < 0.01
    
    def test_comprehensive_risk_metrics(self):
        """Test comprehensive risk metrics calculation"""
        entry_price = 100.0
        stop_loss = 95.0
        signal_type = "LONG"
        
        metrics = self.risk_manager.calculate_comprehensive_risk_metrics(
            entry_price, stop_loss, signal_type, self.account_balance
        )
        
        # Verify basic calculations
        assert metrics.optimal_position_size > 0
        assert len(metrics.tp_levels) == 3
        assert len(metrics.tp_risk_rewards) == 3
        assert len(metrics.tp_profit_amounts) == 3
        assert metrics.risk_amount_usd > 0
        assert metrics.risk_percentage > 0
        
        # Verify TP levels are correct
        expected_tps = [105.0, 110.0, 115.0]
        for actual, expected in zip(metrics.tp_levels, expected_tps):
            assert abs(actual - expected) < 0.01
        
        # Verify R:R ratios
        expected_ratios = [1.0, 2.0, 3.0]
        for actual, expected in zip(metrics.tp_risk_rewards, expected_ratios):
            assert abs(actual - expected) < 0.01
    
    def test_trade_risk_validation_pass(self):
        """Test trade risk validation - should pass"""
        metrics = RiskMetrics(
            optimal_position_size=20.0,
            max_position_size=100.0,
            min_position_size=1.0,
            risk_amount_usd=100.0,
            risk_percentage=1.0,
            tp_levels=[105.0, 110.0, 115.0],
            tp_risk_rewards=[1.0, 2.0, 3.0],
            tp_profit_amounts=[100.0, 200.0, 300.0]
        )
        
        is_valid, issues = self.risk_manager.validate_trade_risk(metrics)
        
        assert is_valid == True
        assert len(issues) == 0
    
    def test_trade_risk_validation_fail_low_rr(self):
        """Test trade risk validation - should fail due to low R:R"""
        metrics = RiskMetrics(
            optimal_position_size=20.0,
            max_position_size=100.0,
            min_position_size=1.0,
            risk_amount_usd=100.0,
            risk_percentage=1.0,
            tp_levels=[102.0],  # Low R:R
            tp_risk_rewards=[0.4],  # Below minimum 1.5
            tp_profit_amounts=[40.0]
        )
        
        is_valid, issues = self.risk_manager.validate_trade_risk(metrics)
        
        assert is_valid == False
        assert len(issues) > 0
        assert "Risk/reward ratio" in issues[0]
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with zero entry price
        size = self.risk_manager.calculate_position_size(0, 95.0, 10000.0)
        assert size == 0.0
        
        # Test with equal entry and stop loss
        size = self.risk_manager.calculate_position_size(100.0, 100.0, 10000.0)
        assert size == 0.0
        
        # Test with negative account balance
        size = self.risk_manager.calculate_position_size(100.0, 95.0, -1000.0)
        assert size == 0.0
        
        # Test invalid signal type
        tps = self.risk_manager.generate_take_profit_levels(100.0, 95.0, "INVALID")
        assert len(tps) == 0
    
    def test_profit_potential_calculation(self):
        """Test profit potential calculations"""
        entry_price = 100.0
        stop_loss = 95.0
        
        metrics = self.risk_manager.calculate_comprehensive_risk_metrics(
            entry_price, stop_loss, "LONG", self.account_balance
        )
        
        # Calculate expected total profit
        total_profit = metrics.get_total_profit_potential()
        assert total_profit > 0
        
        # Verify weighted R:R calculation
        weighted_rr = metrics.get_weighted_risk_reward()
        assert weighted_rr > 0
        # Should be approximately 2.0 (weighted average of 1, 2, 3)
        assert abs(weighted_rr - 2.0) < 0.1


def test_millionaire_scenario():
    """Test a realistic millionaire trading scenario"""
    print("\n🏛️ Testing MILLIONAIRE Trading Scenario...")
    
    # Setup
    config = AIValidatorConfig(default_risk_percentage=1.0)
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
    
    print(f"🎯 Take Profits:")
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
    
    # Assertions for millionaire scenario
    assert metrics.optimal_position_size > 0
    assert len(metrics.tp_levels) == 3
    assert all(rr >= 1.0 for rr in metrics.tp_risk_rewards)
    assert total_profit > metrics.risk_amount_usd  # Profit > Risk
    assert is_valid == True
    
    print("🏆 MILLIONAIRE SCENARIO: PASSED!")


if __name__ == "__main__":
    # Run the millionaire test
    test_millionaire_scenario()
    
    # Run all tests
    pytest.main([__file__, "-v"])