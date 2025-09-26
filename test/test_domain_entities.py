#!/usr/bin/env python3
"""
Test Domain Entities
Validates all domain entities work correctly
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.domain.entities.smc_analysis import SMCAnalysis, OrderBlock, FairValueGap, LiquiditySweep
from src.core.domain.entities.signal import Signal
from src.core.domain.entities.market_data import MultiTimeframeData, MarketContext, OHLCV, TimeframeData
from src.core.domain.entities.position import Position, PositionId, PositionSide, PositionStatus
from src.core.domain.value_objects.money import Money
from src.core.domain.value_objects.symbol import Symbol
from src.core.domain.value_objects.timeframe import Timeframe
from src.core.domain.value_objects.signal_types import SignalType, SignalStrength, SMCRecommendation


def test_value_objects():
    """Test all value objects"""
    print("🏛️ Testing Value Objects...")
    
    # Test Money
    money1 = Money.from_float(100.50)
    money2 = Money.from_float(50.25)
    assert money1 + money2 == Money.from_float(150.75)
    assert money1 > money2
    print("✅ Money value object - PASSED")
    
    # Test Symbol
    symbol = Symbol("BTCUSDT")
    assert symbol.get_base_asset() == "BTC"
    assert symbol.get_quote_asset() == "USDT"
    assert symbol.is_crypto_pair()
    print("✅ Symbol value object - PASSED")
    
    # Test Timeframe
    tf1 = Timeframe("1h")
    tf2 = Timeframe("4h")
    assert tf2.is_higher_than(tf1)
    assert tf1.to_minutes() == 60
    print("✅ Timeframe value object - PASSED")
    
    # Test Signal Types
    assert SignalType.LONG.is_entry_signal()
    assert SignalStrength.from_confidence(0.8) == SignalStrength.STRONG
    print("✅ Signal Types - PASSED")
    
    print("✅ All Value Objects tests passed!\n")


def test_smc_analysis():
    """Test SMC Analysis entity"""
    print("🏛️ Testing SMC Analysis Entity...")
    
    # Create test order blocks
    order_block = OrderBlock(
        price_level=Decimal('50000'),
        volume=Decimal('1000'),
        timestamp=datetime.utcnow(),
        block_type="BULLISH",
        strength="STRONG"
    )
    
    # Create test fair value gap
    fvg = FairValueGap(
        upper_price=Decimal('50100'),
        lower_price=Decimal('49900'),
        timestamp=datetime.utcnow(),
        gap_type="BULLISH"
    )
    
    # Create test liquidity sweep
    sweep = LiquiditySweep(
        price_level=Decimal('49800'),
        timestamp=datetime.utcnow(),
        sweep_type="BULLISH",
        volume=Decimal('500'),
        significance="HIGH"
    )
    
    # Create SMC Analysis with high scores (should give final_score >= 8.5)
    smc = SMCAnalysis(
        liquidity_score=Decimal('9.0'),  # 9.0 * 0.3 = 2.7
        structure_score=Decimal('9.0'),  # 9.0 * 0.3 = 2.7
        momentum_score=Decimal('8.5'),   # 8.5 * 0.2 = 1.7
        risk_reward_score=Decimal('9.0'), # 9.0 * 0.2 = 1.8
                                         # Total = 8.9
        order_blocks_detected=[order_block],
        fair_value_gaps=[fvg],
        liquidity_sweeps=[sweep],
        analysis_reasoning="Strong bullish setup with institutional support",
        confidence_level=Decimal('0.85'),
        timestamp=datetime.utcnow(),
        market_structure="BULLISH",
        trend_strength="STRONG",
        metadata={}
    )
    
    # Test calculations
    assert smc.final_score > Decimal('8.0')
    assert smc.get_recommendation() == SMCRecommendation.STRONG_BUY
    assert smc.should_enter_trade()
    assert smc.get_entry_confidence() == "HIGH"
    
    print("✅ SMC Analysis Entity - PASSED")
    print(f"   Final Score: {smc.final_score}")
    print(f"   Recommendation: {smc.get_recommendation().value}")
    print("✅ SMC Analysis tests passed!\n")


def test_signal_entity():
    """Test Signal entity"""
    print("🏛️ Testing Signal Entity...")
    
    # Create test SMC analysis
    smc = SMCAnalysis(
        liquidity_score=Decimal('8.0'),
        structure_score=Decimal('8.0'),
        momentum_score=Decimal('8.0'),
        risk_reward_score=Decimal('8.0'),
        order_blocks_detected=[],
        fair_value_gaps=[],
        liquidity_sweeps=[],
        analysis_reasoning="Test analysis",
        confidence_level=Decimal('0.8'),
        timestamp=datetime.utcnow(),
        market_structure="BULLISH",
        trend_strength="STRONG",
        metadata={}
    )
    
    # Create entry signal
    signal = Signal.create_entry_signal(
        symbol=Symbol("BTCUSDT"),
        signal_type=SignalType.LONG,
        entry_price=Money.from_float(50000),
        stop_loss=Money.from_float(49000),
        take_profit=Money.from_float(52000),
        timeframe=Timeframe("4h"),
        confidence=Decimal('0.8'),
        reason="Strong bullish setup",
        ai_analysis=smc
    )
    
    # Test signal properties
    assert signal.is_valid_entry()
    assert signal.should_execute()
    assert signal.risk_reward_ratio() == Decimal('2.0')  # 2000/1000 = 2:1
    assert signal.should_notify()
    
    print("✅ Signal Entity - PASSED")
    print(f"   Risk/Reward: {signal.risk_reward_ratio()}")
    print(f"   Should Execute: {signal.should_execute()}")
    print("✅ Signal Entity tests passed!\n")


def test_position_entity():
    """Test Position entity"""
    print("🏛️ Testing Position Entity...")
    
    # Create open position
    position = Position.create_open_position(
        position_id=PositionId("test-pos-1"),
        symbol=Symbol("BTCUSDT"),
        side=PositionSide.LONG,
        entry_price=Money.from_float(50000),
        quantity=Decimal('0.1'),
        stop_loss=Money.from_float(49000),
        take_profit=Money.from_float(52000)
    )
    
    # Test position properties
    assert position.status == PositionStatus.OPEN
    assert position.get_position_value() == Money.from_float(5000)  # 50000 * 0.1
    assert position.get_risk_reward_ratio() == Decimal('2.0')
    
    # Test PnL calculation
    current_price = Money.from_float(51000)
    unrealized_pnl = position.calculate_unrealized_pnl(current_price)
    assert unrealized_pnl == Money.from_float(100)  # (51000 - 50000) * 0.1
    
    # Test stop loss check
    assert not position.is_stop_loss_hit(current_price)
    assert position.is_stop_loss_hit(Money.from_float(48000))
    
    # Test position closing
    closed_position = position.close_position(
        exit_price=Money.from_float(52000),
        close_reason="TAKE_PROFIT"
    )
    
    assert closed_position.status == PositionStatus.CLOSED
    assert closed_position.realized_pnl == Money.from_float(200)  # (52000 - 50000) * 0.1
    
    print("✅ Position Entity - PASSED")
    print(f"   Position Value: {position.get_position_value()}")
    print(f"   Unrealized PnL: {unrealized_pnl}")
    print(f"   Realized PnL: {closed_position.realized_pnl}")
    print("✅ Position Entity tests passed!\n")


def test_market_data_entities():
    """Test Market Data entities"""
    print("🏛️ Testing Market Data Entities...")
    
    # Create test OHLCV data
    candle = OHLCV(
        timestamp=datetime.utcnow(),
        open=Decimal('50000'),
        high=Decimal('50500'),
        low=Decimal('49500'),
        close=Decimal('50200'),
        volume=Decimal('1000')
    )
    
    assert candle.is_bullish
    assert candle.body_size == Decimal('200')
    assert candle.upper_wick == Decimal('300')
    
    # Create timeframe data
    timeframe_data = TimeframeData(
        symbol=Symbol("BTCUSDT"),
        timeframe=Timeframe("1h"),
        candles=[candle],
        indicators={},
        last_updated=datetime.utcnow()
    )
    
    assert timeframe_data.current_price == Decimal('50200')
    
    # Create market context
    context = MarketContext(
        timestamp=datetime.utcnow(),
        overall_sentiment="BULLISH",
        sentiment_strength="STRONG",
        volatility_level="MEDIUM",
        volume_profile="NORMAL",
        risk_on_off="RISK_ON",
        news_impact="LOW",
        economic_events=[],
        market_phase="MARKUP",
        metadata={}
    )
    
    assert context.is_favorable_for_trading()
    assert context.get_risk_adjustment_factor() == Decimal('1.0')
    
    print("✅ Market Data Entities - PASSED")
    print(f"   Candle: {candle.open} -> {candle.close} ({'BULLISH' if candle.is_bullish else 'BEARISH'})")
    print(f"   Market Context: {context.overall_sentiment} {context.sentiment_strength}")
    print("✅ Market Data Entities tests passed!\n")


def main():
    """Run all domain entity tests"""
    print("🚀 DOMAIN ENTITIES TESTS")
    print("=" * 50)
    
    try:
        test_value_objects()
        test_smc_analysis()
        test_signal_entity()
        test_position_entity()
        test_market_data_entities()
        
        print("🏛️ ALL DOMAIN ENTITY TESTS PASSED!")
        print("Domain entities are ready for battle!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"💀 DOMAIN ENTITY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)