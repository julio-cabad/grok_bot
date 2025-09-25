#!/usr/bin/env python3
"""
Trading Strategies - Spartan Code Edition
High-performance, scalable strategies for multi-crypto institutional trading
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"

class SignalStrength(Enum):
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"

@dataclass
class TradingSignal:
    """
    Spartan Trading Signal
    Contains all necessary information for trade execution
    """
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    confidence: float  # 0-1
    timestamp: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'reason': self.reason
        }

class StrategyManager:
    """
    Spartan Strategy Manager
    Manages multiple trading strategies for multi-crypto operations
    """

    def __init__(self):
        self.logger = logging.getLogger("StrategyManager")
        self.logger.info("🏛️ Strategy Manager initialized")

    def squeeze_magic_strategy(self, df: Any, symbol: str) -> TradingSignal:
        """
        Squeeze + Magic Trend Strategy
        LONG: if Squeeze is MAROON or LIME and MagicTrend is BLUE
        SHORT: if Squeeze is MAROON or LIME and MagicTrend is RED

        Args:
            df: DataFrame with technical indicators
            symbol: Trading symbol

        Returns:
            TradingSignal object
        """
        try:
            # Get latest values
            current_data = df.iloc[-1]
            squeeze_color = current_data.get('squeeze_color')
            magic_color = current_data.get('MagicTrend_Color')
            close_price = current_data.get('close')

            # Strategy logic
            signal_type = SignalType.WAIT
            strength = SignalStrength.WEAK
            confidence = 0.5
            reason = "No clear signal"

            # LONG conditions
            if squeeze_color in ['MAROON', 'LIME'] and magic_color == 'BLUE':
                signal_type = SignalType.LONG
                strength = SignalStrength.MEDIUM
                confidence = 0.7
                reason = f"Squeeze {squeeze_color} + Magic BLUE = LONG setup"

            # SHORT conditions
            elif squeeze_color in ['MAROON', 'LIME'] and magic_color == 'RED':
                signal_type = SignalType.SHORT
                strength = SignalStrength.MEDIUM
                confidence = 0.7
                reason = f"Squeeze {squeeze_color} + Magic RED = SHORT setup"

            # Calculate risk management (basic)
            entry_price = close_price
            stop_loss = None
            take_profit = None
            rr_ratio = None

            if signal_type != SignalType.WAIT:
                # Basic SL/TP calculation
                atr = current_data.get('ATR', close_price * 0.02)  # fallback ATR
                if signal_type == SignalType.LONG:
                    stop_loss = close_price - (atr * 1.5)
                    take_profit = close_price + (atr * 3)
                else:  # SHORT
                    stop_loss = close_price + (atr * 1.5)
                    take_profit = close_price - (atr * 3)

                if stop_loss and take_profit:
                    risk = abs(close_price - stop_loss)
                    reward = abs(take_profit - close_price)
                    rr_ratio = reward / risk if risk > 0 else None

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr_ratio,
                confidence=confidence,
                timestamp="2025-09-25",  # Would use datetime
                reason=reason
            )

            self.logger.info(f"📊 Signal generated for {symbol}: {signal_type.value} ({confidence:.1%})")
            return signal

        except Exception as e:
            self.logger.error(f"💀 Strategy calculation failed for {symbol}: {str(e)}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.WAIT,
                strength=SignalStrength.WEAK,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                risk_reward_ratio=None,
                confidence=0.0,
                timestamp="2025-09-25",
                reason=f"Error: {str(e)}"
            )

    def execute_strategy(self, strategy_name: str, df: Any, symbol: str) -> TradingSignal:
        """
        Execute a specific strategy

        Args:
            strategy_name: Name of the strategy to execute
            df: DataFrame with indicators
            symbol: Trading symbol

        Returns:
            TradingSignal
        """
        if strategy_name == "squeeze_magic":
            return self.squeeze_magic_strategy(df, symbol)
        else:
            self.logger.warning(f"Unknown strategy: {strategy_name}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.WAIT,
                strength=SignalStrength.WEAK,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                risk_reward_ratio=None,
                confidence=0.0,
                timestamp="2025-09-25",
                reason=f"Unknown strategy: {strategy_name}"
            )
