#!/usr/bin/env python3
"""
Trading Strategies - Spartan Code Edition
High-performance, scalable strategies for multi-crypto institutional trading
"""

import logging
import sys
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from config.settings import TIMEZONE, POSITION_SIZE, time_frame, taker_fee

if TYPE_CHECKING:
    from notifications.telegram_notifier import TelegramNotifier

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"
    EXIT = "EXIT"

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
    position_opened_at: Optional[str] = None

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
            'reason': self.reason,
            'position_opened_at': self.position_opened_at,
        }

class StrategyManager:
    """
    Spartan Strategy Manager
    Manages multiple trading strategies for multi-crypto operations
    """

    LONG_SQUEEZE_COLORS = {"MAROON", "LIME"}
    SHORT_SQUEEZE_COLORS = {"RED", "GREEN"}

    def __init__(
        self,
        notifier: Optional["TelegramNotifier"] = None,
        notify_on_open: bool = False,
        notify_on_close: bool = False,
        notify_alerts: bool = False,
    ):
        self.logger = logging.getLogger("StrategyManager")
        self.logger.setLevel(logging.WARNING)
        self.last_alerted_signal: Dict[str, SignalType] = {}
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.notifier = notifier
        self.notify_on_open = notify_on_open and notifier is not None
        self.notify_on_close = notify_on_close and notifier is not None
        self.notify_alerts = notify_alerts and notifier is not None
        self.timezone = ZoneInfo(TIMEZONE)
        self.fee_rate = taker_fee

    def _trigger_alert(self, symbol: str, signal_type: SignalType, reason: str) -> None:
        """Play an audible alert and log the new trading signal."""
        try:
            sys.stdout.write('\a')
            sys.stdout.flush()
        except Exception:
            pass
        label = "Cierre" if signal_type == SignalType.EXIT else "Señal"
        print(f"🔔 ALERTA ESPARTANA {symbol}: {label} {signal_type.value}")
        print(f"   ➤ Motivo: {reason}")

        should_notify_alert = self.notifier and self.notify_alerts

        if should_notify_alert:
            if signal_type in (SignalType.LONG, SignalType.SHORT) and self.notify_on_open:
                should_notify_alert = False
            elif signal_type == SignalType.EXIT and self.notify_on_close:
                should_notify_alert = False

        if should_notify_alert:
            try:
                title = f"{label.upper()} {signal_type.value}"
                body = f"{symbol}: {reason}"
                self.notifier.notify_alert(title=title, message=body)
            except Exception as exc:
                self.logger.error(f"Telegram alert notification failed for {symbol}: {exc}")

    def _calculate_quantity(self, entry_price: Optional[float]) -> Optional[float]:
        if entry_price is None or entry_price <= 0:
            return None
        try:
            return POSITION_SIZE / entry_price
        except ZeroDivisionError:
            return None

    def _notify_position_opened(
        self,
        symbol: str,
        signal_type: SignalType,
        entry_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> None:
        if not (self.notifier and self.notify_on_open):
            return

        quantity = self._calculate_quantity(entry_price)
        if quantity is None or entry_price is None:
            return

        try:
            self.notifier.notify_position_opened(
                symbol=symbol,
                side=signal_type.value,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=stop_loss if stop_loss is not None else 0.0,
                take_profit=take_profit if take_profit is not None else 0.0,
                timeframe=time_frame,
            )
        except Exception as exc:
            self.logger.error(f"Telegram open notification failed for {symbol}: {exc}")

    def _notify_position_closed(
        self,
        symbol: str,
        position: Dict[str, Any],
        close_price: Optional[float],
        exit_reason: str,
    ) -> None:
        if not (self.notifier and self.notify_on_close):
            return

        if close_price is None:
            return

        entry_price = position.get('entry_price')
        if entry_price is None or entry_price <= 0:
            return

        quantity = position.get('quantity')
        if quantity is None:
            quantity = self._calculate_quantity(entry_price)

        if quantity is None or quantity <= 0:
            return

        entry_time_dt = position.get('opened_at_dt')
        if entry_time_dt is None:
            entry_str = position.get('opened_at')
            if entry_str:
                try:
                    entry_time_dt = datetime.strptime(entry_str, "%Y-%m-%d %H:%M").replace(tzinfo=self.timezone)
                except ValueError:
                    entry_time_dt = datetime.now(self.timezone)
            else:
                entry_time_dt = datetime.now(self.timezone)

        exit_time_dt = datetime.now(self.timezone)

        if position['type'] == SignalType.LONG:
            gross_pnl = (close_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - close_price) * quantity

        total_commissions = (entry_price + close_price) * quantity * self.fee_rate
        real_pnl = gross_pnl - total_commissions

        reason_code_map = {
            "EXIT STOP LOSE": "STOP_LOSS",
            "EXIT BY COLOR": "COLOR_CHANGE",
        }
        close_reason = reason_code_map.get(exit_reason, exit_reason.replace(" ", "_").upper())

        try:
            self.notifier.notify_position_closed(
                symbol=symbol,
                side=position['type'].value,
                entry_price=entry_price,
                exit_price=close_price,
                quantity=quantity,
                gross_pnl=gross_pnl,
                real_pnl=real_pnl,
                total_commissions=total_commissions,
                close_reason=close_reason,
                entry_time=entry_time_dt,
                exit_time=exit_time_dt,
                timeframe=time_frame,
            )
        except Exception as exc:
            self.logger.error(f"Telegram close notification failed for {symbol}: {exc}")

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
            momentum_color = current_data.get('momentum_color')
            magic_color = current_data.get('MagicTrend_Color')
            close_price = current_data.get('close')

            position = self.open_positions.get(symbol)

            entry_price = None
            stop_loss = None
            take_profit = None
            rr_ratio = None
            entry_time_str = None
            entry_time_dt = None
            position_quantity = None

            if position:
                entry_price = position.get('entry_price')
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')
                rr_ratio = position.get('risk_reward')
                entry_time_str = position.get('opened_at')
                entry_time_dt = position.get('opened_at_dt')
                position_quantity = position.get('quantity')

            signal_type = SignalType.WAIT
            strength = SignalStrength.WEAK
            confidence = 0.5
            reason = "NO OPEN"

            if position:
                pos_type = position['type']
                exit_reason = None

                if close_price is not None:
                    if pos_type == SignalType.LONG:
                        if stop_loss is not None and close_price <= stop_loss:
                            exit_reason = "EXIT STOP LOSE"
                        elif squeeze_color not in self.LONG_SQUEEZE_COLORS or magic_color != 'BLUE':
                            exit_reason = "EXIT BY COLOR"
                    elif pos_type == SignalType.SHORT:
                        if stop_loss is not None and close_price >= stop_loss:
                            exit_reason = "EXIT STOP LOSE"
                        elif squeeze_color not in self.SHORT_SQUEEZE_COLORS or magic_color != 'RED':
                            exit_reason = "EXIT BY COLOR"
                else:
                    if pos_type == SignalType.LONG and (
                        squeeze_color not in self.LONG_SQUEEZE_COLORS or magic_color != 'BLUE'
                    ):
                        exit_reason = "EXIT BY COLOR"
                    elif pos_type == SignalType.SHORT and (
                        squeeze_color not in self.SHORT_SQUEEZE_COLORS or magic_color != 'RED'
                    ):
                        exit_reason = "EXIT BY COLOR"

                if exit_reason:
                    if position and self.notifier and self.notify_on_close:
                        self._notify_position_closed(symbol, position, close_price, exit_reason)
                    signal_type = SignalType.EXIT
                    strength = SignalStrength.STRONG
                    confidence = 0.8
                    reason = exit_reason
                    self.open_positions.pop(symbol, None)
                else:
                    signal_type = SignalType.WAIT
                    strength = SignalStrength.MEDIUM
                    confidence = 0.6
                    reason = pos_type.value
            else:
                if squeeze_color in self.LONG_SQUEEZE_COLORS and magic_color == 'BLUE' and close_price is not None:
                    signal_type = SignalType.LONG
                    strength = SignalStrength.MEDIUM
                    confidence = 0.7
                    reason = "LONG"
                elif squeeze_color in self.SHORT_SQUEEZE_COLORS and magic_color == 'RED' and close_price is not None:
                    signal_type = SignalType.SHORT
                    strength = SignalStrength.MEDIUM
                    confidence = 0.7
                    reason = "SHORT"

                if signal_type in (SignalType.LONG, SignalType.SHORT) and close_price is not None:
                    atr = current_data.get('ATR')
                    if atr is None:
                        atr = close_price * 0.02

                    if signal_type == SignalType.LONG:
                        stop_loss = close_price - (atr * 1.5)
                        take_profit = close_price + (atr * 3)
                    else:
                        stop_loss = close_price + (atr * 1.5)
                        take_profit = close_price - (atr * 3)

                    if stop_loss is not None and take_profit is not None:
                        risk = abs(close_price - stop_loss)
                        reward = abs(take_profit - close_price)
                        rr_ratio = reward / risk if risk > 0 else None

                    entry_price = close_price
                    entry_time_dt = datetime.now(self.timezone)
                    entry_time_str = entry_time_dt.strftime("%Y-%m-%d %H:%M")
                    quantity = self._calculate_quantity(entry_price)
                    self.open_positions[symbol] = {
                        'type': signal_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward': rr_ratio,
                        'opened_at': entry_time_str,
                        'opened_at_dt': entry_time_dt,
                        'quantity': quantity,
                    }
                    if self.notifier and self.notify_on_open:
                        self._notify_position_opened(
                            symbol,
                            signal_type,
                            entry_price,
                            stop_loss,
                            take_profit,
                        )
                elif signal_type in (SignalType.LONG, SignalType.SHORT):
                    signal_type = SignalType.WAIT
                    strength = SignalStrength.WEAK
                    confidence = 0.0
                    reason = "NO OPEN"

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=rr_ratio,
                confidence=confidence,
                timestamp=datetime.utcnow().isoformat(),
                reason=reason,
                position_opened_at=entry_time_str,
            )

            if signal.signal_type in (SignalType.LONG, SignalType.SHORT, SignalType.EXIT):
                previous = self.last_alerted_signal.get(symbol)
                if previous != signal.signal_type:
                    self._trigger_alert(symbol, signal.signal_type, signal.reason)
                self.last_alerted_signal[symbol] = signal.signal_type
            else:
                self.last_alerted_signal[symbol] = SignalType.WAIT

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
                timestamp=datetime.utcnow().isoformat(),
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
                timestamp=datetime.utcnow().isoformat(),
                reason=f"Unknown strategy: {strategy_name}"
            )
