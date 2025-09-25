#!/usr/bin/env python3
"""
Technical Indicators Module - Pure Indicator Calculations
Clean, focused, and efficient indicator functions
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import  Dict, Any
from bnb.binance import RobotBinance

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class TechnicalAnalyzer:
    """
    Pure technical analysis class focused only on indicator calculations
    """

    def __init__(self, symbol: str, timeframe: str = "1h"):
        """
        Initialize the Technical Analyzer

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Candlestick interval ('1m', '5m', '15m', '1h', '4h', '1d')
        """
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.logger = logging.getLogger(f"TechnicalAnalyzer-{self.symbol}")
        self.logger.setLevel(logging.WARNING)
        self.binance_client = RobotBinance(pair=self.symbol, temporality=self.timeframe)
        self.df = None

        self.logger.info(
            f"🏛️ Spartan Analyzer initialized for {self.symbol} on {self.timeframe}"
        )

    def fetch_market_data(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch real market data from Binance

        Args:
            limit: Number of candles to fetch (max 1500)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"⚔️ Fetching {limit} candles for {self.symbol}")

            # Get real market data from Binance
            self.df = self.binance_client.candlestick(limit=limit)

            if self.df.empty:
                raise ValueError(f"No market data received for {self.symbol}")

            self.logger.info(
                f"✅ Retrieved {len(self.df)} candles from {self.df.index[0]} to {self.df.index[-1]}"
            )
            return self.df

        except Exception as e:
            self.logger.error(f"💀 Failed to fetch market data: {str(e)}")
            raise

    def calculate_sma(self, period: int = 20, source: str = "close") -> pd.Series:
        """
        Calculate Simple Moving Average

        Args:
            period: SMA period (default 20)
            source: Price source ('open', 'high', 'low', 'close', 'volume')

        Returns:
            Series with SMA values
        """
        if self.df is None or self.df.empty:
            raise ValueError("No market data available. Call fetch_market_data() first")

        if source not in self.df.columns:
            raise ValueError(
                f"Invalid source '{source}'. Available: {list(self.df.columns)}"
            )

        try:
            sma = ta.sma(self.df[source], length=period)
            self.logger.info(f"📊 SMA({period}) calculated for {source} price")
            return sma

        except Exception as e:
            self.logger.error(f"💀 SMA calculation failed: {str(e)}")
            raise

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            period: RSI period (default 14)

        Returns:
            Series with RSI values
        """
        if self.df is None or self.df.empty:
            raise ValueError("No market data available. Call fetch_market_data() first")

        try:
            rsi = ta.rsi(self.df['close'], length=period)
            self.df[f'RSI_{period}'] = rsi
            self.logger.info(f"📊 RSI({period}) calculated")
            return rsi

        except Exception as e:
            self.logger.error(f"💀 RSI calculation failed: {str(e)}")
            raise

    def calculate_bollinger_bands(self, period: int = 20, std: int = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands

        Args:
            period: BB period (default 20)
            std: Standard deviation multiplier (default 2)

        Returns:
            Dictionary with upper, middle, lower bands
        """
        if self.df is None or self.df.empty:
            raise ValueError("No market data available. Call fetch_market_data() first")

        try:
            bb = ta.bbands(self.df['close'], length=period, std=std)
            # Use generic column names as pandas-ta may vary the suffix
            upper_col = [col for col in bb.columns if 'BBU' in col][0]
            middle_col = [col for col in bb.columns if 'BBM' in col][0]
            lower_col = [col for col in bb.columns if 'BBL' in col][0]
            
            self.df[f'BB_upper_{period}'] = bb[upper_col]
            self.df[f'BB_middle_{period}'] = bb[middle_col]
            self.df[f'BB_lower_{period}'] = bb[lower_col]
            self.logger.info(f"📊 Bollinger Bands({period}, {std}) calculated")
            return {
                'upper': bb[upper_col],
                'middle': bb[middle_col],
                'lower': bb[lower_col]
            }

        except Exception as e:
            self.logger.error(f"💀 Bollinger Bands calculation failed: {str(e)}")
            raise

    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Args:
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if self.df is None or self.df.empty:
            raise ValueError("No market data available. Call fetch_market_data() first")

        try:
            macd = ta.macd(self.df['close'], fast=fast, slow=slow, signal=signal)
            self.df[f'MACD_{fast}_{slow}_{signal}'] = macd[f'MACD_{fast}_{slow}_{signal}']
            self.df[f'MACD_signal_{fast}_{slow}_{signal}'] = macd[f'MACDs_{fast}_{slow}_{signal}']
            self.df[f'MACD_hist_{fast}_{slow}_{signal}'] = macd[f'MACDh_{fast}_{slow}_{signal}']
            self.logger.info(f"📊 MACD({fast}, {slow}, {signal}) calculated")
            return {
                'macd': macd[f'MACD_{fast}_{slow}_{signal}'],
                'signal': macd[f'MACDs_{fast}_{slow}_{signal}'],
                'histogram': macd[f'MACDh_{fast}_{slow}_{signal}']
            }

        except Exception as e:
            self.logger.error(f"💀 MACD calculation failed: {str(e)}")
            raise

    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator

        Args:
            k_period: %K period (default 14)
            d_period: %D period (default 3)
            smooth_k: Smoothing period for %K (default 3)

        Returns:
            Dictionary with %K and %D lines
        """
        if self.df is None or self.df.empty:
            raise ValueError("No market data available. Call fetch_market_data() first")

        try:
            stoch = ta.stoch(self.df['high'], self.df['low'], self.df['close'], k=k_period, d=d_period, smooth_k=smooth_k)
            self.df[f'STOCH_K_{k_period}_{d_period}'] = stoch[f'STOCHk_{k_period}_{d_period}_{smooth_k}']
            self.df[f'STOCH_D_{k_period}_{d_period}'] = stoch[f'STOCHd_{k_period}_{d_period}_{smooth_k}']
            self.logger.info(f"📊 Stochastic({k_period}, {d_period}, {smooth_k}) calculated")
            return {
                'k': stoch[f'STOCHk_{k_period}_{d_period}_{smooth_k}'],
                'd': stoch[f'STOCHd_{k_period}_{d_period}_{smooth_k}']
            }

        except Exception as e:
            self.logger.error(f"💀 Stochastic calculation failed: {str(e)}")
            raise

    def trend_magic(
        self, period: int = 100, coeff: float = 1.0, atr_period: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate Trend Magic indicator

        Args:
            period: CCI period (default: 20)
            coeff: ATR multiplier (default: 1.0)
            atr_period: ATR period (default: 5)

        Returns:
            Dictionary with Trend Magic analysis
        """
        if self.df is None or self.df.empty:
            raise ValueError("No market data available. Call fetch_market_data() first")

        try:
            data = self.df.copy()

            # Calculate ATR using SMA (as in PineScript)
            data["ATR"] = ta.atr(
                data["high"],
                data["low"],
                data["close"],
                length=atr_period,
                mamode="sma",
            )

            # Calculate CCI
            data["CCI"] = ta.cci(
                data["high"], data["low"], data["close"], length=period
            )

            # Calculate upper and lower bands
            upT = data["low"] - data["ATR"] * coeff
            downT = data["high"] + data["ATR"] * coeff

            # Initialize MagicTrend
            data["MagicTrend"] = 0.0

            # Calculate MagicTrend
            for i in range(len(data)):
                if i == 0:
                    if data["CCI"].iloc[i] >= 0:
                        data.loc[data.index[i], "MagicTrend"] = upT.iloc[i]
                    else:
                        data.loc[data.index[i], "MagicTrend"] = downT.iloc[i]
                else:
                    if data["CCI"].iloc[i] >= 0:
                        data.loc[data.index[i], "MagicTrend"] = max(
                            upT.iloc[i], data["MagicTrend"].iloc[i - 1]
                        )
                    else:
                        data.loc[data.index[i], "MagicTrend"] = min(
                            downT.iloc[i], data["MagicTrend"].iloc[i - 1]
                        )

            # Assign colors based on trend (CCI >= 0)
            data["MagicTrend_Color"] = np.where(data["CCI"] >= 0, "BLUE", "RED")

            # Detect buy and sell signals
            data["BuySignal"] = False
            data["SellSignal"] = False

            for i in range(1, len(data)):
                # Buy signal: when low crosses above MagicTrend
                if (
                    data["low"].iloc[i - 1] <= data["MagicTrend"].iloc[i - 1]
                    and data["low"].iloc[i] > data["MagicTrend"].iloc[i]
                ):
                    data.loc[data.index[i], "BuySignal"] = True

                # Sell signal: when high crosses below MagicTrend
                if (
                    data["high"].iloc[i - 1] >= data["MagicTrend"].iloc[i - 1]
                    and data["high"].iloc[i] < data["MagicTrend"].iloc[i]
                ):
                    data.loc[data.index[i], "SellSignal"] = True

            # Update self.df with the calculated columns
            self.df = data

            # Get current values
            current = data.iloc[-1]

            # Determine trend strength
            if current["MagicTrend_Color"] == "BLUE":
                trend_status = "🔵 BULLISH MAGIC"
                trend_emoji = "🐂"
            else:
                trend_status = "🔴 BEARISH MAGIC"
                trend_emoji = "🐻"

            # Calculate distance from trend line
            distance_pct = (
                (current["close"] - current["MagicTrend"]) / current["MagicTrend"]
            ) * 100

            result = {
                "magic_trend_value": current["MagicTrend"],
                "color": current["MagicTrend_Color"],
                "trend_status": trend_status,
                "trend_emoji": trend_emoji,
                "cci_value": current["CCI"],
                "atr_value": current["ATR"],
                "distance_pct": distance_pct,
                "buy_signal": current["BuySignal"],
                "sell_signal": current["SellSignal"],
                "current_price": current["close"],
            }

            self.logger.info(f"🔮 Trend Magic calculated: {trend_status}")
            return result

        except Exception as e:
            self.logger.error(f"💀 Trend Magic calculation failed: {str(e)}")
            raise

    def squeeze_momentum(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        use_true_range: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate Squeeze Momentum Indicator by LazyBear

        Args:
            bb_length: Bollinger Bands length (default: 20)
            bb_mult: Bollinger Bands multiplier (default: 2.0)
            kc_length: Keltner Channel length (default: 20)
            kc_mult: Keltner Channel multiplier (default: 1.5)
            use_true_range: Use True Range for KC calculation (default: True)

        Returns:
            Dictionary with Squeeze Momentum analysis
        """
        if self.df is None or self.df.empty:
            raise ValueError("No market data available. Call fetch_market_data() first")

        try:
            data = self.df.copy()
            source = data["close"]

            # Calculate Bollinger Bands
            basis = ta.sma(source, length=bb_length)
            dev = bb_mult * ta.stdev(source, length=bb_length)
            upper_bb = basis + dev
            lower_bb = basis - dev

            # Calculate Keltner Channels
            ma = ta.sma(source, length=kc_length)

            if use_true_range:
                # Calculate True Range
                tr_data = ta.true_range(data["high"], data["low"], data["close"])
                range_ma = ta.sma(tr_data, length=kc_length)
            else:
                # Use High - Low
                range_data = data["high"] - data["low"]
                range_ma = ta.sma(range_data, length=kc_length)

            upper_kc = ma + range_ma * kc_mult
            lower_kc = ma - range_ma * kc_mult

            # Squeeze conditions
            sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
            sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
            no_sqz = ~sqz_on & ~sqz_off

            # Calculate momentum value using linear regression
            # Equivalent to: linreg(source - avg(avg(highest(high, kc_length), lowest(low, kc_length)), sma(close, kc_length)), kc_length, 0)

            # Calculate highest and lowest over kc_length periods
            highest_high = data["high"].rolling(window=kc_length).max()
            lowest_low = data["low"].rolling(window=kc_length).min()

            # Calculate the average of highest/lowest and sma of close
            avg_hl = (highest_high + lowest_low) / 2
            sma_close = ta.sma(data["close"], length=kc_length)
            avg_base = (avg_hl + sma_close) / 2

            # Linear regression of (source - avg_base)
            momentum_source = source - avg_base

            # Calculate linear regression manually (equivalent to linreg in PineScript)
            val = pd.Series(index=data.index, dtype=float)

            for i in range(kc_length - 1, len(data)):
                y_values = momentum_source.iloc[i - kc_length + 1 : i + 1].values
                x_values = np.arange(kc_length)

                # Linear regression: y = mx + b, we want the slope (m) * (kc_length - 1) + b
                if len(y_values) == kc_length:
                    slope, intercept = np.polyfit(x_values, y_values, 1)
                    val.iloc[i] = slope * (kc_length - 1) + intercept

            # Get current values
            current_val = val.iloc[-1] if not pd.isna(val.iloc[-1]) else 0
            prev_val = val.iloc[-2] if len(val) > 1 and not pd.isna(val.iloc[-2]) else 0

            # Determine momentum color (equivalent to bcolor in PineScript)
            if current_val > 0:
                if current_val > prev_val:
                    momentum_color = "LIME"  # Bullish increasing
                else:
                    momentum_color = "GREEN"  # Bullish decreasing
            else:
                if current_val < prev_val:
                    momentum_color = "RED"  # Bearish decreasing
                else:
                    momentum_color = "MAROON"  # Bearish increasing

            # Determine squeeze color (equivalent to scolor in PineScript)
            current_sqz_on = sqz_on.iloc[-1] if not pd.isna(sqz_on.iloc[-1]) else False
            current_sqz_off = (
                sqz_off.iloc[-1] if not pd.isna(sqz_off.iloc[-1]) else False
            )
            current_no_sqz = no_sqz.iloc[-1] if not pd.isna(no_sqz.iloc[-1]) else False

            if current_no_sqz:
                squeeze_state_color = "BLUE"  # No squeeze
            elif current_sqz_on:
                squeeze_state_color = "BLACK"  # Squeeze on
            else:
                squeeze_state_color = "GRAY"  # Squeeze off

            # Determine squeeze status
            if current_sqz_on:
                squeeze_status = "🔴 SQUEEZE ON"
            elif current_sqz_off:
                squeeze_status = "🟢 SQUEEZE OFF"
            else:
                squeeze_status = "🔵 NO SQUEEZE"

            # Determine momentum trend
            if momentum_color in ["LIME", "GREEN"]:
                momentum_trend = "🟢 BULLISH"
            else:
                momentum_trend = "🔴 BEARISH"

            # Add columns to data DataFrame
            data['momentum_color'] = momentum_color
            data['momentum_trend'] = momentum_trend
            data['squeeze_color'] = momentum_color
            data['squeeze_state_color'] = squeeze_state_color
            data['squeeze_status'] = squeeze_status

            # Update self.df with the calculated columns
            self.df = data

            result = {
                "momentum_value": current_val,
                "momentum_color": momentum_color,
                "momentum_trend": momentum_trend,
                "squeeze_color": momentum_color,
                "squeeze_state_color": squeeze_state_color,
                "squeeze_status": squeeze_status,
                "squeeze_on": current_sqz_on,
                "squeeze_off": current_sqz_off,
                "no_squeeze": current_no_sqz,
                "bb_upper": upper_bb.iloc[-1],
                "bb_lower": lower_bb.iloc[-1],
                "kc_upper": upper_kc.iloc[-1],
                "kc_lower": lower_kc.iloc[-1],
                "current_price": data["close"].iloc[-1],
            }

            self.logger.info(
                f"🎯 Squeeze Momentum calculated: {squeeze_status} | {momentum_trend}"
            )
            return result

        except Exception as e:
            self.logger.error(f"💀 Squeeze Momentum calculation failed: {str(e)}")
            raise

    def get_squeeze_status(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
    ) -> str:
        """
        Get only the Squeeze status

        Returns:
            'SQUEEZE_ON', 'SQUEEZE_OFF', or 'NO_SQUEEZE'
        """
        squeeze_data = self.squeeze_momentum(bb_length, bb_mult, kc_length, kc_mult)

        if squeeze_data["squeeze_on"]:
            return "SQUEEZE_ON"
        elif squeeze_data["squeeze_off"]:
            return "SQUEEZE_OFF"
        else:
            return "NO_SQUEEZE"
