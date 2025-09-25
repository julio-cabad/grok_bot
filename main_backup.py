#!/usr/bin/env python3
"""
Script de prueba para verificar la conexión con Binance y mostrar datos OHLCV
"""

import logging
from config.settings import *
from bnb.binance import RobotBinance
from indicators.technical_indicators import TechnicalAnalyzer
from ai.kimi_client import GeminiClient
from strategy.strategies import StrategyManager

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Iniciando bot multi-cripto de trading institucional...", time_frame)

    # Initialize strategy manager
    strategy_manager = StrategyManager()

    # Process each symbol
    for symbol in SYMBOLS:
        print(f"n{x27=x27*60}")
        print(f"�� Procesando símbolo: {symbol}")
        print(f"{x27=x27*60}")

        try:
            # Get market data
            robot = RobotBinance(pair=symbol, temporality=time_frame)
            df = robot.candlestick(limit=CANDLES_LIMIT)

            if df.empty:
                print(f"⚠️ No se obtuvieron datos para {symbol}")
                continue

            print(f"✅ Datos obtenidos: {len(df)} velas")

            # Calculate technical indicators
            analyzer = TechnicalAnalyzer(symbol=symbol, timeframe=time_frame)
            analyzer.df = df
            analyzer.trend_magic()
            analyzer.squeeze_momentum()
            analyzer.calculate_rsi()
            analyzer.calculate_bollinger_bands()
            analyzer.calculate_macd()
            analyzer.calculate_stochastic()
            df = analyzer.df

            # Execute strategy
            signal = strategy_manager.execute_strategy("squeeze_magic", df, symbol)

            # Display signal
            print(f"🎯 SEÑAL GENERADA: {signal.signal_type.value} | Fuerza: {signal.strength.value} | Confianza: {signal.confidence:.1%}")
            print(f"💰 Precio entrada: ${signal.entry_price:.2f}" if signal.entry_price else "💰 Precio entrada: N/A")
            print(f"🛑 Stop Loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "🛑 Stop Loss: N/A")
            print(f"🎯 Take Profit: ${signal.take_profit:.2f}" if signal.take_profit else "🎯 Take Profit: N/A")
            print(f"📊 Risk/Reward: {signal.risk_reward_ratio:.2f}" if signal.risk_reward_ratio else "📊 Risk/Reward: N/A")
            print(f"📝 Razón: {signal.reason}")

            # AI Analysis (only for BTC for now to avoid spam)
            if symbol == "BTCUSDT":
                print("n🤖 Análisis Institucional con Gemini...")
                current_data = df.iloc[-1]
                data_summary = f"""
                DATOS ACTUALES DE {symbol} (4h):
                - Precio actual: ${current_data[x27closex27]:.2f}
                - RSI (14): {current_data[x27RSI_14x27]:.2f}
                - MACD: {current_data[x27MACD_12_26_9x27]:.4f}
                - Bollinger Bands: Upper {current_data[x27BB_upper_20x27]:.2f}, Middle {current_data[x27BB_middle_20x27]:.2f}, Lower {current_data[x27BB_lower_20x27]:.2f}
                - Estocástico: %K {current_data[x27STOCH_K_14_3x27]:.2f}, %D {current_data[x27STOCH_D_14_3x27]:.2f}
                - Trend Magic: {current_data[x27MagicTrend_Colorx27]} ({current_data[x27MagicTrendx27]:.2f})
                - Squeeze Momentum: {current_data[x27squeeze_colorx27]} ({current_data[x27momentum_colorx27]})
                """

                try:
                    gemini = GeminiClient()
                    analysis = gemini.analyze_market_data(data_summary)
                    print("📊 ANÁLISIS INSTITUCIONAL GEMINI:")
                    print("=" * 80)
                    print(analysis["analysis"])
                    print("=" * 80)
                except Exception as e:
                    print(f"❌ Error con Gemini: {e}")

        except Exception as e:
            print(f"❌ Error procesando {symbol}: {e}")

    print(f"n🎯 Bot multi-cripto completado! Procesados {len(SYMBOLS)} símbolos.")