#!/usr/bin/env python3
"""
Bot Multi-Cripto de Trading Institucional - Spartan Code Edition
Monitoreo continuo 24/7 con análisis AI condicional
"""

import logging
from config.settings import *
from bnb.binance import RobotBinance
from indicators.technical_indicators import TechnicalAnalyzer
from ai.kimi_client import GeminiClient
from strategy.strategies import StrategyManager, SignalType
import time

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_symbols(strategy_manager: StrategyManager):
    """Process all configured symbols"""
    print(f"\n🚀 Procesando {len(SYMBOLS)} símbolos en timeframe {time_frame}...")
    
    for symbol in SYMBOLS:  # Process all symbols
        print(f"\n{'='*60}")
        print(f"📊 Analizando: {symbol}")
        print(f"{'='*60}")

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
            print(f"🎯 SEÑAL: {signal.signal_type.value} | Fuerza: {signal.strength.value} | Confianza: {signal.confidence:.1%}")
            print(f"💰 Entrada: ${signal.entry_price:.2f}" if signal.entry_price else "💰 Entrada: N/A")
            print(f"🛑 SL: ${signal.stop_loss:.2f}" if signal.stop_loss else "🛑 SL: N/A")
            print(f"🎯 TP: ${signal.take_profit:.2f}" if signal.take_profit else "🎯 TP: N/A")
            print(f"📊 R/R: {signal.risk_reward_ratio:.2f}" if signal.risk_reward_ratio else "📊 R/R: N/A")
            print(f"📝 Razón: {signal.reason}")

            # AI Analysis only if valid signal (LONG/SHORT)
            if signal.signal_type != SignalType.WAIT:
                print("\n🤖 ACTIVANDO ANÁLISIS INSTITUCIONAL GEMINI...")
                current_data = df.iloc[-1]
                data_summary = f"""
                SEÑAL GENERADA PARA {symbol}:
                - Tipo: {signal.signal_type.value}
                - Precio Entrada: ${signal.entry_price:.2f}
                - Stop Loss: ${signal.stop_loss:.2f}
                - Take Profit: ${signal.take_profit:.2f}
                - Risk/Reward: {signal.risk_reward_ratio:.2f}
                - Confianza: {signal.confidence:.1%}
                - Razón: {signal.reason}

                DATOS TÉCNICOS ACTUALES DE {symbol} ({time_frame}):
                - Precio actual: ${current_data['close']:.2f}
                - RSI (14): {current_data['RSI_14']:.2f}
                - MACD: {current_data['MACD_12_26_9']:.4f}
                - Bollinger Bands: Upper {current_data['BB_upper_20']:.2f}, Middle {current_data['BB_middle_20']:.2f}, Lower {current_data['BB_lower_20']:.2f}
                - Estocástico: %K {current_data['STOCH_K_14_3']:.2f}, %D {current_data['STOCH_D_14_3']:.2f}
                - Trend Magic: {current_data['MagicTrend_Color']} ({current_data['MagicTrend']:.2f})
                - Squeeze Momentum: {current_data['squeeze_color']} ({current_data['momentum_color']})
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

def main():
    print("⚔️ BOT ESPARTANO MULTI-CRIPTO - MODO INSTITUCIONAL ⚔️")
    print(f"📊 Símbolos: {len(SYMBOLS)} | Timeframe: {time_frame}")
    print(f"🔄 Intervalo: {CHECK_INTERVAL_SECONDS} segundos | Loop Infinito: {ENABLE_INFINITE_LOOP}")
    print("=" * 80)

    # Initialize strategy manager
    strategy_manager = StrategyManager()

    if ENABLE_INFINITE_LOOP:
        print("🔥 INICIANDO MODO 24/7 - PRESIONA CTRL+C PARA DETENER")
        iteration = 0

        while True:
            iteration += 1
            print(f"\n🕐 ITERACIÓN #{iteration} - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

            try:
                process_symbols(strategy_manager)
                print(f"\n💤 Durmiendo {CHECK_INTERVAL_SECONDS} segundos hasta próxima verificación...")
                time.sleep(CHECK_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                print("\n🛑 Bot detenido por usuario")
                break
            except Exception as e:
                print(f"❌ Error en iteración {iteration}: {e}")
                print("⏳ Esperando 30 segundos antes de reintentar...")
                time.sleep(30)
    else:
        print("🔄 EJECUTANDO UNA SOLA ITERACIÓN")
        process_symbols(strategy_manager)

    print("\n🏛️ Bot finalizado exitosamente!")

if __name__ == "__main__":
    main()
