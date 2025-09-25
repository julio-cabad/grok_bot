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
from presenters.console_table import render_table
import time

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

COLOR_EMOJI_MAP = {
    "LIME": "🟢",
    "GREEN": "🥦",
    "MAROON": "🟤",
    "BROWN": "🟤",
    "RED": "🔴",
    "BLUE": "🔵",
    "BLACK": "⚫️",
    "GRAY": "⚪️",
    "GREY": "⚪️"
}


def format_color(color: str) -> str:
    if not color:
        return "N/A"

    color_upper = str(color).upper()
    emoji = COLOR_EMOJI_MAP.get(color_upper)
    if emoji:
        return f"{emoji} {color_upper}"
    return color_upper

def process_symbols(strategy_manager: StrategyManager):
    """Process all configured symbols"""
    print(f"\n🚀 Procesando {len(SYMBOLS)} símbolos en timeframe {time_frame}...")
    table_rows = []

    for symbol in SYMBOLS[:10]:  # Process all symbols
        try:
            # Get market data
            robot = RobotBinance(pair=symbol, temporality=time_frame)
            df = robot.candlestick(limit=CANDLES_LIMIT)

            if df.empty:
                print(f"⚠️ No se obtuvieron datos para {symbol}")
                continue

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

            # Extract current indicator context
            current_data = df.iloc[-1]

            squeeze_color = current_data.get('squeeze_color', 'N/A')
            momentum_color = current_data.get('momentum_color', 'N/A')
            trend_magic_value = current_data.get('MagicTrend')
            trend_magic_color = current_data.get('MagicTrend_Color', 'N/A')
            current_price = current_data.get('close')

            trend_magic_display = "N/A"
            if trend_magic_value is not None and trend_magic_value == trend_magic_value:
                trend_magic_display = f"{trend_magic_value:.2f}"

            price_display = "N/A"
            if current_price is not None and current_price == current_price:
                price_display = f"${current_price:.2f}"

            entry_display = "-"
            sl_display = "-"
            tp_display = "-"
            if signal.entry_price is not None and signal.entry_price == signal.entry_price:
                entry_display = f"${signal.entry_price:.2f}"
            if signal.stop_loss is not None and signal.stop_loss == signal.stop_loss:
                sl_display = f"${signal.stop_loss:.2f}"
            if signal.take_profit is not None and signal.take_profit == signal.take_profit:
                tp_display = f"${signal.take_profit:.2f}"

            table_rows.append(
                (
                    symbol,
                    format_color(momentum_color),
                    format_color(trend_magic_color),
                    format_color(squeeze_color),
                    trend_magic_display,
                    price_display,
                    entry_display,
                    sl_display,
                    tp_display,
                    signal.reason,
                )
            )

            # AI Analysis only if valid signal (LONG/SHORT)
            if signal.signal_type in (SignalType.LONG, SignalType.SHORT):
                print("\n🤖 ACTIVANDO ANÁLISIS INSTITUCIONAL GEMINI...")
                entry = signal.entry_price if signal.entry_price is not None else current_price
                sl = signal.stop_loss if signal.stop_loss is not None else current_price
                tp = signal.take_profit if signal.take_profit is not None else current_price
                rr = signal.risk_reward_ratio if signal.risk_reward_ratio is not None else 0.0

                data_summary = f"""
                SEÑAL GENERADA PARA {symbol}:
                - Tipo: {signal.signal_type.value}
                - Precio Entrada: ${entry:.2f}
                - Stop Loss: ${sl:.2f}
                - Take Profit: ${tp:.2f}
                - Risk/Reward: {rr:.2f}
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
                    #gemini = GeminiClient()
                    #analysis = gemini.analyze_market_data(data_summary)
                    #print("📊 ANÁLISIS INSTITUCIONAL GEMINI:")
                    print("=" * 80)
                    #print(analysis["analysis"])
                    print("=" * 80)
                except Exception as e:
                    print(f"❌ Error con Gemini: {e}")

        except Exception as e:
            print(f"❌ Error procesando {symbol}: {e}")

    if table_rows:
        print("\n📊 RESUMEN DE INDICADORES")
        render_table(
            [
                "Símbolo",
                "Momentum",
                "Trend Magic",
                "Squeeze",
                "Trend Magic Valor",
                "Precio actual",
                "Entrada",
                "Stop Loss",
                "Take Profit",
                "Razón",
            ],
            table_rows,
        )

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
