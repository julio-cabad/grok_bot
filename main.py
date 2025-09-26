#!/usr/bin/env python3
"""
Bot Multi-Cripto de Trading Institucional - Spartan Code Edition
Monitoreo continuo 24/7 con análisis AI condicional
"""

import logging
from config.settings import *
from bnb.binance import RobotBinance
from indicators.technical_indicators import TechnicalAnalyzer
from ai.gemini_client import GeminiClient
from strategy.strategies import StrategyManager, SignalType
from presenters.console_table import render_table
from monitoring.performance_monitor import get_performance_monitor, TimingContext
from cache.market_data_cache import get_market_cache
from notifications.telegram_config import (
    ENABLE_NOTIFICATIONS,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    NOTIFY_POSITION_OPENED,
    NOTIFY_POSITION_CLOSED,
    NOTIFY_ALERTS,
)
from notifications.telegram_notifier import TelegramNotifier
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
    """Process all configured symbols with performance monitoring"""
    performance_monitor = get_performance_monitor()
    market_cache = get_market_cache()
    
    print(f"\n🚀 Procesando {len(SYMBOLS)} símbolos en timeframe {time_frame}...")
    table_rows = []
    
    # Start total cycle timing
    cycle_start = time.time()

    for symbol in SYMBOLS[:10]:  # Process all symbols
        with TimingContext(performance_monitor, 'symbol_processing', {'symbol': symbol}):
            try:
                performance_monitor.increment_counter('symbols_processed')
                # Get market data with timing
                with TimingContext(performance_monitor, 'api_calls', {'symbol': symbol, 'timeframe': time_frame}):
                    robot = RobotBinance(pair=symbol, temporality=time_frame)
                    df = robot.candlestick(limit=CANDLES_LIMIT)
                    
                    # Track API call efficiency
                    performance_monitor.increment_counter('api_calls_total')
                    
                    # Check if data came from cache
                    cached_data = market_cache.get(symbol, time_frame, CANDLES_LIMIT)
                    if cached_data is not None and len(cached_data) == len(df):
                        performance_monitor.increment_counter('api_calls_cached')

                if df.empty:
                    print(f"⚠️ No se obtuvieron datos para {symbol}")
                    performance_monitor.increment_counter('errors')
                    continue

                # Calculate technical indicators with timing
                with TimingContext(performance_monitor, 'indicator_calculations', {'symbol': symbol}):
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
                
                # Track signal generation
                if signal.signal_type in (SignalType.LONG, SignalType.SHORT):
                    performance_monitor.increment_counter('signals_generated')

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
                opened_at_display = "-"
                if signal.entry_price is not None and signal.entry_price == signal.entry_price:
                    entry_display = f"${signal.entry_price:.2f}"
                if signal.stop_loss is not None and signal.stop_loss == signal.stop_loss:
                    sl_display = f"${signal.stop_loss:.2f}"
                if signal.take_profit is not None and signal.take_profit == signal.take_profit:
                    tp_display = f"${signal.take_profit:.2f}"
                if signal.position_opened_at:
                    opened_at_display = signal.position_opened_at

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
                        opened_at_display,
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
                performance_monitor.increment_counter('errors')

    # Record total cycle time
    cycle_duration = time.time() - cycle_start
    performance_monitor.record_timing('total_cycle', cycle_duration, {'symbols_count': len(SYMBOLS[:10])})

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
                "Hora Entrada",
                "Razón",
            ],
            table_rows,
        )
        
        # Show performance summary every 10 cycles
        if performance_monitor.counters.get('symbols_processed', 0) % 100 == 0:
            print("\n⚡ PERFORMANCE SUMMARY")
            performance_monitor.log_performance_summary()
            
            # Show cache stats
            cache_stats = market_cache.get_stats()
            print(f"💾 Cache: {cache_stats['hit_rate_percent']:.1f}% hit rate, {cache_stats['cache_entries']} entries, {cache_stats['total_size_mb']:.1f}MB")
            
            # Cleanup expired cache entries
            expired_count = market_cache.cleanup_expired()
            if expired_count > 0:
                print(f"🧹 Cleaned {expired_count} expired cache entries")

def main():
    print("⚔️ BOT ESPARTANO MULTI-CRIPTO - MODO INSTITUCIONAL ⚔️")
    print(f"📊 Símbolos: {len(SYMBOLS)} | Timeframe: {time_frame}")
    print(f"🔄 Intervalo: {CHECK_INTERVAL_SECONDS} segundos | Loop Infinito: {ENABLE_INFINITE_LOOP}")
    
    # Show optimization status
    if USE_HIGHER_TIMEFRAMES:
        print(f"🚀 TIMEFRAMES SUPERIORES ACTIVADOS - Menos ruido, mejor calidad")
    if USE_AI_VALIDATION:
        print(f"🤖 VALIDACIÓN IA ACTIVADA - Threshold: {AI_CONFIDENCE_THRESHOLD}")
    else:
        print(f"📊 VALIDACIÓN IA DESACTIVADA - Solo análisis técnico")
    
    print("=" * 80)

    notifier = None
    notify_open = False
    notify_close = False
    notify_alerts = False

    if ENABLE_NOTIFICATIONS:
        try:
            notifier = TelegramNotifier(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
            notify_open = NOTIFY_POSITION_OPENED
            notify_close = NOTIFY_POSITION_CLOSED
            notify_alerts = NOTIFY_ALERTS
        except Exception as exc:
            logging.error(f"No se pudo inicializar TelegramNotifier: {exc}")
            notifier = None

    # Initialize strategy manager
    strategy_manager = StrategyManager(
        notifier=notifier,
        notify_on_open=notify_open,
        notify_on_close=notify_close,
        notify_alerts=notify_alerts,
    )

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
