#!/usr/bin/env python3
"""
Script de prueba para verificar la conexión con Binance y mostrar datos OHLCV
"""

import logging
from config.settings import *
from bnb.binance import RobotBinance
from indicators.technical_indicators import TechnicalAnalyzer
from ai.kimi_client import GeminiClient

# Configurar logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Iniciando prueba de conexión con Binance...", time_frame)

    # Elegir un símbolo para probar (usando el primero de la lista)
    symbol = SYMBOLS[0]  # BTCUSDT
    print(f"📊 Probando con símbolo: {symbol}")

    # Inicializar el cliente de Binance
    try:
        robot = RobotBinance(pair=symbol, temporality=time_frame)
        print("✅ Conexión exitosa con Binance!")
    except Exception as e:
        print(f"❌ Error al conectar con Binance: {e}")
        return

    # Probar obtener precio actual
    try:
        price = robot.symbol_price(symbol)
        if price:
            print(f"💰 Precio actual de {symbol}: ${price:.2f}")
        else:
            print(f"⚠️ No se pudo obtener el precio de {symbol}")
    except Exception as e:
        print(f"❌ Error al obtener precio: {e}")

    # Obtener datos OHLCV
    try:
        print(f"\n📈 Obteniendo datos OHLCV de {symbol} (últimas {CANDLES_LIMIT} velas de {time_frame})...")
        df = robot.candlestick(limit=CANDLES_LIMIT)

        if df.empty:
            print("⚠️ No se obtuvieron datos OHLCV")
        else:
            print(f"✅ Datos obtenidos: {len(df)} velas")

            # Integrar Trend Magic al DataFrame
            analyzer = TechnicalAnalyzer(symbol=symbol, timeframe=time_frame)
            analyzer.df = df
            analyzer.trend_magic()
            analyzer.squeeze_momentum()
            analyzer.calculate_rsi()
            analyzer.calculate_bollinger_bands()
            analyzer.calculate_macd()
            analyzer.calculate_stochastic()
            df = analyzer.df

            print("\n📊 Últimas 10 velas OHLCV con Indicadores Técnicos:")

            # Mostrar las últimas 10 velas en formato tabular
            df = df.drop(columns=["ATR"])
            recent_data = df.tail(10)
            print(f"{'Timestamp':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<12} {'MagicTrend':<12} {'Color':^6} {'Momentum':^8} {'Squeeze':^7} {'RSI':^6} {'BB_Mid':<10} {'BB_Up':<10} {'BB_Low':<10} {'MACD':<10} {'Stoch_K':<8} {'Stoch_D':<8}")
            print("-" * 170)

            for timestamp, row in recent_data.iterrows():
                print(f"{timestamp.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{row['open']:<10.4f} "
                      f"{row['high']:<10.4f} "
                      f"{row['low']:<10.4f} "
                      f"{row['close']:<10.4f} "
                      f"{row['volume']:<12.2f} "
                      f"{row['MagicTrend']:<12.4f} "
                      f"{row['MagicTrend_Color']:^6} "
                      f"{row['momentum_color']:^8} "
                      f"{row['squeeze_color']:^7} "
                      f"{row['RSI_14']:^6.2f} "
                      f"{row['BB_middle_20']:<10.4f} "
                      f"{row['BB_upper_20']:<10.4f} "
                      f"{row['BB_lower_20']:<10.4f} "
                      f"{row['MACD_12_26_9']:<10.4f} "
                      f"{row['STOCH_K_14_3']:<8.2f} "
                      f"{row['STOCH_D_14_3']:<8.2f}")

            # Resumen estadístico básico
            print("\n📈 Resumen estadístico:")
            print(f"Precio promedio: ${df['close'].mean():.2f}")
            print(f"Precio máximo: ${df['high'].max():.2f}")
            print(f"Precio mínimo: ${df['low'].min():.2f}")
            print(f"Volumen total: {df['volume'].sum():.2f}")

    except Exception as e:
        print(f"❌ Error al obtener datos OHLCV: {e}")

    print("\n🎯 Prueba completada!")

    # Prueba de Gemini AI
    print("\n🤖 Probando conexión con Gemini AI...")
    try:
        gemini = GeminiClient()
        response = gemini.query("¿Qué modelo de IA eres?")
        print(f"🎯 Respuesta de Gemini: {response}")

        # Análisis institucional con datos técnicos
        print("\n🔍 Realizando análisis institucional con Gemini...")
        current_data = df.iloc[-1]  # Última vela
        data_summary = f"""
        DATOS ACTUALES DE BTC/USDT (4h):
        - Precio actual: ${current_data['close']:.2f}
        - RSI (14): {current_data['RSI_14']:.2f}
        - MACD: {current_data['MACD_12_26_9']:.4f}
        - MACD Signal: {current_data['MACD_signal_12_26_9']:.4f}
        - MACD Histogram: {current_data['MACD_hist_12_26_9']:.4f}
        - Bollinger Bands:
          * Upper: {current_data['BB_upper_20']:.2f}
          * Middle: {current_data['BB_middle_20']:.2f}
          * Lower: {current_data['BB_lower_20']:.2f}
        - Estocástico:
          * %K: {current_data['STOCH_K_14_3']:.2f}
          * %D: {current_data['STOCH_D_14_3']:.2f}
        - Trend Magic: {current_data['MagicTrend_Color']} ({current_data['MagicTrend']:.2f})
        - Squeeze Momentum: {current_data['squeeze_color']} ({current_data['momentum_color']})

        ESTADÍSTICAS GENERALES:
        - Precio promedio: ${df['close'].mean():.2f}
        - Precio máximo: ${df['high'].max():.2f}
        - Precio mínimo: ${df['low'].min():.2f}
        - Volumen promedio: {df['volume'].mean():.0f}
        """

        analysis = gemini.analyze_market_data(data_summary)
        print("📊 ANÁLISIS INSTITUCIONAL GEMINI:")
        print("=" * 80)
        print(analysis["analysis"])
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error con Gemini AI: {e}")

if __name__ == "__main__":
    main()
