#!/usr/bin/env python3
"""
Test Real Market Data
Obtiene datos reales de Binance para las últimas 500 velas con todos los indicadores
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_binance_data(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """
    Obtiene datos reales de Binance
    
    Args:
        symbol: Par de trading (ej: BTCUSDT, ETHUSDT)
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Número de velas (máximo 1000)
    """
    print(f"📡 Obteniendo datos reales de Binance...")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Limit: {limit} velas")
    
    try:
        # Binance API endpoint
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float).round(2)
        df['high'] = df['high'].astype(float).round(2)
        df['low'] = df['low'].astype(float).round(2)
        df['close'] = df['close'].astype(float).round(2)
        df['volume'] = df['volume'].astype(float).round(0)
        
        # Keep only OHLCV columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ Datos obtenidos exitosamente:")
        print(f"   Velas: {len(df)}")
        print(f"   Rango: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
        print(f"   Precio: ${df['close'].iloc[0]:.2f} - ${df['close'].iloc[-1]:.2f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error obteniendo datos de Binance: {e}")
        return None
    except Exception as e:
        print(f"❌ Error procesando datos: {e}")
        return None

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade todos los indicadores técnicos que usa nuestro sistema"""
    
    print("📊 Calculando indicadores técnicos...")
    
    df = df.copy()
    
    # MACD (12, 26, 9) - Exactamente como en technical_indicators.py
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD_12_26_9'] = (exp1 - exp2).round(4)
    df['MACD_signal_12_26_9'] = df['MACD_12_26_9'].ewm(span=9).mean().round(4)
    df['MACD_hist_12_26_9'] = (df['MACD_12_26_9'] - df['MACD_signal_12_26_9']).round(4)
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = (100 - (100 / (1 + rs))).round(2)
    
    # Stochastic (14, 3, 3) - Exactamente como en technical_indicators.py
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    k_percent = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['STOCH_K_14_3'] = k_percent.rolling(window=3).mean().round(2)
    df['STOCH_D_14_3'] = df['STOCH_K_14_3'].rolling(window=3).mean().round(2)
    
    # Bollinger Bands (20, 2) - Exactamente como en technical_indicators.py
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper_20'] = (bb_middle + (bb_std * 2)).round(2)
    df['BB_middle_20'] = bb_middle.round(2)
    df['BB_lower_20'] = (bb_middle - (bb_std * 2)).round(2)
    
    print("✅ Indicadores calculados:")
    print("   • MACD (12, 26, 9)")
    print("   • RSI (14)")
    print("   • Stochastic (14, 3, 3)")
    print("   • Bollinger Bands (20, 2)")
    
    return df

def print_dataframe_summary(df: pd.DataFrame):
    """Imprime resumen del DataFrame"""
    print("\n📊 RESUMEN DEL DATAFRAME:")
    print("=" * 80)
    print(f"📈 Total de velas: {len(df)}")
    print(f"💰 Rango de precios: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"📊 Precio actual: ${df['close'].iloc[-1]:.2f}")
    print(f"📅 Período: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")
    
    # Estadísticas de indicadores actuales
    latest = df.iloc[-1]
    print(f"\n🔍 INDICADORES ACTUALES:")
    print(f"   MACD: {latest['MACD_12_26_9']:.4f}")
    print(f"   MACD Signal: {latest['MACD_signal_12_26_9']:.4f}")
    print(f"   MACD Hist: {latest['MACD_hist_12_26_9']:.4f}")
    print(f"   RSI: {latest['RSI_14']:.2f}")
    print(f"   Stoch %K: {latest['STOCH_K_14_3']:.2f}")
    print(f"   Stoch %D: {latest['STOCH_D_14_3']:.2f}")
    print(f"   BB Upper: ${latest['BB_upper_20']:.2f}")
    print(f"   BB Middle: ${latest['BB_middle_20']:.2f}")
    print(f"   BB Lower: ${latest['BB_lower_20']:.2f}")

def main():
    """Función principal"""
    print("🚀 TEST DATOS REALES DE MERCADO")
    print("=" * 80)
    print(f"🕐 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuración (igual que nuestro sistema)
    SYMBOL = "BTCUSDT"
    INTERVAL = "1h"  # 1 hora como usa el sistema
    LIMIT = 400      # 500 velas como usa el sistema
    
    try:
        # 1. Obtener datos reales de Binance
        df = get_binance_data(SYMBOL, INTERVAL, LIMIT)
        
        if df is None:
            print("❌ No se pudieron obtener datos. Terminando.")
            return
        
        # 2. Añadir indicadores técnicos
        df_with_indicators = add_technical_indicators(df)
        
        # 3. Limpiar datos (remover NaN)
        df_clean = df_with_indicators.dropna()
        
        print(f"\n📊 Datos limpios: {len(df_clean)} velas (removidas {len(df_with_indicators) - len(df_clean)} con NaN)")
        
        # 4. Mostrar resumen
        print_dataframe_summary(df_clean)
        
        # 5. Configurar pandas para mejor visualización
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        # 6. Mostrar las últimas 20 velas para comparativas
        print(f"\n🔍 ÚLTIMAS 20 VELAS CON INDICADORES:")
        print("-" * 120)
        print(df_clean.tail(300).to_string(index=False))
        
        # 7. Guardar en archivo CSV para análisis posterior
        filename = f"real_market_data_{SYMBOL}_{INTERVAL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_clean.to_csv(filename, index=False)
        print(f"\n💾 Datos guardados en: {filename}")
        
        print(f"\n✅ Test completado exitosamente!")
        print(f"🎯 Datos listos para comparativas con otros sistemas")
        
    except Exception as e:
        print(f"💥 Error en el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()