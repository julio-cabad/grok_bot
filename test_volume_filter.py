#!/usr/bin/env python3
"""
Test del VolumeFilter con datos reales
Verifica que el filtro funcione correctamente
"""

import pandas as pd
import numpy as np
from ai.filters.base.filter_config import FilterConfig
from ai.filters.volume.volume_filter import VolumeFilter

def create_test_data():
    """Crea datos de prueba simulando datos reales de mercado"""
    # Simular 100 velas con volúmenes variables
    np.random.seed(42)  # Para reproducibilidad
    
    # Volúmenes base con variación realista
    base_volume = 1000000
    volumes = []
    
    for i in range(100):
        # Simular diferentes condiciones de volumen
        if i < 20:
            # Volumen normal
            vol = base_volume * np.random.uniform(0.8, 1.2)
        elif i < 40:
            # Período de volumen bajo
            vol = base_volume * np.random.uniform(0.3, 0.7)
        elif i < 60:
            # Período de volumen alto
            vol = base_volume * np.random.uniform(1.5, 2.5)
        else:
            # Volumen muy bajo
            vol = base_volume * np.random.uniform(0.1, 0.4)
        
        volumes.append(vol)
    
    # Crear DataFrame con estructura similar a datos reales
    data = {
        'open': [50000 + np.random.uniform(-100, 100) for _ in range(100)],
        'high': [50100 + np.random.uniform(-100, 100) for _ in range(100)],
        'low': [49900 + np.random.uniform(-100, 100) for _ in range(100)],
        'close': [50000 + np.random.uniform(-100, 100) for _ in range(100)],
        'volume': volumes
    }
    
    return pd.DataFrame(data)

def test_volume_filter():
    """Test principal del VolumeFilter"""
    print("🧪 Testing VolumeFilter con datos simulados...")
    
    # Crear configuración
    config = FilterConfig()
    
    # Crear filtro
    volume_filter = VolumeFilter(config)
    
    # Crear datos de prueba
    df = create_test_data()
    
    print(f"📊 Datos creados: {len(df)} velas")
    print(f"   Volumen actual: {df['volume'].iloc[-1]:,.0f}")
    print(f"   Volumen promedio últimas 20: {df['volume'].tail(21).iloc[:-1].mean():,.0f}")
    
    # Preparar market_data
    market_data = {'df': df}
    
    # Test con diferentes scores
    test_cases = [
        ("BTCUSDT", "SHORT", 10.0),  # Score alto con volumen bajo
        ("BTCUSDT", "LONG", 5.0),    # Score medio
        ("BTCUSDT", "SHORT", 3.0),   # Score bajo
    ]
    
    for symbol, signal_type, score in test_cases:
        print(f"\n🔍 Testing {symbol} {signal_type} con score {score}")
        
        # Aplicar filtro
        result = volume_filter.apply(symbol, signal_type, score, market_data)
        
        print(f"   Resultado: {result.filter_name}")
        print(f"   Aplicado: {result.applied}")
        print(f"   Ajuste: {result.score_adjustment:+.1f}")
        print(f"   Razón: {result.reason}")
        if result.warning:
            print(f"   ⚠️ Advertencia: {result.warning}")
        if result.metrics:
            print(f"   📊 Ratio volumen: {result.metrics.get('volume_ratio', 0):.2f}")
    
    # Test de análisis completo
    print(f"\n📈 Análisis completo de volumen:")
    analysis = volume_filter.get_volume_analysis(df)
    
    if 'error' not in analysis:
        print(f"   Ratio actual: {analysis['volume_ratio']:.2f}")
        print(f"   Percentil: {analysis['volume_metrics']['volume_percentile']:.1f}%")
        print(f"   Tendencia: {analysis['volume_metrics']['volume_trend']}")
        print(f"   Es spike: {analysis['volume_metrics']['is_volume_spike']}")
        print(f"   Es seco: {analysis['volume_metrics']['is_volume_dry']}")
    
    # Estadísticas del filtro
    print(f"\n📊 Estadísticas del filtro:")
    stats = volume_filter.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Test de VolumeFilter completado!")

if __name__ == "__main__":
    test_volume_filter()