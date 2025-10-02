"""
Configuración de Filtros de Trading Inteligentes
Archivo de configuración centralizada para todos los filtros del sistema
"""

# Configuración principal del sistema de filtros
FILTER_SETTINGS = {
    # Configuración global
    "enabled": True,  # Habilitar/deshabilitar todo el sistema de filtros
    "mode": "enforcement",  # "warning" (solo alerta) o "enforcement" (aplica penalizaciones)
    
    # Configuración del filtro de volumen
    "volume_filter": {
        "enabled": True,
        "min_volume_threshold": 0.8,  # 80% del promedio - score máximo 6
        "low_volume_threshold": 0.5,  # 50% del promedio - score máximo 4
        "high_volume_bonus_threshold": 1.2,  # 120% del promedio - bonus +0.5
        "lookback_periods": 20,  # Períodos para calcular promedio de volumen
        "max_score_low_volume": 4,  # Score máximo con volumen muy bajo
        "max_score_min_volume": 6,  # Score máximo con volumen bajo
        "volume_bonus": 0.5  # Bonus por volumen alto
    },
    
    # Configuración del filtro de momentum
    "momentum_filter": {
        "enabled": True,
        "conflict_penalty": -3.0,  # Penalización por conflicto direccional
        "neutral_zone_threshold": 0.01,  # MACD entre -0.01 y +0.01 = zona neutral
        "strong_momentum_threshold": 0.05,  # MACD > 0.05 = momentum fuerte
        "momentum_bonus": 0.5,  # Bonus por momentum fuerte alineado
        "macd_fast_period": 12,  # Período EMA rápida para MACD
        "macd_slow_period": 26,  # Período EMA lenta para MACD
        "macd_signal_period": 9  # Período para línea de señal MACD
    },
    
    # Configuración del filtro de estructura
    "structure_filter": {
        "enabled": True,
        "structure_penalty": -2.0,  # Penalización por trade contra estructura
        "structure_bonus": 1.0,  # Bonus por trade alineado con estructura
        "lookback_periods": 20,  # Períodos para análisis de estructura
        "min_swing_strength": 3,  # Mínimo HH/HL o LH/LL para confirmar tendencia
        "bos_invalidation_enabled": True,  # Habilitar invalidación por Break of Structure
        "sideways_threshold": 0.02  # 2% - threshold para detectar mercado lateral
    },
    
    # Configuración del filtro de correlación BTC
    "btc_correlation_filter": {
        "enabled": True,
        "btc_penalty": -1.0,  # Penalización por momentum BTC contrario
        "btc_bonus": 0.5,  # Bonus por momentum BTC alineado
        "min_timeframe": "4h",  # Timeframe mínimo para aplicar filtro
        "correlation_threshold": 0.7,  # Correlación mínima para aplicar filtro
        "btc_symbol": "BTCUSDT",  # Símbolo de Bitcoin a monitorear
        "momentum_threshold": 0.01  # Threshold para momentum BTC neutral
    },
    
    # Configuraciones globales de límites
    "max_total_penalty": -5.0,  # Penalización máxima total permitida
    "min_final_score": 0.0,  # Score mínimo después de filtros
    "max_final_score": 10.0,  # Score máximo después de filtros
    
    # Configuración de logging
    "log_level": "INFO",  # Nivel de logging: DEBUG, INFO, WARNING, ERROR
    "detailed_logging": True  # Habilitar logging detallado de cada filtro
}


# Configuraciones preestablecidas para diferentes modos de operación
CONSERVATIVE_MODE = {
    **FILTER_SETTINGS,
    "mode": "enforcement",
    "volume_filter": {
        **FILTER_SETTINGS["volume_filter"],
        "min_volume_threshold": 0.9,  # Más estricto con volumen
        "momentum_penalty": -4.0  # Penalización mayor
    },
    "momentum_filter": {
        **FILTER_SETTINGS["momentum_filter"],
        "conflict_penalty": -4.0  # Más conservador
    },
    "structure_filter": {
        **FILTER_SETTINGS["structure_filter"],
        "structure_penalty": -3.0  # Penalización mayor contra estructura
    }
}

AGGRESSIVE_MODE = {
    **FILTER_SETTINGS,
    "mode": "enforcement",
    "volume_filter": {
        **FILTER_SETTINGS["volume_filter"],
        "min_volume_threshold": 0.6,  # Menos estricto con volumen
        "momentum_penalty": -2.0  # Penalización menor
    },
    "momentum_filter": {
        **FILTER_SETTINGS["momentum_filter"],
        "conflict_penalty": -2.0  # Menos conservador
    },
    "structure_filter": {
        **FILTER_SETTINGS["structure_filter"],
        "structure_penalty": -1.0  # Penalización menor contra estructura
    }
}

WARNING_MODE = {
    **FILTER_SETTINGS,
    "mode": "warning"  # Solo advertencias, sin penalizaciones reales
}


# Función para obtener configuración según el modo
def get_filter_config(mode: str = "default"):
    """
    Obtiene configuración de filtros según el modo especificado
    
    Args:
        mode: Modo de configuración ("default", "conservative", "aggressive", "warning")
        
    Returns:
        Diccionario de configuración
    """
    modes = {
        "default": FILTER_SETTINGS,
        "conservative": CONSERVATIVE_MODE,
        "aggressive": AGGRESSIVE_MODE,
        "warning": WARNING_MODE
    }
    
    if mode not in modes:
        raise ValueError(f"Modo desconocido: {mode}. Modos disponibles: {list(modes.keys())}")
    
    return modes[mode]