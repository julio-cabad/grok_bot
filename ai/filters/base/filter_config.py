"""
Sistema de configuración para filtros de trading
Proporciona configuración flexible y validada para todos los filtros
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os
from pathlib import Path


@dataclass
class VolumeConfig:
    """Configuración para el filtro de volumen"""
    enabled: bool = True
    min_volume_threshold: float = 0.8  # 80% del promedio
    low_volume_threshold: float = 0.5  # 50% del promedio
    high_volume_bonus_threshold: float = 1.2  # 120% del promedio
    lookback_periods: int = 20
    max_score_low_volume: int = 4  # Score máximo con volumen muy bajo
    max_score_min_volume: int = 6  # Score máximo con volumen bajo
    volume_bonus: float = 0.5  # Bonus por volumen alto
    
    def __post_init__(self):
        """Validación de configuración de volumen"""
        if not 0.1 <= self.min_volume_threshold <= 2.0:
            raise ValueError(f"min_volume_threshold debe estar entre 0.1 y 2.0, recibido: {self.min_volume_threshold}")
        
        if not 0.1 <= self.low_volume_threshold <= 1.0:
            raise ValueError(f"low_volume_threshold debe estar entre 0.1 y 1.0, recibido: {self.low_volume_threshold}")
        
        if self.low_volume_threshold >= self.min_volume_threshold:
            raise ValueError("low_volume_threshold debe ser menor que min_volume_threshold")
        
        if not 5 <= self.lookback_periods <= 100:
            raise ValueError(f"lookback_periods debe estar entre 5 y 100, recibido: {self.lookback_periods}")


@dataclass
class MomentumConfig:
    """Configuración para el filtro de momentum"""
    enabled: bool = True
    conflict_penalty: float = -3.0  # Penalización por conflicto direccional
    neutral_zone_threshold: float = 0.01  # MACD entre -threshold y +threshold = zona neutral
    strong_momentum_threshold: float = 0.05  # MACD > threshold = momentum fuerte
    momentum_bonus: float = 0.5  # Bonus por momentum fuerte alineado
    macd_fast_period: int = 12  # Período EMA rápida para MACD
    macd_slow_period: int = 26  # Período EMA lenta para MACD
    macd_signal_period: int = 9  # Período para línea de señal MACD
    
    def __post_init__(self):
        """Validación de configuración de momentum"""
        if not -10.0 <= self.conflict_penalty <= 0.0:
            raise ValueError(f"conflict_penalty debe estar entre -10.0 y 0.0, recibido: {self.conflict_penalty}")
        
        if not 0.001 <= self.neutral_zone_threshold <= 0.1:
            raise ValueError(f"neutral_zone_threshold debe estar entre 0.001 y 0.1, recibido: {self.neutral_zone_threshold}")
        
        if self.strong_momentum_threshold <= self.neutral_zone_threshold:
            raise ValueError("strong_momentum_threshold debe ser mayor que neutral_zone_threshold")
        
        if not 0.0 <= self.momentum_bonus <= 5.0:
            raise ValueError(f"momentum_bonus debe estar entre 0.0 y 5.0, recibido: {self.momentum_bonus}")
        
        if not 5 <= self.macd_fast_period <= 20:
            raise ValueError(f"macd_fast_period debe estar entre 5 y 20, recibido: {self.macd_fast_period}")
        
        if not 20 <= self.macd_slow_period <= 50:
            raise ValueError(f"macd_slow_period debe estar entre 20 y 50, recibido: {self.macd_slow_period}")
        
        if not 5 <= self.macd_signal_period <= 15:
            raise ValueError(f"macd_signal_period debe estar entre 5 y 15, recibido: {self.macd_signal_period}")
        
        if self.macd_fast_period >= self.macd_slow_period:
            raise ValueError("macd_fast_period debe ser menor que macd_slow_period")


@dataclass
class StructureConfig:
    """Configuración para el filtro de estructura"""
    enabled: bool = True
    structure_penalty: float = -2.0  # Penalización por trade contra estructura
    structure_bonus: float = 1.0  # Bonus por trade con estructura
    lookback_periods: int = 20  # Períodos para análisis de estructura
    min_swing_strength: int = 3  # Mínimo número de HH/HL para confirmar tendencia
    bos_invalidation_enabled: bool = True  # Habilitar invalidación por Break of Structure
    sideways_threshold: float = 0.02  # Threshold para detectar mercado lateral (2%)
    
    def __post_init__(self):
        """Validación de configuración de estructura"""
        if not -10.0 <= self.structure_penalty <= 0.0:
            raise ValueError(f"structure_penalty debe estar entre -10.0 y 0.0, recibido: {self.structure_penalty}")
        
        if not 0.0 <= self.structure_bonus <= 5.0:
            raise ValueError(f"structure_bonus debe estar entre 0.0 y 5.0, recibido: {self.structure_bonus}")
        
        if not 10 <= self.lookback_periods <= 100:
            raise ValueError(f"lookback_periods debe estar entre 10 y 100, recibido: {self.lookback_periods}")
        
        if not 2 <= self.min_swing_strength <= 10:
            raise ValueError(f"min_swing_strength debe estar entre 2 y 10, recibido: {self.min_swing_strength}")


@dataclass
class BTCConfig:
    """Configuración para el filtro de correlación BTC"""
    enabled: bool = True
    btc_penalty: float = -1.0  # Penalización por momentum BTC contrario
    btc_bonus: float = 0.5  # Bonus por momentum BTC alineado
    min_timeframe: str = "4h"  # Timeframe mínimo para aplicar filtro
    correlation_threshold: float = 0.7  # Threshold de correlación mínima
    btc_symbol: str = "BTCUSDT"  # Símbolo de Bitcoin a monitorear
    momentum_threshold: float = 0.01  # Threshold para momentum BTC neutral
    
    def __post_init__(self):
        """Validación de configuración BTC"""
        if not -10.0 <= self.btc_penalty <= 0.0:
            raise ValueError(f"btc_penalty debe estar entre -10.0 y 0.0, recibido: {self.btc_penalty}")
        
        if not 0.0 <= self.btc_bonus <= 5.0:
            raise ValueError(f"btc_bonus debe estar entre 0.0 y 5.0, recibido: {self.btc_bonus}")
        
        valid_timeframes = ["1h", "2h", "4h", "6h", "8h", "12h", "1d"]
        if self.min_timeframe not in valid_timeframes:
            raise ValueError(f"min_timeframe debe ser uno de {valid_timeframes}, recibido: {self.min_timeframe}")
        
        if not 0.1 <= self.correlation_threshold <= 1.0:
            raise ValueError(f"correlation_threshold debe estar entre 0.1 y 1.0, recibido: {self.correlation_threshold}")


@dataclass
class FilterConfig:
    """
    Configuración principal del sistema de filtros
    
    Attributes:
        enabled: Si el sistema de filtros está habilitado globalmente
        mode: Modo de operación ("warning" o "enforcement")
        volume: Configuración del filtro de volumen
        momentum: Configuración del filtro de momentum
        structure: Configuración del filtro de estructura
        btc_correlation: Configuración del filtro de correlación BTC
        max_total_penalty: Penalización máxima total permitida
        min_final_score: Score mínimo después de filtros
    """
    enabled: bool = True
    mode: str = "enforcement"  # "warning" o "enforcement"
    
    # Configuraciones de filtros individuales
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    btc_correlation: BTCConfig = field(default_factory=BTCConfig)
    
    # Configuraciones globales
    max_total_penalty: float = -5.0  # Penalización máxima total
    min_final_score: float = 0.0  # Score mínimo después de filtros
    max_final_score: float = 10.0  # Score máximo después de filtros
    
    # Configuración de logging
    log_level: str = "INFO"
    detailed_logging: bool = True
    
    def __post_init__(self):
        """Validación de configuración global"""
        valid_modes = ["warning", "enforcement"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode debe ser uno de {valid_modes}, recibido: {self.mode}")
        
        if not -20.0 <= self.max_total_penalty <= 0.0:
            raise ValueError(f"max_total_penalty debe estar entre -20.0 y 0.0, recibido: {self.max_total_penalty}")
        
        if not 0.0 <= self.min_final_score <= 10.0:
            raise ValueError(f"min_final_score debe estar entre 0.0 y 10.0, recibido: {self.min_final_score}")
        
        if not 0.0 <= self.max_final_score <= 10.0:
            raise ValueError(f"max_final_score debe estar entre 0.0 y 10.0, recibido: {self.max_final_score}")
        
        if self.min_final_score >= self.max_final_score:
            raise ValueError("min_final_score debe ser menor que max_final_score")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level debe ser uno de {valid_log_levels}, recibido: {self.log_level}")
    
    def is_filter_enabled(self, filter_name: str) -> bool:
        """
        Verifica si un filtro específico está habilitado
        
        Args:
            filter_name: Nombre del filtro ("volume", "momentum", "structure", "btc_correlation")
            
        Returns:
            True si el filtro está habilitado
        """
        if not self.enabled:
            return False
        
        filter_configs = {
            "volume": self.volume,
            "momentum": self.momentum,
            "structure": self.structure,
            "btc_correlation": self.btc_correlation
        }
        
        if filter_name not in filter_configs:
            raise ValueError(f"Filtro desconocido: {filter_name}")
        
        return filter_configs[filter_name].enabled
    
    def get_filter_config(self, filter_name: str):
        """
        Obtiene la configuración de un filtro específico
        
        Args:
            filter_name: Nombre del filtro
            
        Returns:
            Configuración del filtro solicitado
        """
        filter_configs = {
            "volume": self.volume,
            "momentum": self.momentum,
            "structure": self.structure,
            "btc_correlation": self.btc_correlation
        }
        
        if filter_name not in filter_configs:
            raise ValueError(f"Filtro desconocido: {filter_name}")
        
        return filter_configs[filter_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario"""
        return {
            'enabled': self.enabled,
            'mode': self.mode,
            'volume': {
                'enabled': self.volume.enabled,
                'min_volume_threshold': self.volume.min_volume_threshold,
                'low_volume_threshold': self.volume.low_volume_threshold,
                'high_volume_bonus_threshold': self.volume.high_volume_bonus_threshold,
                'lookback_periods': self.volume.lookback_periods,
                'max_score_low_volume': self.volume.max_score_low_volume,
                'max_score_min_volume': self.volume.max_score_min_volume,
                'volume_bonus': self.volume.volume_bonus
            },
            'momentum': {
                'enabled': self.momentum.enabled,
                'momentum_penalty': self.momentum.momentum_penalty,
                'neutral_threshold': self.momentum.neutral_threshold,
                'strong_momentum_threshold': self.momentum.strong_momentum_threshold,
                'extra_penalty_strong_momentum': self.momentum.extra_penalty_strong_momentum
            },
            'structure': {
                'enabled': self.structure.enabled,
                'structure_penalty': self.structure.structure_penalty,
                'structure_bonus': self.structure.structure_bonus,
                'lookback_periods': self.structure.lookback_periods,
                'min_swing_strength': self.structure.min_swing_strength,
                'bos_invalidation_enabled': self.structure.bos_invalidation_enabled,
                'sideways_threshold': self.structure.sideways_threshold
            },
            'btc_correlation': {
                'enabled': self.btc_correlation.enabled,
                'btc_penalty': self.btc_correlation.btc_penalty,
                'btc_bonus': self.btc_correlation.btc_bonus,
                'min_timeframe': self.btc_correlation.min_timeframe,
                'correlation_threshold': self.btc_correlation.correlation_threshold,
                'btc_symbol': self.btc_correlation.btc_symbol,
                'momentum_threshold': self.btc_correlation.momentum_threshold
            },
            'max_total_penalty': self.max_total_penalty,
            'min_final_score': self.min_final_score,
            'max_final_score': self.max_final_score,
            'log_level': self.log_level,
            'detailed_logging': self.detailed_logging
        }


def load_filter_config(config_path: Optional[str] = None) -> FilterConfig:
    """
    Carga configuración de filtros desde archivo
    
    Args:
        config_path: Ruta al archivo de configuración, None para usar default
        
    Returns:
        Configuración de filtros cargada y validada
    """
    if config_path is None:
        # Buscar archivo de configuración en ubicaciones estándar
        possible_paths = [
            "config/filter_settings.py",
            "filter_settings.py",
            os.path.join(os.path.dirname(__file__), "../../../config/filter_settings.py")
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        try:
            # Importar configuración desde archivo Python
            import importlib.util
            spec = importlib.util.spec_from_file_location("filter_settings", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            if hasattr(config_module, 'FILTER_SETTINGS'):
                settings = config_module.FILTER_SETTINGS
                return _create_config_from_dict(settings)
            else:
                raise ValueError("Archivo de configuración no contiene FILTER_SETTINGS")
                
        except Exception as e:
            raise ValueError(f"Error cargando configuración desde {config_path}: {e}")
    
    # Retornar configuración por defecto si no se encuentra archivo
    return FilterConfig()


def _create_config_from_dict(settings: Dict[str, Any]) -> FilterConfig:
    """Crea FilterConfig desde diccionario de configuración"""
    
    # Configuración de volumen
    volume_config = VolumeConfig()
    if 'volume_filter' in settings:
        vol_settings = settings['volume_filter']
        volume_config = VolumeConfig(
            enabled=vol_settings.get('enabled', True),
            min_volume_threshold=vol_settings.get('min_volume_threshold', 0.8),
            low_volume_threshold=vol_settings.get('low_volume_threshold', 0.5),
            high_volume_bonus_threshold=vol_settings.get('high_volume_bonus_threshold', 1.2),
            lookback_periods=vol_settings.get('lookback_periods', 20),
            max_score_low_volume=vol_settings.get('max_score_low_volume', 4),
            max_score_min_volume=vol_settings.get('max_score_min_volume', 6),
            volume_bonus=vol_settings.get('volume_bonus', 0.5)
        )
    
    # Configuración de momentum
    momentum_config = MomentumConfig()
    if 'momentum_filter' in settings:
        mom_settings = settings['momentum_filter']
        momentum_config = MomentumConfig(
            enabled=mom_settings.get('enabled', True),
            momentum_penalty=mom_settings.get('momentum_penalty', -3.0),
            neutral_threshold=mom_settings.get('neutral_threshold', 0.01),
            strong_momentum_threshold=mom_settings.get('strong_momentum_threshold', 0.05),
            extra_penalty_strong_momentum=mom_settings.get('extra_penalty_strong_momentum', -1.0)
        )
    
    # Configuración de estructura
    structure_config = StructureConfig()
    if 'structure_filter' in settings:
        struct_settings = settings['structure_filter']
        structure_config = StructureConfig(
            enabled=struct_settings.get('enabled', True),
            structure_penalty=struct_settings.get('structure_penalty', -2.0),
            structure_bonus=struct_settings.get('structure_bonus', 1.0),
            lookback_periods=struct_settings.get('lookback_periods', 20),
            min_swing_strength=struct_settings.get('min_swing_strength', 3),
            bos_invalidation_enabled=struct_settings.get('bos_invalidation_enabled', True),
            sideways_threshold=struct_settings.get('sideways_threshold', 0.02)
        )
    
    # Configuración de BTC
    btc_config = BTCConfig()
    if 'btc_correlation_filter' in settings:
        btc_settings = settings['btc_correlation_filter']
        btc_config = BTCConfig(
            enabled=btc_settings.get('enabled', True),
            btc_penalty=btc_settings.get('btc_penalty', -1.0),
            btc_bonus=btc_settings.get('btc_bonus', 0.5),
            min_timeframe=btc_settings.get('min_timeframe', "4h"),
            correlation_threshold=btc_settings.get('correlation_threshold', 0.7),
            btc_symbol=btc_settings.get('btc_symbol', "BTCUSDT"),
            momentum_threshold=btc_settings.get('momentum_threshold', 0.01)
        )
    
    # Configuración principal
    return FilterConfig(
        enabled=settings.get('enabled', True),
        mode=settings.get('mode', 'enforcement'),
        volume=volume_config,
        momentum=momentum_config,
        structure=structure_config,
        btc_correlation=btc_config,
        max_total_penalty=settings.get('max_total_penalty', -5.0),
        min_final_score=settings.get('min_final_score', 0.0),
        max_final_score=settings.get('max_final_score', 10.0),
        log_level=settings.get('log_level', 'INFO'),
        detailed_logging=settings.get('detailed_logging', True)
    )