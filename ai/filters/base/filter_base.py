"""
Clase base para todos los filtros de trading
Define la interfaz común que deben implementar todos los filtros
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging

from .filter_result import FilterResult
from .filter_config import FilterConfig


class FilterBase(ABC):
    """
    Clase base abstracta para todos los filtros de trading
    
    Define la interfaz común y funcionalidad compartida
    """
    
    def __init__(self, config: FilterConfig, filter_name: str):
        """
        Inicializa el filtro base
        
        Args:
            config: Configuración global de filtros
            filter_name: Nombre único del filtro
        """
        self.config = config
        self.filter_name = filter_name
        self.logger = logging.getLogger(f"Filter.{filter_name}")
        
        # Configurar nivel de logging
        if hasattr(config, 'log_level'):
            self.logger.setLevel(getattr(logging, config.log_level))
        
        # Estadísticas del filtro
        self.applications_count = 0
        self.penalties_applied = 0
        self.bonuses_applied = 0
        self.total_adjustment = 0.0
    
    @abstractmethod
    def apply(self, symbol: str, signal_type: str, current_score: float, 
              market_data: Dict[str, Any]) -> FilterResult:
        """
        Aplica el filtro a una señal de trading
        
        Args:
            symbol: Símbolo de la criptomoneda (ej: BTCUSDT)
            signal_type: Tipo de señal (LONG/SHORT)
            current_score: Score actual antes de aplicar este filtro
            market_data: Datos de mercado necesarios para el análisis
            
        Returns:
            FilterResult con el resultado de aplicar el filtro
        """
        pass
    
    @abstractmethod
    def is_applicable(self, symbol: str, signal_type: str, 
                     market_data: Dict[str, Any]) -> bool:
        """
        Determina si el filtro es aplicable para la señal dada
        
        Args:
            symbol: Símbolo de la criptomoneda
            signal_type: Tipo de señal (LONG/SHORT)
            market_data: Datos de mercado
            
        Returns:
            True si el filtro debe aplicarse
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Verifica si el filtro está habilitado en la configuración
        
        Returns:
            True si el filtro está habilitado
        """
        return (self.config.enabled and 
                self.config.is_filter_enabled(self.filter_name.lower()))
    
    def get_filter_config(self):
        """
        Obtiene la configuración específica de este filtro
        
        Returns:
            Configuración del filtro
        """
        return self.config.get_filter_config(self.filter_name.lower())
    
    def is_warning_mode(self) -> bool:
        """
        Verifica si el filtro está en modo warning (solo alerta)
        
        Returns:
            True si está en modo warning
        """
        return self.config.mode == "warning"
    
    def log_application(self, filter_result: FilterResult, symbol: str, 
                       signal_type: str, additional_info: Optional[str] = None):
        """
        Registra la aplicación del filtro
        
        Args:
            filter_result: Resultado de aplicar el filtro
            symbol: Símbolo analizado
            signal_type: Tipo de señal
            additional_info: Información adicional para logging
        """
        # Actualizar estadísticas
        self.applications_count += 1
        if filter_result.applied:
            if filter_result.score_adjustment < 0:
                self.penalties_applied += 1
            elif filter_result.score_adjustment > 0:
                self.bonuses_applied += 1
            self.total_adjustment += filter_result.score_adjustment
        
        # Log según configuración
        if self.config.detailed_logging:
            status = "APLICADO" if filter_result.applied else "OMITIDO"
            adjustment_text = f"{filter_result.score_adjustment:+.1f}" if filter_result.applied else "0.0"
            
            log_message = (f"{self.filter_name} {symbol} {signal_type}: {status} "
                          f"({adjustment_text}) - {filter_result.reason}")
            
            if additional_info:
                log_message += f" | {additional_info}"
            
            if filter_result.warning:
                self.logger.warning(f"⚠️ {log_message} | WARNING: {filter_result.warning}")
            elif filter_result.applied:
                if filter_result.score_adjustment < 0:
                    self.logger.info(f"📉 {log_message}")
                elif filter_result.score_adjustment > 0:
                    self.logger.info(f"📈 {log_message}")
                else:
                    self.logger.info(f"➡️ {log_message}")
            else:
                self.logger.debug(f"⏭️ {log_message}")
    
    def validate_market_data(self, market_data: Dict[str, Any], 
                           required_fields: list) -> bool:
        """
        Valida que los datos de mercado contengan los campos requeridos
        
        Args:
            market_data: Datos de mercado a validar
            required_fields: Lista de campos requeridos
            
        Returns:
            True si todos los campos están presentes
        """
        missing_fields = []
        for field in required_fields:
            if field not in market_data or market_data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            self.logger.error(f"Datos faltantes en {self.filter_name}: {missing_fields}")
            return False
        
        return True
    
    def safe_get_dataframe(self, market_data: Dict[str, Any], 
                          key: str = 'df') -> Optional[pd.DataFrame]:
        """
        Obtiene DataFrame de forma segura desde market_data
        
        Args:
            market_data: Datos de mercado
            key: Clave del DataFrame en market_data
            
        Returns:
            DataFrame si existe, None si no
        """
        df = market_data.get(key)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.warning(f"DataFrame no disponible o vacío en {self.filter_name}")
            return None
        return df
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del filtro
        
        Returns:
            Diccionario con estadísticas del filtro
        """
        return {
            'filter_name': self.filter_name,
            'applications_count': self.applications_count,
            'penalties_applied': self.penalties_applied,
            'bonuses_applied': self.bonuses_applied,
            'total_adjustment': self.total_adjustment,
            'avg_adjustment': (self.total_adjustment / max(1, self.applications_count)),
            'penalty_rate': (self.penalties_applied / max(1, self.applications_count)) * 100,
            'bonus_rate': (self.bonuses_applied / max(1, self.applications_count)) * 100,
            'enabled': self.is_enabled(),
            'warning_mode': self.is_warning_mode()
        }
    
    def reset_statistics(self):
        """Reinicia las estadísticas del filtro"""
        self.applications_count = 0
        self.penalties_applied = 0
        self.bonuses_applied = 0
        self.total_adjustment = 0.0
        self.logger.info(f"Estadísticas de {self.filter_name} reiniciadas")
    
    def __str__(self) -> str:
        """Representación en string del filtro"""
        status = "HABILITADO" if self.is_enabled() else "DESHABILITADO"
        mode = "WARNING" if self.is_warning_mode() else "ENFORCEMENT"
        return f"{self.filter_name} ({status}, {mode})"
    
    def __repr__(self) -> str:
        """Representación detallada del filtro"""
        return (f"{self.__class__.__name__}(name='{self.filter_name}', "
                f"enabled={self.is_enabled()}, applications={self.applications_count})")