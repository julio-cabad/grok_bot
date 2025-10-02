"""
Filtro de Volumen para Trading Inteligente
Valida que haya suficiente volumen institucional antes de aprobar trades de alto score
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ..base.filter_base import FilterBase
from ..base.filter_result import FilterResult, create_no_filter_result, create_penalty_result, create_bonus_result
from ..base.filter_config import FilterConfig


class VolumeFilter(FilterBase):
    """
    Filtro de Volumen - Valida participación institucional
    
    Funcionalidad:
    - Calcula ratio de volumen actual vs promedio histórico
    - Penaliza trades con volumen bajo (resistencias/soportes falsos)
    - Bonifica trades con volumen alto (confirmación institucional)
    - Evita Score 10/10 en setups sin volumen real
    """
    
    def __init__(self, config: FilterConfig):
        """Inicializa el filtro de volumen"""
        super().__init__(config, "VolumeFilter")
        self.volume_config = config.get_filter_config("volume")
        
        # Estadísticas específicas del filtro
        self.low_volume_rejections = 0
        self.high_volume_bonuses = 0
        self.total_volume_analyzed = 0
        
        self.logger.info(f"🔊 VolumeFilter inicializado - Threshold: {self.volume_config.min_volume_threshold:.1f}x")
    
    def is_applicable(self, symbol: str, signal_type: str, market_data: Dict[str, Any]) -> bool:
        """
        Determina si el filtro de volumen es aplicable
        
        Args:
            symbol: Símbolo de la criptomoneda
            signal_type: Tipo de señal (LONG/SHORT)
            market_data: Datos de mercado
            
        Returns:
            True si el filtro debe aplicarse
        """
        if not self.is_enabled():
            return False
        
        # Verificar que tengamos datos de volumen
        df = self.safe_get_dataframe(market_data)
        if df is None:
            return False
        
        # Verificar que tengamos suficientes datos históricos
        if len(df) < self.volume_config.lookback_periods:
            self.logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(df)} < {self.volume_config.lookback_periods}")
            return False
        
        # Verificar que tengamos columna de volumen
        if 'volume' not in df.columns:
            self.logger.warning(f"⚠️ Columna 'volume' faltante para {symbol}")
            return False
        
        return True
    
    def apply(self, symbol: str, signal_type: str, current_score: float, 
              market_data: Dict[str, Any]) -> FilterResult:
        """
        Aplica el filtro de volumen a una señal de trading
        
        Args:
            symbol: Símbolo de la criptomoneda
            signal_type: Tipo de señal (LONG/SHORT)
            current_score: Score actual antes de aplicar este filtro
            market_data: Datos de mercado con DataFrame
            
        Returns:
            FilterResult con el resultado del análisis de volumen
        """
        try:
            # Verificar aplicabilidad
            if not self.is_applicable(symbol, signal_type, market_data):
                result = create_no_filter_result(
                    "VolumeFilter", 
                    "Filtro no aplicable - datos insuficientes"
                )
                self.log_application(result, symbol, signal_type)
                return result
            
            # Obtener DataFrame
            df = self.safe_get_dataframe(market_data)
            
            # Calcular ratio de volumen
            volume_ratio = self.calculate_volume_ratio(df)
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = self.calculate_average_volume(df)
            
            # Actualizar estadísticas
            self.total_volume_analyzed += 1
            
            # Crear métricas detalladas para logging
            metrics = {
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume_20': avg_volume,
                'lookback_periods': self.volume_config.lookback_periods,
                'price': float(df['close'].iloc[-1]),
                'symbol': symbol
            }
            
            # Determinar acción basada en ratio de volumen
            result = self._evaluate_volume_ratio(volume_ratio, current_score, metrics)
            
            # Log detallado para comparación con TradingView
            additional_info = (f"Vol: {current_volume:,.0f} | Avg: {avg_volume:,.0f} | "
                             f"Ratio: {volume_ratio:.2f}x | Price: ${metrics['price']:.4f}")
            
            self.log_application(result, symbol, signal_type, additional_info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error en VolumeFilter para {symbol}: {e}")
            return create_no_filter_result(
                "VolumeFilter", 
                f"Error en análisis: {str(e)[:50]}"
            )
    
    def calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """
        Calcula el ratio del volumen actual vs promedio histórico
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Ratio de volumen (1.0 = promedio, >1.0 = alto, <1.0 = bajo)
        """
        try:
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = self.calculate_average_volume(df)
            
            if avg_volume <= 0:
                self.logger.warning("⚠️ Volumen promedio es 0, usando ratio 1.0")
                return 1.0
            
            ratio = current_volume / avg_volume
            
            # Limitar ratio a rangos razonables (evitar outliers extremos)
            ratio = max(0.01, min(ratio, 10.0))
            
            return ratio
            
        except Exception as e:
            self.logger.error(f"❌ Error calculando ratio de volumen: {e}")
            return 1.0  # Ratio neutral en caso de error
    
    def calculate_average_volume(self, df: pd.DataFrame) -> float:
        """
        Calcula el volumen promedio usando los últimos N períodos
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Volumen promedio
        """
        try:
            # Usar los últimos N períodos (excluyendo el actual)
            lookback = self.volume_config.lookback_periods
            volume_data = df['volume'].iloc[-(lookback+1):-1]  # Excluir vela actual
            
            if len(volume_data) == 0:
                # Fallback: usar todos los datos disponibles
                volume_data = df['volume'].iloc[:-1]
            
            # Calcular promedio, excluyendo outliers extremos
            avg_volume = float(volume_data.mean())
            
            # Validar resultado
            if avg_volume <= 0 or np.isnan(avg_volume):
                # Fallback: usar mediana
                avg_volume = float(volume_data.median())
            
            return max(avg_volume, 1.0)  # Mínimo 1 para evitar división por 0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculando volumen promedio: {e}")
            return 1000000.0  # Fallback conservador
    
    def _evaluate_volume_ratio(self, volume_ratio: float, current_score: float, 
                              metrics: Dict[str, Any]) -> FilterResult:
        """
        Evalúa el ratio de volumen y determina penalizaciones/bonificaciones
        
        Args:
            volume_ratio: Ratio de volumen calculado
            current_score: Score actual
            metrics: Métricas para logging
            
        Returns:
            FilterResult con la decisión del filtro
        """
        # Thresholds de configuración
        min_threshold = self.volume_config.min_volume_threshold
        low_threshold = self.volume_config.low_volume_threshold
        high_threshold = self.volume_config.high_volume_bonus_threshold
        
        # Evaluar volumen muy bajo (crítico)
        if volume_ratio < low_threshold:
            self.low_volume_rejections += 1
            
            # Limitar score máximo para volumen muy bajo
            max_score = self.volume_config.max_score_low_volume
            penalty = 0.0
            
            if current_score > max_score:
                penalty = current_score - max_score
            
            warning = f"Volumen crítico: {volume_ratio:.2f}x < {low_threshold:.2f}x"
            
            return create_penalty_result(
                "VolumeFilter",
                penalty,
                f"Volumen muy bajo: {volume_ratio:.2f}x promedio",
                warning=warning,
                metrics=metrics
            )
        
        # Evaluar volumen bajo (penalización moderada)
        elif volume_ratio < min_threshold:
            self.low_volume_rejections += 1
            
            # Limitar score máximo para volumen bajo
            max_score = self.volume_config.max_score_min_volume
            penalty = 0.0
            
            if current_score > max_score:
                penalty = current_score - max_score
            
            warning = f"Volumen bajo: {volume_ratio:.2f}x < {min_threshold:.2f}x"
            
            return create_penalty_result(
                "VolumeFilter",
                penalty,
                f"Volumen bajo: {volume_ratio:.2f}x promedio",
                warning=warning,
                metrics=metrics
            )
        
        # Evaluar volumen alto (bonificación)
        elif volume_ratio >= high_threshold:
            self.high_volume_bonuses += 1
            
            bonus = self.volume_config.volume_bonus
            
            return create_bonus_result(
                "VolumeFilter",
                bonus,
                f"Volumen alto: {volume_ratio:.2f}x promedio",
                metrics=metrics
            )
        
        # Volumen normal (sin ajuste)
        else:
            return FilterResult(
                filter_name="VolumeFilter",
                applied=True,
                score_adjustment=0.0,
                reason=f"Volumen adecuado: {volume_ratio:.2f}x promedio",
                metrics=metrics
            )
    
    def get_volume_analysis_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera resumen detallado del análisis de volumen para debugging
        
        Args:
            df: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con análisis completo de volumen
        """
        try:
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = self.calculate_average_volume(df)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Análisis de tendencia de volumen
            recent_volumes = df['volume'].tail(5).tolist()
            volume_trend = "CRECIENTE" if recent_volumes[-1] > recent_volumes[0] else "DECRECIENTE"
            
            # Percentiles de volumen
            volume_percentile = (df['volume'] <= current_volume).mean() * 100
            
            return {
                'current_volume': current_volume,
                'avg_volume_20': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'volume_percentile': volume_percentile,
                'recent_volumes': recent_volumes,
                'min_threshold': self.volume_config.min_volume_threshold,
                'low_threshold': self.volume_config.low_volume_threshold,
                'high_threshold': self.volume_config.high_volume_bonus_threshold,
                'recommendation': self._get_volume_recommendation(volume_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error en análisis de volumen: {e}")
            return {'error': str(e)}
    
    def _get_volume_recommendation(self, volume_ratio: float) -> str:
        """Genera recomendación basada en ratio de volumen"""
        if volume_ratio < self.volume_config.low_volume_threshold:
            return "EVITAR - Volumen muy bajo, posible setup falso"
        elif volume_ratio < self.volume_config.min_volume_threshold:
            return "PRECAUCIÓN - Volumen bajo, reducir confianza"
        elif volume_ratio >= self.volume_config.high_volume_bonus_threshold:
            return "EXCELENTE - Volumen alto, confirmación institucional"
        else:
            return "ACEPTABLE - Volumen dentro de rango normal"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas específicas del filtro de volumen"""
        base_stats = super().get_statistics()
        
        volume_stats = {
            'low_volume_rejections': self.low_volume_rejections,
            'high_volume_bonuses': self.high_volume_bonuses,
            'total_volume_analyzed': self.total_volume_analyzed,
            'low_volume_rate': (self.low_volume_rejections / max(1, self.total_volume_analyzed)) * 100,
            'high_volume_rate': (self.high_volume_bonuses / max(1, self.total_volume_analyzed)) * 100,
            'config': {
                'min_threshold': self.volume_config.min_volume_threshold,
                'low_threshold': self.volume_config.low_volume_threshold,
                'high_threshold': self.volume_config.high_volume_bonus_threshold,
                'lookback_periods': self.volume_config.lookback_periods
            }
        }
        
        return {**base_stats, **volume_stats}