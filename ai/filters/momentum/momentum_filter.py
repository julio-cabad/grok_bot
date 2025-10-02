"""
Filtro de Momentum - Detección de Conflictos Direccionales
Analiza el momentum del mercado vs la dirección de la señal para evitar trades contra tendencia
"""

import pandas as pd
from typing import Dict, Any, Optional
import numpy as np

from ..base.filter_base import FilterBase
from ..base.filter_result import FilterResult, create_no_filter_result, create_penalty_result, create_bonus_result
from ..base.filter_config import FilterConfig


class MomentumFilter(FilterBase):
    """
    Filtro de Momentum - Detecta conflictos direccionales
    
    Funcionalidad:
    - Calcula MACD histogram para determinar momentum
    - Penaliza trades contra momentum dominante
    - Identifica zona neutral donde momentum es incierto
    - Evita trades en contra de la tendencia principal
    
    Reglas:
    - SHORT + MACD positivo = Conflicto (-3 puntos)
    - LONG + MACD negativo = Conflicto (-3 puntos)
    - MACD entre -0.01 y +0.01 = Zona neutral (sin penalización)
    - Trade alineado con momentum fuerte = Bonus (+0.5 puntos)
    """
    
    def __init__(self, config: FilterConfig):
        super().__init__(config, "momentum")
        self.momentum_config = config.get_filter_config("momentum")
        
        # Configuración específica
        self.conflict_penalty = self.momentum_config.conflict_penalty
        self.neutral_zone_threshold = self.momentum_config.neutral_zone_threshold
        self.strong_momentum_threshold = self.momentum_config.strong_momentum_threshold
        self.momentum_bonus = self.momentum_config.momentum_bonus
        self.macd_fast_period = self.momentum_config.macd_fast_period
        self.macd_slow_period = self.momentum_config.macd_slow_period
        self.macd_signal_period = self.momentum_config.macd_signal_period
        
        self.logger.info(f"📈 MomentumFilter inicializado - Penalty: {self.conflict_penalty} | Neutral: ±{self.neutral_zone_threshold}")
    
    def is_applicable(self, symbol: str, signal_type: str, market_data: Dict[str, Any]) -> bool:
        """
        Determina si el filtro de momentum es aplicable
        
        Args:
            symbol: Símbolo de la criptomoneda
            signal_type: Tipo de señal (LONG/SHORT)
            market_data: Datos de mercado
            
        Returns:
            True si el filtro debe aplicarse
        """
        # Verificar que tenemos datos suficientes
        required_fields = ['df']
        if not self.validate_market_data(market_data, required_fields):
            return False
        
        df = self.safe_get_dataframe(market_data, 'df')
        if df is None:
            return False
        
        # Necesitamos al menos datos para calcular MACD
        min_periods = max(self.macd_slow_period + self.macd_signal_period, 50)
        if len(df) < min_periods:
            self.logger.warning(f"⚠️ Datos insuficientes para MACD en {symbol}: {len(df)} < {min_periods}")
            return False
        
        # Verificar que tenemos columna de precios
        if 'close' not in df.columns:
            self.logger.warning(f"⚠️ Columna 'close' faltante en datos de {symbol}")
            return False
        
        return True
    
    def apply(self, symbol: str, signal_type: str, current_score: float, 
              market_data: Dict[str, Any]) -> FilterResult:
        """
        Aplica el filtro de momentum a una señal de trading
        
        Args:
            symbol: Símbolo de la criptomoneda (ej: BTCUSDT)
            signal_type: Tipo de señal (LONG/SHORT)
            current_score: Score actual antes de aplicar este filtro
            market_data: Datos de mercado con DataFrame
            
        Returns:
            FilterResult con el resultado de aplicar el filtro
        """
        try:
            # Verificar si el filtro es aplicable
            if not self.is_applicable(symbol, signal_type, market_data):
                result = create_no_filter_result(
                    "MomentumFilter", 
                    "Datos insuficientes para análisis MACD"
                )
                self.log_application(result, symbol, signal_type)
                return result
            
            df = market_data['df']
            
            # Calcular MACD y analizar momentum
            momentum_analysis = self._analyze_momentum(df, symbol)
            
            # Determinar acción basada en conflicto direccional
            filter_result = self._determine_momentum_action(
                momentum_analysis, signal_type, current_score, symbol
            )
            
            # Log detallado para análisis
            self._log_detailed_momentum_analysis(momentum_analysis, filter_result, symbol, signal_type)
            
            # Log aplicación
            self.log_application(filter_result, symbol, signal_type, 
                               f"MACD Hist: {momentum_analysis['macd_histogram']:.4f}")
            
            return filter_result
            
        except Exception as e:
            self.logger.error(f"❌ Error en MomentumFilter para {symbol}: {e}")
            return create_no_filter_result(
                "MomentumFilter", 
                f"Error en análisis: {str(e)[:50]}"
            )
    
    def _analyze_momentum(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Analiza el momentum usando MACD histogram
        
        Args:
            df: DataFrame con datos de mercado
            symbol: Símbolo para logging
            
        Returns:
            Diccionario con análisis de momentum
        """
        # Calcular MACD
        macd_data = self._calculate_macd(df['close'])
        
        # Obtener valores actuales
        current_macd = float(macd_data['macd'].iloc[-1])
        current_signal = float(macd_data['signal'].iloc[-1])
        current_histogram = float(macd_data['histogram'].iloc[-1])
        
        # Analizar tendencia del histogram (últimos 3 períodos)
        recent_histogram = macd_data['histogram'].tail(3)
        histogram_trend = self._analyze_histogram_trend(recent_histogram)
        
        # Determinar fuerza del momentum
        momentum_strength = self._classify_momentum_strength(current_histogram)
        
        # Determinar dirección del momentum
        momentum_direction = self._get_momentum_direction(current_histogram)
        
        # Calcular estadísticas adicionales
        histogram_stats = self._calculate_histogram_statistics(macd_data['histogram'])
        
        return {
            'macd': current_macd,
            'signal': current_signal,
            'macd_histogram': current_histogram,
            'histogram_trend': histogram_trend,
            'momentum_strength': momentum_strength,
            'momentum_direction': momentum_direction,
            'histogram_stats': histogram_stats,
            'symbol': symbol,
            'is_neutral_zone': abs(current_histogram) <= self.neutral_zone_threshold,
            'is_strong_momentum': abs(current_histogram) >= self.strong_momentum_threshold
        }
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calcula MACD usando medias móviles exponenciales
        
        Args:
            prices: Serie de precios de cierre
            
        Returns:
            Diccionario con MACD, Signal y Histogram
        """
        # Calcular EMAs
        ema_fast = prices.ewm(span=self.macd_fast_period).mean()
        ema_slow = prices.ewm(span=self.macd_slow_period).mean()
        
        # Calcular MACD line
        macd_line = ema_fast - ema_slow
        
        # Calcular Signal line
        signal_line = macd_line.ewm(span=self.macd_signal_period).mean()
        
        # Calcular Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _analyze_histogram_trend(self, recent_histogram: pd.Series) -> str:
        """
        Analiza la tendencia del histogram en períodos recientes
        
        Args:
            recent_histogram: Últimos valores del histogram
            
        Returns:
            Tendencia: INCREASING, DECREASING, FLAT
        """
        if len(recent_histogram) < 2:
            return "INSUFFICIENT_DATA"
        
        # Calcular cambios
        changes = recent_histogram.diff().dropna()
        
        if len(changes) == 0:
            return "FLAT"
        
        # Determinar tendencia dominante
        positive_changes = (changes > 0).sum()
        negative_changes = (changes < 0).sum()
        
        if positive_changes > negative_changes:
            return "INCREASING"
        elif negative_changes > positive_changes:
            return "DECREASING"
        else:
            return "FLAT"
    
    def _classify_momentum_strength(self, histogram_value: float) -> str:
        """
        Clasifica la fuerza del momentum
        
        Args:
            histogram_value: Valor actual del histogram
            
        Returns:
            Clasificación: STRONG, MODERATE, WEAK, NEUTRAL
        """
        abs_value = abs(histogram_value)
        
        if abs_value <= self.neutral_zone_threshold:
            return "NEUTRAL"
        elif abs_value >= self.strong_momentum_threshold:
            return "STRONG"
        elif abs_value >= self.strong_momentum_threshold * 0.5:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _get_momentum_direction(self, histogram_value: float) -> str:
        """
        Determina la dirección del momentum
        
        Args:
            histogram_value: Valor actual del histogram
            
        Returns:
            Dirección: BULLISH, BEARISH, NEUTRAL
        """
        if histogram_value > self.neutral_zone_threshold:
            return "BULLISH"
        elif histogram_value < -self.neutral_zone_threshold:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_histogram_statistics(self, histogram_series: pd.Series) -> Dict[str, float]:
        """
        Calcula estadísticas del histogram para contexto adicional
        
        Args:
            histogram_series: Serie completa del histogram
            
        Returns:
            Estadísticas del histogram
        """
        recent_data = histogram_series.tail(20)  # Últimos 20 períodos
        
        return {
            'mean': float(recent_data.mean()),
            'std': float(recent_data.std()),
            'min': float(recent_data.min()),
            'max': float(recent_data.max()),
            'current_percentile': float((recent_data < histogram_series.iloc[-1]).sum() / len(recent_data) * 100)
        }
    
    def _determine_momentum_action(self, momentum_analysis: Dict[str, Any], 
                                 signal_type: str, current_score: float, symbol: str) -> FilterResult:
        """
        Determina la acción a tomar basada en el análisis de momentum
        
        Args:
            momentum_analysis: Resultado del análisis de momentum
            signal_type: Tipo de señal (LONG/SHORT)
            current_score: Score actual
            symbol: Símbolo
            
        Returns:
            FilterResult con la acción determinada
        """
        histogram = momentum_analysis['macd_histogram']
        momentum_direction = momentum_analysis['momentum_direction']
        momentum_strength = momentum_analysis['momentum_strength']
        is_neutral = momentum_analysis['is_neutral_zone']
        is_strong = momentum_analysis['is_strong_momentum']
        
        # Modo warning - solo alertas
        if self.is_warning_mode():
            conflict_detected = self._detect_momentum_conflict(signal_type, momentum_direction, is_neutral)
            
            if conflict_detected:
                return FilterResult(
                    filter_name="MomentumFilter",
                    applied=True,
                    score_adjustment=0.0,  # No penalización en modo warning
                    reason=f"⚠️ WARNING: Conflicto direccional - {signal_type} vs {momentum_direction}",
                    warning=f"Momentum {momentum_direction} contra señal {signal_type} en {symbol}",
                    metrics=momentum_analysis
                )
            elif is_strong and not is_neutral:
                return FilterResult(
                    filter_name="MomentumFilter",
                    applied=True,
                    score_adjustment=0.0,  # No bonus en modo warning
                    reason=f"✅ INFO: Momentum fuerte alineado - {signal_type} + {momentum_direction}",
                    metrics=momentum_analysis
                )
            else:
                return create_no_filter_result(
                    "MomentumFilter",
                    f"Momentum {momentum_strength} {momentum_direction} - Sin conflicto"
                )
        
        # Modo enforcement - aplicar penalizaciones/bonificaciones
        
        # Zona neutral - sin acción
        if is_neutral:
            return create_no_filter_result(
                "MomentumFilter",
                f"Zona neutral MACD ({histogram:.4f}) - Sin penalización"
            )
        
        # Detectar conflicto direccional
        conflict_detected = self._detect_momentum_conflict(signal_type, momentum_direction, is_neutral)
        
        if conflict_detected:
            # Conflicto detectado - penalización
            return create_penalty_result(
                "MomentumFilter",
                self.conflict_penalty,
                f"Conflicto direccional: {signal_type} vs momentum {momentum_direction} ({histogram:.4f})",
                warning=f"⚠️ Trade contra momentum dominante - Riesgo alto",
                metrics=momentum_analysis
            )
        
        # Trade alineado con momentum fuerte - bonificación
        if is_strong:
            return create_bonus_result(
                "MomentumFilter",
                self.momentum_bonus,
                f"Momentum fuerte alineado: {signal_type} + {momentum_direction} ({histogram:.4f})",
                metrics=momentum_analysis
            )
        
        # Momentum moderado/débil alineado - sin acción
        return create_no_filter_result(
            "MomentumFilter",
            f"Momentum {momentum_strength} {momentum_direction} alineado - Sin ajuste"
        )
    
    def _detect_momentum_conflict(self, signal_type: str, momentum_direction: str, is_neutral: bool) -> bool:
        """
        Detecta si hay conflicto entre la señal y el momentum
        
        Args:
            signal_type: Tipo de señal (LONG/SHORT)
            momentum_direction: Dirección del momentum (BULLISH/BEARISH/NEUTRAL)
            is_neutral: Si estamos en zona neutral
            
        Returns:
            True si hay conflicto
        """
        if is_neutral or momentum_direction == "NEUTRAL":
            return False
        
        # Conflictos específicos
        if signal_type == "SHORT" and momentum_direction == "BULLISH":
            return True
        
        if signal_type == "LONG" and momentum_direction == "BEARISH":
            return True
        
        return False
    
    def _log_detailed_momentum_analysis(self, momentum_analysis: Dict[str, Any], 
                                      filter_result: FilterResult, symbol: str, signal_type: str):
        """
        Log detallado para análisis de momentum
        
        Args:
            momentum_analysis: Análisis de momentum
            filter_result: Resultado del filtro
            symbol: Símbolo
            signal_type: Tipo de señal
        """
        self.logger.info("📈 " + "="*60)
        self.logger.info(f"📈 ANÁLISIS DE MOMENTUM DETALLADO - {symbol} {signal_type}")
        self.logger.info("📈 " + "="*60)
        
        # Datos MACD
        self.logger.info(f"📈 MACD Line: {momentum_analysis['macd']:.6f}")
        self.logger.info(f"📈 Signal Line: {momentum_analysis['signal']:.6f}")
        self.logger.info(f"📈 Histogram: {momentum_analysis['macd_histogram']:.6f}")
        
        # Análisis de momentum
        self.logger.info(f"📈 Dirección: {momentum_analysis['momentum_direction']}")
        self.logger.info(f"📈 Fuerza: {momentum_analysis['momentum_strength']}")
        self.logger.info(f"📈 Tendencia Histogram: {momentum_analysis['histogram_trend']}")
        self.logger.info(f"📈 Zona Neutral: {momentum_analysis['is_neutral_zone']}")
        self.logger.info(f"📈 Momentum Fuerte: {momentum_analysis['is_strong_momentum']}")
        
        # Estadísticas
        stats = momentum_analysis['histogram_stats']
        self.logger.info(f"📈 Histogram Promedio (20p): {stats['mean']:.6f}")
        self.logger.info(f"📈 Histogram Std: {stats['std']:.6f}")
        self.logger.info(f"📈 Histogram Rango: {stats['min']:.6f} - {stats['max']:.6f}")
        self.logger.info(f"📈 Percentil Actual: {stats['current_percentile']:.1f}%")
        
        # Thresholds configurados
        self.logger.info(f"📈 Zona Neutral: ±{self.neutral_zone_threshold}")
        self.logger.info(f"📈 Momentum Fuerte: ±{self.strong_momentum_threshold}")
        self.logger.info(f"📈 Penalización Conflicto: {self.conflict_penalty}")
        self.logger.info(f"📈 Bonus Alineado: {self.momentum_bonus}")
        
        # Resultado del filtro
        if filter_result.applied:
            action = "PENALIZACIÓN" if filter_result.score_adjustment < 0 else "BONIFICACIÓN" if filter_result.score_adjustment > 0 else "WARNING"
            self.logger.info(f"📈 Acción: {action} ({filter_result.score_adjustment:+.1f})")
            self.logger.info(f"📈 Razón: {filter_result.reason}")
            if filter_result.warning:
                self.logger.info(f"📈 Advertencia: {filter_result.warning}")
        else:
            self.logger.info(f"📈 Acción: SIN FILTRO APLICADO")
            self.logger.info(f"📈 Razón: {filter_result.reason}")
        
        self.logger.info("📈 " + "="*60)
    
    def get_momentum_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtiene análisis de momentum para uso externo
        
        Args:
            df: DataFrame con datos de mercado
            
        Returns:
            Análisis completo de momentum
        """
        if len(df) < max(self.macd_slow_period + self.macd_signal_period, 50):
            return {}
        
        return self._analyze_momentum(df, "EXTERNAL_ANALYSIS")
    
    def __str__(self) -> str:
        """Representación en string del filtro"""
        return (f"MomentumFilter(penalty: {self.conflict_penalty}, neutral: ±{self.neutral_zone_threshold}, "
                f"strong: ±{self.strong_momentum_threshold}, enabled: {self.is_enabled()})")