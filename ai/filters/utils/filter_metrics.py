"""
Sistema de métricas para filtros de trading
Calcula y mantiene métricas de performance en tiempo real
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import threading
from dataclasses import dataclass

from ..base.filter_result import FilterSummary


@dataclass
class FilterMetrics:
    """
    Métricas de performance de filtros
    
    Attributes:
        total_trades: Total de trades analizados
        trades_approved: Trades aprobados después de filtros
        trades_rejected: Trades rechazados por filtros
        trades_with_warnings: Trades con advertencias
        avg_score_adjustment: Ajuste promedio de score
        filter_usage: Uso de cada filtro individual
        win_rate_improvement: Mejora estimada en win rate
    """
    total_trades: int = 0
    trades_approved: int = 0
    trades_rejected: int = 0
    trades_with_warnings: int = 0
    avg_score_adjustment: float = 0.0
    filter_usage: Dict[str, int] = None
    win_rate_improvement: float = 0.0
    
    def __post_init__(self):
        if self.filter_usage is None:
            self.filter_usage = {}
    
    @property
    def approval_rate(self) -> float:
        """Tasa de aprobación de trades"""
        if self.total_trades == 0:
            return 0.0
        return (self.trades_approved / self.total_trades) * 100
    
    @property
    def rejection_rate(self) -> float:
        """Tasa de rechazo de trades"""
        if self.total_trades == 0:
            return 0.0
        return (self.trades_rejected / self.total_trades) * 100
    
    @property
    def warning_rate(self) -> float:
        """Tasa de trades con advertencias"""
        if self.total_trades == 0:
            return 0.0
        return (self.trades_with_warnings / self.total_trades) * 100


class FilterMetricsCalculator:
    """
    Calculadora de métricas para el sistema de filtros
    Thread-safe y optimizada para uso en tiempo real
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._trade_history: List[FilterSummary] = []
        self._cached_metrics: Optional[FilterMetrics] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)  # Cache por 5 minutos
    
    def add_trade_result(self, filter_summary: FilterSummary):
        """
        Añade un resultado de trade para cálculo de métricas
        
        Args:
            filter_summary: Resumen de filtros aplicados al trade
        """
        with self._lock:
            self._trade_history.append(filter_summary)
            
            # Mantener solo últimos 1000 trades para performance
            if len(self._trade_history) > 1000:
                self._trade_history = self._trade_history[-1000:]
            
            # Invalidar cache
            self._cached_metrics = None
            self._cache_timestamp = None
    
    def get_current_metrics(self, force_recalculate: bool = False) -> FilterMetrics:
        """
        Obtiene métricas actuales del sistema
        
        Args:
            force_recalculate: Forzar recálculo ignorando cache
            
        Returns:
            Métricas actuales del sistema
        """
        now = datetime.now()
        
        # Usar cache si está disponible y no ha expirado
        if (not force_recalculate and 
            self._cached_metrics and 
            self._cache_timestamp and 
            now - self._cache_timestamp < self._cache_duration):
            return self._cached_metrics
        
        with self._lock:
            metrics = self._calculate_metrics()
            
            # Actualizar cache
            self._cached_metrics = metrics
            self._cache_timestamp = now
            
            return metrics
    
    def _calculate_metrics(self) -> FilterMetrics:
        """Calcula métricas basadas en el historial actual"""
        if not self._trade_history:
            return FilterMetrics()
        
        total_trades = len(self._trade_history)
        trades_approved = 0
        trades_rejected = 0
        trades_with_warnings = 0
        total_score_adjustment = 0.0
        filter_usage = defaultdict(int)
        
        for trade in self._trade_history:
            # Contar decisiones
            if trade.trade_decision == "APPROVED":
                trades_approved += 1
            elif trade.trade_decision == "REJECTED":
                trades_rejected += 1
            
            if trade.has_warnings:
                trades_with_warnings += 1
            
            # Sumar ajustes de score
            total_score_adjustment += trade.total_adjustment
            
            # Contar uso de filtros
            for filter_result in trade.get_applied_filters():
                filter_usage[filter_result.filter_name] += 1
        
        # Calcular promedio de ajuste
        avg_score_adjustment = total_score_adjustment / total_trades if total_trades > 0 else 0.0
        
        # Estimar mejora de win rate (simplificado)
        # Asumimos que trades rechazados tenían baja probabilidad de éxito
        estimated_bad_trades_filtered = trades_rejected
        win_rate_improvement = (estimated_bad_trades_filtered / total_trades) * 15.0 if total_trades > 0 else 0.0
        
        return FilterMetrics(
            total_trades=total_trades,
            trades_approved=trades_approved,
            trades_rejected=trades_rejected,
            trades_with_warnings=trades_with_warnings,
            avg_score_adjustment=avg_score_adjustment,
            filter_usage=dict(filter_usage),
            win_rate_improvement=win_rate_improvement
        )
    
    def get_filter_performance(self, filter_name: str, days: int = 7) -> Dict[str, Any]:
        """
        Obtiene métricas de performance de un filtro específico
        
        Args:
            filter_name: Nombre del filtro a analizar
            days: Número de días hacia atrás a considerar
            
        Returns:
            Diccionario con métricas del filtro
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            # Filtrar trades recientes
            recent_trades = [
                trade for trade in self._trade_history
                if trade.timestamp >= cutoff_date
            ]
            
            if not recent_trades:
                return {
                    'filter_name': filter_name,
                    'period_days': days,
                    'trades_analyzed': 0,
                    'times_applied': 0,
                    'avg_adjustment': 0.0,
                    'application_rate': 0.0
                }
            
            # Analizar aplicación del filtro
            times_applied = 0
            total_adjustment = 0.0
            
            for trade in recent_trades:
                filter_result = trade.get_filter_by_name(filter_name)
                if filter_result and filter_result.applied:
                    times_applied += 1
                    total_adjustment += filter_result.score_adjustment
            
            avg_adjustment = total_adjustment / times_applied if times_applied > 0 else 0.0
            application_rate = (times_applied / len(recent_trades)) * 100
            
            return {
                'filter_name': filter_name,
                'period_days': days,
                'trades_analyzed': len(recent_trades),
                'times_applied': times_applied,
                'avg_adjustment': avg_adjustment,
                'application_rate': application_rate,
                'total_adjustment': total_adjustment
            }
    
    def get_comparison_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtiene métricas comparativas para evaluar efectividad
        
        Args:
            days: Número de días hacia atrás a considerar
            
        Returns:
            Métricas comparativas del sistema
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._lock:
            recent_trades = [
                trade for trade in self._trade_history
                if trade.timestamp >= cutoff_date
            ]
            
            if not recent_trades:
                return {}
            
            # Separar trades por decisión
            approved_trades = [t for t in recent_trades if t.trade_decision == "APPROVED"]
            rejected_trades = [t for t in recent_trades if t.trade_decision == "REJECTED"]
            
            # Calcular scores promedio
            avg_original_score = sum(t.original_score for t in recent_trades) / len(recent_trades)
            avg_final_score = sum(t.final_score for t in recent_trades) / len(recent_trades)
            
            avg_approved_score = (sum(t.final_score for t in approved_trades) / len(approved_trades) 
                                 if approved_trades else 0.0)
            avg_rejected_score = (sum(t.final_score for t in rejected_trades) / len(rejected_trades) 
                                 if rejected_trades else 0.0)
            
            return {
                'period_days': days,
                'total_trades': len(recent_trades),
                'approved_trades': len(approved_trades),
                'rejected_trades': len(rejected_trades),
                'avg_original_score': avg_original_score,
                'avg_final_score': avg_final_score,
                'avg_approved_score': avg_approved_score,
                'avg_rejected_score': avg_rejected_score,
                'score_improvement': avg_final_score - avg_original_score,
                'approval_rate': (len(approved_trades) / len(recent_trades)) * 100
            }
    
    def clear_history(self):
        """Limpia el historial de trades"""
        with self._lock:
            self._trade_history.clear()
            self._cached_metrics = None
            self._cache_timestamp = None


# Instancia global del calculador de métricas
filter_metrics_calculator = FilterMetricsCalculator()