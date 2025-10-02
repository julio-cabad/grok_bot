"""
Sistema de logging especializado para filtros de trading
Proporciona logging estructurado y métricas en tiempo real
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import threading

from ..base.filter_result import FilterResult, FilterSummary


class FilterLogger:
    """
    Logger especializado para filtros de trading con métricas integradas
    
    Características:
    - Logging estructurado por filtro
    - Métricas en tiempo real
    - Estadísticas históricas
    - Thread-safe para uso concurrente
    """
    
    def __init__(self, max_history_size: int = 1000):
        """
        Inicializa el logger de filtros
        
        Args:
            max_history_size: Máximo número de registros a mantener en memoria
        """
        self.logger = logging.getLogger("FilterSystem")
        self.max_history_size = max_history_size
        
        # Thread-safe collections para métricas
        self._lock = threading.Lock()
        self._filter_history = deque(maxlen=max_history_size)
        self._daily_stats = defaultdict(lambda: {
            'trades_analyzed': 0,
            'trades_approved': 0,
            'trades_rejected': 0,
            'trades_with_warnings': 0,
            'total_score_adjustment': 0.0,
            'filters_applied': defaultdict(int),
            'rejection_reasons': defaultdict(int)
        })
        
        # Configurar formato de logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura el formato de logging para filtros"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_filter_application(self, filter_summary: FilterSummary):
        """
        Registra la aplicación completa de filtros para una señal
        
        Args:
            filter_summary: Resumen completo de filtros aplicados
        """
        with self._lock:
            # Añadir a historial
            self._filter_history.append(filter_summary)
            
            # Actualizar estadísticas diarias
            today = datetime.now().date().isoformat()
            stats = self._daily_stats[today]
            
            stats['trades_analyzed'] += 1
            
            if filter_summary.trade_decision == "APPROVED":
                stats['trades_approved'] += 1
            elif filter_summary.trade_decision == "REJECTED":
                stats['trades_rejected'] += 1
                if filter_summary.rejection_reason:
                    stats['rejection_reasons'][filter_summary.rejection_reason] += 1
            
            if filter_summary.has_warnings:
                stats['trades_with_warnings'] += 1
            
            stats['total_score_adjustment'] += filter_summary.total_adjustment
            
            # Contar filtros aplicados
            for filter_result in filter_summary.get_applied_filters():
                stats['filters_applied'][filter_result.filter_name] += 1
        
        # Log principal
        self.logger.info(
            f"🔍 FILTROS {filter_summary.symbol} {filter_summary.signal_type}: "
            f"{filter_summary.get_summary_text()}"
        )
        
        # Log detallado de cada filtro aplicado
        for filter_result in filter_summary.get_applied_filters():
            if filter_result.score_adjustment != 0:
                adjustment_emoji = "📉" if filter_result.score_adjustment < 0 else "📈"
                self.logger.info(
                    f"  {adjustment_emoji} {filter_result.filter_name}: "
                    f"{filter_result.score_adjustment:+.1f} - {filter_result.reason}"
                )
            
            if filter_result.warning:
                self.logger.warning(f"  ⚠️ {filter_result.filter_name}: {filter_result.warning}")
    
    def log_filter_error(self, filter_name: str, error: Exception, symbol: str, 
                        additional_context: Optional[Dict[str, Any]] = None):
        """
        Registra errores de filtros individuales
        
        Args:
            filter_name: Nombre del filtro que falló
            error: Excepción capturada
            symbol: Símbolo que se estaba analizando
            additional_context: Contexto adicional para debugging
        """
        error_info = {
            'filter_name': filter_name,
            'symbol': symbol,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_context:
            error_info['context'] = additional_context
        
        self.logger.error(
            f"❌ ERROR {filter_name} en {symbol}: {error} | "
            f"Context: {json.dumps(additional_context or {}, ensure_ascii=False)}"
        )
        
        # Actualizar estadísticas de errores
        with self._lock:
            today = datetime.now().date().isoformat()
            if 'errors' not in self._daily_stats[today]:
                self._daily_stats[today]['errors'] = defaultdict(int)
            self._daily_stats[today]['errors'][filter_name] += 1
    
    def log_filter_stats(self, period: str = "daily"):
        """
        Registra estadísticas periódicas de filtros
        
        Args:
            period: Período de estadísticas ("daily", "weekly")
        """
        if period == "daily":
            self._log_daily_stats()
        elif period == "weekly":
            self._log_weekly_stats()
    
    def _log_daily_stats(self):
        """Registra estadísticas del día actual"""
        today = datetime.now().date().isoformat()
        
        with self._lock:
            if today not in self._daily_stats:
                self.logger.info("📊 Sin actividad de filtros hoy")
                return
            
            stats = self._daily_stats[today]
            
            # Calcular métricas
            total_trades = stats['trades_analyzed']
            if total_trades == 0:
                return
            
            approval_rate = (stats['trades_approved'] / total_trades) * 100
            rejection_rate = (stats['trades_rejected'] / total_trades) * 100
            warning_rate = (stats['trades_with_warnings'] / total_trades) * 100
            avg_adjustment = stats['total_score_adjustment'] / total_trades
            
            # Log estadísticas principales
            self.logger.info("📊 ESTADÍSTICAS DIARIAS DE FILTROS:")
            self.logger.info(f"   📈 Trades analizados: {total_trades}")
            self.logger.info(f"   ✅ Aprobados: {stats['trades_approved']} ({approval_rate:.1f}%)")
            self.logger.info(f"   ❌ Rechazados: {stats['trades_rejected']} ({rejection_rate:.1f}%)")
            self.logger.info(f"   ⚠️ Con advertencias: {stats['trades_with_warnings']} ({warning_rate:.1f}%)")
            self.logger.info(f"   📊 Ajuste promedio: {avg_adjustment:+.2f} puntos")
            
            # Log filtros más activos
            if stats['filters_applied']:
                most_active = sorted(stats['filters_applied'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                self.logger.info("   🔥 Filtros más activos:")
                for filter_name, count in most_active:
                    percentage = (count / total_trades) * 100
                    self.logger.info(f"      {filter_name}: {count} ({percentage:.1f}%)")
            
            # Log razones de rechazo principales
            if stats['rejection_reasons']:
                top_reasons = sorted(stats['rejection_reasons'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                self.logger.info("   🚫 Principales razones de rechazo:")
                for reason, count in top_reasons:
                    self.logger.info(f"      {reason}: {count}")
    
    def _log_weekly_stats(self):
        """Registra estadísticas de la semana"""
        # Calcular fechas de la semana
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        
        weekly_stats = {
            'trades_analyzed': 0,
            'trades_approved': 0,
            'trades_rejected': 0,
            'total_score_adjustment': 0.0,
            'filters_applied': defaultdict(int)
        }
        
        with self._lock:
            # Sumar estadísticas de la semana
            for i in range(7):
                date_key = (week_start + timedelta(days=i)).isoformat()
                if date_key in self._daily_stats:
                    day_stats = self._daily_stats[date_key]
                    weekly_stats['trades_analyzed'] += day_stats['trades_analyzed']
                    weekly_stats['trades_approved'] += day_stats['trades_approved']
                    weekly_stats['trades_rejected'] += day_stats['trades_rejected']
                    weekly_stats['total_score_adjustment'] += day_stats['total_score_adjustment']
                    
                    for filter_name, count in day_stats['filters_applied'].items():
                        weekly_stats['filters_applied'][filter_name] += count
        
        total_trades = weekly_stats['trades_analyzed']
        if total_trades == 0:
            self.logger.info("📊 Sin actividad de filtros esta semana")
            return
        
        approval_rate = (weekly_stats['trades_approved'] / total_trades) * 100
        avg_adjustment = weekly_stats['total_score_adjustment'] / total_trades
        
        self.logger.info("📊 ESTADÍSTICAS SEMANALES DE FILTROS:")
        self.logger.info(f"   📈 Trades analizados: {total_trades}")
        self.logger.info(f"   ✅ Tasa de aprobación: {approval_rate:.1f}%")
        self.logger.info(f"   📊 Ajuste promedio: {avg_adjustment:+.2f} puntos")
    
    def get_recent_history(self, limit: int = 50) -> List[FilterSummary]:
        """
        Obtiene el historial reciente de filtros aplicados
        
        Args:
            limit: Número máximo de registros a retornar
            
        Returns:
            Lista de FilterSummary más recientes
        """
        with self._lock:
            return list(self._filter_history)[-limit:]
    
    def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un día específico
        
        Args:
            date: Fecha en formato YYYY-MM-DD, None para hoy
            
        Returns:
            Diccionario con estadísticas del día
        """
        if date is None:
            date = datetime.now().date().isoformat()
        
        with self._lock:
            return dict(self._daily_stats.get(date, {}))
    
    def clear_old_stats(self, days_to_keep: int = 30):
        """
        Limpia estadísticas antiguas para liberar memoria
        
        Args:
            days_to_keep: Número de días de estadísticas a mantener
        """
        cutoff_date = datetime.now().date() - timedelta(days=days_to_keep)
        
        with self._lock:
            dates_to_remove = [
                date for date in self._daily_stats.keys()
                if datetime.fromisoformat(date).date() < cutoff_date
            ]
            
            for date in dates_to_remove:
                del self._daily_stats[date]
        
        self.logger.info(f"🧹 Limpiadas estadísticas anteriores a {cutoff_date}")


# Instancia global del logger para uso en todo el sistema
filter_logger = FilterLogger()