"""
FilterManager - Coordinador Principal del Sistema de Filtros
Gestiona la aplicación secuencial de todos los filtros de trading
"""

from typing import Dict, Any, List, Optional
import time
from datetime import datetime

from ..base.filter_config import FilterConfig, load_filter_config
from ..base.filter_result import FilterResult, FilterSummary
from ..utils.filter_logger import filter_logger
from ..utils.filter_metrics import filter_metrics_calculator

# Importar filtros específicos
from ..volume.volume_filter import VolumeFilter
from ..momentum.momentum_filter import MomentumFilter


class FilterManager:
    """
    Coordinador principal del sistema de filtros
    
    Responsabilidades:
    - Coordinar aplicación secuencial de filtros
    - Generar FilterSummary completo
    - Manejar errores de filtros individuales
    - Aplicar límites globales de penalización
    - Logging y métricas centralizadas
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el FilterManager
        
        Args:
            config_path: Ruta al archivo de configuración, None para default
        """
        # Cargar configuración
        self.config = load_filter_config(config_path)
        
        # Inicializar filtros habilitados
        self.filters = []
        self._initialize_filters()
        
        # Estadísticas del manager
        self.total_analyses = 0
        self.total_approved = 0
        self.total_rejected = 0
        self.start_time = datetime.now()
        
        filter_logger.logger.info(f"🎛️ FilterManager inicializado - {len(self.filters)} filtros activos")
        filter_logger.logger.info(f"📋 Modo: {self.config.mode.upper()} | Filtros: {[f.filter_name for f in self.filters]}")
    
    def _initialize_filters(self):
        """Inicializa todos los filtros habilitados"""
        self.filters = []
        
        # VolumeFilter
        if self.config.is_filter_enabled("volume"):
            try:
                volume_filter = VolumeFilter(self.config)
                self.filters.append(volume_filter)
                filter_logger.logger.info("✅ VolumeFilter inicializado")
            except Exception as e:
                filter_logger.logger.error(f"❌ Error inicializando VolumeFilter: {e}")
        
        # MomentumFilter
        if self.config.is_filter_enabled("momentum"):
            try:
                momentum_filter = MomentumFilter(self.config)
                self.filters.append(momentum_filter)
                filter_logger.logger.info("✅ MomentumFilter inicializado")
            except Exception as e:
                filter_logger.logger.error(f"❌ Error inicializando MomentumFilter: {e}")
        
        # TODO: Añadir otros filtros cuando estén implementados
        
        if len(self.filters) == 0:
            filter_logger.logger.warning("⚠️ Ningún filtro habilitado - FilterManager en modo pasivo")
    
    def apply_filters(self, symbol: str, signal_type: str, initial_score: float, 
                     market_data: Dict[str, Any]) -> FilterSummary:
        """
        Aplica todos los filtros secuencialmente a una señal de trading
        
        Args:
            symbol: Símbolo de la criptomoneda (ej: BTCUSDT)
            signal_type: Tipo de señal (LONG/SHORT)
            initial_score: Score inicial antes de filtros
            market_data: Datos de mercado (debe incluir DataFrame en 'df')
            
        Returns:
            FilterSummary con resultado completo de todos los filtros
        """
        start_time = time.time()
        self.total_analyses += 1
        
        try:
            # Validar entrada
            if not self._validate_input(symbol, signal_type, initial_score, market_data):
                return self._create_error_summary(symbol, signal_type, initial_score, 
                                                "Datos de entrada inválidos")
            
            # Si no hay filtros habilitados, retornar sin cambios
            if not self.filters:
                return self._create_no_filters_summary(symbol, signal_type, initial_score)
            
            # Aplicar filtros secuencialmente
            filter_results = []
            current_score = initial_score
            
            for filter_instance in self.filters:
                try:
                    # Aplicar filtro individual
                    result = filter_instance.apply(symbol, signal_type, current_score, market_data)
                    filter_results.append(result)
                    
                    # Actualizar score si el filtro se aplicó
                    if result.applied and not self.config.is_warning_mode():
                        current_score += result.score_adjustment
                        
                        # Aplicar límites globales
                        current_score = max(self.config.min_final_score, 
                                          min(current_score, self.config.max_final_score))
                    
                except Exception as e:
                    # Error en filtro individual - continuar con otros
                    filter_logger.log_filter_error(filter_instance.filter_name, e, symbol)
                    
                    error_result = FilterResult(
                        filter_name=filter_instance.filter_name,
                        applied=False,
                        score_adjustment=0.0,
                        reason=f"Error en filtro: {str(e)[:30]}...",
                        warning=f"Filtro falló: {str(e)}"
                    )
                    filter_results.append(error_result)
            
            # Calcular ajuste total
            total_adjustment = sum(r.score_adjustment for r in filter_results if r.applied)
            
            # En modo warning, no aplicar ajustes reales
            if self.config.is_warning_mode():
                final_score = initial_score
                total_adjustment = 0.0
            else:
                final_score = current_score
            
            # Determinar decisión final
            trade_decision, rejection_reason = self._determine_trade_decision(
                final_score, filter_results
            )
            
            # Crear resumen completo
            summary = FilterSummary(
                symbol=symbol,
                signal_type=signal_type,
                original_score=initial_score,
                final_score=final_score,
                total_adjustment=total_adjustment,
                filters_applied=filter_results,
                trade_decision=trade_decision,
                rejection_reason=rejection_reason
            )
            
            # Actualizar estadísticas
            if trade_decision == "APPROVED":
                self.total_approved += 1
            elif trade_decision == "REJECTED":
                self.total_rejected += 1
            
            # Logging y métricas
            filter_logger.log_filter_application(summary)
            filter_metrics_calculator.add_trade_result(summary)
            
            return summary
            
        except Exception as e:
            filter_logger.logger.error(f"❌ Error crítico en FilterManager para {symbol}: {e}")
            return self._create_error_summary(symbol, signal_type, initial_score, str(e))
    
    def _validate_input(self, symbol: str, signal_type: str, initial_score: float, 
                       market_data: Dict[str, Any]) -> bool:
        """Valida los datos de entrada"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        if signal_type not in ["LONG", "SHORT"]:
            return False
        
        if not isinstance(initial_score, (int, float)) or initial_score < 0 or initial_score > 10:
            return False
        
        if not isinstance(market_data, dict):
            return False
        
        return True
    
    def _determine_trade_decision(self, final_score: float, 
                                filter_results: List[FilterResult]) -> tuple[str, Optional[str]]:
        """
        Determina la decisión final del trade basada en score y filtros
        
        Returns:
            Tuple de (decisión, razón_de_rechazo)
        """
        # Verificar si hay advertencias críticas
        critical_warnings = [r for r in filter_results if r.warning and "crítico" in r.warning.lower()]
        if critical_warnings:
            return "REJECTED", f"Advertencia crítica: {critical_warnings[0].warning}"
        
        # Decisión basada en score final
        # Nota: El threshold real se aplica en el sistema principal, aquí solo reportamos
        if final_score >= 7.5:  # Usar threshold del sistema principal
            return "APPROVED", None
        else:
            return "REJECTED", f"Score final insuficiente: {final_score:.1f} < 7.5"
    
    def _create_no_filters_summary(self, symbol: str, signal_type: str, 
                                  initial_score: float) -> FilterSummary:
        """Crea resumen cuando no hay filtros habilitados"""
        return FilterSummary(
            symbol=symbol,
            signal_type=signal_type,
            original_score=initial_score,
            final_score=initial_score,
            total_adjustment=0.0,
            filters_applied=[],
            trade_decision="APPROVED",
            rejection_reason=None
        )
    
    def _create_error_summary(self, symbol: str, signal_type: str, initial_score: float, 
                             error_message: str) -> FilterSummary:
        """Crea resumen de error"""
        error_result = FilterResult(
            filter_name="FilterManager",
            applied=False,
            score_adjustment=0.0,
            reason=f"Error del sistema: {error_message}",
            warning="Sistema de filtros falló"
        )
        
        return FilterSummary(
            symbol=symbol,
            signal_type=signal_type,
            original_score=initial_score,
            final_score=initial_score,  # Sin cambios en caso de error
            total_adjustment=0.0,
            filters_applied=[error_result],
            trade_decision="APPROVED",  # Conservador: permitir trade si filtros fallan
            rejection_reason=None
        )
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del FilterManager"""
        uptime = datetime.now() - self.start_time
        
        return {
            'total_analyses': self.total_analyses,
            'total_approved': self.total_approved,
            'total_rejected': self.total_rejected,
            'approval_rate': (self.total_approved / max(1, self.total_analyses)) * 100,
            'rejection_rate': (self.total_rejected / max(1, self.total_analyses)) * 100,
            'active_filters': len(self.filters),
            'filter_names': [f.filter_name for f in self.filters],
            'uptime_seconds': uptime.total_seconds(),
            'mode': self.config.mode,
            'filters_enabled': self.config.enabled
        }
    
    def get_all_filter_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de todos los filtros"""
        stats = {
            'manager': self.get_manager_statistics(),
            'filters': {}
        }
        
        for filter_instance in self.filters:
            stats['filters'][filter_instance.filter_name] = filter_instance.get_statistics()
        
        return stats
    
    def reload_config(self, config_path: Optional[str] = None):
        """Recarga la configuración y reinicializa filtros"""
        try:
            old_config = self.config
            self.config = load_filter_config(config_path)
            
            # Reinicializar filtros si la configuración cambió
            self._initialize_filters()
            
            filter_logger.logger.info(f"🔄 Configuración recargada - {len(self.filters)} filtros activos")
            
        except Exception as e:
            filter_logger.logger.error(f"❌ Error recargando configuración: {e}")
            # Mantener configuración anterior en caso de error
            self.config = old_config
    
    def is_enabled(self) -> bool:
        """Verifica si el sistema de filtros está habilitado"""
        return self.config.enabled and len(self.filters) > 0
    
    def is_warning_mode(self) -> bool:
        """Verifica si está en modo warning"""
        return self.config.mode == "warning"
    
    def __str__(self) -> str:
        """Representación en string del FilterManager"""
        status = "HABILITADO" if self.is_enabled() else "DESHABILITADO"
        mode = self.config.mode.upper()
        return f"FilterManager ({status}, {mode}, {len(self.filters)} filtros)"