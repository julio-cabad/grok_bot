"""
Modelos de datos para resultados de filtros de trading
Contiene las estructuras de datos que usan todos los filtros
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from datetime import datetime


@dataclass
class FilterResult:
    """
    Resultado de aplicar un filtro individual
    
    Attributes:
        filter_name: Nombre del filtro aplicado
        applied: Si el filtro se aplicó o se saltó
        score_adjustment: Ajuste aplicado al score (-3 a +1)
        reason: Razón del ajuste en español
        warning: Advertencia opcional para el usuario
        metrics: Métricas adicionales del filtro
    """
    filter_name: str
    applied: bool
    score_adjustment: float
    reason: str
    warning: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario para logging"""
        return {
            'filter_name': self.filter_name,
            'applied': self.applied,
            'score_adjustment': self.score_adjustment,
            'reason': self.reason,
            'warning': self.warning,
            'metrics': self.metrics
        }
    
    def to_json(self) -> str:
        """Convierte el resultado a JSON para persistencia"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class FilterSummary:
    """
    Resumen completo de todos los filtros aplicados a una señal
    
    Attributes:
        symbol: Símbolo analizado (ej: BTCUSDT)
        signal_type: Tipo de señal (LONG/SHORT)
        original_score: Score original antes de filtros
        final_score: Score final después de filtros
        total_adjustment: Suma total de ajustes aplicados
        filters_applied: Lista de todos los filtros ejecutados
        trade_decision: Decisión final (APPROVED/REJECTED/WARNING)
        rejection_reason: Razón específica si fue rechazado
        timestamp: Momento del análisis
    """
    symbol: str
    signal_type: str
    original_score: float
    final_score: float
    total_adjustment: float
    filters_applied: List[FilterResult]
    trade_decision: str  # "APPROVED", "REJECTED", "WARNING"
    rejection_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validación después de inicialización"""
        # Validar que el score final sea consistente
        expected_final = self.original_score + self.total_adjustment
        if abs(self.final_score - expected_final) > 0.01:
            raise ValueError(f"Score final inconsistente: {self.final_score} vs esperado {expected_final}")
        
        # Validar decisión de trade
        valid_decisions = ["APPROVED", "REJECTED", "WARNING"]
        if self.trade_decision not in valid_decisions:
            raise ValueError(f"Decisión inválida: {self.trade_decision}. Debe ser una de: {valid_decisions}")
    
    @property
    def score_improvement(self) -> float:
        """Calcula la mejora (o empeoramiento) del score"""
        return self.final_score - self.original_score
    
    @property
    def filters_count(self) -> int:
        """Número de filtros que se aplicaron realmente"""
        return len([f for f in self.filters_applied if f.applied])
    
    @property
    def has_warnings(self) -> bool:
        """Indica si algún filtro generó advertencias"""
        return any(f.warning for f in self.filters_applied)
    
    def get_applied_filters(self) -> List[FilterResult]:
        """Retorna solo los filtros que se aplicaron"""
        return [f for f in self.filters_applied if f.applied]
    
    def get_filter_by_name(self, filter_name: str) -> Optional[FilterResult]:
        """Busca un filtro específico por nombre"""
        for filter_result in self.filters_applied:
            if filter_result.filter_name == filter_name:
                return filter_result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resumen a diccionario para logging"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'original_score': self.original_score,
            'final_score': self.final_score,
            'total_adjustment': self.total_adjustment,
            'score_improvement': self.score_improvement,
            'filters_applied': [f.to_dict() for f in self.filters_applied],
            'filters_count': self.filters_count,
            'trade_decision': self.trade_decision,
            'rejection_reason': self.rejection_reason,
            'has_warnings': self.has_warnings,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convierte el resumen a JSON para persistencia"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def get_summary_text(self) -> str:
        """Genera resumen en texto para notificaciones"""
        applied_filters = self.get_applied_filters()
        
        if not applied_filters:
            return f"Sin filtros aplicados - Score: {self.original_score:.1f}"
        
        filter_names = [f.filter_name for f in applied_filters]
        adjustment_text = f"{self.total_adjustment:+.1f}" if self.total_adjustment != 0 else "0.0"
        
        return (f"Filtros aplicados: {', '.join(filter_names)} | "
                f"Score: {self.original_score:.1f} → {self.final_score:.1f} "
                f"({adjustment_text}) | {self.trade_decision}")


# Funciones de utilidad para crear resultados comunes
def create_no_filter_result(filter_name: str, reason: str = "Filtro no aplicable") -> FilterResult:
    """Crea un resultado cuando el filtro no se aplica"""
    return FilterResult(
        filter_name=filter_name,
        applied=False,
        score_adjustment=0.0,
        reason=reason
    )


def create_penalty_result(filter_name: str, penalty: float, reason: str, 
                         warning: Optional[str] = None, metrics: Optional[Dict] = None) -> FilterResult:
    """Crea un resultado con penalización"""
    return FilterResult(
        filter_name=filter_name,
        applied=True,
        score_adjustment=-abs(penalty),  # Asegurar que sea negativo
        reason=reason,
        warning=warning,
        metrics=metrics or {}
    )


def create_bonus_result(filter_name: str, bonus: float, reason: str, 
                       metrics: Optional[Dict] = None) -> FilterResult:
    """Crea un resultado con bonificación"""
    return FilterResult(
        filter_name=filter_name,
        applied=True,
        score_adjustment=abs(bonus),  # Asegurar que sea positivo
        reason=reason,
        metrics=metrics or {}
    )