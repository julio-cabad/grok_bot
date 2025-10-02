"""
Sistema de Filtros de Trading Inteligentes
Módulo principal para filtros que mejoran la precisión del trading automatizado
"""

from .base.filter_result import FilterResult, FilterSummary
from .base.filter_config import FilterConfig

__all__ = [
    'FilterResult', 
    'FilterSummary',
    'FilterConfig'
]

# FilterManager se importará cuando esté implementado
# from .manager.filter_manager import FilterManager