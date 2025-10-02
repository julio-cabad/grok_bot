"""
Clases base y modelos de datos para el sistema de filtros
"""

from .filter_result import FilterResult, FilterSummary
from .filter_config import FilterConfig
from .filter_base import FilterBase

__all__ = [
    'FilterResult',
    'FilterSummary', 
    'FilterConfig',
    'FilterBase'
]