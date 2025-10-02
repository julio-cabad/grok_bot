"""
Filtro de Volumen para Trading Inteligente
Valida volumen institucional vs retail antes de aprobar trades
"""

from .volume_filter import VolumeFilter

__all__ = ['VolumeFilter']