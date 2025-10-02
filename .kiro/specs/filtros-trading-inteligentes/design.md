# Documento de Diseño - Filtros de Trading Inteligentes

## Visión General

El sistema de Filtros de Trading Inteligentes se implementará como una capa de validación que se ejecuta después del análisis técnico inicial pero antes de la asignación final del score. Los filtros operarán de manera secuencial, aplicando penalizaciones o bonificaciones según las condiciones del mercado.

## Arquitectura

### Diagrama de Flujo del Sistema

```
Datos de Mercado → Análisis Técnico Inicial → Filtros Inteligentes → Score Final → Decisión de Trade
                                                      ↓
                                            [Volumen] → [Momentum] → [Estructura] → [BTC Correlation]
```

### Componentes Principales

1. **FilterManager**: Coordinador principal de todos los filtros
2. **VolumeFilter**: Validación de volumen institucional
3. **MomentumFilter**: Detección de momentum contrario
4. **StructureFilter**: Análisis de estructura de mercado
5. **BTCCorrelationFilter**: Monitoreo de correlación con Bitcoin
6. **FilterConfig**: Sistema de configuración centralizada
7. **FilterLogger**: Sistema de logging y métricas

## Componentes e Interfaces

### 1. FilterManager (Coordinador Principal)

```python
class FilterManager:
    def __init__(self, config: FilterConfig):
        self.filters = [
            VolumeFilter(config.volume),
            MomentumFilter(config.momentum), 
            StructureFilter(config.structure),
            BTCCorrelationFilter(config.btc_correlation)
        ]
        self.logger = FilterLogger()
    
    def apply_filters(self, 
                     symbol: str, 
                     signal_type: str, 
                     initial_score: float, 
                     market_data: Dict) -> FilterResult:
        """Aplica todos los filtros secuencialmente"""
        
    def get_filter_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de todos los filtros"""
```

### 2. VolumeFilter (Filtro de Volumen)

```python
class VolumeFilter:
    def __init__(self, config: VolumeConfig):
        self.min_volume_threshold = config.min_volume_threshold  # 0.8 (80%)
        self.low_volume_threshold = config.low_volume_threshold  # 0.5 (50%)
        self.high_volume_bonus_threshold = config.high_volume_bonus  # 1.2 (120%)
        
    def apply(self, symbol: str, market_data: Dict) -> FilterResult:
        """
        Calcula ratio de volumen actual vs promedio 20 períodos
        Aplica penalizaciones o bonificaciones según thresholds
        """
        
    def calculate_volume_ratio(self, current_volume: float, df: pd.DataFrame) -> float:
        """Calcula ratio volumen actual vs promedio 20 períodos"""
        
    def get_volume_penalty(self, volume_ratio: float) -> float:
        """Determina penalización basada en ratio de volumen"""
```

### 3. MomentumFilter (Filtro de Momentum)

```python
class MomentumFilter:
    def __init__(self, config: MomentumConfig):
        self.momentum_penalty = config.momentum_penalty  # -3 puntos
        self.neutral_threshold = config.neutral_threshold  # 0.01
        
    def apply(self, signal_type: str, market_data: Dict) -> FilterResult:
        """
        Detecta momentum contrario usando MACD histogram
        Aplica penalización automática si hay conflicto
        """
        
    def detect_momentum_conflict(self, signal_type: str, macd_hist: float) -> bool:
        """Detecta si hay conflicto entre señal y momentum"""
        
    def is_momentum_neutral(self, macd_hist: float) -> bool:
        """Determina si momentum es neutral"""
```

### 4. StructureFilter (Filtro de Estructura)

```python
class StructureFilter:
    def __init__(self, config: StructureConfig):
        self.structure_penalty = config.structure_penalty  # -2 puntos
        self.structure_bonus = config.structure_bonus  # +1 punto
        self.lookback_periods = config.lookback_periods  # 20
        
    def apply(self, signal_type: str, market_data: Dict) -> FilterResult:
        """
        Detecta estructura de mercado y aplica penalizaciones/bonificaciones
        """
        
    def detect_market_structure(self, df: pd.DataFrame) -> str:
        """
        Detecta estructura: UPTREND, DOWNTREND, SIDEWAYS
        Usando patrones HH/HL vs LH/LL
        """
        
    def find_swing_points(self, df: pd.DataFrame) -> Dict[str, List]:
        """Identifica swing highs y swing lows"""
        
    def detect_break_of_structure(self, df: pd.DataFrame, current_structure: str) -> bool:
        """Detecta cambio de estructura de mercado"""
```

### 5. BTCCorrelationFilter (Filtro de Correlación BTC)

```python
class BTCCorrelationFilter:
    def __init__(self, config: BTCConfig):
        self.btc_penalty = config.btc_penalty  # -1 punto
        self.btc_bonus = config.btc_bonus  # +0.5 puntos
        self.min_timeframe = config.min_timeframe  # "4h"
        
    def apply(self, symbol: str, signal_type: str, timeframe: str, market_data: Dict) -> FilterResult:
        """
        Aplica filtro de correlación BTC solo para timeframes 4H+
        Excluye Bitcoin de su propio análisis
        """
        
    def should_apply_btc_filter(self, symbol: str, timeframe: str) -> bool:
        """Determina si aplicar filtro BTC"""
        
    def get_btc_momentum(self, timeframe: str) -> float:
        """Obtiene momentum de BTC usando MACD histogram"""
        
    def detect_btc_conflict(self, signal_type: str, btc_momentum: float) -> bool:
        """Detecta conflicto entre señal y momentum BTC"""
```

## Modelos de Datos

### FilterResult

```python
@dataclass
class FilterResult:
    filter_name: str
    applied: bool
    score_adjustment: float
    reason: str
    warning: Optional[str] = None
    metrics: Dict[str, Any] = None
```

### FilterConfig

```python
@dataclass
class FilterConfig:
    # Configuración global
    enabled: bool = True
    mode: str = "enforcement"  # "warning" o "enforcement"
    
    # Configuraciones específicas
    volume: VolumeConfig
    momentum: MomentumConfig
    structure: StructureConfig
    btc_correlation: BTCConfig
```

### FilterSummary

```python
@dataclass
class FilterSummary:
    original_score: float
    final_score: float
    total_adjustment: float
    filters_applied: List[FilterResult]
    trade_decision: str  # "APPROVED", "REJECTED", "WARNING"
    rejection_reason: Optional[str] = None
```

## Manejo de Errores

### Estrategia de Fallback

1. **Error en filtro individual**: Continuar con otros filtros, registrar error
2. **Error en datos de mercado**: Usar valores por defecto conservadores
3. **Error en datos BTC**: Desactivar filtro BTC temporalmente
4. **Error crítico del sistema**: Modo conservador (score máximo 6)

### Logging de Errores

```python
class FilterLogger:
    def log_filter_application(self, symbol: str, filter_summary: FilterSummary):
        """Registra aplicación completa de filtros"""
        
    def log_filter_error(self, filter_name: str, error: Exception, symbol: str):
        """Registra errores de filtros individuales"""
        
    def log_filter_stats(self, period: str = "daily"):
        """Registra estadísticas periódicas de filtros"""
```

## Estrategia de Testing

### Testing Unitario

1. **Cada filtro individualmente** con datos sintéticos
2. **Casos edge**: volumen cero, datos faltantes, valores extremos
3. **Configuraciones diferentes**: thresholds, modos warning/enforcement

### Testing de Integración

1. **FilterManager completo** con datos reales históricos
2. **Secuencia de filtros** y interacciones entre ellos
3. **Performance** con múltiples símbolos simultáneos

### Testing de Validación

1. **Backtesting** con datos históricos de 6 meses
2. **A/B testing** entre sistema actual y con filtros
3. **Paper trading** por 2-4 semanas antes de implementación

### Métricas de Testing

```python
class FilterMetrics:
    def __init__(self):
        self.trades_filtered = 0
        self.trades_approved = 0
        self.filter_impact = {}  # Por filtro individual
        self.win_rate_improvement = 0.0
        self.avg_score_adjustment = 0.0
```

## Implementación Gradual

### Fase 1: Filtro de Volumen (Semana 1)
- Implementar solo VolumeFilter
- Modo "warning" por 3 días
- Modo "enforcement" por 4 días
- Análisis de impacto

### Fase 2: Filtro de Momentum (Semana 2)
- Añadir MomentumFilter
- Combinación Volumen + Momentum
- Monitoreo de interacciones

### Fase 3: Filtro de Estructura (Semana 3)
- Añadir StructureFilter
- Validar detección de tendencias
- Ajustar thresholds según resultados

### Fase 4: Filtro BTC (Semana 4)
- Añadir BTCCorrelationFilter
- Sistema completo operativo
- Optimización final de parámetros

## Configuración del Sistema

### Archivo de Configuración (config/filter_settings.py)

```python
FILTER_SETTINGS = {
    "enabled": True,
    "mode": "enforcement",  # "warning" o "enforcement"
    
    "volume_filter": {
        "enabled": True,
        "min_volume_threshold": 0.8,  # 80% del promedio
        "low_volume_threshold": 0.5,  # 50% del promedio
        "high_volume_bonus_threshold": 1.2,  # 120% del promedio
        "lookback_periods": 20
    },
    
    "momentum_filter": {
        "enabled": True,
        "momentum_penalty": -3.0,
        "neutral_threshold": 0.01
    },
    
    "structure_filter": {
        "enabled": True,
        "structure_penalty": -2.0,
        "structure_bonus": 1.0,
        "lookback_periods": 20,
        "min_swing_strength": 3
    },
    
    "btc_correlation_filter": {
        "enabled": True,
        "btc_penalty": -1.0,
        "btc_bonus": 0.5,
        "min_timeframe": "4h",
        "correlation_threshold": 0.7
    }
}
```

## Monitoreo y Métricas

### Dashboard de Filtros

1. **Trades filtrados por día/semana**
2. **Impacto de cada filtro individual**
3. **Win rate antes/después de filtros**
4. **Score distribution** antes/después
5. **Tiempo de procesamiento** de filtros

### Alertas del Sistema

1. **Filtro fallando frecuentemente**
2. **Demasiados trades rechazados** (>80%)
3. **Win rate deteriorándose** después de filtros
4. **Errores en obtención de datos BTC**

Este diseño proporciona una base sólida para implementar los filtros de manera modular, testeable y configurable, permitiendo optimización continua basada en resultados reales.