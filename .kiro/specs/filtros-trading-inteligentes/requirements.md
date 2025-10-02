# Documento de Requisitos - Filtros de Trading Inteligentes

## Introducción

Este proyecto implementa filtros críticos para mejorar la precisión del sistema de trading automatizado, reduciendo trades de baja calidad y aumentando el win rate. Los filtros se basan en análisis de volumen, momentum, estructura de mercado y correlación con BTC.

**Problema Actual**: El sistema genera Score 10/10 que fallan inmediatamente en múltiples criptomonedas, indicando calibración incorrecta y falta de filtros fundamentales.

**Objetivo**: Implementar 4 filtros esenciales que mejoren el win rate del 55% actual al 70% estimado, reduciendo drawdown y aumentando confiabilidad.

## Requisitos

### Requisito 1: Sistema de Filtro de Volumen

**Historia de Usuario**: Como trader automatizado, quiero que el sistema valide el volumen antes de asignar scores altos, para evitar trades en resistencias/soportes falsos sin participación institucional.

#### Criterios de Aceptación

1. CUANDO el volumen actual sea menor al 80% del promedio de 20 períodos, ENTONCES el sistema NO DEBERÁ asignar Score superior a 6
2. CUANDO el volumen actual sea menor al 50% del promedio de 20 períodos, ENTONCES el sistema NO DEBERÁ asignar Score superior a 4
3. CUANDO el volumen sea superior al 120% del promedio, ENTONCES el sistema PODRÁ otorgar bonus de +0.5 puntos al score
4. EL sistema DEBERÁ mostrar advertencia "Low Volume Setup" cuando el volumen sea insuficiente
5. EL sistema DEBERÁ calcular el promedio de volumen usando los últimos 20 períodos del timeframe actual

### Requisito 2: Sistema de Penalización por Momentum Contrario

**Historia de Usuario**: Como trader automatizado, quiero que el sistema penalice automáticamente trades que van contra el momentum dominante, para evitar "nadar contra corriente" en cualquier criptomoneda.

#### Criterios de Aceptación

1. CUANDO se genere señal SHORT y el MACD histogram sea positivo, ENTONCES el sistema DEBERÁ restar 3 puntos del score
2. CUANDO se genere señal LONG y el MACD histogram sea negativo, ENTONCES el sistema DEBERÁ restar 3 puntos del score
3. CUANDO el momentum sea neutral (MACD histogram entre -0.01 y +0.01), ENTONCES NO se aplicará penalización
4. EL sistema DEBERÁ registrar en logs cuando aplique penalización por momentum contrario
5. LA penalización DEBERÁ ser automática sin excepciones manuales

### Requisito 3: Sistema de Detección de Estructura de Mercado

**Historia de Usuario**: Como trader automatizado, quiero que el sistema identifique la estructura básica del mercado (tendencia alcista/bajista) para penalizar trades counter-trend y bonificar trades with-trend.

#### Criterios de Aceptación

1. CUANDO se detecten 3 Higher Highs y 3 Higher Lows consecutivos, ENTONCES el sistema DEBERÁ clasificar como "UPTREND"
2. CUANDO se detecten 3 Lower Highs y 3 Lower Lows consecutivos, ENTONCES el sistema DEBERÁ clasificar como "DOWNTREND"
3. CUANDO se genere señal SHORT en UPTREND confirmado, ENTONCES el sistema DEBERÁ restar 2 puntos del score
4. CUANDO se genere señal LONG en DOWNTREND confirmado, ENTONCES el sistema DEBERÁ restar 2 puntos del score
5. CUANDO se genere señal alineada con la estructura, ENTONCES el sistema PODRÁ otorgar bonus de +1 punto
6. EL sistema DEBERÁ detectar Break of Structure (BOS) cuando se rompa el patrón de HH/HL o LH/LL
7. CUANDO se detecte BOS, ENTONCES el sistema DEBERÁ invalidar trades abiertos contra la nueva estructura

### Requisito 4: Sistema de Monitoreo de Correlación BTC

**Historia de Usuario**: Como trader de criptomonedas, quiero que el sistema considere el momentum de BTC antes de generar señales, ya que la mayoría de criptomonedas siguen a BTC el 70-80% del tiempo en timeframes altos.

#### Criterios de Aceptación

1. CUANDO el timeframe sea 4H o superior, ENTONCES el sistema DEBERÁ aplicar filtro de correlación BTC
2. CUANDO BTC tenga momentum alcista y se genere señal SHORT en cualquier criptomoneda, ENTONCES el sistema DEBERÁ restar 1 punto del score
3. CUANDO BTC tenga momentum bajista y se genere señal LONG en cualquier criptomoneda, ENTONCES el sistema DEBERÁ restar 1 punto del score
4. CUANDO BTC momentum esté alineado con la señal, ENTONCES el sistema PODRÁ otorgar bonus de +0.5 puntos
5. EL sistema DEBERÁ obtener datos de BTC/USDT del mismo exchange y timeframe
6. EL momentum de BTC DEBERÁ calcularse usando MACD histogram de BTC
7. PARA timeframes menores a 4H, el filtro BTC NO se aplicará
8. SI la criptomoneda analizada ES Bitcoin, ENTONCES este filtro NO se aplicará

### Requisito 5: Sistema de Configuración y Logging

**Historia de Usuario**: Como desarrollador del sistema, quiero poder configurar los filtros y monitorear su impacto para optimizar los parámetros según los resultados.

#### Criterios de Aceptación

1. EL sistema DEBERÁ permitir habilitar/deshabilitar cada filtro independientemente
2. EL sistema DEBERÁ registrar en logs el score original y el score final después de filtros
3. EL sistema DEBERÁ mostrar qué filtros se aplicaron y su impacto en cada análisis
4. EL sistema DEBERÁ mantener estadísticas de trades rechazados por cada filtro
5. EL sistema DEBERÁ permitir ajustar los thresholds de cada filtro via configuración
6. CUANDO un trade sea rechazado por filtros, ENTONCES el sistema DEBERÁ explicar la razón específica

### Requisito 6: Sistema de Implementación Gradual

**Historia de Usuario**: Como operador del sistema, quiero implementar los filtros gradualmente para validar su efectividad sin perder oportunidades reales de trading.

#### Criterios de Aceptación

1. EL sistema DEBERÁ permitir activar filtros en fases: Volumen → Momentum → Estructura → BTC
2. CADA filtro DEBERÁ poder configurarse en modo "warning" (solo alerta) o "enforcement" (aplicar penalización)
3. EL sistema DEBERÁ mantener métricas comparativas entre trades con y sin filtros
4. EL sistema DEBERÁ generar reportes semanales del impacto de cada filtro
5. CUANDO se active un nuevo filtro, ENTONCES el sistema DEBERÁ monitorear por 1 semana antes del siguiente