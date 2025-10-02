# Plan de Implementación - Filtros de Trading Inteligentes

## Tareas de Implementación

- [ ] 1. Crear infraestructura base del sistema de filtros
  - Implementar clases base y interfaces principales
  - Crear sistema de configuración centralizada
  - Establecer estructura de logging y métricas
  - _Requisitos: 5.1, 5.2, 5.3_

- [ ] 1.1 Implementar FilterResult y modelos de datos base
  - Crear dataclasses para FilterResult, FilterConfig, FilterSummary
  - Implementar validación de tipos y valores por defecto
  - Añadir métodos de serialización para logging
  - _Requisitos: 5.2, 5.4_

- [ ] 1.2 Crear FilterLogger con métricas integradas
  - Implementar logging estructurado para cada filtro
  - Crear sistema de métricas en tiempo real
  - Añadir exportación de estadísticas para análisis
  - _Requisitos: 5.2, 5.3, 5.4_

- [ ] 1.3 Implementar FilterConfig con validación
  - Crear sistema de configuración flexible por filtro
  - Implementar validación de parámetros de configuración
  - Añadir carga desde archivo config/filter_settings.py
  - _Requisitos: 5.1, 5.5, 6.2_

- [ ] 2. Implementar VolumeFilter (Filtro de Volumen)
  - Crear clase VolumeFilter con cálculo de ratios de volumen
  - Implementar penalizaciones basadas en thresholds configurables
  - Añadir detección de volumen institucional vs retail
  - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2.1 Desarrollar cálculo de ratio de volumen
  - Implementar calculate_volume_ratio usando promedio 20 períodos
  - Añadir manejo de datos faltantes o inconsistentes
  - Crear validación de calidad de datos de volumen
  - _Requisitos: 1.5_

- [ ] 2.2 Implementar sistema de penalizaciones por volumen
  - Crear get_volume_penalty con thresholds configurables
  - Implementar lógica: <80% = max score 6, <50% = max score 4
  - Añadir bonus para volumen alto (>120% = +0.5 puntos)
  - _Requisitos: 1.1, 1.2, 1.3_

- [ ] 2.3 Añadir advertencias de volumen bajo
  - Implementar detección y logging de "Low Volume Setup"
  - Crear alertas cuando volumen sea insuficiente
  - Integrar advertencias en el sistema de notificaciones
  - _Requisitos: 1.4_

- [ ] 3. Implementar MomentumFilter (Filtro de Momentum)
  - Crear clase MomentumFilter con detección de momentum contrario
  - Implementar penalización automática para conflictos direccionales
  - Añadir manejo de momentum neutral
  - _Requisitos: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 3.1 Desarrollar detección de conflictos de momentum
  - Implementar detect_momentum_conflict usando MACD histogram
  - Crear lógica: SHORT + MACD positivo = conflicto
  - Añadir lógica: LONG + MACD negativo = conflicto
  - _Requisitos: 2.1, 2.2_

- [ ] 3.2 Implementar penalización automática
  - Crear aplicación automática de -3 puntos por conflicto
  - Implementar zona neutral (MACD entre -0.01 y +0.01)
  - Añadir logging detallado de penalizaciones aplicadas
  - _Requisitos: 2.3, 2.4, 2.5_

- [ ] 4. Implementar StructureFilter (Filtro de Estructura)
  - Crear clase StructureFilter con detección de tendencias
  - Implementar identificación de patrones HH/HL vs LH/LL
  - Añadir detección de Break of Structure (BOS)
  - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 4.1 Desarrollar detección de swing points
  - Implementar find_swing_points para identificar pivots
  - Crear algoritmo de detección de swing highs y lows
  - Añadir filtrado de ruido y confirmación de pivots
  - _Requisitos: 3.1, 3.2_

- [ ] 4.2 Implementar clasificación de estructura de mercado
  - Crear detect_market_structure: UPTREND, DOWNTREND, SIDEWAYS
  - Implementar lógica: 3 HH + 3 HL = UPTREND
  - Implementar lógica: 3 LH + 3 LL = DOWNTREND
  - _Requisitos: 3.1, 3.2_

- [ ] 4.3 Añadir penalizaciones y bonificaciones por estructura
  - Implementar penalización: SHORT en UPTREND = -2 puntos
  - Implementar penalización: LONG en DOWNTREND = -2 puntos
  - Añadir bonus: trade alineado con estructura = +1 punto
  - _Requisitos: 3.3, 3.4, 3.5_

- [ ] 4.4 Implementar detección de Break of Structure
  - Crear detect_break_of_structure para cambios de tendencia
  - Implementar invalidación de trades contra nueva estructura
  - Añadir alertas cuando se detecte cambio estructural
  - _Requisitos: 3.6, 3.7_

- [ ] 5. Implementar BTCCorrelationFilter (Filtro de Correlación BTC)
  - Crear clase BTCCorrelationFilter con obtención de datos BTC
  - Implementar filtrado solo para timeframes 4H+ y excluir Bitcoin
  - Añadir penalizaciones por momentum BTC contrario
  - _Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

- [ ] 5.1 Desarrollar obtención de datos BTC
  - Implementar get_btc_momentum usando mismo exchange y timeframe
  - Crear cálculo de MACD histogram para BTC
  - Añadir cache de datos BTC para optimizar performance
  - _Requisitos: 4.5, 4.6_

- [ ] 5.2 Implementar lógica de aplicación condicional
  - Crear should_apply_btc_filter: solo timeframes 4H+
  - Implementar exclusión cuando symbol == "BTCUSDT"
  - Añadir configuración de timeframes mínimos
  - _Requisitos: 4.1, 4.7, 4.8_

- [ ] 5.3 Añadir detección de conflictos BTC
  - Implementar detect_btc_conflict para momentum contrario
  - Crear penalización: BTC alcista + ALT SHORT = -1 punto
  - Crear penalización: BTC bajista + ALT LONG = -1 punto
  - Añadir bonus: BTC alineado = +0.5 puntos
  - _Requisitos: 4.2, 4.3, 4.4_

- [ ] 6. Implementar FilterManager (Coordinador Principal)
  - Crear clase FilterManager que coordine todos los filtros
  - Implementar aplicación secuencial de filtros
  - Añadir generación de FilterSummary completo
  - _Requisitos: 5.1, 5.2, 5.3, 5.4_

- [ ] 6.1 Desarrollar apply_filters con procesamiento secuencial
  - Implementar ejecución ordenada: Volumen → Momentum → Estructura → BTC
  - Crear acumulación de ajustes de score
  - Añadir manejo de errores por filtro individual
  - _Requisitos: 5.1, 5.2_

- [ ] 6.2 Implementar generación de FilterSummary
  - Crear resumen completo con score original vs final
  - Implementar decisión final: APPROVED, REJECTED, WARNING
  - Añadir razones detalladas de rechazo
  - _Requisitos: 5.3, 5.4_

- [ ] 6.3 Añadir sistema de estadísticas integradas
  - Implementar get_filter_stats con métricas por filtro
  - Crear tracking de trades filtrados vs aprobados
  - Añadir cálculo de win rate improvement
  - _Requisitos: 5.4_

- [ ] 7. Integrar filtros en el sistema principal de trading
  - Modificar AIValidatorUltra para usar FilterManager
  - Integrar filtros en el flujo de validación de señales
  - Actualizar sistema de notificaciones con información de filtros
  - _Requisitos: 5.1, 5.2, 5.3_

- [ ] 7.1 Modificar validate_signal para incluir filtros
  - Integrar FilterManager después del análisis técnico inicial
  - Modificar ValidationResult para incluir FilterSummary
  - Actualizar logging para mostrar impacto de filtros
  - _Requisitos: 5.2, 5.3_

- [ ] 7.2 Actualizar sistema de notificaciones Telegram
  - Modificar telegram_notifier para mostrar filtros aplicados
  - Añadir información de score original vs final
  - Incluir razones de rechazo por filtros en notificaciones
  - _Requisitos: 5.3, 5.4_

- [ ] 8. Implementar sistema de configuración y testing
  - Crear archivo config/filter_settings.py con todos los parámetros
  - Implementar modo "warning" vs "enforcement" por filtro
  - Añadir sistema de testing unitario para cada filtro
  - _Requisitos: 5.5, 6.1, 6.2, 6.3_

- [ ] 8.1 Crear configuración centralizada
  - Implementar FILTER_SETTINGS con todos los parámetros
  - Añadir validación de configuración al inicio
  - Crear sistema de recarga de configuración en caliente
  - _Requisitos: 5.5_

- [ ] 8.2 Implementar modos de operación flexibles
  - Crear modo "warning": solo alerta, no penaliza
  - Crear modo "enforcement": aplica penalizaciones reales
  - Añadir configuración individual por filtro
  - _Requisitos: 6.2_

- [ ] 8.3 Desarrollar suite de testing unitario
  - Crear tests para cada filtro con datos sintéticos
  - Implementar tests de casos edge y manejo de errores
  - Añadir tests de integración para FilterManager completo
  - _Requisitos: 6.3_

- [ ] 9. Implementar sistema de implementación gradual
  - Crear sistema de activación por fases
  - Implementar monitoreo comparativo entre versiones
  - Añadir generación de reportes de impacto
  - _Requisitos: 6.1, 6.3, 6.4, 6.5_

- [ ] 9.1 Desarrollar activación por fases
  - Implementar habilitación gradual: Volumen → Momentum → Estructura → BTC
  - Crear período de observación de 1 semana por fase
  - Añadir rollback automático si métricas empeoran
  - _Requisitos: 6.1, 6.5_

- [ ] 9.2 Crear sistema de monitoreo comparativo
  - Implementar A/B testing entre sistema actual y con filtros
  - Crear métricas de comparación: win rate, drawdown, Sharpe ratio
  - Añadir dashboard de métricas en tiempo real
  - _Requisitos: 6.3, 6.4_

- [ ] 9.3 Implementar generación de reportes automáticos
  - Crear reportes semanales de impacto por filtro
  - Implementar alertas cuando métricas se desvíen
  - Añadir recomendaciones automáticas de ajuste de parámetros
  - _Requisitos: 6.4_

- [ ] 10. Realizar testing y validación completa del sistema
  - Ejecutar backtesting con datos históricos de 6 meses
  - Realizar paper trading por 2-4 semanas
  - Validar mejora de win rate y reducción de drawdown
  - _Requisitos: Todos los requisitos_

- [ ] 10.1 Ejecutar backtesting histórico completo
  - Probar sistema con datos de últimos 6 meses
  - Comparar métricas: trades filtrados, win rate, profit factor
  - Validar que Score 10 tenga realmente 85%+ win rate
  - _Requisitos: Validación general_

- [ ] 10.2 Realizar paper trading en vivo
  - Ejecutar sistema completo en modo paper por 2-4 semanas
  - Monitorear performance en tiempo real
  - Ajustar parámetros basado en resultados reales
  - _Requisitos: Validación en vivo_

- [ ] 10.3 Generar reporte final de validación
  - Documentar mejoras obtenidas vs sistema original
  - Crear recomendaciones para optimización continua
  - Preparar sistema para deployment en producción
  - _Requisitos: Documentación y deployment_