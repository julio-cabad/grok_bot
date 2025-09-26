# Test Directory - Trading Bot Refactoring

Esta carpeta contiene todos los componentes refactorizados que estamos desarrollando.

## Estructura

```
test/
├── config/           # Configuración y feature flags
├── migration/        # Sistema de migración gradual
├── core/            # Entidades de dominio
├── infrastructure/  # Adaptadores externos
├── application/     # Servicios de aplicación
└── tests/          # Tests unitarios e integración
```

## Principio

**El bot actual sigue funcionando normalmente mientras desarrollamos aquí.**

Cada componente se desarrolla, testea y valida en esta carpeta antes de ser integrado al sistema principal mediante feature flags.