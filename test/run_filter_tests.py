#!/usr/bin/env python3
"""
Script para ejecutar todos los tests del sistema de filtros
Ejecuta tests de infraestructura y tests con datos reales
"""

import unittest
import sys
import os
from datetime import datetime

# Añadir path del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """Ejecuta todos los tests del sistema de filtros"""
    
    print("🧪 EJECUTANDO TESTS DEL SISTEMA DE FILTROS")
    print("=" * 60)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print("=" * 60)
    
    # Descubrir y ejecutar todos los tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_filter*.py')
    
    # Configurar runner con verbosidad
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Ejecutar tests
    print("\n🚀 INICIANDO EJECUCIÓN DE TESTS...")
    print("-" * 60)
    
    result = runner.run(suite)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE EJECUCIÓN")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    success = total_tests - failures - errors - skipped
    
    print(f"✅ Tests exitosos: {success}/{total_tests}")
    print(f"❌ Tests fallidos: {failures}")
    print(f"💥 Tests con errores: {errors}")
    print(f"⏭️ Tests omitidos: {skipped}")
    
    success_rate = (success / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Tasa de éxito: {success_rate:.1f}%")
    
    # Detalles de fallos si los hay
    if result.failures:
        print(f"\n❌ DETALLES DE FALLOS ({len(result.failures)}):")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"🔴 {test}")
            print(f"   {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 DETALLES DE ERRORES ({len(result.errors)}):")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"🔴 {test}")
            error_line = traceback.split('\n')[-2] if '\n' in traceback else traceback
            print(f"   {error_line.strip()}")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    if success_rate >= 95:
        print("🎉 ¡Excelente! La infraestructura está lista para producción")
    elif success_rate >= 80:
        print("✅ Buena cobertura, revisar fallos menores antes de continuar")
    elif success_rate >= 60:
        print("⚠️ Algunos problemas detectados, revisar antes de continuar")
    else:
        print("🚨 Problemas serios detectados, revisar infraestructura")
    
    print("\n🔧 PRÓXIMOS PASOS:")
    if success_rate >= 90:
        print("   1. ✅ Infraestructura validada")
        print("   2. 🚀 Continuar con implementación de filtros específicos")
        print("   3. 📊 Comenzar con VolumeFilter")
    else:
        print("   1. 🔍 Revisar y corregir fallos detectados")
        print("   2. 🧪 Re-ejecutar tests hasta 90%+ éxito")
        print("   3. 📋 Validar configuración y dependencias")
    
    print("=" * 60)
    
    # Retornar código de salida
    return 0 if success_rate >= 90 else 1


def run_specific_test(test_name):
    """Ejecuta un test específico"""
    
    print(f"🧪 EJECUTANDO TEST ESPECÍFICO: {test_name}")
    print("=" * 60)
    
    # Cargar test específico
    loader = unittest.TestLoader()
    
    try:
        if '.' in test_name:
            # Formato: module.TestClass.test_method
            suite = loader.loadTestsFromName(test_name)
        else:
            # Formato: test_file
            suite = loader.loadTestsFromName(f"{test_name}")
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return 0 if result.wasSuccessful() else 1
        
    except Exception as e:
        print(f"❌ Error cargando test {test_name}: {e}")
        return 1


if __name__ == '__main__':
    # Verificar argumentos
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        exit_code = run_specific_test(test_name)
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)