#!/usr/bin/env python3
"""
Tests para la infraestructura base del sistema de filtros
Valida modelos de datos, configuración y logging con datos reales
"""

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

# Importar módulos a testear
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ai.filters.base.filter_result import (
        FilterResult, FilterSummary, 
        create_no_filter_result, create_penalty_result, create_bonus_result
    )
    from ai.filters.base.filter_config import (
        FilterConfig, VolumeConfig, MomentumConfig, StructureConfig, BTCConfig,
        load_filter_config
    )
    from ai.filters.utils.filter_logger import FilterLogger
    from ai.filters.utils.filter_metrics import FilterMetrics, FilterMetricsCalculator
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    sys.exit(1)


class TestFilterResult(unittest.TestCase):
    """Tests para FilterResult y FilterSummary"""
    
    def test_filter_result_creation(self):
        """Test creación básica de FilterResult"""
        result = FilterResult(
            filter_name="TestFilter",
            applied=True,
            score_adjustment=-2.0,
            reason="Test penalty",
            warning="Test warning",
            metrics={'test_metric': 123}
        )
        
        self.assertEqual(result.filter_name, "TestFilter")
        self.assertTrue(result.applied)
        self.assertEqual(result.score_adjustment, -2.0)
        self.assertEqual(result.reason, "Test penalty")
        self.assertEqual(result.warning, "Test warning")
        self.assertEqual(result.metrics['test_metric'], 123)
    
    def test_filter_result_serialization(self):
        """Test serialización de FilterResult"""
        result = FilterResult(
            filter_name="VolumeFilter",
            applied=True,
            score_adjustment=-1.5,
            reason="Volumen bajo detectado"
        )
        
        # Test to_dict
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['filter_name'], "VolumeFilter")
        self.assertEqual(result_dict['score_adjustment'], -1.5)
        
        # Test to_json
        result_json = result.to_json()
        self.assertIsInstance(result_json, str)
        parsed = json.loads(result_json)
        self.assertEqual(parsed['filter_name'], "VolumeFilter")
    
    def test_filter_summary_creation_and_validation(self):
        """Test creación y validación de FilterSummary"""
        filter_results = [
            FilterResult("VolumeFilter", True, -2.0, "Volumen bajo"),
            FilterResult("MomentumFilter", True, -1.0, "Momentum contrario"),
            FilterResult("StructureFilter", False, 0.0, "No aplicable")
        ]
        
        summary = FilterSummary(
            symbol="BTCUSDT",
            signal_type="SHORT",
            original_score=8.0,
            final_score=5.0,
            total_adjustment=-3.0,
            filters_applied=filter_results,
            trade_decision="REJECTED",
            rejection_reason="Score final muy bajo"
        )
        
        # Validar propiedades calculadas
        self.assertEqual(summary.score_improvement, -3.0)
        self.assertEqual(summary.filters_count, 2)  # Solo 2 aplicados
        self.assertFalse(summary.has_warnings)
        
        # Test get_applied_filters
        applied = summary.get_applied_filters()
        self.assertEqual(len(applied), 2)
        self.assertEqual(applied[0].filter_name, "VolumeFilter")
        
        # Test get_filter_by_name
        volume_filter = summary.get_filter_by_name("VolumeFilter")
        self.assertIsNotNone(volume_filter)
        self.assertEqual(volume_filter.score_adjustment, -2.0)
        
        nonexistent = summary.get_filter_by_name("NonExistent")
        self.assertIsNone(nonexistent)
    
    def test_filter_summary_validation_error(self):
        """Test que FilterSummary valide consistencia de scores"""
        filter_results = [
            FilterResult("TestFilter", True, -2.0, "Test")
        ]
        
        # Score final inconsistente debería fallar
        with self.assertRaises(ValueError):
            FilterSummary(
                symbol="BTCUSDT",
                signal_type="LONG",
                original_score=8.0,
                final_score=5.0,  # Debería ser 6.0 (8.0 - 2.0)
                total_adjustment=-2.0,
                filters_applied=filter_results,
                trade_decision="APPROVED"
            )
        
        # Decisión inválida debería fallar
        with self.assertRaises(ValueError):
            FilterSummary(
                symbol="BTCUSDT",
                signal_type="LONG",
                original_score=8.0,
                final_score=6.0,
                total_adjustment=-2.0,
                filters_applied=filter_results,
                trade_decision="INVALID_DECISION"
            )
    
    def test_utility_functions(self):
        """Test funciones de utilidad para crear resultados"""
        # Test create_no_filter_result
        no_filter = create_no_filter_result("TestFilter", "No aplicable")
        self.assertFalse(no_filter.applied)
        self.assertEqual(no_filter.score_adjustment, 0.0)
        
        # Test create_penalty_result
        penalty = create_penalty_result("TestFilter", 2.5, "Penalización test")
        self.assertTrue(penalty.applied)
        self.assertEqual(penalty.score_adjustment, -2.5)  # Debe ser negativo
        
        # Test create_bonus_result
        bonus = create_bonus_result("TestFilter", 1.5, "Bonus test")
        self.assertTrue(bonus.applied)
        self.assertEqual(bonus.score_adjustment, 1.5)  # Debe ser positivo


class TestFilterConfig(unittest.TestCase):
    """Tests para sistema de configuración"""
    
    def test_volume_config_validation(self):
        """Test validación de VolumeConfig"""
        # Configuración válida
        config = VolumeConfig(
            min_volume_threshold=0.8,
            low_volume_threshold=0.5,
            lookback_periods=20
        )
        self.assertEqual(config.min_volume_threshold, 0.8)
        
        # Threshold inválido debería fallar
        with self.assertRaises(ValueError):
            VolumeConfig(min_volume_threshold=5.0)  # Muy alto
        
        with self.assertRaises(ValueError):
            VolumeConfig(low_volume_threshold=0.9, min_volume_threshold=0.8)  # low >= min
    
    def test_momentum_config_validation(self):
        """Test validación de MomentumConfig"""
        # Configuración válida
        config = MomentumConfig(
            momentum_penalty=-3.0,
            neutral_threshold=0.01
        )
        self.assertEqual(config.momentum_penalty, -3.0)
        
        # Penalización positiva debería fallar
        with self.assertRaises(ValueError):
            MomentumConfig(momentum_penalty=3.0)
        
        # Threshold muy alto debería fallar
        with self.assertRaises(ValueError):
            MomentumConfig(neutral_threshold=0.5)
    
    def test_filter_config_main(self):
        """Test configuración principal FilterConfig"""
        config = FilterConfig(
            enabled=True,
            mode="enforcement"
        )
        
        # Test is_filter_enabled
        self.assertTrue(config.is_filter_enabled("volume"))
        self.assertTrue(config.is_filter_enabled("momentum"))
        
        # Deshabilitar globalmente
        config.enabled = False
        self.assertFalse(config.is_filter_enabled("volume"))
        
        # Test get_filter_config
        config.enabled = True
        volume_config = config.get_filter_config("volume")
        self.assertIsInstance(volume_config, VolumeConfig)
        
        # Filtro inexistente debería fallar
        with self.assertRaises(ValueError):
            config.get_filter_config("nonexistent")
    
    def test_config_serialization(self):
        """Test serialización de configuración"""
        config = FilterConfig()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn('enabled', config_dict)
        self.assertIn('volume', config_dict)
        self.assertIn('momentum', config_dict)
        
        # Verificar estructura de sub-configuraciones
        self.assertIn('min_volume_threshold', config_dict['volume'])
        self.assertIn('momentum_penalty', config_dict['momentum'])
    
    def test_load_config_from_file(self):
        """Test carga de configuración desde archivo"""
        # Crear archivo temporal de configuración
        config_content = '''
FILTER_SETTINGS = {
    "enabled": True,
    "mode": "warning",
    "volume_filter": {
        "enabled": True,
        "min_volume_threshold": 0.9
    },
    "momentum_filter": {
        "enabled": False,
        "momentum_penalty": -4.0
    }
}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            # Cargar configuración
            config = load_filter_config(temp_path)
            
            self.assertTrue(config.enabled)
            self.assertEqual(config.mode, "warning")
            self.assertEqual(config.volume.min_volume_threshold, 0.9)
            self.assertFalse(config.momentum.enabled)
            self.assertEqual(config.momentum.momentum_penalty, -4.0)
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_default(self):
        """Test carga de configuración por defecto"""
        config = load_filter_config("nonexistent_file.py")
        
        # Debería cargar configuración por defecto
        self.assertIsInstance(config, FilterConfig)
        self.assertTrue(config.enabled)
        self.assertEqual(config.mode, "enforcement")


class TestFilterLogger(unittest.TestCase):
    """Tests para sistema de logging"""
    
    def setUp(self):
        """Setup para cada test"""
        self.logger = FilterLogger(max_history_size=100)
    
    def test_filter_logger_initialization(self):
        """Test inicialización del logger"""
        self.assertIsNotNone(self.logger.logger)
        self.assertEqual(self.logger.max_history_size, 100)
        self.assertEqual(len(self.logger._filter_history), 0)
    
    def test_log_filter_application(self):
        """Test logging de aplicación de filtros"""
        # Crear FilterSummary de prueba
        filter_results = [
            FilterResult("VolumeFilter", True, -2.0, "Volumen bajo"),
            FilterResult("MomentumFilter", True, 1.0, "Momentum favorable")
        ]
        
        summary = FilterSummary(
            symbol="ETHUSDT",
            signal_type="LONG",
            original_score=7.0,
            final_score=6.0,
            total_adjustment=-1.0,
            filters_applied=filter_results,
            trade_decision="APPROVED"
        )
        
        # Log la aplicación
        self.logger.log_filter_application(summary)
        
        # Verificar que se guardó en historial
        self.assertEqual(len(self.logger._filter_history), 1)
        
        # Verificar estadísticas
        today = datetime.now().date().isoformat()
        stats = self.logger._daily_stats[today]
        self.assertEqual(stats['trades_analyzed'], 1)
        self.assertEqual(stats['trades_approved'], 1)
        self.assertEqual(stats['filters_applied']['VolumeFilter'], 1)
        self.assertEqual(stats['filters_applied']['MomentumFilter'], 1)
    
    def test_log_filter_error(self):
        """Test logging de errores"""
        error = ValueError("Test error")
        context = {"test_data": "test_value"}
        
        self.logger.log_filter_error("TestFilter", error, "BTCUSDT", context)
        
        # Verificar que se registró el error
        today = datetime.now().date().isoformat()
        if 'errors' in self.logger._daily_stats[today]:
            self.assertEqual(self.logger._daily_stats[today]['errors']['TestFilter'], 1)
    
    def test_get_recent_history(self):
        """Test obtención de historial reciente"""
        # Añadir varios registros
        for i in range(5):
            filter_results = [FilterResult("TestFilter", True, -1.0, f"Test {i}")]
            summary = FilterSummary(
                symbol=f"TEST{i}USDT",
                signal_type="LONG",
                original_score=8.0,
                final_score=7.0,
                total_adjustment=-1.0,
                filters_applied=filter_results,
                trade_decision="APPROVED"
            )
            self.logger.log_filter_application(summary)
        
        # Obtener historial
        recent = self.logger.get_recent_history(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1].symbol, "TEST4USDT")  # Más reciente
    
    def test_daily_stats(self):
        """Test estadísticas diarias"""
        today = datetime.now().date().isoformat()
        
        # Inicialmente vacío
        stats = self.logger.get_daily_stats()
        self.assertEqual(len(stats), 0)
        
        # Añadir algunos registros
        for decision in ["APPROVED", "REJECTED", "APPROVED"]:
            filter_results = [FilterResult("TestFilter", True, -1.0, "Test")]
            summary = FilterSummary(
                symbol="TESTUSDT",
                signal_type="LONG",
                original_score=8.0,
                final_score=7.0,
                total_adjustment=-1.0,
                filters_applied=filter_results,
                trade_decision=decision
            )
            self.logger.log_filter_application(summary)
        
        # Verificar estadísticas
        stats = self.logger.get_daily_stats()
        self.assertEqual(stats['trades_analyzed'], 3)
        self.assertEqual(stats['trades_approved'], 2)
        self.assertEqual(stats['trades_rejected'], 1)


class TestFilterMetrics(unittest.TestCase):
    """Tests para sistema de métricas"""
    
    def setUp(self):
        """Setup para cada test"""
        self.calculator = FilterMetricsCalculator()
    
    def test_metrics_calculation(self):
        """Test cálculo de métricas básicas"""
        # Añadir algunos trades de prueba
        for i in range(10):
            decision = "APPROVED" if i % 2 == 0 else "REJECTED"
            filter_results = [
                FilterResult("VolumeFilter", True, -1.0, "Test"),
                FilterResult("MomentumFilter", i % 3 == 0, -0.5 if i % 3 == 0 else 0.0, "Test")
            ]
            
            summary = FilterSummary(
                symbol=f"TEST{i}USDT",
                signal_type="LONG",
                original_score=8.0,
                final_score=7.0 if decision == "APPROVED" else 5.0,
                total_adjustment=-1.0 if decision == "APPROVED" else -3.0,
                filters_applied=filter_results,
                trade_decision=decision
            )
            self.calculator.add_trade_result(summary)
        
        # Obtener métricas
        metrics = self.calculator.get_current_metrics()
        
        self.assertEqual(metrics.total_trades, 10)
        self.assertEqual(metrics.trades_approved, 5)
        self.assertEqual(metrics.trades_rejected, 5)
        self.assertEqual(metrics.approval_rate, 50.0)
        self.assertEqual(metrics.rejection_rate, 50.0)
        self.assertIn('VolumeFilter', metrics.filter_usage)
        self.assertEqual(metrics.filter_usage['VolumeFilter'], 10)  # Aplicado en todos
    
    def test_filter_performance(self):
        """Test métricas de performance por filtro"""
        # Añadir trades con diferentes filtros
        for i in range(5):
            filter_results = [
                FilterResult("VolumeFilter", True, -2.0, "Volumen bajo"),
                FilterResult("MomentumFilter", i < 3, -1.0 if i < 3 else 0.0, "Test")
            ]
            
            # Calcular score final correcto
            total_adj = -3.0 if i < 3 else -2.0
            final_score = 8.0 + total_adj
            
            summary = FilterSummary(
                symbol=f"TEST{i}USDT",
                signal_type="LONG",
                original_score=8.0,
                final_score=final_score,
                total_adjustment=total_adj,
                filters_applied=filter_results,
                trade_decision="APPROVED"
            )
            self.calculator.add_trade_result(summary)
        
        # Obtener performance del VolumeFilter
        volume_perf = self.calculator.get_filter_performance("VolumeFilter", days=1)
        
        self.assertEqual(volume_perf['filter_name'], "VolumeFilter")
        self.assertEqual(volume_perf['trades_analyzed'], 5)
        self.assertEqual(volume_perf['times_applied'], 5)
        self.assertEqual(volume_perf['application_rate'], 100.0)
        self.assertEqual(volume_perf['avg_adjustment'], -2.0)
        
        # Obtener performance del MomentumFilter
        momentum_perf = self.calculator.get_filter_performance("MomentumFilter", days=1)
        
        self.assertEqual(momentum_perf['times_applied'], 3)  # Solo aplicado en 3 de 5
        self.assertEqual(momentum_perf['application_rate'], 60.0)
    
    def test_comparison_metrics(self):
        """Test métricas comparativas"""
        # Añadir trades con diferentes scores
        approved_scores = [8.0, 7.5, 7.0]
        rejected_scores = [5.0, 4.5]
        
        for score in approved_scores:
            filter_results = [FilterResult("TestFilter", True, -1.0, "Test")]
            summary = FilterSummary(
                symbol="TESTUSDT",
                signal_type="LONG",
                original_score=score + 1.0,
                final_score=score,
                total_adjustment=-1.0,
                filters_applied=filter_results,
                trade_decision="APPROVED"
            )
            self.calculator.add_trade_result(summary)
        
        for score in rejected_scores:
            filter_results = [FilterResult("TestFilter", True, -2.0, "Test")]
            summary = FilterSummary(
                symbol="TESTUSDT",
                signal_type="LONG",
                original_score=score + 2.0,
                final_score=score,
                total_adjustment=-2.0,
                filters_applied=filter_results,
                trade_decision="REJECTED"
            )
            self.calculator.add_trade_result(summary)
        
        # Obtener métricas comparativas
        comparison = self.calculator.get_comparison_metrics(days=1)
        
        self.assertEqual(comparison['total_trades'], 5)
        self.assertEqual(comparison['approved_trades'], 3)
        self.assertEqual(comparison['rejected_trades'], 2)
        self.assertEqual(comparison['approval_rate'], 60.0)
        
        # Verificar scores promedio
        expected_avg_approved = sum(approved_scores) / len(approved_scores)
        expected_avg_rejected = sum(rejected_scores) / len(rejected_scores)
        
        self.assertAlmostEqual(comparison['avg_approved_score'], expected_avg_approved, places=2)
        self.assertAlmostEqual(comparison['avg_rejected_score'], expected_avg_rejected, places=2)


if __name__ == '__main__':
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reducir ruido en tests
    
    # Ejecutar tests
    unittest.main(verbosity=2)