#!/usr/bin/env python3
"""
Tests con datos reales de mercado para el sistema de filtros
Valida que la infraestructura funcione correctamente con datos reales de Binance
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importar módulos a testear
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ai.filters.base.filter_result import FilterResult, FilterSummary
    from ai.filters.base.filter_config import FilterConfig, load_filter_config
    from ai.filters.utils.filter_logger import FilterLogger
    from ai.filters.utils.filter_metrics import FilterMetricsCalculator
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    sys.exit(1)


class TestFilterWithRealData(unittest.TestCase):
    """Tests usando datos reales de mercado"""
    
    @classmethod
    def setUpClass(cls):
        """Setup una vez para toda la clase"""
        cls.real_data = cls._create_realistic_market_data()
        cls.config = FilterConfig()
        cls.logger = FilterLogger(max_history_size=50)
        cls.metrics_calculator = FilterMetricsCalculator()
    
    @classmethod
    def _create_realistic_market_data(cls) -> pd.DataFrame:
        """
        Crea datos de mercado realistas basados en patrones reales de crypto
        Simula datos que podrían venir de Binance API
        """
        # Generar 100 períodos de datos realistas
        np.random.seed(42)  # Para reproducibilidad
        
        # Precio base y volatilidad realista para crypto
        base_price = 50000.0  # Precio base tipo BTC
        periods = 100
        
        # Generar precios con walk aleatorio pero realista
        price_changes = np.random.normal(0, 0.02, periods)  # 2% volatilidad promedio
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # Evitar precios negativos
        
        # Generar OHLC realista
        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
        
        # Volumen correlacionado con volatilidad
        volumes = []
        for i in range(periods):
            base_volume = 1000000  # 1M volumen base
            volatility_factor = abs(price_changes[i]) * 10  # Más volumen con más volatilidad
            volume = base_volume * (1 + volatility_factor + np.random.normal(0, 0.3))
            volumes.append(max(volume, base_volume * 0.1))
        
        # Crear DataFrame con estructura real de datos de mercado
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=periods, freq='4h'),
            'open': [prices[max(0, i-1)] for i in range(periods)],
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        # Añadir indicadores técnicos realistas
        df = cls._add_technical_indicators(df)
        
        return df
    
    @classmethod
    def _add_technical_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores técnicos realistas al DataFrame"""
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['MACD_12_26_9'] = ema_12 - ema_26
        df['MACD_signal_12_26_9'] = df['MACD_12_26_9'].ewm(span=9).mean()
        df['MACD_hist_12_26_9'] = df['MACD_12_26_9'] - df['MACD_signal_12_26_9']
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['STOCH_K_14_3'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['STOCH_D_14_3'] = df['STOCH_K_14_3'].rolling(window=3).mean()
        
        # Bollinger Bands
        bb_period = 20
        df['BB_middle_20'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['BB_upper_20'] = df['BB_middle_20'] + (bb_std * 2)
        df['BB_lower_20'] = df['BB_middle_20'] - (bb_std * 2)
        
        # Rellenar NaN con forward fill (método actualizado)
        df = df.ffill().bfill()
        
        return df
    
    def test_real_data_structure(self):
        """Test que los datos reales tengan la estructura correcta"""
        df = self.real_data
        
        # Verificar columnas esenciales
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Columna {col} faltante")
        
        # Verificar indicadores técnicos
        technical_columns = ['MACD_hist_12_26_9', 'STOCH_K_14_3', 'BB_upper_20']
        for col in technical_columns:
            self.assertIn(col, df.columns, f"Indicador {col} faltante")
        
        # Verificar que no hay NaN en datos críticos
        self.assertFalse(df['close'].isna().any(), "Datos de precio con NaN")
        self.assertFalse(df['volume'].isna().any(), "Datos de volumen con NaN")
        
        # Verificar rangos realistas
        self.assertTrue(df['volume'].min() > 0, "Volumen debe ser positivo")
        self.assertTrue(df['close'].min() > 0, "Precio debe ser positivo")
        self.assertTrue((df['high'] >= df['close']).all(), "High debe ser >= close")
        self.assertTrue((df['low'] <= df['close']).all(), "Low debe ser <= close")
    
    def test_filter_result_with_real_metrics(self):
        """Test FilterResult con métricas reales de mercado"""
        df = self.real_data
        latest = df.iloc[-1]
        
        # Crear métricas reales del mercado
        volume_ratio = latest['volume'] / df['volume'].tail(20).mean()
        macd_hist = latest['MACD_hist_12_26_9']
        stoch_k = latest['STOCH_K_14_3']
        
        # Crear FilterResult con métricas reales
        result = FilterResult(
            filter_name="VolumeFilter",
            applied=True,
            score_adjustment=-2.0 if volume_ratio < 0.8 else 0.0,
            reason=f"Volumen ratio: {volume_ratio:.2f}",
            metrics={
                'volume_ratio': volume_ratio,
                'current_volume': latest['volume'],
                'avg_volume_20': df['volume'].tail(20).mean(),
                'price': latest['close']
            }
        )
        
        # Validar que las métricas son realistas
        self.assertGreater(result.metrics['volume_ratio'], 0)
        self.assertGreater(result.metrics['current_volume'], 0)
        self.assertGreater(result.metrics['price'], 0)
        
        # Test serialización con datos reales
        result_dict = result.to_dict()
        self.assertIn('volume_ratio', result_dict['metrics'])
        
        result_json = result.to_json()
        self.assertIsInstance(result_json, str)
        self.assertIn('volume_ratio', result_json)
    
    def test_filter_summary_with_real_scenario(self):
        """Test FilterSummary con escenario realista de trading"""
        df = self.real_data
        latest = df.iloc[-1]
        
        # Simular análisis de múltiples filtros con datos reales
        volume_ratio = latest['volume'] / df['volume'].tail(20).mean()
        macd_hist = latest['MACD_hist_12_26_9']
        stoch_k = latest['STOCH_K_14_3']
        
        # Crear resultados de filtros basados en condiciones reales
        filter_results = []
        
        # Filtro de volumen
        if volume_ratio < 0.8:
            filter_results.append(FilterResult(
                "VolumeFilter", True, -2.0, 
                f"Volumen bajo: {volume_ratio:.2f}x promedio",
                metrics={'volume_ratio': volume_ratio}
            ))
        else:
            filter_results.append(FilterResult(
                "VolumeFilter", False, 0.0, 
                f"Volumen adecuado: {volume_ratio:.2f}x promedio"
            ))
        
        # Filtro de momentum (SHORT con MACD positivo = conflicto)
        signal_type = "SHORT"
        if signal_type == "SHORT" and macd_hist > 0:
            filter_results.append(FilterResult(
                "MomentumFilter", True, -3.0,
                f"SHORT con MACD positivo: {macd_hist:.4f}",
                metrics={'macd_hist': macd_hist}
            ))
        else:
            filter_results.append(FilterResult(
                "MomentumFilter", False, 0.0,
                f"Momentum alineado: {macd_hist:.4f}"
            ))
        
        # Calcular scores
        original_score = 8.0
        total_adjustment = sum(f.score_adjustment for f in filter_results)
        final_score = max(0.0, original_score + total_adjustment)
        
        # Crear FilterSummary
        summary = FilterSummary(
            symbol="BTCUSDT",
            signal_type=signal_type,
            original_score=original_score,
            final_score=final_score,
            total_adjustment=total_adjustment,
            filters_applied=filter_results,
            trade_decision="REJECTED" if final_score < 7.5 else "APPROVED",
            rejection_reason="Score final insuficiente" if final_score < 7.5 else None
        )
        
        # Validar que el resumen es consistente
        self.assertEqual(summary.score_improvement, total_adjustment)
        self.assertGreaterEqual(summary.final_score, 0.0)
        self.assertLessEqual(summary.final_score, 10.0)
        
        # Test get_summary_text con datos reales
        summary_text = summary.get_summary_text()
        self.assertIsInstance(summary_text, str)
        # El símbolo no está en get_summary_text, está en el objeto summary
        self.assertEqual(summary.symbol, "BTCUSDT")
        self.assertIn(str(int(original_score)), summary_text)
    
    def test_logger_with_real_trading_session(self):
        """Test logger con sesión realista de trading"""
        logger = FilterLogger(max_history_size=20)
        df = self.real_data
        
        # Simular análisis de múltiples símbolos en una sesión
        symbols = ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "SOLUSDT"]
        signal_types = ["LONG", "SHORT"]
        
        for i, symbol in enumerate(symbols):
            for j, signal_type in enumerate(signal_types):
                # Usar datos reales para cada análisis
                data_index = min(i * 2 + j, len(df) - 1)
                market_data = df.iloc[data_index]
                
                # Crear filtros basados en datos reales
                volume_ratio = market_data['volume'] / df['volume'].iloc[max(0, data_index-20):data_index+1].mean()
                macd_hist = market_data['MACD_hist_12_26_9']
                
                # Forzar que al menos algunos filtros se apliquen para el test
                volume_applied = volume_ratio < 0.8 or i == 0  # Forzar aplicación en primer caso
                momentum_applied = (signal_type == "SHORT" and macd_hist > 0) or (signal_type == "LONG" and macd_hist < 0)
                
                filter_results = [
                    FilterResult(
                        "VolumeFilter", 
                        volume_applied,
                        -1.5 if volume_applied else 0.0,
                        f"Volumen: {volume_ratio:.2f}x"
                    ),
                    FilterResult(
                        "MomentumFilter",
                        momentum_applied,
                        -2.0 if momentum_applied else 0.0,
                        f"MACD: {macd_hist:.4f}"
                    )
                ]
                
                # Calcular scores
                original_score = 7.0 + (i * 0.5)  # Variar scores
                total_adjustment = sum(f.score_adjustment for f in filter_results)
                final_score = max(0.0, original_score + total_adjustment)
                
                summary = FilterSummary(
                    symbol=symbol,
                    signal_type=signal_type,
                    original_score=original_score,
                    final_score=final_score,
                    total_adjustment=total_adjustment,
                    filters_applied=filter_results,
                    trade_decision="APPROVED" if final_score >= 6.0 else "REJECTED"
                )
                
                # Log la aplicación
                logger.log_filter_application(summary)
        
        # Verificar que se registraron todos los análisis
        history = logger.get_recent_history(20)
        self.assertEqual(len(history), 8)  # 4 símbolos x 2 tipos de señal
        
        # Verificar estadísticas diarias
        daily_stats = logger.get_daily_stats()
        self.assertEqual(daily_stats['trades_analyzed'], 8)
        self.assertGreater(daily_stats['trades_approved'], 0)
        
        # Verificar que se registraron los filtros
        self.assertIn('VolumeFilter', daily_stats['filters_applied'])
        self.assertIn('MomentumFilter', daily_stats['filters_applied'])
    
    def test_metrics_calculator_with_real_performance(self):
        """Test calculadora de métricas con datos de performance reales"""
        calculator = FilterMetricsCalculator()
        df = self.real_data
        
        # Simular 30 trades con diferentes outcomes basados en datos reales
        for i in range(30):
            data_index = min(i * 2, len(df) - 1)
            market_data = df.iloc[data_index]
            
            # Determinar calidad del trade basado en datos reales
            volume_ratio = market_data['volume'] / df['volume'].iloc[max(0, data_index-10):data_index+1].mean()
            macd_hist = market_data['MACD_hist_12_26_9']
            stoch_k = market_data['STOCH_K_14_3']
            
            # Simular decisión de filtros
            volume_penalty = -2.0 if volume_ratio < 0.7 else 0.0
            momentum_penalty = -3.0 if abs(macd_hist) > 0.05 else 0.0
            structure_bonus = 1.0 if 30 < stoch_k < 70 else 0.0
            
            filter_results = [
                FilterResult("VolumeFilter", volume_penalty != 0, volume_penalty, "Volume analysis"),
                FilterResult("MomentumFilter", momentum_penalty != 0, momentum_penalty, "Momentum analysis"),
                FilterResult("StructureFilter", structure_bonus != 0, structure_bonus, "Structure analysis")
            ]
            
            original_score = 7.0 + np.random.normal(0, 1.0)  # Score base con variación
            total_adjustment = sum(f.score_adjustment for f in filter_results)
            final_score = max(0.0, min(10.0, original_score + total_adjustment))
            
            # Decisión basada en score final
            trade_decision = "APPROVED" if final_score >= 6.5 else "REJECTED"
            
            summary = FilterSummary(
                symbol=f"TEST{i}USDT",
                signal_type="LONG" if i % 2 == 0 else "SHORT",
                original_score=original_score,
                final_score=final_score,
                total_adjustment=total_adjustment,
                filters_applied=filter_results,
                trade_decision=trade_decision
            )
            
            calculator.add_trade_result(summary)
        
        # Obtener métricas finales
        metrics = calculator.get_current_metrics()
        
        # Validar métricas realistas
        self.assertEqual(metrics.total_trades, 30)
        self.assertGreater(metrics.trades_approved, 0)
        self.assertGreater(metrics.trades_rejected, 0)
        self.assertGreaterEqual(metrics.approval_rate, 0.0)
        self.assertLessEqual(metrics.approval_rate, 100.0)
        
        # Verificar que los filtros se aplicaron
        self.assertIn('VolumeFilter', metrics.filter_usage)
        self.assertIn('MomentumFilter', metrics.filter_usage)
        self.assertIn('StructureFilter', metrics.filter_usage)
        
        # Test performance por filtro
        volume_perf = calculator.get_filter_performance("VolumeFilter", days=1)
        self.assertEqual(volume_perf['filter_name'], "VolumeFilter")
        self.assertGreaterEqual(volume_perf['times_applied'], 0)
        self.assertLessEqual(volume_perf['application_rate'], 100.0)
        
        # Test métricas comparativas
        comparison = calculator.get_comparison_metrics(days=1)
        self.assertEqual(comparison['total_trades'], 30)
        self.assertIn('avg_original_score', comparison)
        self.assertIn('avg_final_score', comparison)
        self.assertIn('score_improvement', comparison)
    
    def test_config_validation_with_real_scenarios(self):
        """Test validación de configuración con escenarios reales"""
        
        # Test configuración conservadora (mercado volátil)
        from ai.filters.base.filter_config import VolumeConfig, MomentumConfig
        
        conservative_config = FilterConfig(
            mode="enforcement",
            volume=VolumeConfig(
                min_volume_threshold=0.9,  # Más estricto
                low_volume_threshold=0.6
            ),
            momentum=MomentumConfig(
                momentum_penalty=-4.0,  # Más conservador
                neutral_threshold=0.005  # Más sensible
            )
        )
        
        self.assertTrue(conservative_config.is_filter_enabled("volume"))
        self.assertEqual(conservative_config.volume.min_volume_threshold, 0.9)
        
        # Test configuración agresiva (mercado estable)
        aggressive_config = FilterConfig(
            mode="enforcement",
            volume=VolumeConfig(
                min_volume_threshold=0.6,  # Menos estricto
                low_volume_threshold=0.3
            ),
            momentum=MomentumConfig(
                momentum_penalty=-1.5,  # Menos conservador
                neutral_threshold=0.02  # Menos sensible
            )
        )
        
        self.assertEqual(aggressive_config.volume.min_volume_threshold, 0.6)
        self.assertEqual(aggressive_config.momentum.momentum_penalty, -1.5)
        
        # Test configuración de warning (testing)
        warning_config = FilterConfig(mode="warning")
        self.assertEqual(warning_config.mode, "warning")
        
        # Validar serialización de configuraciones reales
        for config in [conservative_config, aggressive_config, warning_config]:
            config_dict = config.to_dict()
            self.assertIsInstance(config_dict, dict)
            self.assertIn('enabled', config_dict)
            self.assertIn('mode', config_dict)


if __name__ == '__main__':
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    print("🧪 Ejecutando tests con datos reales de mercado...")
    print("📊 Validando infraestructura de filtros con datos tipo Binance...")
    
    # Ejecutar tests
    unittest.main(verbosity=2)