#!/usr/bin/env python3
"""
Performance Monitor - SPARTAN EDITION
Ultra-lightweight performance monitoring for trading bot optimization
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque


@dataclass
class PerformanceMetric:
    """Single performance metric"""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    SPARTAN PERFORMANCE MONITOR
    - Lightweight metrics collection
    - Real-time performance analysis
    - Automatic optimization suggestions
    - Zero external dependencies
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize Performance Monitor
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.logger = logging.getLogger("PerformanceMonitor")
        
        # Metrics storage
        self.metrics: Dict[str, deque] = {}
        
        # Performance counters
        self.counters: Dict[str, int] = {
            'api_calls_total': 0,
            'api_calls_cached': 0,
            'symbols_processed': 0,
            'signals_generated': 0,
            'ai_validations': 0,
            'errors': 0
        }
        
        # Timing data
        self.timings: Dict[str, List[float]] = {
            'symbol_processing': [],
            'api_calls': [],
            'indicator_calculations': [],
            'ai_validations': [],
            'total_cycle': []
        }
        
        self.logger.info("🏛️ Spartan Performance Monitor initialized")
    
    def record_timing(self, metric_name: str, duration: float, metadata: Dict[str, Any] = None) -> None:
        """Record timing metric"""
        if metric_name not in self.timings:
            self.timings[metric_name] = []
        
        self.timings[metric_name].append(duration)
        
        # Keep only recent timings
        if len(self.timings[metric_name]) > self.max_history:
            self.timings[metric_name] = self.timings[metric_name][-self.max_history:]
        
        # Store detailed metric
        metric = PerformanceMetric(
            name=metric_name,
            value=duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.max_history)
        
        self.metrics[metric_name].append(metric)
    
    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """Increment performance counter"""
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        
        self.counters[counter_name] += amount
    
    def get_average_timing(self, metric_name: str, last_n: int = None) -> float:
        """Get average timing for a metric"""
        if metric_name not in self.timings or not self.timings[metric_name]:
            return 0.0
        
        timings = self.timings[metric_name]
        if last_n:
            timings = timings[-last_n:]
        
        return sum(timings) / len(timings)
    
    def get_percentile_timing(self, metric_name: str, percentile: float = 95) -> float:
        """Get percentile timing for a metric"""
        if metric_name not in self.timings or not self.timings[metric_name]:
            return 0.0
        
        timings = sorted(self.timings[metric_name])
        index = int(len(timings) * (percentile / 100))
        return timings[min(index, len(timings) - 1)]
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        total_calls = self.counters.get('api_calls_total', 0)
        cached_calls = self.counters.get('api_calls_cached', 0)
        
        if total_calls == 0:
            return 0.0
        
        return (cached_calls / total_calls) * 100
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'counters': self.counters.copy(),
            'cache_hit_rate': round(self.get_cache_hit_rate(), 2),
            'average_timings': {},
            'p95_timings': {},
            'performance_score': 0,
            'recommendations': []
        }
        
        # Calculate timing statistics
        for metric_name in self.timings:
            if self.timings[metric_name]:
                summary['average_timings'][metric_name] = round(
                    self.get_average_timing(metric_name), 3
                )
                summary['p95_timings'][metric_name] = round(
                    self.get_percentile_timing(metric_name, 95), 3
                )
        
        # Calculate performance score (0-100)
        score = self._calculate_performance_score()
        summary['performance_score'] = score
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        summary['recommendations'] = recommendations
        
        return summary
    
    def _calculate_performance_score(self) -> int:
        """Calculate overall performance score (0-100)"""
        score = 100
        
        # Deduct points for slow performance
        avg_symbol_time = self.get_average_timing('symbol_processing')
        if avg_symbol_time > 2.0:  # Target: < 2 seconds per symbol
            score -= min(30, (avg_symbol_time - 2.0) * 10)
        
        # Deduct points for low cache hit rate
        cache_hit_rate = self.get_cache_hit_rate()
        if cache_hit_rate < 80:  # Target: > 80% hit rate
            score -= min(20, (80 - cache_hit_rate) / 2)
        
        # Deduct points for AI validation slowness
        avg_ai_time = self.get_average_timing('ai_validations')
        if avg_ai_time > 10.0:  # Target: < 10 seconds for AI
            score -= min(25, (avg_ai_time - 10.0) * 2)
        
        # Deduct points for errors
        error_rate = self.counters.get('errors', 0) / max(1, self.counters.get('symbols_processed', 1))
        if error_rate > 0.05:  # Target: < 5% error rate
            score -= min(15, error_rate * 100)
        
        return max(0, int(score))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Symbol processing time
        avg_symbol_time = self.get_average_timing('symbol_processing')
        if avg_symbol_time > 3.0:
            recommendations.append(f"⚠️ Symbol processing is slow ({avg_symbol_time:.1f}s avg). Consider reducing CANDLES_LIMIT or optimizing indicators.")
        elif avg_symbol_time > 2.0:
            recommendations.append(f"💡 Symbol processing could be faster ({avg_symbol_time:.1f}s avg). Current performance is acceptable.")
        
        # Cache performance
        cache_hit_rate = self.get_cache_hit_rate()
        if cache_hit_rate < 60:
            recommendations.append(f"🚨 Cache hit rate is low ({cache_hit_rate:.1f}%). Check cache TTL settings and data freshness requirements.")
        elif cache_hit_rate < 80:
            recommendations.append(f"⚠️ Cache hit rate could be better ({cache_hit_rate:.1f}%). Consider increasing cache TTL for your timeframe.")
        else:
            recommendations.append(f"✅ Excellent cache performance ({cache_hit_rate:.1f}% hit rate)!")
        
        # AI validation performance
        avg_ai_time = self.get_average_timing('ai_validations')
        if avg_ai_time > 15.0:
            recommendations.append(f"🚨 AI validation is very slow ({avg_ai_time:.1f}s avg). Consider reducing AI_TIMEOUT_SECONDS or optimizing prompts.")
        elif avg_ai_time > 10.0:
            recommendations.append(f"⚠️ AI validation is slow ({avg_ai_time:.1f}s avg). Monitor for timeout issues.")
        elif avg_ai_time > 0:
            recommendations.append(f"✅ AI validation performance is good ({avg_ai_time:.1f}s avg).")
        
        # API call efficiency
        total_calls = self.counters.get('api_calls_total', 0)
        symbols_processed = self.counters.get('symbols_processed', 0)
        if symbols_processed > 0:
            calls_per_symbol = total_calls / symbols_processed
            if calls_per_symbol > 1.5:
                recommendations.append(f"⚠️ High API calls per symbol ({calls_per_symbol:.1f}). Cache may not be working optimally.")
            elif calls_per_symbol <= 1.0:
                recommendations.append(f"✅ Excellent API efficiency ({calls_per_symbol:.1f} calls per symbol)!")
        
        # Error rate
        error_rate = self.counters.get('errors', 0) / max(1, symbols_processed)
        if error_rate > 0.1:
            recommendations.append(f"🚨 High error rate ({error_rate:.1%}). Check logs for recurring issues.")
        elif error_rate > 0.05:
            recommendations.append(f"⚠️ Moderate error rate ({error_rate:.1%}). Monitor for stability issues.")
        
        # Overall recommendations
        if not recommendations:
            recommendations.append("🏛️ Performance is excellent! No optimizations needed.")
        
        return recommendations
    
    def log_performance_summary(self) -> None:
        """Log performance summary to console"""
        summary = self.get_performance_summary()
        
        self.logger.info("📊 PERFORMANCE SUMMARY")
        self.logger.info(f"   Score: {summary['performance_score']}/100")
        self.logger.info(f"   Cache Hit Rate: {summary['cache_hit_rate']:.1f}%")
        
        if 'symbol_processing' in summary['average_timings']:
            self.logger.info(f"   Avg Symbol Time: {summary['average_timings']['symbol_processing']:.2f}s")
        
        if 'ai_validations' in summary['average_timings']:
            self.logger.info(f"   Avg AI Time: {summary['average_timings']['ai_validations']:.2f}s")
        
        self.logger.info(f"   API Calls: {summary['counters']['api_calls_total']} ({summary['counters']['api_calls_cached']} cached)")
        
        # Log top recommendation
        if summary['recommendations']:
            self.logger.info(f"   💡 {summary['recommendations'][0]}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics and counters"""
        self.metrics.clear()
        self.counters = {key: 0 for key in self.counters}
        self.timings = {key: [] for key in self.timings}
        
        self.logger.info("🔄 Performance metrics reset")


# Context manager for timing operations
class TimingContext:
    """Context manager for automatic timing measurement"""
    
    def __init__(self, monitor: PerformanceMonitor, metric_name: str, metadata: Dict[str, Any] = None):
        self.monitor = monitor
        self.metric_name = metric_name
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_timing(self.metric_name, duration, self.metadata)


# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor