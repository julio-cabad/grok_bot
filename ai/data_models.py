#!/usr/bin/env python3
"""
Data Models - Spartan Trading System
Enhanced data structures for maximum profitability and precision
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum


class SignalStrength(Enum):
    """Signal strength classification for institutional precision"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY" 
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketZone(Enum):
    """Premium/Discount zone classification"""
    PREMIUM = "PREMIUM"      # >70.5% - Sell zone
    EQUILIBRIUM = "EQUILIBRIUM"  # 29.5% - 70.5% - Wait zone
    DISCOUNT = "DISCOUNT"    # <29.5% - Buy zone


# ValidationResult is defined in ai_validator.py to avoid duplication
# SMCAnalysis is handled directly by FastSMCAnalyzer as dictionaries


@dataclass
class RiskMetrics:
    """Comprehensive risk management calculations"""
    # Position Sizing
    optimal_position_size: float
    max_position_size: float
    min_position_size: float
    
    # Risk Calculations
    risk_amount_usd: float
    risk_percentage: float
    
    # Take Profit Analysis
    tp_levels: List[float] = field(default_factory=list)
    tp_risk_rewards: List[float] = field(default_factory=list)
    tp_profit_amounts: List[float] = field(default_factory=list)
    
    # Volatility Adjustments
    volatility_level: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    volatility_multiplier: float = 1.0
    adjusted_position_size: float = 0.0
    
    # Portfolio Impact
    portfolio_risk_percentage: float = 0.0
    correlation_adjustment: float = 1.0
    
    def get_total_profit_potential(self) -> float:
        """Calculate total profit potential across all TPs"""
        return sum(self.tp_profit_amounts)
    
    def get_weighted_risk_reward(self) -> float:
        """Calculate weighted average risk/reward ratio"""
        if not self.tp_risk_rewards:
            return 0.0
        
        # Weight by position allocation (33%, 33%, 34% typically)
        weights = [0.33, 0.33, 0.34][:len(self.tp_risk_rewards)]
        weighted_sum = sum(rr * w for rr, w in zip(self.tp_risk_rewards, weights))
        return weighted_sum


# MarketContext is handled directly by FastSMCAnalyzer as dictionaries


@dataclass
class AIValidatorConfig:
    """Configuration for the AI Validator system - Optimized for profits"""
    # AI Analysis Settings
    confidence_threshold: float = 7.5  # Minimum score to enter trades
    timeout_seconds: int = 30  # Max time for AI analysis
    max_retries: int = 3  # Retry attempts for failed analyses
    use_threading: bool = True  # Enable threading for timeout protection
    
    # Risk Management Settings
    default_risk_percentage: float = 1.0  # Default risk per trade
    max_risk_percentage: float = 2.0  # Maximum risk per trade
    min_risk_reward_ratio: float = 1.5  # Minimum R:R to enter
    position_size_limits: Tuple[float, float] = (0.1, 10.0)  # Min/Max position multipliers
    
    # Volatility Adjustments
    high_volatility_reduction: float = 0.5  # Reduce position by 50% in high vol
    low_volatility_increase: float = 1.2  # Increase position by 20% in low vol
    volatility_lookback_periods: int = 20  # Periods for volatility calculation
    
    # SMC Analysis Settings
    order_block_lookback: int = 20  # Candles to look back for order blocks
    fvg_min_gap_percentage: float = 0.1  # Minimum gap size for FVG detection
    premium_threshold: float = 0.705  # 70.5% for premium zone
    discount_threshold: float = 0.295  # 29.5% for discount zone
    liquidity_detection_periods: int = 50  # Periods for liquidity analysis
    
    # Cache Settings
    cache_ttl_minutes: int = 5  # Cache time-to-live
    cache_max_size: int = 1000  # Maximum cache entries
    cache_cleanup_frequency: int = 100  # Cleanup every N queries
    
    # Performance Settings
    max_concurrent_analyses: int = 5  # Max parallel AI analyses
    memory_limit_mb: int = 500  # Memory limit for cache
    performance_monitoring: bool = True  # Enable performance tracking
    
    # Notification Settings
    notify_high_confidence: bool = True  # Notify for scores >8.5
    notify_warnings: bool = True  # Include warnings in notifications
    detailed_analysis_in_notifications: bool = True  # Include full analysis
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 10:
            issues.append("Confidence threshold must be between 0 and 10")
        
        if self.default_risk_percentage <= 0 or self.default_risk_percentage > 5:
            issues.append("Default risk percentage must be between 0 and 5")
        
        if self.min_risk_reward_ratio < 1.0:
            issues.append("Minimum risk/reward ratio must be >= 1.0")
        
        if self.timeout_seconds <= 0:
            issues.append("Timeout seconds must be positive")
        
        return issues
    
    def get_volatility_multiplier(self, volatility_level: str) -> float:
        """Get position size multiplier based on volatility"""
        multipliers = {
            "HIGH": self.high_volatility_reduction,
            "MEDIUM": 1.0,
            "LOW": self.low_volatility_increase
        }
        return multipliers.get(volatility_level, 1.0)


# Global configuration instance
DEFAULT_CONFIG = AIValidatorConfig()