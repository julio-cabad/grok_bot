#!/usr/bin/env python3
"""
Risk Manager - Spartan Trading System
Intelligent risk management for maximum profitability and capital preservation
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

from ..data_models import RiskMetrics, AIValidatorConfig, DEFAULT_CONFIG


class RiskManager:
    """
    Elite Risk Manager for institutional-grade position sizing and profit optimization
    
    Features:
    - Dynamic position sizing based on account balance and risk
    - Multiple take profit levels for maximum profit capture
    - Volatility-based position adjustments
    - Risk/reward validation and optimization
    - Portfolio-level risk management
    """
    
    def __init__(self, config: AIValidatorConfig = None):
        """Initialize the Risk Manager with configuration"""
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("RiskManager")
        
        # Performance tracking
        self.total_calculations = 0
        self.average_calculation_time = 0.0
        
        self.logger.info(
            f"💰 RiskManager initialized - "
            f"Default Risk: {self.config.default_risk_percentage}%, "
            f"Min R:R: {self.config.min_risk_reward_ratio}"
        )
    
    def calculate_position_size(self, 
                              entry_price: float,
                              stop_loss: float,
                              account_balance: float,
                              risk_percentage: float = None) -> float:
        """
        Calculate optimal position size using institutional risk management
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss level
            account_balance: Total account balance
            risk_percentage: Risk percentage (default from config)
            
        Returns:
            Optimal position size in units
        """
        try:
            # Input validation
            if entry_price <= 0 or stop_loss <= 0 or account_balance <= 0:
                raise ValueError("All prices and balance must be positive")
            
            if entry_price == stop_loss:
                raise ValueError("Entry price cannot equal stop loss")
            
            # Use default risk if not provided
            risk_pct = risk_percentage or self.config.default_risk_percentage
            
            # Validate risk percentage
            if risk_pct <= 0 or risk_pct > self.config.max_risk_percentage:
                self.logger.warning(f"Risk percentage {risk_pct}% outside limits, using default")
                risk_pct = self.config.default_risk_percentage
            
            # Calculate risk amount in USD
            risk_amount = account_balance * (risk_pct / 100)
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            # Calculate base position size
            position_size = risk_amount / risk_per_unit
            
            # Apply position size limits
            min_size = account_balance * self.config.position_size_limits[0] / entry_price
            max_size = account_balance * self.config.position_size_limits[1] / entry_price
            
            position_size = max(min_size, min(position_size, max_size))
            
            self.logger.debug(
                f"Position sizing: Entry=${entry_price:.4f}, SL=${stop_loss:.4f}, "
                f"Risk=${risk_amount:.2f} ({risk_pct}%), Size={position_size:.6f}"
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def generate_take_profit_levels(self,
                                  entry_price: float,
                                  stop_loss: float,
                                  signal_type: str,
                                  custom_ratios: List[float] = None) -> List[float]:
        """
        Generate multiple take profit levels for maximum profit capture
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss level
            signal_type: 'LONG' or 'SHORT'
            custom_ratios: Custom R:R ratios (default: [1.0, 2.0, 3.0])
            
        Returns:
            List of take profit levels
        """
        try:
            # Input validation
            if entry_price <= 0 or stop_loss <= 0:
                raise ValueError("Entry and stop loss prices must be positive")
            
            if signal_type not in ['LONG', 'SHORT']:
                raise ValueError("Signal type must be 'LONG' or 'SHORT'")
            
            # Use default ratios if not provided
            ratios = custom_ratios or [1.0, 2.0, 3.0]
            
            # Calculate risk (distance from entry to stop loss)
            risk = abs(entry_price - stop_loss)
            
            take_profits = []
            
            for ratio in ratios:
                if signal_type == 'LONG':
                    tp = entry_price + (risk * ratio)
                else:  # SHORT
                    tp = entry_price - (risk * ratio)
                
                # Ensure TP is positive
                if tp > 0:
                    take_profits.append(tp)
            
            self.logger.debug(
                f"Generated TPs for {signal_type}: Entry=${entry_price:.4f}, "
                f"SL=${stop_loss:.4f}, TPs={[f'${tp:.4f}' for tp in take_profits]}"
            )
            
            return take_profits
            
        except Exception as e:
            self.logger.error(f"Take profit generation failed: {e}")
            return []
    
    def calculate_risk_reward_ratios(self,
                                   entry_price: float,
                                   stop_loss: float,
                                   take_profits: List[float]) -> List[float]:
        """
        Calculate risk/reward ratios for each take profit level
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss level
            take_profits: List of take profit levels
            
        Returns:
            List of risk/reward ratios
        """
        try:
            if not take_profits:
                return []
            
            risk = abs(entry_price - stop_loss)
            if risk == 0:
                return []
            
            ratios = []
            for tp in take_profits:
                reward = abs(tp - entry_price)
                ratio = reward / risk
                ratios.append(ratio)
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Risk/reward calculation failed: {e}")
            return []
    
    def detect_volatility_from_data(self, 
                                   df: pd.DataFrame, 
                                   lookback_periods: int = None) -> Dict[str, Any]:
        """
        Detect volatility level from price data using advanced statistical methods
        
        Args:
            df: DataFrame with OHLCV data
            lookback_periods: Number of periods to analyze (default from config)
            
        Returns:
            Dictionary with volatility analysis
        """
        try:
            lookback = lookback_periods or self.config.volatility_lookback_periods
            
            if len(df) < lookback:
                self.logger.warning(f"Insufficient data for volatility analysis: {len(df)} < {lookback}")
                return {
                    'level': 'MEDIUM',
                    'percentile': 50.0,
                    'std_dev': 0.0,
                    'atr_ratio': 1.0,
                    'expanding': False
                }
            
            # Get recent data
            recent_data = df.tail(lookback)
            
            # Calculate price returns
            returns = recent_data['close'].pct_change().dropna()
            
            # Calculate standard deviation (volatility)
            volatility_std = returns.std()
            
            # Calculate ATR-based volatility
            high_low = recent_data['high'] - recent_data['low']
            high_close = abs(recent_data['high'] - recent_data['close'].shift(1))
            low_close = abs(recent_data['low'] - recent_data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.mean()
            atr_ratio = atr / recent_data['close'].mean()
            
            # Calculate volatility percentile (compare to longer history)
            if len(df) >= lookback * 3:
                historical_data = df.tail(lookback * 3)
                historical_returns = historical_data['close'].pct_change().dropna()
                
                # Rolling volatility calculation
                rolling_vol = historical_returns.rolling(window=lookback).std()
                current_vol_percentile = (rolling_vol < volatility_std).sum() / len(rolling_vol) * 100
            else:
                current_vol_percentile = 50.0  # Default to median
            
            # Determine if volatility is expanding
            if len(returns) >= 10:
                recent_vol = returns.tail(5).std()
                older_vol = returns.iloc[-10:-5].std()
                volatility_expanding = recent_vol > older_vol * 1.2
            else:
                volatility_expanding = False
            
            # Classify volatility level using multiple criteria
            vol_score = 0
            
            # Score based on standard deviation
            if volatility_std > 0.04:
                vol_score += 2  # Very high
            elif volatility_std > 0.025:
                vol_score += 1  # High
            elif volatility_std < 0.008:
                vol_score -= 2  # Very low
            elif volatility_std < 0.015:
                vol_score -= 1  # Low
            
            # Score based on ATR ratio
            if atr_ratio > 0.03:
                vol_score += 2  # Very high
            elif atr_ratio > 0.02:
                vol_score += 1  # High
            elif atr_ratio < 0.008:
                vol_score -= 2  # Very low
            elif atr_ratio < 0.012:
                vol_score -= 1  # Low
            
            # Score based on percentile (if available)
            if current_vol_percentile > 85:
                vol_score += 1
            elif current_vol_percentile < 25:
                vol_score -= 1
            
            # Final classification
            if vol_score >= 2:
                volatility_level = 'HIGH'
            elif vol_score <= -2:
                volatility_level = 'LOW'
            else:
                volatility_level = 'MEDIUM'
            
            volatility_analysis = {
                'level': volatility_level,
                'percentile': current_vol_percentile,
                'std_dev': volatility_std,
                'atr_ratio': atr_ratio,
                'expanding': volatility_expanding,
                'raw_atr': atr,
                'lookback_periods': lookback
            }
            
            self.logger.debug(
                f"Volatility analysis: Level={volatility_level}, "
                f"Percentile={current_vol_percentile:.1f}%, "
                f"StdDev={volatility_std:.4f}, ATR={atr_ratio:.4f}"
            )
            
            return volatility_analysis
            
        except Exception as e:
            self.logger.error(f"Volatility detection failed: {e}")
            return {
                'level': 'MEDIUM',
                'percentile': 50.0,
                'std_dev': 0.0,
                'atr_ratio': 1.0,
                'expanding': False
            }
    
    def adjust_for_volatility(self,
                            position_size: float,
                            volatility_analysis: Dict[str, Any] = None,
                            volatility_level: str = None,
                            volatility_percentile: float = None) -> float:
        """
        Adjust position size based on comprehensive volatility analysis
        
        Args:
            position_size: Base position size
            volatility_analysis: Complete volatility analysis from detect_volatility_from_data()
            volatility_level: Manual volatility level ('HIGH', 'MEDIUM', 'LOW')
            volatility_percentile: Manual volatility percentile (0-100)
            
        Returns:
            Adjusted position size
        """
        try:
            # Use volatility analysis if provided, otherwise use manual inputs
            if volatility_analysis:
                vol_level = volatility_analysis['level']
                vol_percentile = volatility_analysis['percentile']
                expanding = volatility_analysis.get('expanding', False)
                atr_ratio = volatility_analysis.get('atr_ratio', 1.0)
            else:
                vol_level = volatility_level or 'MEDIUM'
                vol_percentile = volatility_percentile or 50.0
                expanding = False
                atr_ratio = 1.0
            
            # Get base multiplier from config
            base_multiplier = self.config.get_volatility_multiplier(vol_level)
            
            # Fine-tune based on percentile
            percentile_adjustment = 1.0
            if vol_percentile > 95:  # Extreme volatility (top 5%)
                percentile_adjustment = 0.6
            elif vol_percentile > 85:  # Very high volatility
                percentile_adjustment = 0.75
            elif vol_percentile > 75:  # High volatility
                percentile_adjustment = 0.9
            elif vol_percentile < 15:  # Very low volatility
                percentile_adjustment = 1.25
            elif vol_percentile < 25:  # Low volatility
                percentile_adjustment = 1.15
            
            # Additional adjustment for expanding volatility
            expansion_adjustment = 1.0
            if expanding:
                expansion_adjustment = 0.85  # Reduce size when volatility is expanding
                self.logger.debug("Volatility expanding - applying additional reduction")
            
            # ATR-based adjustment
            atr_adjustment = 1.0
            if atr_ratio > 0.03:  # Very high ATR
                atr_adjustment = 0.8
            elif atr_ratio > 0.02:  # High ATR
                atr_adjustment = 0.9
            elif atr_ratio < 0.008:  # Very low ATR
                atr_adjustment = 1.2
            elif atr_ratio < 0.012:  # Low ATR
                atr_adjustment = 1.1
            
            # Combine all adjustments
            final_multiplier = base_multiplier * percentile_adjustment * expansion_adjustment * atr_adjustment
            
            # Apply safety limits (don't go below 0.3x or above 2.0x)
            final_multiplier = max(0.3, min(final_multiplier, 2.0))
            
            adjusted_size = position_size * final_multiplier
            
            self.logger.debug(
                f"Advanced volatility adjustment: "
                f"Level={vol_level}, Percentile={vol_percentile:.1f}%, "
                f"Base={base_multiplier:.2f}, Percentile={percentile_adjustment:.2f}, "
                f"Expansion={expansion_adjustment:.2f}, ATR={atr_adjustment:.2f}, "
                f"Final={final_multiplier:.2f}, "
                f"Size: {position_size:.6f} → {adjusted_size:.6f}"
            )
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Volatility adjustment failed: {e}")
            return position_size
    
    def calculate_comprehensive_risk_metrics(self,
                                           entry_price: float,
                                           stop_loss: float,
                                           signal_type: str,
                                           account_balance: float,
                                           risk_percentage: float = None,
                                           df: pd.DataFrame = None,
                                           volatility_level: str = "MEDIUM",
                                           volatility_percentile: float = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a trade
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss level
            signal_type: 'LONG' or 'SHORT'
            account_balance: Account balance
            risk_percentage: Risk percentage
            df: DataFrame with OHLCV data for volatility analysis
            volatility_level: Manual volatility level (used if df not provided)
            volatility_percentile: Manual volatility percentile
            
        Returns:
            Comprehensive RiskMetrics object
        """
        try:
            # Calculate base position size
            base_position_size = self.calculate_position_size(
                entry_price, stop_loss, account_balance, risk_percentage
            )
            
            # Generate take profit levels
            tp_levels = self.generate_take_profit_levels(
                entry_price, stop_loss, signal_type
            )
            
            # Calculate risk/reward ratios
            rr_ratios = self.calculate_risk_reward_ratios(
                entry_price, stop_loss, tp_levels
            )
            
            # Detect volatility from data if available
            volatility_analysis = None
            if df is not None:
                volatility_analysis = self.detect_volatility_from_data(df)
                actual_volatility_level = volatility_analysis['level']
            else:
                actual_volatility_level = volatility_level
            
            # Adjust for volatility
            adjusted_position_size = self.adjust_for_volatility(
                base_position_size, 
                volatility_analysis=volatility_analysis,
                volatility_level=volatility_level,
                volatility_percentile=volatility_percentile
            )
            
            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss) * adjusted_position_size
            actual_risk_pct = (risk_amount / account_balance) * 100
            
            # Calculate profit amounts for each TP
            tp_profit_amounts = []
            for tp in tp_levels:
                profit_per_unit = abs(tp - entry_price)
                # Assume equal allocation across TPs (33%, 33%, 34%)
                allocation = adjusted_position_size / len(tp_levels)
                profit_amount = profit_per_unit * allocation
                tp_profit_amounts.append(profit_amount)
            
            # Calculate position size limits
            min_size = account_balance * self.config.position_size_limits[0] / entry_price
            max_size = account_balance * self.config.position_size_limits[1] / entry_price
            
            # Get volatility multiplier and analysis
            vol_multiplier = self.config.get_volatility_multiplier(actual_volatility_level)
            
            # Store volatility analysis in metrics
            volatility_data = volatility_analysis or {
                'level': actual_volatility_level,
                'percentile': volatility_percentile or 50.0,
                'std_dev': 0.0,
                'atr_ratio': 1.0,
                'expanding': False
            }
            
            return RiskMetrics(
                optimal_position_size=adjusted_position_size,
                max_position_size=max_size,
                min_position_size=min_size,
                risk_amount_usd=risk_amount,
                risk_percentage=actual_risk_pct,
                tp_levels=tp_levels,
                tp_risk_rewards=rr_ratios,
                tp_profit_amounts=tp_profit_amounts,
                volatility_level=actual_volatility_level,
                volatility_multiplier=vol_multiplier,
                adjusted_position_size=adjusted_position_size,
                portfolio_risk_percentage=actual_risk_pct,
                correlation_adjustment=1.0  # Default, can be enhanced later
            )
            
        except Exception as e:
            self.logger.error(f"Comprehensive risk calculation failed: {e}")
            return RiskMetrics(
                optimal_position_size=0.0,
                max_position_size=0.0,
                min_position_size=0.0,
                risk_amount_usd=0.0,
                risk_percentage=0.0
            )
    
    def validate_trade_risk(self,
                          risk_metrics: RiskMetrics,
                          max_portfolio_risk: float = 5.0) -> Tuple[bool, List[str]]:
        """
        Validate if trade meets risk management criteria
        
        Args:
            risk_metrics: Risk metrics to validate
            max_portfolio_risk: Maximum portfolio risk percentage
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check minimum risk/reward ratio
            if risk_metrics.tp_risk_rewards:
                min_rr = min(risk_metrics.tp_risk_rewards)
                if min_rr < self.config.min_risk_reward_ratio:
                    issues.append(f"Risk/reward ratio {min_rr:.2f} below minimum {self.config.min_risk_reward_ratio}")
            
            # Check portfolio risk
            if risk_metrics.portfolio_risk_percentage > max_portfolio_risk:
                issues.append(f"Portfolio risk {risk_metrics.portfolio_risk_percentage:.2f}% exceeds maximum {max_portfolio_risk}%")
            
            # Check position size limits
            if risk_metrics.optimal_position_size <= 0:
                issues.append("Position size is zero or negative")
            
            if risk_metrics.optimal_position_size > risk_metrics.max_position_size:
                issues.append("Position size exceeds maximum limit")
            
            # Check if we have take profits
            if not risk_metrics.tp_levels:
                issues.append("No take profit levels generated")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                self.logger.debug("Trade risk validation: PASSED")
            else:
                self.logger.warning(f"Trade risk validation: FAILED - {'; '.join(issues)}")
            
            return is_valid, issues
            
        except Exception as e:
            self.logger.error(f"Risk validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the risk manager"""
        return {
            'total_calculations': self.total_calculations,
            'average_calculation_time': self.average_calculation_time,
            'config': {
                'default_risk_percentage': self.config.default_risk_percentage,
                'min_risk_reward_ratio': self.config.min_risk_reward_ratio,
                'position_size_limits': self.config.position_size_limits
            }
        }