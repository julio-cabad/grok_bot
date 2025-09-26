#!/usr/bin/env python3
"""
Configuration Manager - Spartan Trading System
Centralized configuration management for maximum flexibility and control
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import json

from .data_models import AIValidatorConfig


class ConfigurationManager:
    """
    Elite Configuration Manager for the Spartan Trading System
    
    Features:
    - Load configuration from multiple sources
    - Runtime configuration updates
    - Configuration validation
    - Environment variable overrides
    - Configuration persistence
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager"""
        self.logger = logging.getLogger("ConfigManager")
        self.config_file = config_file or "ai_validator_config.json"
        self._config = None
        self._config_history = []
        
        # Load configuration
        self.reload_configuration()
        
        self.logger.info(f"🔧 ConfigurationManager initialized - File: {self.config_file}")
    
    def get_config(self) -> AIValidatorConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self._load_default_config()
        return self._config
    
    def reload_configuration(self) -> AIValidatorConfig:
        """Reload configuration from all sources"""
        try:
            # Start with default configuration
            config = self._load_default_config()
            
            # Override with file configuration if exists
            file_config = self._load_file_config()
            if file_config:
                config = self._merge_configs(config, file_config)
            
            # Override with environment variables
            env_config = self._load_env_config()
            if env_config:
                config = self._merge_configs(config, env_config)
            
            # Override with settings.py if available
            settings_config = self._load_settings_config()
            if settings_config:
                config = self._merge_configs(config, settings_config)
            
            # Validate configuration
            issues = config.validate_config()
            if issues:
                self.logger.warning(f"Configuration validation issues: {issues}")
                for issue in issues:
                    self.logger.warning(f"  - {issue}")
            
            # Store configuration
            old_config = self._config
            self._config = config
            
            # Track configuration changes
            if old_config:
                self._track_config_changes(old_config, config)
            
            self.logger.info("✅ Configuration reloaded successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reload configuration: {e}")
            if self._config is None:
                self._config = self._load_default_config()
            return self._config
    
    def update_config(self, **kwargs) -> bool:
        """
        Update configuration at runtime
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            current_config = self.get_config()
            
            # Create new configuration with updates
            config_dict = asdict(current_config)
            
            # Apply updates
            for key, value in kwargs.items():
                if hasattr(current_config, key):
                    config_dict[key] = value
                    self.logger.info(f"🔄 Config updated: {key} = {value}")
                else:
                    self.logger.warning(f"⚠️ Unknown config parameter: {key}")
            
            # Create new config object
            new_config = AIValidatorConfig(**config_dict)
            
            # Validate new configuration
            issues = new_config.validate_config()
            if issues:
                self.logger.error(f"❌ Configuration validation failed: {issues}")
                return False
            
            # Store old config for history
            self._config_history.append(self._config)
            
            # Apply new configuration
            self._config = new_config
            
            # Optionally persist to file
            self._save_config_to_file()
            
            self.logger.info("✅ Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for monitoring"""
        config = self.get_config()
        return {
            'ai_settings': {
                'confidence_threshold': config.confidence_threshold,
                'timeout_seconds': config.timeout_seconds,
                'max_retries': config.max_retries,
                'use_threading': config.use_threading
            },
            'risk_management': {
                'default_risk_percentage': config.default_risk_percentage,
                'max_risk_percentage': config.max_risk_percentage,
                'min_risk_reward_ratio': config.min_risk_reward_ratio,
                'position_size_limits': config.position_size_limits
            },
            'volatility_settings': {
                'high_volatility_reduction': config.high_volatility_reduction,
                'low_volatility_increase': config.low_volatility_increase,
                'volatility_lookback_periods': config.volatility_lookback_periods
            },
            'smc_settings': {
                'order_block_lookback': config.order_block_lookback,
                'fvg_min_gap_percentage': config.fvg_min_gap_percentage,
                'premium_threshold': config.premium_threshold,
                'discount_threshold': config.discount_threshold
            },
            'cache_settings': {
                'cache_ttl_minutes': config.cache_ttl_minutes,
                'cache_max_size': config.cache_max_size,
                'cache_cleanup_frequency': config.cache_cleanup_frequency
            },
            'performance_settings': {
                'max_concurrent_analyses': config.max_concurrent_analyses,
                'memory_limit_mb': config.memory_limit_mb,
                'performance_monitoring': config.performance_monitoring
            }
        }
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self._config_history.append(self._config)
            self._config = self._load_default_config()
            self.logger.info("🔄 Configuration reset to defaults")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to reset configuration: {e}")
            return False
    
    def rollback_config(self) -> bool:
        """Rollback to previous configuration"""
        try:
            if not self._config_history:
                self.logger.warning("⚠️ No configuration history available for rollback")
                return False
            
            previous_config = self._config_history.pop()
            self._config = previous_config
            self.logger.info("🔄 Configuration rolled back to previous version")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to rollback configuration: {e}")
            return False
    
    def _load_default_config(self) -> AIValidatorConfig:
        """Load default configuration"""
        return AIValidatorConfig()
    
    def _load_file_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self.logger.debug(f"📁 Loaded configuration from {self.config_file}")
                return config_data
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load config file {self.config_file}: {e}")
        return None
    
    def _load_env_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Map environment variables to config parameters
        env_mappings = {
            'AI_CONFIDENCE_THRESHOLD': ('confidence_threshold', float),
            'AI_TIMEOUT_SECONDS': ('timeout_seconds', int),
            'AI_MAX_RETRIES': ('max_retries', int),
            'AI_USE_THREADING': ('use_threading', lambda x: x.lower() == 'true'),
            'RISK_DEFAULT_PERCENTAGE': ('default_risk_percentage', float),
            'RISK_MAX_PERCENTAGE': ('max_risk_percentage', float),
            'RISK_MIN_RR_RATIO': ('min_risk_reward_ratio', float),
            'CACHE_TTL_MINUTES': ('cache_ttl_minutes', int),
            'CACHE_MAX_SIZE': ('cache_max_size', int),
            'SMC_ORDER_BLOCK_LOOKBACK': ('order_block_lookback', int),
            'SMC_FVG_MIN_GAP_PCT': ('fvg_min_gap_percentage', float),
            'SMC_PREMIUM_THRESHOLD': ('premium_threshold', float),
            'SMC_DISCOUNT_THRESHOLD': ('discount_threshold', float)
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    env_config[config_key] = converter(env_value)
                    self.logger.debug(f"🌍 Environment override: {config_key} = {env_config[config_key]}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Invalid environment variable {env_var}={env_value}: {e}")
        
        return env_config if env_config else None
    
    def _load_settings_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from settings.py"""
        try:
            # Try to import settings from the main config
            from config.settings import (
                AI_CONFIDENCE_THRESHOLD, AI_TIMEOUT_SECONDS, USE_AI_VALIDATION
            )
            
            settings_config = {}
            
            # Map settings.py variables to config
            if hasattr(locals(), 'AI_CONFIDENCE_THRESHOLD'):
                settings_config['confidence_threshold'] = AI_CONFIDENCE_THRESHOLD
            
            if hasattr(locals(), 'AI_TIMEOUT_SECONDS'):
                settings_config['timeout_seconds'] = AI_TIMEOUT_SECONDS
            
            if hasattr(locals(), 'USE_AI_VALIDATION'):
                # This could control whether AI validation is enabled
                pass
            
            if settings_config:
                self.logger.debug("⚙️ Loaded configuration from settings.py")
                return settings_config
                
        except ImportError:
            self.logger.debug("📝 No settings.py configuration found")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load settings.py configuration: {e}")
        
        return None
    
    def _merge_configs(self, base_config: AIValidatorConfig, override_dict: Dict[str, Any]) -> AIValidatorConfig:
        """Merge configuration with overrides"""
        try:
            # Convert base config to dict
            config_dict = asdict(base_config)
            
            # Apply overrides
            for key, value in override_dict.items():
                if key in config_dict:
                    config_dict[key] = value
            
            # Create new config object
            return AIValidatorConfig(**config_dict)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to merge configurations: {e}")
            return base_config
    
    def _save_config_to_file(self) -> bool:
        """Save current configuration to file"""
        try:
            config_dict = asdict(self._config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.debug(f"💾 Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save configuration to file: {e}")
            return False
    
    def _track_config_changes(self, old_config: AIValidatorConfig, new_config: AIValidatorConfig):
        """Track and log configuration changes"""
        old_dict = asdict(old_config)
        new_dict = asdict(new_config)
        
        changes = []
        for key, new_value in new_dict.items():
            old_value = old_dict.get(key)
            if old_value != new_value:
                changes.append(f"{key}: {old_value} → {new_value}")
        
        if changes:
            self.logger.info(f"🔄 Configuration changes detected:")
            for change in changes:
                self.logger.info(f"  - {change}")


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigurationManager:
    """Get or create global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_file)
    return _config_manager

def get_config() -> AIValidatorConfig:
    """Get current configuration (convenience function)"""
    return get_config_manager().get_config()

def update_config(**kwargs) -> bool:
    """Update configuration at runtime (convenience function)"""
    return get_config_manager().update_config(**kwargs)

def reload_config() -> AIValidatorConfig:
    """Reload configuration from all sources (convenience function)"""
    return get_config_manager().reload_configuration()