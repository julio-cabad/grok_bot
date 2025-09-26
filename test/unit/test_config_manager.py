#!/usr/bin/env python3
"""
Configuration Manager Tests - Spartan Trading System
Comprehensive tests for the configuration management system
"""

import sys
import os
import json
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.config_manager import ConfigurationManager, get_config_manager, get_config, update_config
from ai.data_models import AIValidatorConfig


class TestConfigurationManager:
    """Test suite for Configuration Manager"""
    
    def setup_method(self):
        """Setup for each test"""
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.config_manager = ConfigurationManager(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup after each test"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_default_configuration(self):
        """Test default configuration loading"""
        config = self.config_manager.get_config()
        
        assert isinstance(config, AIValidatorConfig)
        assert config.confidence_threshold == 7.5
        assert config.timeout_seconds == 30
        assert config.default_risk_percentage == 1.0
        assert config.min_risk_reward_ratio == 1.5
        print("✅ Default configuration test passed")
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = AIValidatorConfig(
            confidence_threshold=15.0,  # Invalid: > 10
            default_risk_percentage=-1.0,  # Invalid: negative
            min_risk_reward_ratio=0.5  # Invalid: < 1.0
        )
        
        issues = config.validate_config()
        assert len(issues) >= 3  # Should have at least 3 validation issues
        print(f"✅ Configuration validation test passed - Found {len(issues)} issues")
    
    def test_runtime_configuration_update(self):
        """Test runtime configuration updates"""
        # Get initial config
        initial_threshold = self.config_manager.get_config().confidence_threshold
        
        # Update configuration
        success = self.config_manager.update_config(
            confidence_threshold=8.5,
            timeout_seconds=45
        )
        
        assert success == True
        
        # Verify updates
        updated_config = self.config_manager.get_config()
        assert updated_config.confidence_threshold == 8.5
        assert updated_config.timeout_seconds == 45
        
        print("✅ Runtime configuration update test passed")
    
    def test_invalid_configuration_update(self):
        """Test invalid configuration update rejection"""
        # Try to update with invalid values
        success = self.config_manager.update_config(
            confidence_threshold=15.0,  # Invalid
            default_risk_percentage=-1.0  # Invalid
        )
        
        assert success == False
        
        # Verify configuration wasn't changed
        config = self.config_manager.get_config()
        assert config.confidence_threshold != 15.0
        assert config.default_risk_percentage != -1.0
        
        print("✅ Invalid configuration update rejection test passed")
    
    def test_file_configuration_loading(self):
        """Test loading configuration from JSON file"""
        # Create test configuration file
        test_config = {
            'confidence_threshold': 8.0,
            'timeout_seconds': 60,
            'default_risk_percentage': 2.0
        }
        
        with open(self.temp_file.name, 'w') as f:
            json.dump(test_config, f)
        
        # Reload configuration
        config = self.config_manager.reload_configuration()
        
        assert config.confidence_threshold == 8.0
        assert config.timeout_seconds == 60
        assert config.default_risk_percentage == 2.0
        
        print("✅ File configuration loading test passed")
    
    def test_environment_variable_override(self):
        """Test environment variable configuration override"""
        # Set environment variables
        os.environ['AI_CONFIDENCE_THRESHOLD'] = '9.0'
        os.environ['AI_TIMEOUT_SECONDS'] = '90'
        os.environ['RISK_DEFAULT_PERCENTAGE'] = '1.5'
        
        try:
            # Reload configuration
            config = self.config_manager.reload_configuration()
            
            assert config.confidence_threshold == 9.0
            assert config.timeout_seconds == 90
            assert config.default_risk_percentage == 1.5
            
            print("✅ Environment variable override test passed")
            
        finally:
            # Clean up environment variables
            for var in ['AI_CONFIDENCE_THRESHOLD', 'AI_TIMEOUT_SECONDS', 'RISK_DEFAULT_PERCENTAGE']:
                if var in os.environ:
                    del os.environ[var]
    
    def test_configuration_rollback(self):
        """Test configuration rollback functionality"""
        # Get initial config
        initial_threshold = self.config_manager.get_config().confidence_threshold
        
        # Update configuration
        self.config_manager.update_config(confidence_threshold=8.5)
        assert self.config_manager.get_config().confidence_threshold == 8.5
        
        # Rollback configuration
        success = self.config_manager.rollback_config()
        assert success == True
        
        # Verify rollback
        rolled_back_config = self.config_manager.get_config()
        assert rolled_back_config.confidence_threshold == initial_threshold
        
        print("✅ Configuration rollback test passed")
    
    def test_configuration_reset(self):
        """Test configuration reset to defaults"""
        # Update configuration
        self.config_manager.update_config(
            confidence_threshold=8.5,
            timeout_seconds=90
        )
        
        # Reset to defaults
        success = self.config_manager.reset_to_defaults()
        assert success == True
        
        # Verify reset
        config = self.config_manager.get_config()
        default_config = AIValidatorConfig()
        
        assert config.confidence_threshold == default_config.confidence_threshold
        assert config.timeout_seconds == default_config.timeout_seconds
        
        print("✅ Configuration reset test passed")
    
    def test_configuration_summary(self):
        """Test configuration summary generation"""
        summary = self.config_manager.get_config_summary()
        
        # Verify summary structure
        assert 'ai_settings' in summary
        assert 'risk_management' in summary
        assert 'volatility_settings' in summary
        assert 'smc_settings' in summary
        assert 'cache_settings' in summary
        assert 'performance_settings' in summary
        
        # Verify some key values
        assert 'confidence_threshold' in summary['ai_settings']
        assert 'default_risk_percentage' in summary['risk_management']
        
        print("✅ Configuration summary test passed")
    
    def test_global_configuration_functions(self):
        """Test global configuration convenience functions"""
        # Test get_config function
        config = get_config()
        assert isinstance(config, AIValidatorConfig)
        
        # Test update_config function
        success = update_config(confidence_threshold=8.0)
        assert success == True
        
        updated_config = get_config()
        assert updated_config.confidence_threshold == 8.0
        
        print("✅ Global configuration functions test passed")


def test_millionaire_configuration_scenario():
    """Test a realistic millionaire trading configuration scenario"""
    print("\n🏛️ Testing MILLIONAIRE Configuration Scenario...")
    
    # Create configuration manager
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        config_manager = ConfigurationManager(temp_file.name)
        
        try:
            # Test high-performance trading configuration
            millionaire_config = {
                'confidence_threshold': 8.0,  # Higher threshold for quality
                'timeout_seconds': 20,  # Faster for high-frequency
                'max_retries': 2,  # Quick retries
                'default_risk_percentage': 0.5,  # Conservative risk
                'min_risk_reward_ratio': 2.0,  # Higher R:R requirement
                'cache_ttl_minutes': 3,  # Shorter cache for fresh data
                'max_concurrent_analyses': 10,  # Higher concurrency
                'performance_monitoring': True  # Full monitoring
            }
            
            # Update configuration
            success = config_manager.update_config(**millionaire_config)
            assert success == True
            
            # Verify configuration
            config = config_manager.get_config()
            assert config.confidence_threshold == 8.0
            assert config.timeout_seconds == 20
            assert config.default_risk_percentage == 0.5
            assert config.min_risk_reward_ratio == 2.0
            
            # Test configuration summary
            summary = config_manager.get_config_summary()
            print(f"📊 AI Settings: Threshold={summary['ai_settings']['confidence_threshold']}, Timeout={summary['ai_settings']['timeout_seconds']}s")
            print(f"💰 Risk Settings: Default={summary['risk_management']['default_risk_percentage']}%, Min R:R={summary['risk_management']['min_risk_reward_ratio']}")
            print(f"🚀 Performance: Concurrent={summary['performance_settings']['max_concurrent_analyses']}, Monitoring={summary['performance_settings']['performance_monitoring']}")
            
            # Test environment override
            os.environ['AI_CONFIDENCE_THRESHOLD'] = '8.5'
            config = config_manager.reload_configuration()
            assert config.confidence_threshold == 8.5
            
            print("🏆 MILLIONAIRE CONFIGURATION: SUCCESS!")
            
        finally:
            # Cleanup
            try:
                os.unlink(temp_file.name)
                if 'AI_CONFIDENCE_THRESHOLD' in os.environ:
                    del os.environ['AI_CONFIDENCE_THRESHOLD']
            except:
                pass


if __name__ == "__main__":
    # Run millionaire scenario test
    test_millionaire_configuration_scenario()
    
    # Run basic tests
    test_manager = TestConfigurationManager()
    test_manager.setup_method()
    
    try:
        test_manager.test_default_configuration()
        test_manager.test_configuration_validation()
        test_manager.test_runtime_configuration_update()
        test_manager.test_invalid_configuration_update()
        test_manager.test_file_configuration_loading()
        test_manager.test_environment_variable_override()
        test_manager.test_configuration_rollback()
        test_manager.test_configuration_reset()
        test_manager.test_configuration_summary()
        test_manager.test_global_configuration_functions()
        
        print("\n🏛️ ALL CONFIGURATION MANAGER TESTS PASSED! 🏛️")
        
    finally:
        test_manager.teardown_method()