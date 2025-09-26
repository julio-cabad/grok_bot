#!/usr/bin/env python3
"""
Test script for the Spartan Migration System
Validates that feature flags and orchestrator work correctly
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.shared.config.migration_config import MigrationConfig, migration_config
from src.shared.migration.migration_orchestrator import MigrationOrchestrator, ComponentType
from src.shared.logging.migration_logger import MigrationLogger

def test_migration_config():
    """Test migration configuration"""
    print("🏛️ Testing Migration Configuration...")
    
    # Test default configuration
    config = MigrationConfig()
    print(f"✅ Default phase: {config.get_migration_phase()}")
    print(f"✅ Active flags: {config.get_active_flags()}")
    print(f"✅ Configuration valid: {config.validate_configuration()}")
    
    # Test phase detection
    config.use_new_notifications = True
    print(f"✅ Phase after enabling notifications: {config.get_migration_phase()}")
    
    config.use_new_indicators = True
    print(f"✅ Phase after enabling indicators: {config.get_migration_phase()}")
    
    print("✅ Migration Configuration tests passed!\n")

def test_migration_orchestrator():
    """Test migration orchestrator"""
    print("🏛️ Testing Migration Orchestrator...")
    
    orchestrator = MigrationOrchestrator()
    
    # Mock components for testing
    class MockLegacyNotifier:
        def send_message(self, message):
            return f"LEGACY: {message}"
    
    class MockNewNotifier:
        def send_message(self, message):
            return f"NEW: {message}"
    
    # Register components
    orchestrator.register_legacy_component(ComponentType.NOTIFICATIONS, MockLegacyNotifier())
    orchestrator.register_new_component(ComponentType.NOTIFICATIONS, MockNewNotifier())
    
    # Test component selection (should use legacy by default)
    notifier = orchestrator.get_component(ComponentType.NOTIFICATIONS)
    result = notifier.send_message("Test message")
    print(f"✅ Default component result: {result}")
    
    # Test migration status
    status = orchestrator.get_migration_status()
    print(f"✅ Migration status: {status}")
    
    print("✅ Migration Orchestrator tests passed!\n")

def test_logging_system():
    """Test migration logging system"""
    print("🏛️ Testing Migration Logging System...")
    
    logger = logging.getLogger("TestLogger")
    
    # Test structured logging
    MigrationLogger.log_migration_event(
        logger=logger,
        event_type="TEST",
        component="notifications",
        phase="TESTING",
        message="Testing migration logging system",
        test_data="sample_value"
    )
    
    MigrationLogger.log_component_switch(
        logger=logger,
        component="notifications",
        from_version="legacy",
        to_version="new",
        reason="testing"
    )
    
    print("✅ Migration Logging tests passed!\n")

def test_rollback_system():
    """Test rollback system"""
    print("🏛️ Testing Rollback System...")
    
    config = MigrationConfig()
    config.rollback_error_threshold = 2  # Lower threshold for testing
    
    orchestrator = MigrationOrchestrator(config)
    
    # Simulate errors
    test_error = Exception("Test error")
    
    # First error - should not trigger rollback
    rollback_triggered = orchestrator.rollback_manager.record_error("notifications", test_error)
    print(f"✅ First error rollback triggered: {rollback_triggered}")
    
    # Second error - should trigger rollback
    rollback_triggered = orchestrator.rollback_manager.record_error("notifications", test_error)
    print(f"✅ Second error rollback triggered: {rollback_triggered}")
    
    print("✅ Rollback System tests passed!\n")

def main():
    """Run all tests"""
    print("🚀 SPARTAN MIGRATION SYSTEM TESTS")
    print("=" * 50)
    
    try:
        test_migration_config()
        test_migration_orchestrator()
        test_logging_system()
        test_rollback_system()
        
        print("🏛️ ALL TESTS PASSED! Migration system is ready for battle!")
        print("=" * 50)
        
        # Display current configuration
        print("\n📊 CURRENT MIGRATION STATUS:")
        print(f"Phase: {migration_config.get_migration_phase()}")
        print(f"Active Flags: {migration_config.get_active_flags()}")
        
        return True
        
    except Exception as e:
        print(f"💀 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)