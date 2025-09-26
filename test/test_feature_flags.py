#!/usr/bin/env python3
"""
Test Feature Flags System
Validates that all feature flags work correctly for gradual migration
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_all_migration_phases():
    """Test all migration phases with different flag combinations"""
    print("🏛️ Testing All Migration Phases...")
    
    from src.shared.config.migration_config import MigrationConfig
    
    # Phase 0 - All Legacy
    config = MigrationConfig()
    assert config.get_migration_phase() == "PHASE_0_PREPARATION"
    print("✅ Phase 0 (All Legacy) - PASSED")
    
    # Phase A - Only Notifications
    config.use_new_notifications = True
    assert config.get_migration_phase() == "PHASE_A_NOTIFICATIONS"
    print("✅ Phase A (Notifications) - PASSED")
    
    # Phase B - Notifications + Indicators
    config.use_new_indicators = True
    assert config.get_migration_phase() == "PHASE_B_INDICATORS"
    print("✅ Phase B (Notifications + Indicators) - PASSED")
    
    # Phase C - Notifications + Indicators + Strategies
    config.use_new_strategies = True
    assert config.get_migration_phase() == "PHASE_C_STRATEGIES"
    print("✅ Phase C (Notifications + Indicators + Strategies) - PASSED")
    
    # Phase D - All + AI
    config.use_ai_validation = True
    assert config.get_migration_phase() == "PHASE_D_AI_SMC"
    print("✅ Phase D (All + AI SMC) - PASSED")
    
    # Phase E - Complete Refactored
    config.use_refactored_version = True
    assert config.get_migration_phase() == "PHASE_E_COMPLETE"
    print("✅ Phase E (Complete Refactored) - PASSED")
    
    print("✅ All Migration Phases tests passed!\n")

def test_component_selection():
    """Test that orchestrator selects correct components based on flags"""
    print("🏛️ Testing Component Selection...")
    
    from src.shared.migration.migration_orchestrator import MigrationOrchestrator, ComponentType
    from src.shared.config.migration_config import MigrationConfig
    
    # Test with different configurations
    config = MigrationConfig()
    orchestrator = MigrationOrchestrator(config)
    
    # Mock components
    class MockLegacy:
        def get_type(self): return "LEGACY"
    
    class MockNew:
        def get_type(self): return "NEW"
    
    # Register components
    orchestrator.register_legacy_component(ComponentType.NOTIFICATIONS, MockLegacy())
    orchestrator.register_new_component(ComponentType.NOTIFICATIONS, MockNew())
    
    # Test legacy selection (default)
    component = orchestrator.get_component(ComponentType.NOTIFICATIONS)
    assert component.get_type() == "LEGACY"
    print("✅ Legacy component selection - PASSED")
    
    # Test new component selection
    config.use_new_notifications = True
    orchestrator.config = config  # Update config
    component = orchestrator.get_component(ComponentType.NOTIFICATIONS)
    assert component.get_type() == "NEW"
    print("✅ New component selection - PASSED")
    
    print("✅ Component Selection tests passed!\n")

def test_rollback_functionality():
    """Test automatic rollback functionality"""
    print("🏛️ Testing Rollback Functionality...")
    
    from src.shared.migration.migration_orchestrator import MigrationOrchestrator
    from src.shared.config.migration_config import MigrationConfig
    
    config = MigrationConfig()
    config.rollback_error_threshold = 2
    config.use_new_notifications = True  # Start with new component
    
    orchestrator = MigrationOrchestrator(config)
    
    # Simulate errors
    error1 = Exception("First error")
    error2 = Exception("Second error")
    
    # First error should not trigger rollback
    rollback1 = orchestrator.rollback_manager.record_error("notifications", error1)
    assert not rollback1
    assert config.use_new_notifications == True  # Should still be True
    print("✅ First error handling - PASSED")
    
    # Second error should trigger rollback
    rollback2 = orchestrator.rollback_manager.record_error("notifications", error2)
    assert rollback2
    assert config.use_new_notifications == False  # Should be rolled back
    print("✅ Automatic rollback - PASSED")
    
    print("✅ Rollback Functionality tests passed!\n")

def main():
    """Run all feature flag tests"""
    print("🚀 FEATURE FLAGS SYSTEM TESTS")
    print("=" * 50)
    
    try:
        test_all_migration_phases()
        test_component_selection()
        test_rollback_functionality()
        
        print("🏛️ ALL FEATURE FLAG TESTS PASSED!")
        print("Feature flags system is ready for gradual migration!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"💀 FEATURE FLAG TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)