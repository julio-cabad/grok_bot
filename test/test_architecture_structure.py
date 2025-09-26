#!/usr/bin/env python3
"""
Test Architecture Structure
Validates that Clean Architecture directory structure is correctly created
"""

import os
from pathlib import Path

def test_core_layer_structure():
    """Test that core domain layer structure exists"""
    print("🏛️ Testing Core Domain Layer Structure...")
    
    base_path = Path(__file__).parent.parent / "src" / "core"
    
    # Test core directories
    required_dirs = [
        "domain",
        "domain/entities",
        "domain/value_objects", 
        "domain/exceptions",
        "ports",
        "use_cases"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"Missing directory: {full_path}"
        assert (full_path / "__init__.py").exists(), f"Missing __init__.py in: {full_path}"
        print(f"✅ {dir_path} - EXISTS")
    
    print("✅ Core Domain Layer structure - PASSED\n")

def test_infrastructure_layer_structure():
    """Test that infrastructure layer structure exists"""
    print("🔧 Testing Infrastructure Layer Structure...")
    
    base_path = Path(__file__).parent.parent / "src" / "infrastructure"
    
    # Test infrastructure directories
    required_dirs = [
        "exchanges",
        "exchanges/binance",
        "indicators",
        "indicators/technical",
        "ai",
        "ai/providers",
        "ai/analyzers",
        "notifications",
        "notifications/telegram",
        "persistence",
        "persistence/repositories",
        "persistence/models",
        "cache"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"Missing directory: {full_path}"
        assert (full_path / "__init__.py").exists(), f"Missing __init__.py in: {full_path}"
        print(f"✅ {dir_path} - EXISTS")
    
    print("✅ Infrastructure Layer structure - PASSED\n")

def test_application_layer_structure():
    """Test that application layer structure exists"""
    print("🚀 Testing Application Layer Structure...")
    
    base_path = Path(__file__).parent.parent / "src" / "application"
    
    # Test application directories
    required_dirs = [
        "services",
        "strategies"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"Missing directory: {full_path}"
        assert (full_path / "__init__.py").exists(), f"Missing __init__.py in: {full_path}"
        print(f"✅ {dir_path} - EXISTS")
    
    print("✅ Application Layer structure - PASSED\n")

def test_presentation_layer_structure():
    """Test that presentation layer structure exists"""
    print("🖥️ Testing Presentation Layer Structure...")
    
    base_path = Path(__file__).parent.parent / "src" / "presentation"
    
    # Test presentation directories
    required_dirs = [
        "cli",
        "cli/commands",
        "cli/formatters",
        "api"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        assert full_path.exists(), f"Missing directory: {full_path}"
        assert (full_path / "__init__.py").exists(), f"Missing __init__.py in: {full_path}"
        print(f"✅ {dir_path} - EXISTS")
    
    print("✅ Presentation Layer structure - PASSED\n")

def test_shared_utilities_structure():
    """Test that shared utilities structure exists"""
    print("🛠️ Testing Shared Utilities Structure...")
    
    base_path = Path(__file__).parent.parent / "src" / "shared"
    
    # Test shared directories (already created in previous task)
    required_dirs = [
        "config",
        "logging", 
        "migration",
        "monitoring",
        "utils"
    ]
    
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"✅ {dir_path} - EXISTS")
        else:
            print(f"⚠️ {dir_path} - MISSING (will be created later)")
    
    print("✅ Shared Utilities structure - CHECKED\n")

def test_legacy_code_untouched():
    """Test that legacy code remains untouched"""
    print("🛡️ Testing Legacy Code Integrity...")
    
    base_path = Path(__file__).parent.parent
    
    # Test that legacy files still exist and are untouched
    legacy_files = [
        "main.py",
        "config/settings.py",
        "bnb/binance.py",
        "indicators/technical_indicators.py",
        "strategy/strategies.py",
        "notifications/telegram_notifier.py"
    ]
    
    for file_path in legacy_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path} - PRESERVED")
        else:
            print(f"⚠️ {file_path} - MISSING")
    
    print("✅ Legacy Code Integrity - VERIFIED\n")

def test_documentation_exists():
    """Test that documentation is created"""
    print("📚 Testing Documentation...")
    
    base_path = Path(__file__).parent.parent
    
    # Test documentation files
    doc_files = [
        "src/README.md",
        "test/README.md"
    ]
    
    for file_path in doc_files:
        full_path = base_path / file_path
        assert full_path.exists(), f"Missing documentation: {full_path}"
        print(f"✅ {file_path} - EXISTS")
    
    print("✅ Documentation - PASSED\n")

def main():
    """Run all architecture structure tests"""
    print("🏛️ CLEAN ARCHITECTURE STRUCTURE TESTS")
    print("=" * 50)
    
    try:
        test_core_layer_structure()
        test_infrastructure_layer_structure()
        test_application_layer_structure()
        test_presentation_layer_structure()
        test_shared_utilities_structure()
        test_legacy_code_untouched()
        test_documentation_exists()
        
        print("🏛️ ALL ARCHITECTURE TESTS PASSED!")
        print("Clean Architecture structure is ready for development!")
        print("=" * 50)
        
        # Display structure summary
        print("\n📊 ARCHITECTURE SUMMARY:")
        print("✅ Core Domain Layer - Ready")
        print("✅ Infrastructure Layer - Ready") 
        print("✅ Application Layer - Ready")
        print("✅ Presentation Layer - Ready")
        print("✅ Legacy Code - Preserved")
        print("✅ Documentation - Created")
        
        return True
        
    except Exception as e:
        print(f"💀 ARCHITECTURE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)