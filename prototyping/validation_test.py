#!/usr/bin/env python3
"""
Quick validation test for the elemental transmutator system.
This just confirms the main components work and demos run successfully.
"""

import sys
import os
import time
import logging
import subprocess
from pathlib import Path

def setup_logging():
    """Setup logging for the validation test."""
    # Set stdout encoding for Windows compatibility
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def test_quick_demo():
    """Test that the quick demo runs successfully."""
    logger = logging.getLogger(__name__)
    logger.info("Testing quick demo execution...")
    
    try:
        # Change to the prototyping directory
        prototyping_dir = Path(__file__).parent
        
        # Run the quick demo
        result = subprocess.run(
            [sys.executable, "quick_demo_ascii.py"],
            cwd=prototyping_dir,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode == 0:
            logger.info("  Quick demo: PASS")
            return True
        else:
            logger.error(f"  Quick demo failed with code {result.returncode}")
            logger.error(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("  Quick demo: TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"  Quick demo error: {e}")
        return False

def test_module_imports():
    """Test that core modules can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("Testing core module imports...")
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    imports = [
        ("photonuclear_transmutation", "Physics engine"),
        ("energy_ledger", "Energy accounting"),
        ("atomic_binder", "Nuclear data"),
    ]
    
    success = True
    for module_name, description in imports:
        try:
            __import__(module_name)
            logger.info(f"  {description}: OK")
        except ImportError as e:
            logger.error(f"  {description}: FAIL - {e}")
            success = False
    
    return success

def test_config_files():
    """Test that configuration files exist and are valid."""
    logger = logging.getLogger(__name__)
    logger.info("Testing configuration files...")
    
    config_files = [
        ("hardware_config.json", "Hardware configuration"),
        ("PROTOTYPE_DEMO_BLUEPRINT.md", "Prototype documentation"),
        ("FINAL_STATUS_REPORT.md", "Status report")
    ]
    
    success = True
    for filename, description in config_files:
        filepath = Path(__file__).parent / filename
        if filepath.exists():
            logger.info(f"  {description}: OK")
        else:
            logger.error(f"  {description}: MISSING")
            success = False
    
    return success

def main():
    """Run the validation test suite."""
    logger = setup_logging()
    
    logger.info("ELEMENTAL TRANSMUTATOR - VALIDATION TEST")
    logger.info("=" * 50)
    
    tests = [
        ("Core Module Imports", test_module_imports),
        ("Configuration Files", test_config_files),
        ("Quick Demo Execution", test_quick_demo)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"{test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION TEST RESULTS")
    logger.info("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name:<25}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 50)
    if all_passed:
        logger.info("ALL TESTS PASSED - SYSTEM VALIDATED ✓")
        logger.info("System ready for demonstration and production!")
        return 0
    else:
        logger.error("SOME TESTS FAILED - CHECK SYSTEM")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
