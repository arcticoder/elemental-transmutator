#!/usr/bin/env python3
"""
Integration test for the complete elemental transmutator system.
Tests all major components and validates the full pipeline.
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging for the integration test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported successfully."""
    logger = logging.getLogger(__name__)
    logger.info("Testing module imports...")
    
    try:
        import photonuclear_transmutation
        logger.info("  photonuclear_transmutation: OK")
        
        import energy_ledger  
        logger.info("  energy_ledger: OK")
        
        import atomic_binder
        logger.info("  atomic_binder: OK")
        
        from prototyping import gamma_beam_controller
        logger.info("  gamma_beam_controller: OK")
        
        from prototyping import target_cell_monitor
        logger.info("  target_cell_monitor: OK")
        
        from prototyping import data_logger
        logger.info("  data_logger: OK")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def test_physics_engine():
    """Test the photonuclear physics calculations."""
    logger = logging.getLogger(__name__)
    logger.info("Testing physics engine...")
    
    try:
        import photonuclear_transmutation as pn
        
        # Test creating a PhotonuclearTransmutation instance
        config = pn.PhotonuclearConfig(
            feedstock_isotope="Pb-208",
            target_isotope="Au-197",
            gamma_energy_MeV=16.5,
            photon_flux=5e13
        )
        
        transmuter = pn.PhotonuclearTransmutation(config)
        logger.info("  PhotonuclearTransmutation: Initialized OK")
        
        # Test GDR cross-section calculation
        sigma = transmuter.gdr_cross_section()
        logger.info(f"  GDR cross-section: {sigma:.3e} barns")
        
        # Test LV enhancement
        lv_factor = transmuter.lv_enhancement_factor()
        logger.info(f"  LV enhancement factor: {lv_factor:.2f}")
        
        # Test transmutation calculation
        result = transmuter.calculate_transmutation_yield(target_mass=1.0)
        logger.info(f"  Transmutation yield: {result['conversion_efficiency']:.6f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Physics engine test failed: {e}")
        return False

def test_hardware_controllers():
    """Test the hardware control modules."""
    logger = logging.getLogger(__name__)
    logger.info("Testing hardware controllers...")
    
    try:
        import json
        from prototyping.gamma_beam_controller import GammaBeamController
        from prototyping.target_cell_monitor import TargetCellMonitor
        from prototyping.data_logger import DataLogger
        
        # Load config for controllers that need it
        config_path = Path(__file__).parent / "hardware_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test beam controller initialization
        beam = GammaBeamController(config)
        logger.info("  GammaBeamController: Initialized OK")
        
        # Test target monitor initialization
        target = TargetCellMonitor()
        logger.info("  TargetCellMonitor: Initialized OK")
        
        # Test data logger initialization
        data_logger = DataLogger()
        logger.info("  DataLogger: Initialized OK")
        
        # Cleanup
        beam.cleanup()
        target.cleanup()
        data_logger.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Hardware controller test failed: {e}")
        return False

def test_energy_ledger():
    """Test the energy accounting system."""
    logger = logging.getLogger(__name__)
    logger.info("Testing energy ledger...")
    
    try:
        import energy_ledger as el
        
        # Test basic energy ledger
        ledger = el.EnergyLedger()
        
        # Add some operations
        ledger.add_operation('beam_power', 30000, 3600)  # 30kW for 1 hour
        ledger.add_operation('cooling', 5000, 3600)      # 5kW for 1 hour
        
        costs = ledger.get_summary()
        logger.info(f"  Total energy: {ledger.total_energy_kwh:.1f} kWh")
        logger.info(f"  Operations logged: {len(ledger.operations)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Energy ledger test failed: {e}")
        return False

def test_configuration():
    """Test hardware configuration loading."""
    logger = logging.getLogger(__name__)
    logger.info("Testing configuration...")
    
    try:
        import json
        config_path = Path(__file__).parent / "hardware_config.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verify key sections exist
        required_sections = ['laser_system', 'electron_accelerator', 'gamma_conversion', 'beam_monitoring']
        for section in required_sections:
            if section in config:
                logger.info(f"  {section}: OK")
            else:
                logger.error(f"  {section}: Missing")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration test failed: {e}")
        return False

def run_integration_test():
    """Run the complete integration test suite."""
    logger = setup_logging()
    
    # Set stdout encoding for Windows compatibility
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    logger.info("ELEMENTAL TRANSMUTATOR - INTEGRATION TEST")
    logger.info("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Physics Engine", test_physics_engine),
        ("Hardware Controllers", test_hardware_controllers),
        ("Energy Ledger", test_energy_ledger),
        ("Configuration", test_configuration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"{test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{test_name:<20}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 50)
    if all_passed:
        logger.info("ALL TESTS PASSED - SYSTEM READY")
        return 0
    else:
        logger.error("SOME TESTS FAILED - CHECK SYSTEM")
        return 1

if __name__ == "__main__":
    exit_code = run_integration_test()
    sys.exit(exit_code)
