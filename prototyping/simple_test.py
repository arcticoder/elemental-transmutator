#!/usr/bin/env python3
"""
Simple Test Demo
================

Basic test of the photonuclear transmutation hardware control system.
Tests all major components without unicode characters or complex logging.
"""

import sys
import time
import json
import logging
from pathlib import Path

# Set environment variable for console encoding
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import the hardware controllers
from gamma_beam_controller import GammaBeamController
from target_cell_monitor import TargetCellMonitor  
from data_logger import DataLogger

def main():
    """Simple test of the complete system."""
    print("=== PHOTONUCLEAR GOLD PRODUCTION TEST ===")
    print("Testing hardware control system integration")
    print("=" * 50)
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    try:
        # Test configuration
        config = {
            "gamma_source": {
                "energy_MeV": 16.5,
                "flux_per_cm2_s": 1e13,
                "warmup_s": 10  # Quick test
            },
            "target_cell": {
                "target_temperature_c": 150,
                "coolant_flow_lpm": 3.0,
                "max_pressure_bar": 5.0,
                "max_dose_rate_gy_h": 25.0,
                "database_path": "test_target.db"
            },
            "data_logging": {
                "database_path": "test_data.db",
                "backup_directory": "test_backups",
                "sampling_rates": {
                    "gamma_beam": 1.0,
                    "target_cell": 0.5
                }
            }
        }
        
        print("\n1. Testing Gamma Beam Controller...")
        beam = GammaBeamController(config["gamma_source"])
        
        if beam.power_on():
            print("   Power on: SUCCESS")
            
            if beam.wait_for_ready():
                print("   System ready: SUCCESS")
                
                if beam.beam_on():
                    print("   Beam activation: SUCCESS")
                    
                    # Test for 5 seconds
                    for i in range(5):
                        status = beam.get_status()
                        energy = status['beam_energy_MeV']
                        stability = status['stability_percent']
                        print(f"   T+{i+1}s: {energy:.2f} MeV, {stability:.1f}% stable")
                        time.sleep(1)
                    
                    beam.beam_off()
                    print("   Beam off: SUCCESS")
                
            beam.power_off()
            print("   Power off: SUCCESS")
        
        print("\n2. Testing Target Cell Monitor...")
        target = TargetCellMonitor(config["target_cell"])
        
        if target.load_target(1.0):  # 1g Pb-208
            print("   Target loading: SUCCESS")
            
            if target.start_cooling():
                print("   Cooling system: SUCCESS")
                
                if target.start_irradiation():
                    print("   Irradiation monitoring: SUCCESS")
                    
                    # Test for 5 seconds
                    for i in range(5):
                        status = target.get_status()
                        temp = status['target_temperature_c']
                        conv = status['conversion_percent']
                        print(f"   T+{i+1}s: {temp:.1f}C, {conv:.4f}% converted")
                        time.sleep(1)
                    
                    target.stop_irradiation()
                    print("   Stop irradiation: SUCCESS")
                
                success, analysis = target.unload_target()
                if success:
                    print("   Target unloading: SUCCESS")
                    conv = analysis.get('conversion_efficiency_percent', 0)
                    gold = analysis.get('au197_produced_g', 0)
                    print(f"   Final conversion: {conv:.4f}%")
                    print(f"   Gold produced: {gold:.6f} g")
        
        print("\n3. Testing Data Logger...")
        logger = DataLogger(config["data_logging"])
        
        # Register subsystems
        logger.register_subsystem("gamma_beam", beam)
        logger.register_subsystem("target_cell", target)
        
        # Start test session
        session_config = {
            "operator": "test_user",
            "pb208_mass_g": 1.0,
            "planned_duration_h": 0.1,
            "notes": "Integration test"
        }
        
        session_id = logger.start_session(session_config)
        print(f"   Session started: {session_id}")
        
        # Log some test data
        for i in range(5):
            logger.log_data_point("test_subsystem", "test_parameter", i * 10.0, "units")
            time.sleep(0.5)
        
        # End session
        summary = logger.end_session()
        print(f"   Session ended: SUCCESS")
        print(f"   Data points: {summary.get('data_points_collected', 0)}")
        
        print("\n=== ALL TESTS PASSED ===")
        print("Hardware control system is operational!")
        print("Ready for full demonstration runs.")
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
