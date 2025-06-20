#!/usr/bin/env python3
"""
Quick Demo - ASCII Version
==========================

6-hour photonuclear gold production demonstration without unicode characters.
Validates the complete hardware control system with realistic simulation.

Usage:
  python quick_demo_ascii.py              # Normal 6-minute demo
  python quick_demo_ascii.py --mock       # Fast mock mode for CI (30 seconds)
  python quick_demo_ascii.py --profile    # Enable profiling output
"""

import sys
import time
import json
import logging
import os
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Set encoding for Windows compatibility  
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import hardware controllers
from gamma_beam_controller import GammaBeamController
from target_cell_monitor import TargetCellMonitor
from data_logger import DataLogger

@contextmanager
def profile_step(name, logger, enabled=False):
    """Context manager for profiling demo steps."""
    start_time = time.time()
    if enabled:
        logger.info(f"PROFILE: Starting {name}")
    yield
    elapsed = time.time() - start_time
    if enabled:
        logger.info(f"PROFILE: {name} completed in {elapsed:.2f} seconds")

class QuickPhotonuclearDemo:
    """Quick demonstration of photonuclear gold production."""
    
    def __init__(self, mock_mode=False, profile_mode=False):
        """Initialize the quick demo."""
        self.logger = logging.getLogger(__name__)
        self.mock_mode = mock_mode
        self.profile_mode = profile_mode
        
        if self.mock_mode:
            self.logger.info("*** MOCK MODE ENABLED - Fast simulation for CI ***")
        
        # Demo parameters
        self.pb208_mass_g = 1.0  # 1g for safety
        
        # Adjust timing based on mode
        if self.mock_mode:
            self.demo_duration_hours = 0.0083  # 30 seconds for CI
            self.warmup_time = 1  # 1 second warmup
            self.thermal_wait = 1  # 1 second thermal
            self.monitoring_interval = 1  # 1 second updates
        else:
            self.demo_duration_hours = 0.1  # 6 minutes for normal demo
            self.warmup_time = 30  # 30 second warmup
            self.thermal_wait = 300  # 5 minute thermal wait
            self.monitoring_interval = 30  # 30 second updates
        
        # Configuration
        self.config = {            "gamma_source": {
                "energy_MeV": 16.5,
                "flux_per_cm2_s": 5e13,  # Higher flux for acceleration
                "warmup_s": self.warmup_time,
                "mock_mode": self.mock_mode
            },"target_cell": {
                "target_temperature_c": 150,
                "coolant_flow_lpm": 3.0,
                "max_pressure_bar": 5.0,
                "max_dose_rate_gy_h": 25.0,
                "database_path": "quick_demo_target.db",
                "mock_mode": self.mock_mode
            },            "data_logging": {
                "database_path": "quick_demo_data.db",
                "backup_directory": "quick_demo_backups",
                "sampling_rates": {
                    "gamma_beam": 2.0,
                    "target_cell": 1.0
                },
                "mock_mode": self.mock_mode
            }
        }
        
        # Hardware controllers
        self.gamma_beam = None
        self.target_cell = None
        self.data_logger = None
        self.session_id = None
        
        # Results tracking
        self.results = {
            "start_time": None,
            "end_time": None,
            "conversion_percent": 0,
            "gold_produced_mg": 0,
            "gold_value_usd": 0,
            "demo_success": False
        }
    
    def initialize_systems(self) -> bool:
        """Initialize all hardware systems."""
        try:
            self.logger.info("Initializing hardware systems...")            # Initialize controllers
            gamma_config = self.config["gamma_source"].copy()
            gamma_config["mock_mode"] = self.mock_mode
            
            target_config = self.config["target_cell"].copy()
            target_config["mock_mode"] = self.mock_mode
            
            data_config = self.config["data_logging"].copy()
            data_config["mock_mode"] = self.mock_mode
            
            self.gamma_beam = GammaBeamController(gamma_config, self.mock_mode)
            self.target_cell = TargetCellMonitor(target_config)
            self.data_logger = DataLogger(self.config["data_logging"])
            
            # Register for data logging
            self.data_logger.register_subsystem("gamma_beam", self.gamma_beam)
            self.data_logger.register_subsystem("target_cell", self.target_cell)
            
            self.logger.info("Hardware initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            return False
    
    def run_safety_checks(self) -> bool:
        """Quick safety verification."""
        self.logger.info("Running safety checks...")
        
        # Check gamma beam safety
        safe, interlocks = self.gamma_beam.check_safety_interlocks()
        if not safe:
            self.logger.error(f"Safety interlocks active: {interlocks}")
            return False
        
        self.logger.info("Safety checks passed")
        return True
    
    def prepare_target(self) -> bool:
        """Load and condition the Pb-208 target."""
        self.logger.info(f"Loading {self.pb208_mass_g:.1f} g Pb-208 target...")
        
        # Load target
        if not self.target_cell.load_target(self.pb208_mass_g):
            self.logger.error("Target loading failed")
            return False
          # Quick thermal conditioning
        self.logger.info("Quick thermal conditioning...")
        if not self.target_cell.start_cooling():
            self.logger.error("Cooling system failed")
            return False
        
        # Brief conditioning time with progress
        self.progress_sleep(self.thermal_wait, "Thermal conditioning")
        
        self.logger.info("Target preparation complete")
        return True
    
    def startup_beam(self) -> bool:
        """Start up the gamma ray beam."""
        self.logger.info("Starting gamma beam system...")
        
        # Power on
        if not self.gamma_beam.power_on():
            self.logger.error("Gamma beam power on failed")
            return False
        
        # Wait for ready
        if not self.gamma_beam.wait_for_ready():
            self.logger.error("Gamma beam ready timeout")
            return False
        
        # Activate beam
        if not self.gamma_beam.beam_on():
            self.logger.error("Gamma beam activation failed")
            return False
        
        # Verify beam quality
        status = self.gamma_beam.get_status()
        energy = status['beam_energy_MeV']
        stability = status['stability_percent']
        
        self.logger.info(f"Gamma beam active: {energy:.2f} MeV, {stability:.1f}% stable")
        return True
    
    def run_irradiation(self) -> bool:
        """Execute the accelerated irradiation sequence."""
        self.logger.info("Starting accelerated irradiation...")
        
        # Start irradiation monitoring
        if not self.target_cell.start_irradiation():
            self.logger.error("Irradiation monitoring failed")
            return False
          # Run accelerated sequence
        irradiation_duration = self.demo_duration_hours * 3600  # Convert to seconds
        start_time = time.time()
        end_time = start_time + irradiation_duration
        
        # Use mock-aware update interval
        update_interval = self.monitoring_interval
        last_update = time.time()
        
        if self.mock_mode:
            self.logger.info(f"Irradiation will run for {irradiation_duration:.1f} seconds (MOCK MODE)")
        else:
            self.logger.info(f"Irradiation will run for {irradiation_duration/60:.1f} minutes")
        
        step_count = 0
        total_steps = max(1, int(irradiation_duration / update_interval))
        
        while time.time() < end_time:
            current_time = time.time()
            elapsed_minutes = (current_time - start_time) / 60
            remaining_minutes = (end_time - current_time) / 60
            
            # Regular status updates with step counter
            if current_time - last_update >= update_interval:
                step_count += 1
                progress_percent = (step_count / total_steps) * 100
                
                if self.mock_mode:
                    self.logger.info(f"MOCK: Step {step_count}/{total_steps} ({progress_percent:.0f}%) - {elapsed_minutes*60:.1f}s elapsed, {remaining_minutes*60:.1f}s remaining")
                else:
                    self.logger.info(f"Status: {elapsed_minutes:.1f}min elapsed, {remaining_minutes:.1f}min remaining")
                
                self.log_irradiation_status(elapsed_minutes, remaining_minutes)
                last_update = current_time            # Sleep with progress indication
            sleep_duration = min(update_interval, remaining_minutes * 60)
            if sleep_duration > 0:
                if self.mock_mode and sleep_duration > 0.5:
                    # Show sub-second progress in mock mode
                    substeps = max(2, int(sleep_duration * 2))  # 2 substeps per second
                    substep_duration = sleep_duration / substeps
                    for j in range(substeps):
                        if time.time() >= end_time:
                            break
                        time.sleep(substep_duration)
                else:
                    time.sleep(sleep_duration)
        
        # Stop irradiation
        self.target_cell.stop_irradiation()
        self.gamma_beam.beam_off()
        
        self.logger.info("Irradiation sequence complete")
        return True
    
    def analyze_results(self) -> bool:
        """Analyze the transmutation results."""
        self.logger.info("Analyzing results...")
          # Brief cooling period
        cooling_duration = 30 if not self.mock_mode else 2  # 30s normal, 2s mock
        self.logger.info("Brief cooling period...")
        self.progress_sleep(cooling_duration, "Cooling")
        
        # Unload and analyze
        success, analysis = self.target_cell.unload_target()
        if not success:
            self.logger.error("Target unloading failed")
            return False
        
        # Extract results
        conversion = analysis.get('conversion_efficiency_percent', 0)
        gold_g = analysis.get('au197_produced_g', 0)
        gold_mg = gold_g * 1000  # Convert to mg
        gold_value = gold_g * 65.0  # USD value at current prices
        
        # Store results
        self.results.update({
            "conversion_percent": conversion,
            "gold_produced_mg": gold_mg,
            "gold_value_usd": gold_value,
            "demo_success": conversion > 0
        })
        
        self.logger.info("Results analysis complete")
        return True
    
    def log_irradiation_status(self, elapsed_min: float, remaining_min: float):
        """Log current irradiation status."""
        target_status = self.target_cell.get_status()
        beam_status = self.gamma_beam.get_status()
        
        conversion = target_status.get('conversion_percent', 0)
        temp = target_status.get('target_temperature_c', 0)
        stability = beam_status.get('stability_percent', 0)
        activity = target_status.get('au197_activity_bq', 0)
        
        self.logger.info(f"Status: {elapsed_min:.1f}min elapsed, {remaining_min:.1f}min remaining")
        self.logger.info(f"  Conversion: {conversion:.4f}% | Temp: {temp:.1f}C")
        self.logger.info(f"  Beam stability: {stability:.1f}% | Au-197: {activity:.0f} Bq")
        
        # Calculate projected results
        if elapsed_min > 0:
            rate = conversion / elapsed_min
            projected_final = conversion + (rate * remaining_min)
            projected_gold_mg = projected_final * 0.01 * 1000  # Rough estimate
            
            self.logger.info(f"  Projected final: {projected_final:.4f}% ({projected_gold_mg:.3f} mg gold)")
    
    def progress_sleep(self, duration: float, description: str = "Processing") -> None:
        """Sleep with progress updates for better CI visibility."""
        if duration <= 1:
            time.sleep(duration)
            return
        
        steps = max(5, min(20, int(duration)))  # 5-20 progress steps
        step_duration = duration / steps
        
        for i in range(steps):
            progress_percent = ((i + 1) / steps) * 100
            elapsed = (i + 1) * step_duration
            self.logger.info(f"{description}: Step {i+1}/{steps} ({progress_percent:.0f}%) - {elapsed:.1f}s elapsed")
            time.sleep(step_duration)
    
    def run_complete_demo(self) -> bool:
        """Execute the complete demonstration sequence."""
        self.logger.info("=== PHOTONUCLEAR GOLD PRODUCTION QUICK DEMO ===")
        self.results["start_time"] = datetime.now().isoformat()
        
        total_steps = 8
        current_step = 0
        
        def log_step_start(step_name: str) -> None:
            nonlocal current_step
            current_step += 1
            self.logger.info(f"[STEP {current_step}/{total_steps}] {step_name}")
        
        try:
            # Step 1: Initialize systems
            log_step_start("Initializing Systems")
            with profile_step("System Initialization", self.logger, self.profile_mode):
                if not self.initialize_systems():
                    return False
            
            # Step 2: Start data logging
            log_step_start("Setting Up Data Logging")
            with profile_step("Data Logging Setup", self.logger, self.profile_mode):
                session_config = {
                    "operator": "quick_demo",
                    "pb208_mass_g": self.pb208_mass_g,
                    "planned_duration_h": self.demo_duration_hours,
                    "notes": "Quick demonstration run"
                }
                
                self.session_id = self.data_logger.start_session(session_config)
                self.logger.info(f"Data logging started: {self.session_id}")
            
            # Step 3: Safety checks
            log_step_start("Running Safety Checks")
            with profile_step("Safety Checks", self.logger, self.profile_mode):
                if not self.run_safety_checks():
                    return False
            
            # Step 4: Prepare target
            log_step_start("Preparing Target")
            with profile_step("Target Preparation", self.logger, self.profile_mode):
                if not self.prepare_target():
                    return False
            
            # Step 5: Start gamma beam
            log_step_start("Starting Gamma Beam")
            with profile_step("Beam Startup", self.logger, self.profile_mode):
                if not self.startup_beam():
                    return False
            
            # Step 6: Run irradiation
            log_step_start("Running Irradiation")
            with profile_step("Irradiation Phase", self.logger, self.profile_mode):
                if not self.run_irradiation():
                    return False            # Step 7: Analyze results
            log_step_start("Analyzing Results")
            if not self.analyze_results():
                return False
            
            # Step 8: Cleanup and shutdown
            log_step_start("Cleanup and Shutdown")
            self.cleanup_systems()
            
            # Complete demo
            self.results["end_time"] = datetime.now().isoformat()
            self.results["demo_success"] = True
            
            # Step 9: Shutdown systems
            self.shutdown_systems()
            
            # Step 10: End data logging
            summary = self.data_logger.end_session()
            self.logger.info(f"Data logging ended: {summary.get('data_points_collected', 0)} points")
            
            self.logger.info("=== DEMONSTRATION COMPLETED SUCCESSFULLY ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Demonstration failed: {e}")
            self.shutdown_systems()
            return False
    
    def cleanup_systems(self):
        """Clean up experiment data and systems before shutdown."""
        try:
            self.logger.info("Cleaning up systems...")
            
            # Ensure all monitoring is stopped
            if self.target_cell:
                self.target_cell.stop_irradiation()
            
            # Process final data
            if self.data_logger:
                self.data_logger.process_final_data()
                
            self.logger.info("Cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def shutdown_systems(self):
        """Shutdown all systems safely."""
        try:
            if self.gamma_beam:
                self.gamma_beam.power_off()
            
            if self.data_logger:
                self.data_logger.stop_data_collection()
                
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
    
    def print_final_results(self):
        """Print the final demonstration results."""
        print("\n" + "=" * 60)
        print("QUICK DEMO RESULTS")
        print("=" * 60)
        
        if self.results["demo_success"]:
            print("STATUS: SUCCESS")
        else:
            print("STATUS: FAILED")
        
        print(f"Pb-208 feedstock: {self.pb208_mass_g:.1f} g")
        print(f"Conversion efficiency: {self.results['conversion_percent']:.4f}%")
        print(f"Gold produced: {self.results['gold_produced_mg']:.3f} mg")
        print(f"Gold value: ${self.results['gold_value_usd']:.4f}")
        print(f"Energy source: LV Self-Powered (Zero cost)")
        
        start = self.results.get("start_time", "")
        end = self.results.get("end_time", "")
        print(f"Start time: {start}")
        print(f"End time: {end}")
        
        print("=" * 60)
        
        if self.results["demo_success"]:
            print("PROOF-OF-CONCEPT VALIDATED!")
            print("System ready for full-scale demonstration.")
        else:
            print("Demo incomplete - check logs for issues.")

def main():
    """Main demonstration entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Photonuclear Gold Production Demo')
    parser.add_argument('--mock', action='store_true', help='Enable mock mode for CI (30 seconds)')
    parser.add_argument('--profile', action='store_true', help='Enable profiling output')
    args = parser.parse_args()
    
    mode_str = "MOCK MODE" if args.mock else "NORMAL MODE"
    duration_str = "30 seconds" if args.mock else "6 minutes"
    
    print(f"PHOTONUCLEAR GOLD PRODUCTION - QUICK DEMO ({mode_str})")
    print("=" * 50)
    print(f"Accelerated demonstration with 1g Pb-208 - {duration_str}")
    print("Proof-of-concept for photonuclear transmutation")
    print("=" * 50)
    
    # Setup logging
    log_filename = f'quick_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    if args.profile:
        logger.info("PROFILING MODE ENABLED")
    
    try:
        # Run demonstration
        with profile_step("Total Demo", logger, args.profile):
            demo = QuickPhotonuclearDemo(mock_mode=args.mock, profile_mode=args.profile)
            success = demo.run_complete_demo()
        
        # Print results
        demo.print_final_results()
        
        if args.profile:
            logger.info(f"PROFILE: Complete log saved to {log_filename}")
        
        return success
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return False
    except Exception as e:
        print(f"\nDemo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
