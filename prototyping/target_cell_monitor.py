#!/usr/bin/env python3
"""
Target Cell Monitor
==================

Real-time monitoring and control of the Pb-208 target cell during
photonuclear transmutation. Handles temperature control, coolant flow,
material positioning, and transmutation progress tracking.

Key capabilities:
- Pb-208 target temperature and pressure monitoring
- Coolant circulation and thermal management
- Real-time neutron flux and gamma radiation monitoring
- Product isotope detection and quantification
- Material handling and positioning control
- Safety interlocks and emergency procedures
"""

import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import sqlite3

class TargetState(Enum):
    """Target cell operational states."""
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    IRRADIATING = "irradiating"
    COOLING = "cooling"
    UNLOADING = "unloading"
    MAINTENANCE = "maintenance"
    FAULT = "fault"

@dataclass
class TargetMetrics:
    """Real-time target cell metrics."""
    timestamp: float
    state: TargetState
    target_temp_c: float
    coolant_temp_in_c: float
    coolant_temp_out_c: float
    coolant_flow_lpm: float
    pressure_bar: float
    neutron_flux_n_cm2_s: float
    gamma_dose_rate_gy_h: float
    pb208_mass_g: float
    au197_activity_bq: float
    pb207_activity_bq: float
    conversion_percent: float
    irradiation_time_h: float

@dataclass
class IsotopeInventory:
    """Current isotope inventory in target cell."""
    pb208_atoms: float
    pb207_atoms: float
    au197_atoms: float
    tl203_atoms: float
    other_products: Dict[str, float]
    total_activity_bq: float
    heat_generation_w: float

class TargetCellMonitor:
    """
    Advanced target cell monitoring and control system.
    """
    
    def __init__(self, config: Dict):
        """Initialize the target cell monitor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.state = TargetState.EMPTY
        self.irradiation_start = None
        self.metrics_queue = queue.Queue()
        
        # Hardware interfaces (simulated for prototype)
        self.temperature_controller = None
        self.coolant_pump = None
        self.position_controller = None
        self.radiation_monitor = None
        self.spectrometer = None
          # Control parameters
        self.target_temp_c = config.get("target_temperature_c", 200)
        self.coolant_flow_lpm = config.get("coolant_flow_lpm", 5.0)
        self.max_pressure_bar = config.get("max_pressure_bar", 10.0)
        self.max_dose_rate = config.get("max_dose_rate_gy_h", 100.0)
        
        # Mock mode for fast CI testing
        self.mock_mode = config.get("mock_mode", False)
        if self.mock_mode:
            self.logger.info("Target cell monitor running in MOCK MODE")
        
        # Timing parameters (adjusted for mock mode)
        self.equilibrium_timeout = 2 if self.mock_mode else 300  # 2s vs 5min
        self.equilibrium_interval = 0.5 if self.mock_mode else 10  # 0.5s vs 10s intervals
        
        # Material tracking
        self.initial_pb208_g = 0.0
        self.current_inventory = IsotopeInventory(0, 0, 0, 0, {}, 0, 0)
        
        # Data logging
        self.db_path = config.get("database_path", "target_monitoring.db")
        self.init_database()
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger.info("TargetCellMonitor initialized")
    
    def init_database(self):
        """Initialize SQLite database for data logging."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS target_metrics (
                    timestamp REAL PRIMARY KEY,
                    state TEXT,
                    target_temp_c REAL,
                    coolant_temp_in_c REAL,
                    coolant_temp_out_c REAL,
                    coolant_flow_lpm REAL,
                    pressure_bar REAL,
                    neutron_flux REAL,
                    gamma_dose_rate REAL,
                    pb208_mass_g REAL,
                    au197_activity_bq REAL,
                    pb207_activity_bq REAL,
                    conversion_percent REAL,
                    irradiation_time_h REAL
                )
            """)
            
            # Create isotope inventory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS isotope_inventory (
                    timestamp REAL PRIMARY KEY,
                    pb208_atoms REAL,
                    pb207_atoms REAL,
                    au197_atoms REAL,
                    tl203_atoms REAL,
                    total_activity_bq REAL,
                    heat_generation_w REAL
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def load_target(self, pb208_mass_g: float) -> bool:
        """Load Pb-208 target material into the cell."""
        if self.state != TargetState.EMPTY:
            self.logger.error(f"Cannot load target in state: {self.state}")
            return False
        
        try:
            self.logger.info(f"Loading {pb208_mass_g:.3f} g of Pb-208 target...")
            self.state = TargetState.LOADING
            
            # Initialize material tracking
            self.initial_pb208_g = pb208_mass_g
            avogadro = 6.022e23
            pb208_molar_mass = 207.976627  # g/mol
            
            initial_pb208_atoms = (pb208_mass_g / pb208_molar_mass) * avogadro
            self.current_inventory = IsotopeInventory(
                pb208_atoms=initial_pb208_atoms,
                pb207_atoms=0,
                au197_atoms=0,
                tl203_atoms=0,
                other_products={},
                total_activity_bq=0,
                heat_generation_w=0
            )
            
            # Simulate target loading procedure
            time.sleep(5)  # Loading time
            
            # Verify target position
            if self._verify_target_position():
                self.state = TargetState.READY
                self.logger.info("Target loaded successfully")
                return True
            else:
                self.logger.error("Target position verification failed")
                self.state = TargetState.FAULT
                return False
                
        except Exception as e:
            self.logger.error(f"Target loading failed: {e}")
            self.state = TargetState.FAULT
            return False
    
    def start_cooling(self) -> bool:
        """Start coolant circulation and temperature control."""
        try:
            self.logger.info("Starting coolant circulation...")
            
            # Start coolant pump
            # self.coolant_pump.start(self.coolant_flow_lpm)
            
            # Set temperature controller
            # self.temperature_controller.set_target(self.target_temp_c)
              # Wait for thermal equilibrium
            self.logger.info("Waiting for thermal equilibrium...")
            equilibrium_time = 0
            max_wait = self.equilibrium_timeout
            check_interval = self.equilibrium_interval
            
            if self.mock_mode:
                self.logger.info(f"MOCK MODE: Fast thermal equilibrium ({max_wait}s timeout)")
            
            while equilibrium_time < max_wait:
                metrics = self._collect_metrics()
                temp_delta = abs(metrics.target_temp_c - self.target_temp_c)
                
                if temp_delta < 5.0:  # Within 5°C
                    self.logger.info(f"Thermal equilibrium reached: {metrics.target_temp_c:.1f}°C")
                    return True
                
                if self.mock_mode:
                    # Show progress in mock mode
                    progress = (equilibrium_time / max_wait) * 100
                    self.logger.info(f"MOCK: Thermal progress {progress:.0f}% ({equilibrium_time:.1f}s/{max_wait}s)")
                
                time.sleep(check_interval)
                equilibrium_time += check_interval
            
            self.logger.warning("Thermal equilibrium not reached within timeout")
            return True  # Continue anyway for prototype
            
        except Exception as e:
            self.logger.error(f"Cooling startup failed: {e}")
            return False
    
    def start_irradiation(self) -> bool:
        """Begin photonuclear irradiation monitoring."""
        if self.state != TargetState.READY:
            self.logger.error(f"Cannot start irradiation from state: {self.state}")
            return False
        
        try:
            self.logger.info("Starting irradiation monitoring...")
            self.state = TargetState.IRRADIATING
            self.irradiation_start = time.time()
            
            # Start monitoring thread
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.start()
            
            self.logger.info("Irradiation monitoring active")
            return True
            
        except Exception as e:
            self.logger.error(f"Irradiation start failed: {e}")
            self.state = TargetState.FAULT
            return False
    
    def stop_irradiation(self) -> bool:
        """Stop irradiation and begin cooling phase."""
        if self.state != TargetState.IRRADIATING:
            self.logger.warning(f"Stop irradiation called from state: {self.state}")
        
        try:
            self.logger.info("Stopping irradiation...")
            self.state = TargetState.COOLING
            
            # Calculate final results
            if self.irradiation_start:
                total_time_h = (time.time() - self.irradiation_start) / 3600
                self.logger.info(f"Total irradiation time: {total_time_h:.2f} hours")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Irradiation stop failed: {e}")
            return False
    
    def unload_target(self) -> Tuple[bool, Dict]:
        """Unload target and return final product inventory."""
        if self.state not in [TargetState.COOLING, TargetState.READY]:
            self.logger.error(f"Cannot unload target in state: {self.state}")
            return False, {}
        
        try:
            self.logger.info("Unloading target...")
            self.state = TargetState.UNLOADING
            
            # Stop monitoring
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join()
            
            # Final analysis
            final_analysis = self._perform_final_analysis()
            
            # Reset state
            self.state = TargetState.EMPTY
            self.irradiation_start = None
            self.initial_pb208_g = 0.0
            
            self.logger.info("Target unloaded successfully")
            return True, final_analysis
            
        except Exception as e:
            self.logger.error(f"Target unloading failed: {e}")
            self.state = TargetState.FAULT
            return False, {}
    
    def read_metrics(self) -> TargetMetrics:
        """Read current target cell metrics."""
        try:
            if not self.metrics_queue.empty():
                return self.metrics_queue.get_nowait()
            else:
                return self._collect_metrics()
                
        except Exception as e:
            self.logger.error(f"Failed to read metrics: {e}")
            return self._generate_fault_metrics()
    
    def get_status(self) -> Dict:
        """Get comprehensive target cell status."""
        metrics = self.read_metrics()
        
        status = {
            "state": self.state.value,
            "target_temperature_c": metrics.target_temp_c,
            "coolant_flow_lpm": metrics.coolant_flow_lpm,
            "pressure_bar": metrics.pressure_bar,
            "irradiation_time_h": metrics.irradiation_time_h,
            "conversion_percent": metrics.conversion_percent,
            "pb208_remaining_g": metrics.pb208_mass_g,
            "au197_activity_bq": metrics.au197_activity_bq,
            "neutron_flux": metrics.neutron_flux_n_cm2_s,
            "dose_rate_gy_h": metrics.gamma_dose_rate_gy_h,
            "thermal_power_w": self.current_inventory.heat_generation_w
        }
        
        return status
    
    def get_isotope_inventory(self) -> IsotopeInventory:
        """Get current isotope inventory."""
        return self.current_inventory
    
    def emergency_shutdown(self):
        """Emergency shutdown of target cell."""
        self.logger.critical("TARGET CELL EMERGENCY SHUTDOWN")
        self.state = TargetState.FAULT
        
        try:
            # Stop coolant circulation
            # self.coolant_pump.emergency_stop()
            
            # Activate emergency cooling
            # self.temperature_controller.emergency_cooling()
            
            # Stop monitoring
            self.monitoring_active = False
            
        except Exception as e:
            self.logger.critical(f"Emergency shutdown failed: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Update isotope inventory
                self._update_isotope_inventory()
                
                # Check safety limits
                if self._check_safety_limits(metrics):
                    self.emergency_shutdown()
                    break
                
                # Log to database
                self._log_metrics(metrics)
                
                # Store for real-time access
                self.metrics_queue.put(metrics)
                
                time.sleep(1)  # 1 Hz monitoring
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _collect_metrics(self) -> TargetMetrics:
        """Collect real-time metrics from sensors."""
        current_time = time.time()
        
        # Calculate irradiation time
        irradiation_time_h = 0
        if self.irradiation_start and self.state == TargetState.IRRADIATING:
            irradiation_time_h = (current_time - self.irradiation_start) / 3600
        
        # Simulate realistic target behavior during irradiation
        if self.state == TargetState.IRRADIATING:
            # Temperature rises during irradiation due to heat generation
            base_temp = self.target_temp_c + self.current_inventory.heat_generation_w * 0.1
            temp_noise = np.random.normal(0, 2)
            target_temp = base_temp + temp_noise
            
            # Neutron flux from photodisintegration
            neutron_flux = 1e10 + np.random.normal(0, 1e9)
            
            # Gamma dose rate from products
            dose_rate = 10.0 + self.current_inventory.total_activity_bq * 1e-6
            
        else:
            # Baseline conditions
            target_temp = 25.0 + np.random.normal(0, 1)
            neutron_flux = 1e6  # Background
            dose_rate = 0.1
        
        # Calculate conversion percentage
        initial_atoms = self.initial_pb208_g / 207.976627 * 6.022e23
        current_pb208_fraction = self.current_inventory.pb208_atoms / initial_atoms if initial_atoms > 0 else 1.0
        conversion_percent = (1.0 - current_pb208_fraction) * 100
        
        return TargetMetrics(
            timestamp=current_time,
            state=self.state,
            target_temp_c=target_temp,
            coolant_temp_in_c=25.0 + np.random.normal(0, 1),
            coolant_temp_out_c=target_temp - 10.0,
            coolant_flow_lpm=self.coolant_flow_lpm + np.random.normal(0, 0.1),
            pressure_bar=1.0 + np.random.normal(0, 0.1),
            neutron_flux_n_cm2_s=neutron_flux,
            gamma_dose_rate_gy_h=dose_rate,
            pb208_mass_g=self.current_inventory.pb208_atoms * 207.976627 / 6.022e23,
            au197_activity_bq=self.current_inventory.au197_atoms * 0.1,  # Assume 10% are excited
            pb207_activity_bq=self.current_inventory.pb207_atoms * 0.01,  # Some excited states
            conversion_percent=conversion_percent,
            irradiation_time_h=irradiation_time_h
        )
    
    def _update_isotope_inventory(self):
        """Update isotope inventory based on reaction kinetics."""
        if self.state != TargetState.IRRADIATING or not self.irradiation_start:
            return
        
        # Time step
        dt = 1.0  # 1 second
        
        # Reaction rates (simplified)
        gamma_flux = 1e13  # γ/cm²/s from beam
        
        # Pb-208(γ,n)Pb-207 reaction
        pb208_n_cross_section = 0.15e-24  # cm² (GDR peak)
        pb208_reaction_rate = gamma_flux * pb208_n_cross_section
        
        # Calculate reactions per second
        pb208_reactions_per_s = self.current_inventory.pb208_atoms * pb208_reaction_rate
        
        # Update atom counts
        self.current_inventory.pb208_atoms -= pb208_reactions_per_s * dt
        self.current_inventory.pb207_atoms += pb208_reactions_per_s * dt * 0.9  # 90% to Pb-207
        self.current_inventory.au197_atoms += pb208_reactions_per_s * dt * 0.05  # 5% to Au-197
        self.current_inventory.tl203_atoms += pb208_reactions_per_s * dt * 0.05  # 5% to Tl-203
        
        # Update activity (simplified)
        pb207_halflife_s = 22.6 * 365.25 * 24 * 3600  # 22.6 years
        au197_halflife_s = 6.18 * 24 * 3600  # 6.18 days
        
        pb207_decay_constant = 0.693 / pb207_halflife_s
        au197_decay_constant = 0.693 / au197_halflife_s
        
        self.current_inventory.total_activity_bq = (
            self.current_inventory.pb207_atoms * pb207_decay_constant +
            self.current_inventory.au197_atoms * au197_decay_constant
        )
        
        # Heat generation (simplified)
        self.current_inventory.heat_generation_w = self.current_inventory.total_activity_bq * 1e-6
    
    def _check_safety_limits(self, metrics: TargetMetrics) -> bool:
        """Check if any safety limits are exceeded."""
        # Temperature limit
        if metrics.target_temp_c > 500:
            self.logger.error(f"Temperature limit exceeded: {metrics.target_temp_c:.1f}°C")
            return True
        
        # Pressure limit
        if metrics.pressure_bar > self.max_pressure_bar:
            self.logger.error(f"Pressure limit exceeded: {metrics.pressure_bar:.1f} bar")
            return True
        
        # Dose rate limit
        if metrics.gamma_dose_rate_gy_h > self.max_dose_rate:
            self.logger.error(f"Dose rate limit exceeded: {metrics.gamma_dose_rate_gy_h:.1f} Gy/h")
            return True
        
        # Coolant flow minimum
        if metrics.coolant_flow_lpm < 1.0:
            self.logger.error(f"Coolant flow too low: {metrics.coolant_flow_lpm:.1f} L/min")
            return True
        
        return False
    
    def _log_metrics(self, metrics: TargetMetrics):
        """Log metrics to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO target_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                metrics.state.value,
                metrics.target_temp_c,
                metrics.coolant_temp_in_c,
                metrics.coolant_temp_out_c,
                metrics.coolant_flow_lpm,
                metrics.pressure_bar,
                metrics.neutron_flux_n_cm2_s,
                metrics.gamma_dose_rate_gy_h,
                metrics.pb208_mass_g,
                metrics.au197_activity_bq,
                metrics.pb207_activity_bq,
                metrics.conversion_percent,
                metrics.irradiation_time_h
            ))
            
            # Log isotope inventory
            cursor.execute("""
                INSERT INTO isotope_inventory VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                self.current_inventory.pb208_atoms,
                self.current_inventory.pb207_atoms,
                self.current_inventory.au197_atoms,
                self.current_inventory.tl203_atoms,
                self.current_inventory.total_activity_bq,
                self.current_inventory.heat_generation_w
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database logging failed: {e}")
    
    def _perform_final_analysis(self) -> Dict:
        """Perform final analysis of transmutation products."""
        try:
            # Calculate final yields
            avogadro = 6.022e23
            
            au197_mass_g = self.current_inventory.au197_atoms * 196.966569 / avogadro
            pb207_mass_g = self.current_inventory.pb207_atoms * 206.975897 / avogadro
            pb208_remaining_g = self.current_inventory.pb208_atoms * 207.976627 / avogadro
            
            conversion_efficiency = ((self.initial_pb208_g - pb208_remaining_g) / self.initial_pb208_g) * 100
            
            # Economic analysis
            au_price_per_g = 65.0  # USD
            pb_price_per_g = 0.002  # USD
            
            au_value = au197_mass_g * au_price_per_g
            pb_loss_cost = (self.initial_pb208_g - pb208_remaining_g) * pb_price_per_g
            net_value = au_value - pb_loss_cost
            
            return {
                "initial_pb208_g": self.initial_pb208_g,
                "final_pb208_g": pb208_remaining_g,
                "au197_produced_g": au197_mass_g,
                "pb207_produced_g": pb207_mass_g,
                "conversion_efficiency_percent": conversion_efficiency,
                "gold_value_usd": au_value,
                "net_value_usd": net_value,
                "total_activity_bq": self.current_inventory.total_activity_bq,
                "final_heat_generation_w": self.current_inventory.heat_generation_w
            }
            
        except Exception as e:
            self.logger.error(f"Final analysis failed: {e}")
            return {}
    
    def _verify_target_position(self) -> bool:
        """Verify target is properly positioned."""
        # In real implementation, check position sensors
        return True
    
    def _generate_fault_metrics(self) -> TargetMetrics:
        """Generate fault state metrics."""
        return TargetMetrics(
            timestamp=time.time(),
            state=TargetState.FAULT,
            target_temp_c=0,
            coolant_temp_in_c=0,
            coolant_temp_out_c=0,
            coolant_flow_lpm=0,
            pressure_bar=0,
            neutron_flux_n_cm2_s=0,
            gamma_dose_rate_gy_h=0,
            pb208_mass_g=0,
            au197_activity_bq=0,
            pb207_activity_bq=0,
            conversion_percent=0,
            irradiation_time_h=0
        )

    def cleanup(self) -> None:
        """Cleanup method for integration tests."""
        # Stop any active monitoring
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.join()
        # Close database connection
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load hardware configuration
    with open("hardware_config.json", "r") as f:
        config = json.load(f)
    
    # Initialize target monitor
    target = TargetCellMonitor(config["target_cell"])
    
    try:
        # Simulate target operation
        print("=== TARGET CELL OPERATION SEQUENCE ===")
        
        # Load target
        if target.load_target(10.0):  # 10g Pb-208
            print("✓ Target loaded: 10.0 g Pb-208")
            
            # Start cooling
            if target.start_cooling():
                print("✓ Cooling system active")
                
                # Start irradiation
                if target.start_irradiation():
                    print("✓ Irradiation monitoring started")
                    
                    # Monitor for 30 seconds
                    print("Monitoring for 30 seconds...")
                    for i in range(30):
                        time.sleep(1)
                        status = target.get_status()
                        if i % 5 == 0:  # Print every 5 seconds
                            print(f"  {i}s: {status['target_temperature_c']:.1f}°C, "
                                  f"{status['conversion_percent']:.3f}% converted, "
                                  f"{status['au197_activity_bq']:.0f} Bq Au-197")
                    
                    # Stop irradiation
                    target.stop_irradiation()
                    print("✓ Irradiation stopped")
                    
                    # Unload target
                    success, analysis = target.unload_target()
                    if success:
                        print("✓ Target unloaded")
                        print(f"Final analysis: {analysis}")
        
    except KeyboardInterrupt:
        print("\nEmergency shutdown!")
        target.emergency_shutdown()
    
    def cleanup(self) -> None:
        """Cleanup method for integration tests."""
        # Stop any active monitoring
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread:
            self.monitor_thread.join()
        # Close database connection
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
