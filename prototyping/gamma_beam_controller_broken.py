#!/usr/bin/env python3
"""
Gamma Beam Controller
====================

Hardware control interface for the inverse Compton scattering γ-ray source.
Manages beam generation, focusing, monitoring, and safety interlocks for
the photonuclear transmutation experiment.

Key capabilities:
- Inverse Compton scattering beam generation (16.5 MeV)
- Real-time flux monitoring and control
- Safety interlock integration
- Beam stability feedback control
- Power management and thermal monitoring
"""

import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue

class BeamState(Enum):
    """Gamma beam operational states."""
    OFF = "off"
    WARMING_UP = "warming_up"
    READY = "ready"
    ON = "beam_on"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class BeamMetrics:
    """Real-time beam performance metrics."""
    timestamp: float
    state: BeamState
    energy_MeV: float
    flux_per_cm2_s: float
    beam_current_ua: float
    laser_power_kw: float
    electron_current_ma: float
    target_temperature_c: float
    dose_rate_usv_h: float
    stability_percent: float
    uptime_hours: float

class GammaBeamController:
    """
    Advanced gamma-ray beam controller for photonuclear transmutation.
    """
    
    def __init__(self, config: Dict):
        """Initialize the gamma beam controller."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Beam state management
        self.state = BeamState.OFF
        self.start_time = None
        self.metrics_queue = queue.Queue()
        self.safety_interlocks = []
        
        # Hardware interfaces (simulated for prototype)
        self.laser_controller = None
        self.electron_gun = None
        self.beam_monitor = None
        self.safety_system = None
        
        # Control parameters
        self.target_energy = config.get("energy_MeV", 16.5)
        self.target_flux = config.get("flux_per_cm2_s", 1e13)
        self.warmup_time = config.get("warmup_s", 3600)
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger.info(f"GammaBeamController initialized for {self.target_energy:.1f} MeV beam")
    
    def initialize_hardware(self) -> bool:
        """Initialize all hardware components."""
        try:
            # Initialize laser system
            self.logger.info("Initializing 800nm laser system...")
            # self.laser_controller = LaserController(self.config["laser"])
            
            # Initialize electron gun
            self.logger.info("Initializing 100 MeV electron gun...")
            # self.electron_gun = ElectronGun(self.config["electron_beam"])
            
            # Initialize beam monitoring
            self.logger.info("Initializing beam monitoring systems...")
            # self.beam_monitor = BeamMonitor(self.config["monitoring"])
            
            # Initialize safety systems
            self.logger.info("Initializing safety interlocks...")
            # self.safety_system = SafetySystem(self.config["safety"])
            
            self.logger.info("Hardware initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            self.state = BeamState.FAULT
            return False
    
    def check_safety_interlocks(self) -> Tuple[bool, List[str]]:
        """Check all safety interlocks before beam operation."""
        active_interlocks = []
        
        # Check personnel safety
        if not self._check_personnel_clear():
            active_interlocks.append("personnel_in_area")
        
        # Check shielding position
        if not self._check_shielding_closed():
            active_interlocks.append("shielding_open")
        
        # Check target readiness
        if not self._check_target_ready():
            active_interlocks.append("target_not_ready")
        
        # Check cooling systems
        if not self._check_cooling_active():
            active_interlocks.append("cooling_fault")
        
        # Check dose rate monitors
        if not self._check_radiation_monitors():
            active_interlocks.append("radiation_monitor_fault")
        
        # Check emergency stops
        if not self._check_emergency_stops():
            active_interlocks.append("emergency_stop_active")
        
        return len(active_interlocks) == 0, active_interlocks
    
    def power_on(self) -> bool:
        """Power on the gamma beam system with full safety checks."""
        self.logger.info("Powering on gamma beam system...")
        
        if self.state != BeamState.OFF:
            self.logger.warning(f"Cannot power on from state: {self.state}")
            return False
        
        # Check safety interlocks
        safe, interlocks = self.check_safety_interlocks()
        if not safe:
            self.logger.error(f"Safety interlocks active: {interlocks}")
            return False
        
        # Initialize hardware
        if not self.initialize_hardware():
            return False
        
        # Begin warmup sequence
        self.state = BeamState.WARMING_UP
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        
        self.logger.info(f"System warming up... ETA: {self.warmup_time/60:.1f} minutes")
        return True
    
    def wait_for_ready(self) -> bool:
        """Wait for the system to complete warmup and be ready for beam."""
        if self.state != BeamState.WARMING_UP:
            return self.state == BeamState.READY
        
        warmup_start = time.time()
        while time.time() - warmup_start < self.warmup_time:
            time.sleep(10)  # Check every 10 seconds
            
            # Check for faults during warmup
            if self.state == BeamState.FAULT:
                self.logger.error("Fault detected during warmup")
                return False
            
            # Update progress
            elapsed = time.time() - warmup_start
            progress = (elapsed / self.warmup_time) * 100
            self.logger.info(f"Warmup progress: {progress:.1f}%")
        
        # Warmup complete
        self.state = BeamState.READY
        self.logger.info("System ready for beam operation")
        return True
      def beam_on(self) -> bool:
        """Turn on the gamma ray beam."""
        if self.state != BeamState.READY:
            self.logger.error(f"Cannot activate beam from state: {self.state}")
            return False
        
        # Final safety check
        safe, interlocks = self.check_safety_interlocks()
        if not safe:
            self.logger.error(f"Cannot activate beam - interlocks: {interlocks}")
            return False
        
        try:
            # Ramp up laser power
            self.logger.info("Ramping laser to full power...")
            # self.laser_controller.set_power(self.config["laser_power_kw"])
            
            # Activate electron beam
            self.logger.info("Activating electron beam...")
            # self.electron_gun.beam_on()
            
            # Monitor beam formation
            self.logger.info("Monitoring beam formation...")
            time.sleep(5)  # Allow beam to stabilize
            
            # Verify beam parameters
            if self._verify_beam_parameters():
                self.state = BeamState.ON
                self.logger.info(f"Gamma beam ON: {self.target_energy:.1f} MeV, {self.target_flux:.2e} γ/cm²/s")
                return True
            else:
                self.logger.error("Beam parameters out of tolerance")
                self.beam_off()
                return False
                
        except Exception as e:
            self.logger.error(f"Beam activation failed: {e}")
            self.state = BeamState.FAULT
            self.emergency_stop()
            return False
    
    def beam_off(self) -> bool:
        """Turn off the gamma ray beam safely."""
        self.logger.info("Turning off gamma beam...")
        
        try:
            # Ramp down laser power
            # self.laser_controller.set_power(0)
            
            # Turn off electron beam
            # self.electron_gun.beam_off()
            
            # Wait for radiation to decay
            time.sleep(10)
            
            self.state = BeamState.READY
            self.logger.info("Gamma beam OFF")
            return True
            
        except Exception as e:
            self.logger.error(f"Beam shutdown failed: {e}")
            self.emergency_stop()
            return False
    
    def pulse(self, energy_mj: float, rate_hz: float) -> bool:
        """Generate pulsed gamma ray beam."""
        if self.state != BeamState.ON:
            return False
        
        try:
            # Modulate laser for pulsed operation
            pulse_duration = 1.0 / rate_hz
            
            # Calculate required laser parameters
            laser_energy = energy_mj / self.config.get("conversion_efficiency", 0.01)
            
            # Execute pulse
            # self.laser_controller.pulse(laser_energy, pulse_duration)
            
            return True
              except Exception as e:
            self.logger.error(f"Pulse generation failed: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency shutdown of all beam systems."""
        self.logger.critical("EMERGENCY STOP ACTIVATED")
        self.state = BeamState.EMERGENCY_STOP
        
        try:
            # Immediate laser shutdown
            # self.laser_controller.emergency_stop()
            
            # Immediate electron beam shutdown
            # self.electron_gun.emergency_stop()
            
            # Activate beam dumps
            # self.safety_system.activate_beam_dumps()
            
        except Exception as e:
            self.logger.critical(f"Emergency stop failed: {e}")
    
    def power_off(self) -> bool:
        """Complete system shutdown."""
        self.logger.info("Powering down gamma beam system...")
        
        # Stop beam if active
        if self.state == BeamState.ON:
            self.beam_off()
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Shutdown hardware
        try:
            # self.laser_controller.shutdown()
            # self.electron_gun.shutdown()
            # self.beam_monitor.shutdown()
            
            self.state = BeamState.OFF
            self.logger.info("System powered down successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Shutdown failed: {e}")
            return False
    
    def read_metrics(self) -> BeamMetrics:
        """Read current beam performance metrics."""
        try:
            # Get latest metrics from queue
            if not self.metrics_queue.empty():
                return self.metrics_queue.get_nowait()
            else:
                # Return current state if no new metrics
                return self._generate_current_metrics()
                
        except Exception as e:
            self.logger.error(f"Failed to read metrics: {e}")
            return self._generate_fault_metrics()
    
    def get_status(self) -> Dict:
        """Get comprehensive system status."""
        metrics = self.read_metrics()
        safe, interlocks = self.check_safety_interlocks()
        
        return {
            "state": self.state.value,
            "uptime_hours": metrics.uptime_hours,
            "beam_energy_MeV": metrics.energy_MeV,
            "beam_flux": metrics.flux_per_cm2_s,
            "stability_percent": metrics.stability_percent,
            "dose_rate_usv_h": metrics.dose_rate_usv_h,
            "safety_status": "OK" if safe else "INTERLOCKED",
            "active_interlocks": interlocks,
            "target_temperature_c": metrics.target_temperature_c
        }
    
    def _monitoring_loop(self):
        """Background monitoring loop for beam parameters."""
        while self.monitoring_active:
            try:
                # Collect metrics from hardware
                metrics = self._collect_metrics()
                
                # Check for faults
                if self._check_for_faults(metrics):
                    self.logger.warning("Fault detected in beam monitoring")
                    self.state = BeamState.FAULT
                
                # Store metrics
                self.metrics_queue.put(metrics)
                
                time.sleep(1)  # 1 Hz monitoring
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _collect_metrics(self) -> BeamMetrics:
        """Collect real-time metrics from hardware."""
        current_time = time.time()
        uptime = (current_time - self.start_time) / 3600 if self.start_time else 0
        
        # Simulate realistic beam parameters with some noise
        base_energy = self.target_energy
        base_flux = self.target_flux
        
        # Add realistic variations
        energy_noise = np.random.normal(0, 0.1)  # ±0.1 MeV stability
        flux_noise = np.random.normal(0, 0.02)   # ±2% flux stability
        
        return BeamMetrics(
            timestamp=current_time,
            state=self.state,
            energy_MeV=base_energy + energy_noise,
            flux_per_cm2_s=base_flux * (1 + flux_noise),
            beam_current_ua=50.0 + np.random.normal(0, 2),
            laser_power_kw=1.0 + np.random.normal(0, 0.05),
            electron_current_ma=10.0 + np.random.normal(0, 0.5),
            target_temperature_c=85.0 + np.random.normal(0, 5),
            dose_rate_usv_h=0.5 + np.random.normal(0, 0.1),
            stability_percent=98.0 + np.random.normal(0, 1),
            uptime_hours=uptime
        )
    
    def _verify_beam_parameters(self) -> bool:
        """Verify beam is within acceptable parameters."""
        metrics = self.read_metrics()
        
        # Energy tolerance: ±5%
        energy_ok = abs(metrics.energy_MeV - self.target_energy) / self.target_energy < 0.05
        
        # Flux tolerance: ±10%
        flux_ok = abs(metrics.flux_per_cm2_s - self.target_flux) / self.target_flux < 0.10
        
        # Stability requirement: >95%
        stability_ok = metrics.stability_percent > 95.0
        
        return energy_ok and flux_ok and stability_ok
    
    def _check_for_faults(self, metrics: BeamMetrics) -> bool:
        """Check metrics for fault conditions."""
        # Temperature fault
        if metrics.target_temperature_c > 150:
            return True
        
        # Dose rate fault
        if metrics.dose_rate_usv_h > 10:
            return True
        
        # Beam stability fault
        if metrics.stability_percent < 90:
            return True
        
        return False
    
    def _generate_current_metrics(self) -> BeamMetrics:
        """Generate metrics for current state."""
        return self._collect_metrics()
    
    def _generate_fault_metrics(self) -> BeamMetrics:
        """Generate fault state metrics."""
        return BeamMetrics(
            timestamp=time.time(),
            state=BeamState.FAULT,
            energy_MeV=0,
            flux_per_cm2_s=0,
            beam_current_ua=0,
            laser_power_kw=0,
            electron_current_ma=0,
            target_temperature_c=25,
            dose_rate_usv_h=0,
            stability_percent=0,
            uptime_hours=0
        )
    
    # Safety interlock check methods (simulated)
    def _check_personnel_clear(self) -> bool:
        """Check that all personnel are clear of radiation area."""
        # In real implementation, check motion detectors, key card system
        return True
    
    def _check_shielding_closed(self) -> bool:
        """Check that radiation shielding is properly positioned."""
        # In real implementation, check shielding position sensors
        return True
    
    def _check_target_ready(self) -> bool:
        """Check that target is properly positioned and cooled."""
        # In real implementation, check target position and cooling flow
        return True
    
    def _check_cooling_active(self) -> bool:
        """Check that cooling systems are operational."""
        # In real implementation, check coolant flow and temperature
        return True
    
    def _check_radiation_monitors(self) -> bool:
        """Check that radiation monitoring systems are functional."""
        # In real implementation, check detector status and calibration
        return True
    
    def _check_emergency_stops(self) -> bool:
        """Check that no emergency stops are active."""
        # In real implementation, check all emergency stop circuits
        return True

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load hardware configuration
    with open("hardware_config.json", "r") as f:
        config = json.load(f)
    
    # Initialize beam controller
    beam = GammaBeamController(config["gamma_source"])
    
    try:
        # Startup sequence
        print("=== GAMMA BEAM STARTUP SEQUENCE ===")
        if beam.power_on():
            print("✓ Power on successful")
            
            if beam.wait_for_ready():
                print("✓ System ready")
                
                if beam.beam_on():
                    print("✓ Beam active")
                    
                    # Run for 10 seconds with monitoring
                    print("Running beam for 10 seconds...")
                    for i in range(10):
                        time.sleep(1)
                        status = beam.get_status()
                        print(f"  {i+1}s: {status['beam_energy_MeV']:.2f} MeV, "
                              f"{status['beam_flux']:.2e} γ/cm²/s, "
                              f"{status['stability_percent']:.1f}% stable")
                    
                    # Shutdown
                    beam.beam_off()
                    print("✓ Beam off")
                
            beam.power_off()
            print("✓ System shutdown")
            
    except KeyboardInterrupt:
        print("\nEmergency stop activated!")
        beam.emergency_stop()
        beam.power_off()
