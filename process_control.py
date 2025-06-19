#!/usr/bin/env python3
"""
Process Control & Digital Twin Module
=====================================

Comprehensive automation, process control, and digital twin system for the rhodium replicator pilot plant.
Includes real-time monitoring, safety interlocks, batch scheduling, and predictive maintenance.

Author: Advanced Energy Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import json
import datetime
import threading
import time
import queue
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize_scalar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System operation states"""
    OFFLINE = "offline"
    STARTUP = "startup"
    STANDBY = "standby"
    OPERATING = "operating"
    SHUTDOWN = "shutdown"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

class AlarmLevel(Enum):
    """Alarm severity levels"""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ProcessVariable:
    """Process variable with metadata"""
    name: str
    value: float
    units: str
    timestamp: str
    setpoint: Optional[float] = None
    alarm_high: Optional[float] = None
    alarm_low: Optional[float] = None
    trend_data: List[Tuple[str, float]] = None
    quality_status: str = "good"  # 'good', 'uncertain', 'bad'

@dataclass
class ControlLoop:
    """PID control loop configuration"""
    pv_name: str  # Process variable name
    sp_name: str  # Setpoint name
    output_name: str  # Control output name
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    output_min: float = 0.0
    output_max: float = 100.0
    integral_term: float = 0.0
    last_error: float = 0.0
    last_time: float = 0.0

@dataclass
class SafetyInterlock:
    """Safety interlock definition"""
    name: str
    condition: str  # Logical condition string
    action: str  # Action to take when triggered
    priority: int  # Priority level (1=highest)
    enabled: bool = True
    last_triggered: Optional[str] = None
    trigger_count: int = 0

@dataclass
class BatchRecord:
    """Production batch record"""
    batch_id: str
    start_time: str
    end_time: Optional[str]
    target_isotope: str
    feedstock_mass: float  # kg
    beam_energy: float  # MeV
    beam_current: float  # µA
    target_yield: float  # atoms/second
    actual_yield: Optional[float] = None
    efficiency: Optional[float] = None
    quality_grade: Optional[str] = None
    operator_notes: str = ""
    process_data: Dict[str, List] = None

class DigitalTwin:
    """
    Digital twin model of the rhodium replicator system.
    Provides real-time simulation and predictive capabilities.
    """
    
    def __init__(self):
        # Physics model parameters
        self.lv_parameters = {
            'xi': 3.2e-18,
            'eta': 1.1e-19
        }
        
        # System state variables
        self.beam_power = 0.0  # kW
        self.target_temperature = 300.0  # K
        self.coolant_flow = 0.0  # L/min
        self.vacuum_pressure = 1e-6  # Torr
        self.magnetic_field = 0.0  # Tesla
        
        # Process variables
        self.process_variables = {}
        self.initialize_process_variables()
        
    def initialize_process_variables(self):
        """Initialize all process variables"""
        now = datetime.datetime.now().isoformat()
        
        pv_definitions = [
            ("beam_current", 0.0, "µA", 100.0, 150.0, 10.0),
            ("beam_energy", 0.0, "MeV", 200.0, 250.0, 50.0),
            ("target_temp", 300.0, "K", 400.0, 500.0, 250.0),
            ("coolant_flow", 0.0, "L/min", 50.0, 60.0, 30.0),
            ("vacuum_pressure", 1e-3, "Torr", 1e-5, 1e-4, 1e-6),
            ("separation_efficiency", 0.0, "%", 95.0, 99.0, 80.0),
            ("yield_rate", 0.0, "atoms/s", 1e12, 1e13, 1e10),
            ("radiation_level", 0.0, "mSv/h", 10.0, 25.0, 0.0),
            ("shielding_integrity", 100.0, "%", 100.0, 100.0, 95.0),
            ("power_consumption", 0.0, "kW", 100.0, 120.0, 0.0)
        ]
        
        for name, initial, units, alarm_high, sp, alarm_low in pv_definitions:
            self.process_variables[name] = ProcessVariable(
                name=name,
                value=initial,
                units=units,
                timestamp=now,
                setpoint=sp,
                alarm_high=alarm_high,
                alarm_low=alarm_low,
                trend_data=[(now, initial)]
            )
    
    def update_physics_model(self, dt: float = 1.0):
        """Update the physics-based model"""
        
        # Get current beam parameters
        beam_current = self.process_variables["beam_current"].value  # µA
        beam_energy = self.process_variables["beam_energy"].value  # MeV
        
        # Calculate beam power
        beam_power_kw = beam_current * beam_energy * 1e-3  # kW
        self.process_variables["power_consumption"].value = beam_power_kw * 1.2  # Include inefficiencies
        
        # Target heating model
        target_temp = self.process_variables["target_temp"].value
        coolant_flow = self.process_variables["coolant_flow"].value
        
        # Heat generation from beam
        heat_generation = beam_power_kw * 0.8  # 80% converted to heat
        
        # Heat removal by cooling
        heat_removal = coolant_flow * 0.1 * (target_temp - 300.0)  # Simplified cooling model
        
        # Temperature dynamics
        thermal_mass = 10.0  # Effective thermal mass
        dT_dt = (heat_generation - heat_removal) / thermal_mass
        new_temp = target_temp + dT_dt * dt
        
        self.process_variables["target_temp"].value = max(new_temp, 300.0)
        
        # Vacuum pressure dynamics (outgassing)
        vacuum_pressure = self.process_variables["vacuum_pressure"].value
        outgassing_rate = 1e-8 * np.exp((target_temp - 300.0) / 100.0)  # Temperature-dependent
        pumping_speed = 1000.0  # L/s
        
        dP_dt = outgassing_rate - pumping_speed * vacuum_pressure / 1000.0
        new_pressure = vacuum_pressure + dP_dt * dt
        self.process_variables["vacuum_pressure"].value = max(new_pressure, 1e-9)
        
        # Lorentz violation enhanced yield calculation
        if beam_current > 0 and beam_energy > 0:
            # Base cross-section
            sigma_base = 1e-27  # cm²
            
            # LV enhancement
            lv_enhancement = (1.0 + self.lv_parameters['xi'] * (beam_energy / 100.0)**2 + 
                            self.lv_parameters['eta'] * np.log(beam_energy / 10.0))
            
            # Enhanced cross-section
            sigma_enhanced = sigma_base * lv_enhancement
            
            # Target density (atoms/cm³)
            target_density = 8.9e22  # Typical for iron target
            
            # Beam flux (particles/cm²/s)
            beam_flux = beam_current * 6.24e12  # Convert µA to particles/s, assume 1 cm² area
            
            # Reaction rate
            reaction_rate = sigma_enhanced * target_density * beam_flux
            
            # Account for system efficiency
            system_efficiency = self.process_variables["separation_efficiency"].value / 100.0
            actual_yield = reaction_rate * system_efficiency
            
            self.process_variables["yield_rate"].value = actual_yield
        else:
            self.process_variables["yield_rate"].value = 0.0
        
        # Radiation level calculation
        radiation_background = 0.1  # mSv/h
        beam_radiation = beam_power_kw * 0.5  # Proportional to beam power
        self.process_variables["radiation_level"].value = radiation_background + beam_radiation
        
        # Update timestamps
        now = datetime.datetime.now().isoformat()
        for pv in self.process_variables.values():
            pv.timestamp = now
            if pv.trend_data is None:
                pv.trend_data = []
            pv.trend_data.append((now, pv.value))
            
            # Keep only last 1000 points
            if len(pv.trend_data) > 1000:
                pv.trend_data = pv.trend_data[-1000:]
    
    def predict_future_state(self, time_horizon: float = 3600.0) -> Dict[str, float]:
        """Predict system state after time_horizon seconds"""
        
        # Simple linear extrapolation based on current trends
        predictions = {}
        
        for name, pv in self.process_variables.items():
            if len(pv.trend_data) >= 2:
                # Calculate trend
                recent_data = pv.trend_data[-10:]  # Last 10 points
                times = [datetime.datetime.fromisoformat(t).timestamp() for t, v in recent_data]
                values = [v for t, v in recent_data]
                
                # Linear regression
                if len(times) > 1:
                    trend = (values[-1] - values[0]) / (times[-1] - times[0])
                    predicted_value = pv.value + trend * time_horizon
                    predictions[name] = predicted_value
                else:
                    predictions[name] = pv.value
            else:
                predictions[name] = pv.value
        
        return predictions

class ProcessController:
    """
    Main process control system with PID controllers, safety interlocks,
    and batch management.
    """
    
    def __init__(self):
        self.digital_twin = DigitalTwin()
        self.system_state = SystemState.OFFLINE
        
        # Control loops
        self.control_loops: Dict[str, ControlLoop] = {}
        self.initialize_control_loops()
        
        # Safety interlocks
        self.safety_interlocks: List[SafetyInterlock] = []
        self.initialize_safety_interlocks()
        
        # Batch management
        self.current_batch: Optional[BatchRecord] = None
        self.batch_history: List[BatchRecord] = []
        
        # Alarm system
        self.active_alarms: List[Dict[str, Any]] = []
        
        # Control thread
        self.control_thread = None
        self.control_active = False
        self.control_queue = queue.Queue()
        
        # Data logging
        self.log_data = []
        
    def initialize_control_loops(self):
        """Initialize PID control loops"""
        
        control_configs = [
            ("temp_control", "target_temp", "target_temp", "coolant_flow", 2.0, 0.1, 0.5),
            ("vacuum_control", "vacuum_pressure", "vacuum_pressure", "pump_speed", 1000.0, 10.0, 0.0),
            ("beam_control", "yield_rate", "yield_rate", "beam_current", 1e-10, 1e-12, 0.0),
        ]
        
        for name, pv, sp, output, kp, ki, kd in control_configs:
            self.control_loops[name] = ControlLoop(
                pv_name=pv,
                sp_name=sp,
                output_name=output,
                kp=kp,
                ki=ki,
                kd=kd
            )
    
    def initialize_safety_interlocks(self):
        """Initialize safety interlock system"""
        
        interlocks = [
            SafetyInterlock(
                name="High Temperature",
                condition="target_temp > 450.0",
                action="emergency_shutdown",
                priority=1
            ),
            SafetyInterlock(
                name="High Radiation",
                condition="radiation_level > 20.0",
                action="beam_shutdown",
                priority=1
            ),
            SafetyInterlock(
                name="Low Vacuum",
                condition="vacuum_pressure > 1e-4",
                action="beam_shutdown",
                priority=2
            ),
            SafetyInterlock(
                name="Low Coolant Flow",
                condition="coolant_flow < 25.0 and beam_current > 50.0",
                action="reduce_beam_power",
                priority=2
            ),
            SafetyInterlock(
                name="Shielding Integrity",
                condition="shielding_integrity < 98.0",
                action="emergency_shutdown",
                priority=1
            )
        ]
        
        self.safety_interlocks = interlocks
    
    def start_control_system(self):
        """Start the control system"""
        if self.control_active:
            logger.warning("Control system already running")
            return
        
        self.control_active = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        self.system_state = SystemState.STARTUP
        logger.info("Process control system started")
    
    def stop_control_system(self):
        """Stop the control system"""
        self.control_active = False
        if self.control_thread:
            self.control_thread.join(timeout=5.0)
        
        self.system_state = SystemState.OFFLINE
        logger.info("Process control system stopped")
    
    def _control_loop(self):
        """Main control loop (runs in separate thread)"""
        
        control_interval = 1.0  # seconds
        
        while self.control_active:
            try:
                start_time = time.time()
                
                # Update digital twin
                self.digital_twin.update_physics_model(control_interval)
                
                # Check safety interlocks
                self._check_safety_interlocks()
                
                # Update control loops
                self._update_control_loops()
                
                # Check alarms
                self._check_alarms()
                
                # Log data
                self._log_process_data()
                
                # Process any commands from queue
                self._process_commands()
                
                # Sleep for remainder of control interval
                elapsed = time.time() - start_time
                sleep_time = max(0, control_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                time.sleep(1.0)
    
    def _check_safety_interlocks(self):
        """Check all safety interlocks"""
        
        for interlock in self.safety_interlocks:
            if not interlock.enabled:
                continue
            
            # Evaluate condition
            try:
                condition_met = self._evaluate_condition(interlock.condition)
                
                if condition_met:
                    # Trigger interlock
                    self._trigger_interlock(interlock)
                    
            except Exception as e:
                logger.error(f"Error evaluating interlock {interlock.name}: {e}")
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a logical condition string"""
        
        # Create namespace with process variables
        namespace = {}
        for name, pv in self.digital_twin.process_variables.items():
            namespace[name] = pv.value
        
        # Add system state
        namespace['system_state'] = self.system_state.value
        
        try:
            return eval(condition, {"__builtins__": {}}, namespace)
        except:
            return False
    
    def _trigger_interlock(self, interlock: SafetyInterlock):
        """Trigger a safety interlock"""
        
        now = datetime.datetime.now().isoformat()
        interlock.last_triggered = now
        interlock.trigger_count += 1
        
        logger.warning(f"Safety interlock triggered: {interlock.name}")
        
        # Create alarm
        alarm = {
            'timestamp': now,
            'level': AlarmLevel.CRITICAL.value,
            'message': f"Safety interlock: {interlock.name}",
            'condition': interlock.condition,
            'action': interlock.action
        }
        self.active_alarms.append(alarm)
        
        # Execute action
        if interlock.action == "emergency_shutdown":
            self._emergency_shutdown()
        elif interlock.action == "beam_shutdown":
            self._beam_shutdown()
        elif interlock.action == "reduce_beam_power":
            self._reduce_beam_power()
    
    def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        self.system_state = SystemState.EMERGENCY
        
        # Shut down beam immediately
        self.digital_twin.process_variables["beam_current"].value = 0.0
        self.digital_twin.process_variables["beam_energy"].value = 0.0
        
        # Maximum cooling
        self.digital_twin.process_variables["coolant_flow"].value = 60.0
        
        # Stop current batch
        if self.current_batch:
            self._end_batch("emergency_shutdown")
    
    def _beam_shutdown(self):
        """Beam shutdown procedure"""
        logger.warning("Beam shutdown initiated")
        
        # Gradual beam shutdown
        current_beam = self.digital_twin.process_variables["beam_current"].value
        self.digital_twin.process_variables["beam_current"].value = max(0, current_beam * 0.1)
        
        if current_beam < 1.0:
            self.digital_twin.process_variables["beam_current"].value = 0.0
            self.digital_twin.process_variables["beam_energy"].value = 0.0
    
    def _reduce_beam_power(self):
        """Reduce beam power"""
        current_beam = self.digital_twin.process_variables["beam_current"].value
        self.digital_twin.process_variables["beam_current"].value = max(0, current_beam * 0.8)
        logger.info("Beam power reduced for safety")
    
    def _update_control_loops(self):
        """Update all PID control loops"""
        
        current_time = time.time()
        
        for name, loop in self.control_loops.items():
            try:
                # Get process variable and setpoint
                pv = self.digital_twin.process_variables.get(loop.pv_name)
                if not pv or pv.setpoint is None:
                    continue
                
                # Calculate error
                error = pv.setpoint - pv.value
                
                # Time delta
                dt = current_time - loop.last_time if loop.last_time > 0 else 1.0
                
                # PID calculation
                # Proportional term
                p_term = loop.kp * error
                
                # Integral term
                loop.integral_term += error * dt
                i_term = loop.ki * loop.integral_term
                
                # Derivative term
                d_term = loop.kd * (error - loop.last_error) / dt if dt > 0 else 0.0
                
                # Calculate output
                output = p_term + i_term + d_term
                
                # Clamp output
                output = max(loop.output_min, min(loop.output_max, output))
                
                # Apply output (simplified - would interface with actual hardware)
                if loop.output_name in self.digital_twin.process_variables:
                    output_pv = self.digital_twin.process_variables[loop.output_name]
                    # Apply output as a change rather than absolute value for some variables
                    if loop.output_name == "coolant_flow":
                        output_pv.value = output
                    elif loop.output_name == "beam_current":
                        # Gradual beam current adjustment
                        target_current = output * 100.0  # Scale to µA
                        current_current = output_pv.value
                        change_rate = 10.0  # µA/s
                        if abs(target_current - current_current) > change_rate * dt:
                            direction = 1 if target_current > current_current else -1
                            output_pv.value += direction * change_rate * dt
                        else:
                            output_pv.value = target_current
                
                # Update for next iteration
                loop.last_error = error
                loop.last_time = current_time
                
            except Exception as e:
                logger.error(f"Control loop {name} error: {e}")
    
    def _check_alarms(self):
        """Check for alarm conditions"""
        
        now = datetime.datetime.now().isoformat()
        
        for name, pv in self.digital_twin.process_variables.items():
            # High alarm
            if pv.alarm_high and pv.value > pv.alarm_high:
                alarm = {
                    'timestamp': now,
                    'level': AlarmLevel.ALARM.value,
                    'message': f"{pv.name} HIGH: {pv.value:.2f} {pv.units}",
                    'variable': name,
                    'value': pv.value,
                    'limit': pv.alarm_high
                }
                
                # Check if alarm already exists
                if not any(a.get('variable') == name and 'HIGH' in a.get('message', '') 
                          for a in self.active_alarms):
                    self.active_alarms.append(alarm)
            
            # Low alarm
            if pv.alarm_low and pv.value < pv.alarm_low:
                alarm = {
                    'timestamp': now,
                    'level': AlarmLevel.ALARM.value,
                    'message': f"{pv.name} LOW: {pv.value:.2f} {pv.units}",
                    'variable': name,
                    'value': pv.value,
                    'limit': pv.alarm_low
                }
                
                if not any(a.get('variable') == name and 'LOW' in a.get('message', '') 
                          for a in self.active_alarms):
                    self.active_alarms.append(alarm)
        
        # Remove cleared alarms
        self.active_alarms = [alarm for alarm in self.active_alarms 
                             if self._is_alarm_still_active(alarm)]
    
    def _is_alarm_still_active(self, alarm: Dict[str, Any]) -> bool:
        """Check if an alarm condition is still active"""
        
        variable = alarm.get('variable')
        if not variable:
            return True  # Keep non-variable alarms
        
        pv = self.digital_twin.process_variables.get(variable)
        if not pv:
            return False
        
        if 'HIGH' in alarm['message']:
            return pv.alarm_high and pv.value > pv.alarm_high
        elif 'LOW' in alarm['message']:
            return pv.alarm_low and pv.value < pv.alarm_low
        
        return True
    
    def _log_process_data(self):
        """Log process data for historical analysis"""
        
        now = datetime.datetime.now().isoformat()
        
        log_entry = {
            'timestamp': now,
            'system_state': self.system_state.value,
            'process_variables': {name: pv.value for name, pv in self.digital_twin.process_variables.items()},
            'active_alarms_count': len(self.active_alarms),
            'batch_id': self.current_batch.batch_id if self.current_batch else None
        }
        
        self.log_data.append(log_entry)
        
        # Keep only last 10000 entries
        if len(self.log_data) > 10000:
            self.log_data = self.log_data[-10000:]
    
    def _process_commands(self):
        """Process commands from the queue"""
        
        try:
            while not self.control_queue.empty():
                command = self.control_queue.get_nowait()
                self._execute_command(command)
        except queue.Empty:
            pass
    
    def _execute_command(self, command: Dict[str, Any]):
        """Execute a control command"""
        
        cmd_type = command.get('type')
        
        if cmd_type == 'set_setpoint':
            variable = command.get('variable')
            value = command.get('value')
            if variable in self.digital_twin.process_variables:
                self.digital_twin.process_variables[variable].setpoint = value
                logger.info(f"Setpoint changed: {variable} = {value}")
        
        elif cmd_type == 'start_batch':
            self._start_batch(command.get('batch_config', {}))
        
        elif cmd_type == 'end_batch':
            self._end_batch(command.get('reason', 'operator_request'))
        
        elif cmd_type == 'change_state':
            new_state = SystemState(command.get('state'))
            self._change_system_state(new_state)
    
    def start_batch(self, batch_config: Dict[str, Any]) -> str:
        """Start a new production batch"""
        
        if self.current_batch:
            raise ValueError("Batch already in progress")
        
        command = {
            'type': 'start_batch',
            'batch_config': batch_config
        }
        self.control_queue.put(command)
        
        return batch_config.get('batch_id', 'unknown')
    
    def _start_batch(self, batch_config: Dict[str, Any]):
        """Internal batch start implementation"""
        
        batch_id = batch_config.get('batch_id', f"batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.current_batch = BatchRecord(
            batch_id=batch_id,
            start_time=datetime.datetime.now().isoformat(),
            end_time=None,
            target_isotope=batch_config.get('target_isotope', 'Rh-103'),
            feedstock_mass=batch_config.get('feedstock_mass', 1.0),
            beam_energy=batch_config.get('beam_energy', 200.0),
            beam_current=batch_config.get('beam_current', 100.0),
            target_yield=batch_config.get('target_yield', 1e12),
            process_data={}
        )
        
        # Set process setpoints
        self.digital_twin.process_variables["beam_energy"].setpoint = self.current_batch.beam_energy
        self.digital_twin.process_variables["beam_current"].setpoint = self.current_batch.beam_current
        self.digital_twin.process_variables["target_temp"].setpoint = 400.0
        self.digital_twin.process_variables["coolant_flow"].setpoint = 45.0
        
        self.system_state = SystemState.OPERATING
        
        logger.info(f"Batch {batch_id} started")
    
    def end_batch(self, reason: str = "completed") -> Optional[BatchRecord]:
        """End the current batch"""
        
        command = {
            'type': 'end_batch',
            'reason': reason
        }
        self.control_queue.put(command)
        
        return self.current_batch
    
    def _end_batch(self, reason: str):
        """Internal batch end implementation"""
        
        if not self.current_batch:
            return
        
        self.current_batch.end_time = datetime.datetime.now().isoformat()
        
        # Calculate batch metrics
        if self.current_batch.start_time and self.current_batch.end_time:
            start_dt = datetime.datetime.fromisoformat(self.current_batch.start_time)
            end_dt = datetime.datetime.fromisoformat(self.current_batch.end_time)
            duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
            
            # Average yield during batch
            recent_yield = self.digital_twin.process_variables["yield_rate"].value
            self.current_batch.actual_yield = recent_yield
            
            # Calculate efficiency
            if self.current_batch.target_yield > 0:
                self.current_batch.efficiency = (recent_yield / self.current_batch.target_yield) * 100.0
            
            # Determine quality grade
            if self.current_batch.efficiency and self.current_batch.efficiency > 90:
                self.current_batch.quality_grade = "A"
            elif self.current_batch.efficiency and self.current_batch.efficiency > 75:
                self.current_batch.quality_grade = "B"
            else:
                self.current_batch.quality_grade = "C"
        
        # Store batch record
        self.batch_history.append(self.current_batch)
        completed_batch = self.current_batch
        self.current_batch = None
        
        # Return to standby
        self.system_state = SystemState.STANDBY
        
        # Reset setpoints
        self.digital_twin.process_variables["beam_current"].setpoint = 0.0
        
        logger.info(f"Batch {completed_batch.batch_id} ended: {reason}")
    
    def _change_system_state(self, new_state: SystemState):
        """Change system state with appropriate actions"""
        
        old_state = self.system_state
        self.system_state = new_state
        
        logger.info(f"System state changed: {old_state.value} → {new_state.value}")
        
        # State-specific actions
        if new_state == SystemState.SHUTDOWN:
            # Graceful shutdown
            if self.current_batch:
                self._end_batch("shutdown")
            
            # Reduce beam power gradually
            self.digital_twin.process_variables["beam_current"].setpoint = 0.0
        
        elif new_state == SystemState.EMERGENCY:
            self._emergency_shutdown()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'system_state': self.system_state.value,
            'process_variables': {
                name: {
                    'value': pv.value,
                    'units': pv.units,
                    'setpoint': pv.setpoint,
                    'quality': pv.quality_status
                }
                for name, pv in self.digital_twin.process_variables.items()
            },
            'active_alarms': self.active_alarms,
            'current_batch': asdict(self.current_batch) if self.current_batch else None,            'control_loops_status': {
                name: {
                    'pv_value': self.digital_twin.process_variables[loop.pv_name].value,
                    'setpoint': self.digital_twin.process_variables[loop.pv_name].setpoint,
                    'output': self.digital_twin.process_variables[loop.output_name].value if loop.output_name in self.digital_twin.process_variables else 0
                }
                for name, loop in self.control_loops.items()
                if loop.pv_name in self.digital_twin.process_variables
            }
        }
    
    def get_batch_history(self, limit: int = 10) -> List[BatchRecord]:
        """Get recent batch history"""
        return self.batch_history[-limit:]
    
    def predict_maintenance_needs(self) -> Dict[str, Any]:
        """Predict maintenance requirements using digital twin"""
        
        predictions = self.digital_twin.predict_future_state(24 * 3600)  # 24 hours
        
        maintenance_alerts = []
        
        # Check for potential issues
        if predictions.get('target_temp', 0) > 400:
            maintenance_alerts.append({
                'component': 'Target Assembly',
                'issue': 'High temperature trend',
                'priority': 'medium',
                'estimated_time': '7 days'
            })
        
        if predictions.get('vacuum_pressure', 0) > 1e-5:
            maintenance_alerts.append({
                'component': 'Vacuum System',
                'issue': 'Pressure degradation',
                'priority': 'high',
                'estimated_time': '3 days'
            })
        
        # Calculate system health score
        health_metrics = []
        for name, pv in self.digital_twin.process_variables.items():
            if pv.alarm_high and pv.alarm_low:
                # Normalized position within acceptable range
                range_size = pv.alarm_high - pv.alarm_low
                position = (pv.value - pv.alarm_low) / range_size
                health_score = 1.0 - abs(position - 0.5) * 2  # Best at center
                health_metrics.append(health_score)
        
        overall_health = np.mean(health_metrics) if health_metrics else 1.0
        
        return {
            'overall_health_score': overall_health,
            'maintenance_alerts': maintenance_alerts,
            'predictions': predictions,
            'recommended_actions': self._get_maintenance_recommendations(overall_health)
        }
    
    def _get_maintenance_recommendations(self, health_score: float) -> List[str]:
        """Generate maintenance recommendations"""
        
        recommendations = []
        
        if health_score < 0.7:
            recommendations.append("Schedule comprehensive system inspection")
        
        if health_score < 0.8:
            recommendations.append("Review and update preventive maintenance schedule")
        
        # Check specific systems
        temp_trend = [entry['process_variables']['target_temp'] 
                     for entry in self.log_data[-100:] if 'target_temp' in entry['process_variables']]
        
        if len(temp_trend) > 10 and np.mean(temp_trend[-10:]) > np.mean(temp_trend[:10]):
            recommendations.append("Inspect cooling system performance")
        
        return recommendations

def demo_process_control():
    """Demonstrate process control system capabilities"""
    
    print("🤖 PROCESS CONTROL & DIGITAL TWIN DEMO")
    print("=" * 50)
    
    # Initialize process controller
    controller = ProcessController()
    
    print("🚀 Starting control system...")
    controller.start_control_system()
    
    # Wait for system to stabilize
    time.sleep(2)
    
    print("\n📊 Initial system status:")
    status = controller.get_system_status()
    print(f"  System State: {status['system_state']}")
    print(f"  Active Alarms: {len(status['active_alarms'])}")
    
    # Start a production batch
    print("\n🏭 Starting production batch...")
    
    batch_config = {
        'batch_id': 'DEMO_001',
        'target_isotope': 'Rh-103',
        'feedstock_mass': 0.5,  # kg
        'beam_energy': 200.0,   # MeV
        'beam_current': 80.0,   # µA
        'target_yield': 5e11    # atoms/second
    }
    
    batch_id = controller.start_batch(batch_config)
    print(f"  Batch ID: {batch_id}")
    
    # Monitor for several seconds
    print("\n⏱️  Monitoring batch progress...")
    
    for i in range(10):
        time.sleep(1)
        status = controller.get_system_status()
        
        if i % 3 == 0:  # Print every 3 seconds
            pv = status['process_variables']
            print(f"  T+{i+1}s: Beam={pv['beam_current']['value']:.1f}µA, "
                  f"Temp={pv['target_temp']['value']:.1f}K, "
                  f"Yield={pv['yield_rate']['value']:.2e} atoms/s")
    
    # Introduce a disturbance (simulate high temperature)
    print("\n🔥 Simulating temperature disturbance...")
    controller.digital_twin.process_variables["target_temp"].value = 460.0
    
    # Monitor safety response
    time.sleep(2)
    status = controller.get_system_status()
    
    print(f"  Temperature: {status['process_variables']['target_temp']['value']:.1f}K")
    print(f"  Active Alarms: {len(status['active_alarms'])}")
    
    if status['active_alarms']:
        for alarm in status['active_alarms']:
            print(f"    ⚠️ {alarm['message']}")
      # End batch
    print("\n🏁 Ending batch...")
    completed_batch = controller.end_batch("demo_complete")
    
    if completed_batch:
        print(f"  Batch {completed_batch.batch_id} completed")
        if completed_batch.efficiency is not None:
            print(f"  Efficiency: {completed_batch.efficiency:.1f}%")
        else:
            print(f"  Efficiency: Not calculated")
        print(f"  Quality Grade: {completed_batch.quality_grade or 'Not assigned'}")
    
    # Demonstrate predictive maintenance
    print("\n🔧 Predictive maintenance analysis...")
    
    maintenance = controller.predict_maintenance_needs()
    print(f"  System Health Score: {maintenance['overall_health_score']:.2f}")
    
    if maintenance['maintenance_alerts']:
        print("  Maintenance Alerts:")
        for alert in maintenance['maintenance_alerts']:
            print(f"    • {alert['component']}: {alert['issue']} (Priority: {alert['priority']})")
    
    if maintenance['recommended_actions']:
        print("  Recommendations:")
        for action in maintenance['recommended_actions']:
            print(f"    • {action}")
    
    # Generate process data export
    print("\n💾 Generating process data export...")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    export_data = {
        'system_status': status,
        'batch_history': [asdict(batch) for batch in controller.get_batch_history()],
        'process_log': controller.log_data[-100:],  # Last 100 entries
        'maintenance_analysis': maintenance,
        'digital_twin_state': {
            name: asdict(pv) for name, pv in controller.digital_twin.process_variables.items()
        }
    }
    
    export_file = f"process_control_demo_{timestamp}.json"
    with open(export_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"  Export saved: {export_file}")
    
    # Stop control system
    print("\n🛑 Stopping control system...")
    controller.stop_control_system()
    
    print("\n🎉 Process control demonstration complete!")
    
    return controller, export_data

if __name__ == "__main__":
    # Run demonstration
    controller, data = demo_process_control()
