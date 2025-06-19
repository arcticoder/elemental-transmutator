#!/usr/bin/env python3
"""
Stimulated Pair Production Engine: LV-Enhanced Energy-to-Matter Conversion  
=========================================================================

This module implements stimulated pair production using LV-modified thresholds
to convert stored energy into matter through controlled particle creation.

Key Features:
1. Breit-Wheeler laser-laser collision pair production
2. Dynamic Casimir pair spawning in metamaterial cavities  
3. LV-shifted production thresholds for enhanced efficiency
4. Controlled particle beam generation and steering
5. Integration with energy storage and matter assembly systems

Physics:
- Modified pair production threshold: E_th → E_th√(1 - δ_LV)
- Enhanced cross sections through vacuum structure modifications
- Stimulated emission of particle pairs from vacuum fluctuations
- Metamaterial-enhanced field localization for controlled production

Author: LV Energy Converter Framework
"""

import numpy as np
from scipy import integrate, optimize, special
from scipy.special import kv, iv, factorial
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings

# Import energy ledger system
try:
    from .energy_ledger import EnergyLedger, EnergyType
except ImportError:
    from energy_ledger import EnergyLedger, EnergyType

@dataclass
class PairProductionConfig:
    """Configuration for stimulated pair production system."""
    
    # Production method
    production_method: str = "breit_wheeler"     # "breit_wheeler", "dynamic_casimir", "hybrid"
    target_particle_type: str = "electron"      # "electron", "muon", "pion"
    production_rate_target: float = 1e12        # Target pairs per second
    
    # Laser parameters (for Breit-Wheeler)
    laser_wavelength: float = 800e-9            # Laser wavelength (m)
    laser_pulse_energy: float = 1e-3            # Pulse energy (J)
    laser_pulse_duration: float = 1e-12         # Pulse duration (s)
    laser_beam_waist: float = 1e-6              # Beam waist radius (m)
    collision_angle: float = np.pi              # Head-on collision
    
    # Dynamic Casimir parameters
    cavity_length: float = 1e-6                 # Cavity length (m)
    boundary_oscillation_freq: float = 1e12     # THz oscillation
    boundary_oscillation_amplitude: float = 1e-9 # Oscillation amplitude (m)
    metamaterial_enhancement: bool = True        # Use metamaterial cavity
    
    # LV parameters
    mu_lv: float = 1e-17                        # CPT violation coefficient
    alpha_lv: float = 1e-14                     # Lorentz violation coefficient
    beta_lv: float = 1e-11                      # Gravitational LV coefficient
    
    # Field configuration
    magnetic_field_strength: float = 1.0        # Tesla
    electric_field_amplitude: float = 1e10      # V/m
    field_geometry: str = "parallel"            # "parallel", "crossed", "helical"
    
    # Particle collection and steering
    collection_efficiency: float = 0.8          # Particle collection efficiency
    beam_steering_enabled: bool = True          # Enable particle beam steering
    separation_field_strength: float = 1e6      # V/m for e+/e- separation
    
    # Safety and control
    max_field_strength: float = 1e12           # Maximum field strength (V/m)
    radiation_shielding: bool = True            # Enable radiation shielding
    emergency_shutdown_threshold: float = 1e10  # Emergency particle rate (pairs/s)

@dataclass
class ParticleBeamProperties:
    """Properties of produced particle beams."""
    
    particle_type: str               # Type of particles produced
    antiparticle_type: str          # Type of antiparticles produced
    production_rate: float          # Particles per second
    beam_energy: float              # Average particle energy (J)
    beam_divergence: float          # Beam divergence (rad)
    polarization: float             # Beam polarization
    spatial_distribution: str       # "gaussian", "uniform", "ring"

class StimulatedPairEngine:
    """
    Stimulated pair production engine for energy-to-matter conversion.
    
    This class implements controlled conversion of stored energy into
    particle-antiparticle pairs using LV-enhanced production mechanisms.
    """
    
    def __init__(self, config: PairProductionConfig, energy_ledger: EnergyLedger):
        self.config = config
        self.energy_ledger = energy_ledger
        
        # Physical constants
        self.c = 3e8                    # Speed of light (m/s)
        self.hbar = 1.055e-34          # Reduced Planck constant (J⋅s)
        self.e = 1.602e-19             # Elementary charge (C)
        self.m_e = 9.109e-31           # Electron mass (kg)
        self.m_mu = 1.883e-28          # Muon mass (kg)
        self.m_pi = 2.488e-28          # Pion mass (kg)
        self.alpha = 1/137.0           # Fine structure constant
        self.epsilon_0 = 8.854e-12     # Permittivity of free space (F/m)
        
        # Initialize production systems
        self._initialize_particle_properties()
        self._calculate_lv_modifications()
        
        # System state
        self.total_pairs_produced = 0
        self.production_history = []
        self.beam_quality_factor = 1.0
        self.system_temperature = 300  # K
        
    def _initialize_particle_properties(self):
        """Initialize properties of particles that can be produced."""
        self.particle_properties = {
            "electron": {
                "mass": self.m_e,
                "charge": -self.e,
                "antiparticle": "positron",
                "threshold_energy": 2 * self.m_e * self.c**2,  # 2mec²
                "production_cross_section": 6.65e-29  # Breit-Wheeler cross section (m²)
            },
            "muon": {
                "mass": self.m_mu,
                "charge": -self.e,
                "antiparticle": "antimuon",
                "threshold_energy": 2 * self.m_mu * self.c**2,
                "production_cross_section": 6.65e-29 * (self.m_e / self.m_mu)**2
            },
            "pion": {
                "mass": self.m_pi,
                "charge": self.e,  # π+ charge
                "antiparticle": "antipion",
                "threshold_energy": 2 * self.m_pi * self.c**2,
                "production_cross_section": 1e-30  # Approximate
            }
        }
    
    def _calculate_lv_modifications(self):
        """Calculate LV modifications to pair production."""
        particle = self.particle_properties[self.config.target_particle_type]
        
        # Energy scale
        E_threshold = particle["threshold_energy"]
        E_scale = E_threshold / (self.hbar * self.c)
        
        # LV threshold modification: E_th → E_th√(1 - δ_LV)
        lv_threshold_shift = (self.config.mu_lv * E_scale**2 + 
                             self.config.alpha_lv * E_scale +
                             self.config.beta_lv * np.sqrt(E_scale))
        
        # Threshold reduction (makes production easier)
        self.lv_threshold_factor = np.sqrt(max(1 - lv_threshold_shift, 0.1))
        
        # Cross section enhancement through vacuum modifications
        self.lv_cross_section_enhancement = (1 + 0.1 * np.sum([
            self.config.mu_lv * 1e15,
            self.config.alpha_lv * 1e12, 
            self.config.beta_lv * 1e9
        ]))
        
        print(f"LV modifications calculated:")
        print(f"  Threshold factor: {self.lv_threshold_factor:.3f}")
        print(f"  Cross section enhancement: {self.lv_cross_section_enhancement:.3f}×")
    
    def calculate_breit_wheeler_rate(self, laser_energy_1: float, 
                                   laser_energy_2: float) -> float:
        """
        Calculate Breit-Wheeler pair production rate from laser collisions.
        
        Parameters:
        -----------
        laser_energy_1, laser_energy_2 : float
            Energies of colliding laser pulses (J)
            
        Returns:
        --------
        float
            Pair production rate (pairs per collision)
        """
        # Photon energies
        photon_energy_1 = laser_energy_1 / self.config.laser_pulse_duration
        photon_energy_2 = laser_energy_2 / self.config.laser_pulse_duration
        
        # Center-of-mass energy
        cos_theta = np.cos(self.config.collision_angle)
        s = 2 * photon_energy_1 * photon_energy_2 * (1 - cos_theta)
        
        # Threshold check with LV modification
        particle = self.particle_properties[self.config.target_particle_type]
        threshold = particle["threshold_energy"] * self.lv_threshold_factor
        
        if s < threshold:
            return 0.0
        
        # Breit-Wheeler cross section with LV enhancement
        base_cross_section = particle["production_cross_section"]
        lv_enhanced_cross_section = base_cross_section * self.lv_cross_section_enhancement
        
        # Number density of photons in laser pulse
        pulse_volume = np.pi * self.config.laser_beam_waist**2 * self.c * self.config.laser_pulse_duration
        
        n1 = laser_energy_1 / (photon_energy_1 * pulse_volume)
        n2 = laser_energy_2 / (photon_energy_2 * pulse_volume)
        
        # Production rate
        rate = n1 * n2 * lv_enhanced_cross_section * self.c * pulse_volume
        
        return rate
    
    def calculate_dynamic_casimir_rate(self, cavity_energy: float) -> float:
        """
        Calculate dynamic Casimir pair production rate.
        
        Parameters:
        -----------
        cavity_energy : float
            Energy stored in cavity (J)
            
        Returns:
        --------
        float
            Pair production rate (pairs/s)
        """
        # Field amplitude in cavity
        cavity_volume = self.config.cavity_length**3  # Assume cubic cavity
        field_amplitude = np.sqrt(2 * cavity_energy / (self.epsilon_0 * cavity_volume))
        
        # Critical field for pair production (Schwinger limit with LV modification)
        E_schwinger = (self.m_e**2 * self.c**3) / (self.e * self.hbar)
        lv_modified_critical_field = E_schwinger * self.lv_threshold_factor
        
        # Production rate (exponentially suppressed below critical field)
        if field_amplitude < lv_modified_critical_field:
            # Tunneling regime
            exponent = -np.pi * lv_modified_critical_field / field_amplitude
            tunneling_rate = (self.alpha * field_amplitude**2) / (self.hbar * self.c**2) * np.exp(exponent)
        else:
            # Above-threshold production
            tunneling_rate = (self.alpha * field_amplitude**2) / (self.hbar * self.c**2)
        
        # Dynamic enhancement from boundary motion
        oscillation_parameter = (self.config.boundary_oscillation_amplitude * 
                               self.config.boundary_oscillation_freq / self.c)
        
        dynamic_enhancement = 1 + oscillation_parameter**2
        
        # Metamaterial enhancement
        metamaterial_factor = 10.0 if self.config.metamaterial_enhancement else 1.0
        
        total_rate = (tunneling_rate * dynamic_enhancement * metamaterial_factor * 
                     cavity_volume * self.lv_cross_section_enhancement)
        
        return total_rate
    
    def produce_particle_pairs(self, input_energy: float, 
                             production_time: float = 1e-6) -> Dict[str, float]:
        """
        Produce particle pairs from input energy.
        
        Parameters:
        -----------
        input_energy : float
            Input energy for pair production (J)
        production_time : float
            Time duration for production (s)
            
        Returns:
        --------
        Dict[str, float]
            Production results
        """
        particle = self.particle_properties[self.config.target_particle_type]
        
        if self.config.production_method == "breit_wheeler":
            # Split energy between two laser pulses
            laser_energy = input_energy / 2
            production_rate = self.calculate_breit_wheeler_rate(laser_energy, laser_energy)
            
        elif self.config.production_method == "dynamic_casimir":
            # Use energy to drive cavity oscillations
            production_rate = self.calculate_dynamic_casimir_rate(input_energy)
            
        else:  # hybrid
            # Use both methods
            bw_energy = input_energy * 0.7
            dc_energy = input_energy * 0.3
            
            bw_rate = self.calculate_breit_wheeler_rate(bw_energy/2, bw_energy/2)
            dc_rate = self.calculate_dynamic_casimir_rate(dc_energy)
            production_rate = bw_rate + dc_rate
        
        # Total pairs produced
        pairs_produced = production_rate * production_time
        
        # Energy per pair (including kinetic energy)
        energy_per_pair = particle["threshold_energy"] * 1.5  # Include kinetic energy
        
        # Total energy invested in matter
        matter_energy = pairs_produced * energy_per_pair
        
        # Production efficiency 
        production_efficiency = matter_energy / input_energy if input_energy > 0 else 0
        
        # Apply collection efficiency
        collected_pairs = pairs_produced * self.config.collection_efficiency
        collected_matter_energy = collected_pairs * energy_per_pair
        
        # Log production
        self.energy_ledger.log_transaction(
            EnergyType.PAIR_PRODUCTION, -input_energy,
            location="pair_production_chamber", pathway="energy_to_matter"
        )
        
        self.energy_ledger.log_transaction(
            EnergyType.OUTPUT_MATTER_SYNTHESIS, collected_matter_energy,
            location="particle_collector", pathway="matter_output"
        )
        
        # Log losses
        production_losses = input_energy - matter_energy
        collection_losses = matter_energy - collected_matter_energy
        
        if production_losses > 0:
            self.energy_ledger.log_transaction(
                EnergyType.LOSSES_PAIR_EFFICIENCY, -production_losses,
                location="production_chamber", pathway="production_losses"
            )
        
        if collection_losses > 0:
            self.energy_ledger.log_transaction(
                EnergyType.LOSSES_ASSEMBLY, -collection_losses,
                location="particle_collector", pathway="collection_losses"
            )
        
        # Update system state
        self.total_pairs_produced += collected_pairs
        
        # Record production event
        self.production_history.append({
            'input_energy': input_energy,
            'pairs_produced': pairs_produced,
            'collected_pairs': collected_pairs,
            'production_rate': production_rate,
            'efficiency': production_efficiency,
            'method': self.config.production_method,
            'timestamp': len(self.production_history)
        })
        
        # Emergency check
        if production_rate > self.config.emergency_shutdown_threshold:
            warnings.warn("Production rate exceeds safety threshold!")
        
        return {
            'pairs_produced': pairs_produced,
            'collected_pairs': collected_pairs,
            'production_rate': production_rate,
            'matter_energy_created': collected_matter_energy,
            'production_efficiency': production_efficiency,
            'collection_efficiency': self.config.collection_efficiency
        }
    
    def optimize_production_parameters(self, target_rate: float) -> Dict[str, float]:
        """
        Optimize production parameters for target pair production rate.
        
        Parameters:
        -----------
        target_rate : float
            Target production rate (pairs/s)
            
        Returns:
        --------
        Dict[str, float]
            Optimized parameters
        """
        def objective(params):
            if self.config.production_method == "breit_wheeler":
                laser_energy, beam_waist = params
                
                # Update configuration temporarily
                old_energy = self.config.laser_pulse_energy
                old_waist = self.config.laser_beam_waist
                
                self.config.laser_pulse_energy = laser_energy
                self.config.laser_beam_waist = beam_waist
                
                rate = self.calculate_breit_wheeler_rate(laser_energy, laser_energy)
                
                # Restore configuration
                self.config.laser_pulse_energy = old_energy
                self.config.laser_beam_waist = old_waist
                
            else:  # dynamic_casimir
                cavity_energy, oscillation_freq = params
                
                old_freq = self.config.boundary_oscillation_freq
                self.config.boundary_oscillation_freq = oscillation_freq
                
                rate = self.calculate_dynamic_casimir_rate(cavity_energy)
                
                self.config.boundary_oscillation_freq = old_freq
            
            return abs(rate - target_rate)
        
        # Bounds depend on production method
        if self.config.production_method == "breit_wheeler":
            bounds = [
                (1e-6, 1e-1),    # laser_energy (J)
                (1e-9, 1e-3)     # beam_waist (m)
            ]
            x0 = [self.config.laser_pulse_energy, self.config.laser_beam_waist]
        else:
            bounds = [
                (1e-12, 1e-6),   # cavity_energy (J)
                (1e9, 1e15)      # oscillation_freq (Hz)
            ]
            x0 = [1e-9, self.config.boundary_oscillation_freq]
        
        # Optimize
        result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return {
            'optimization_success': result.success,
            'optimal_parameters': result.x,
            'achieved_rate': target_rate - result.fun,
            'parameter_names': ['laser_energy', 'beam_waist'] if self.config.production_method == "breit_wheeler" 
                              else ['cavity_energy', 'oscillation_freq']
        }
    
    def simulate_production_cycle(self, cycle_duration: float, 
                                energy_budget: float) -> Dict[str, np.ndarray]:
        """
        Simulate complete production cycle.
        
        Parameters:
        -----------
        cycle_duration : float
            Duration of production cycle (s)
        energy_budget : float
            Total energy budget (J)
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Simulation results
        """
        time_steps = 1000
        times = np.linspace(0, cycle_duration, time_steps)
        dt = times[1] - times[0]
        
        # Initialize arrays
        production_rates = np.zeros(time_steps)
        cumulative_pairs = np.zeros(time_steps)
        energy_usage = np.zeros(time_steps)
        efficiency_history = np.zeros(time_steps)
        
        # Energy per time step
        energy_per_step = energy_budget / time_steps
        total_pairs = 0
        total_energy_used = 0
        
        for i, t in enumerate(times):
            # Produce pairs this time step
            results = self.produce_particle_pairs(energy_per_step, dt)
            
            production_rates[i] = results['production_rate']
            total_pairs += results['collected_pairs']
            cumulative_pairs[i] = total_pairs
            
            total_energy_used += energy_per_step
            energy_usage[i] = total_energy_used
            
            efficiency_history[i] = results['production_efficiency']
        
        return {
            'times': times,
            'production_rates': production_rates,
            'cumulative_pairs': cumulative_pairs,
            'energy_usage': energy_usage,
            'efficiency_history': efficiency_history,
            'final_pair_count': total_pairs,
            'total_energy_invested': total_energy_used,
            'average_efficiency': np.mean(efficiency_history)
        }
    
    def generate_production_report(self) -> Dict:
        """Generate comprehensive pair production report."""
        if len(self.production_history) == 0:
            return {'error': 'No production events recorded'}
        
        total_input_energy = sum(event['input_energy'] for event in self.production_history)
        total_pairs = sum(event['collected_pairs'] for event in self.production_history)
        avg_efficiency = np.mean([event['efficiency'] for event in self.production_history])
        
        # Calculate matter equivalent mass
        particle = self.particle_properties[self.config.target_particle_type]
        total_matter_mass = total_pairs * 2 * particle["mass"]  # particle + antiparticle
        
        return {
            'production_statistics': {
                'total_events': len(self.production_history),
                'total_pairs_produced': total_pairs,
                'total_input_energy': total_input_energy,
                'average_efficiency': avg_efficiency,
                'total_matter_mass_created': total_matter_mass
            },
            'system_configuration': {
                'production_method': self.config.production_method,
                'target_particle': self.config.target_particle_type,
                'lv_threshold_factor': self.lv_threshold_factor,
                'lv_cross_section_enhancement': self.lv_cross_section_enhancement
            },
            'performance_metrics': {
                'beam_quality_factor': self.beam_quality_factor,
                'collection_efficiency': self.config.collection_efficiency,
                'energy_ledger_balance': self.energy_ledger.calculate_net_energy_gain()
            }
        }

def demo_stimulated_pair_production():
    """Demonstrate stimulated pair production."""
    print("=== Stimulated Pair Production Demo ===")
    
    # Create energy ledger
    ledger = EnergyLedger("Pair_Production_Demo")
    
    # Create configuration
    config = PairProductionConfig(
        production_method="breit_wheeler",
        target_particle_type="electron",
        laser_pulse_energy=1e-3,                # 1 mJ pulse
        laser_pulse_duration=1e-12,             # 1 ps
        laser_beam_waist=1e-6,                  # 1 μm waist
        mu_lv=1e-17,                           # 100× experimental bound
        alpha_lv=1e-14,                        # 100× experimental bound
        beta_lv=1e-11,                         # 100× experimental bound
        collection_efficiency=0.8               # 80% collection
    )
    
    # Initialize pair production engine
    pair_engine = StimulatedPairEngine(config, ledger)
    
    # Test pair production
    print(f"\n=== Pair Production Test ===")
    input_energy = 2e-3  # 2 mJ
    results = pair_engine.produce_particle_pairs(input_energy, production_time=1e-6)
    
    print(f"✓ Pairs Produced: {results['pairs_produced']:.2e}")
    print(f"✓ Collected Pairs: {results['collected_pairs']:.2e}")
    print(f"✓ Production Rate: {results['production_rate']:.2e} pairs/s")
    print(f"✓ Matter Energy Created: {results['matter_energy_created']:.2e} J")
    print(f"✓ Production Efficiency: {results['production_efficiency']:.1%}")
    
    # Test parameter optimization
    print(f"\n=== Parameter Optimization ===")
    target_rate = 1e12  # 1 THz production rate
    optimization = pair_engine.optimize_production_parameters(target_rate)
    
    print(f"✓ Optimization Success: {optimization['optimization_success']}")
    print(f"✓ Achieved Rate: {optimization['achieved_rate']:.2e} pairs/s")
    
    # Test production cycle simulation
    print(f"\n=== Production Cycle Simulation ===")
    cycle_results = pair_engine.simulate_production_cycle(
        cycle_duration=1e-3,    # 1 ms cycle
        energy_budget=1e-2      # 10 mJ budget
    )
    
    print(f"✓ Final Pair Count: {cycle_results['final_pair_count']:.2e}")
    print(f"✓ Total Energy Invested: {cycle_results['total_energy_invested']:.2e} J")
    print(f"✓ Average Efficiency: {cycle_results['average_efficiency']:.1%}")
    
    # Generate report
    report = pair_engine.generate_production_report()
    print(f"\n=== Production Report ===")
    print(f"✓ Total Pairs: {report['production_statistics']['total_pairs_produced']:.2e}")
    print(f"✓ Matter Mass Created: {report['production_statistics']['total_matter_mass_created']:.2e} kg")
    print(f"✓ LV Threshold Factor: {report['system_configuration']['lv_threshold_factor']:.3f}")
    print(f"✓ LV Enhancement: {report['system_configuration']['lv_cross_section_enhancement']:.3f}×")
    
    return pair_engine, report

if __name__ == "__main__":
    demo_stimulated_pair_production()
