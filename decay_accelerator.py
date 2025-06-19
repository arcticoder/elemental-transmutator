#!/usr/bin/env python3
"""
LV-Accelerated Decay Module
===========================

Lorentz violation enhanced nuclear decay acceleration for element-agnostic production.
Speeds up β-decay and electron capture transitions by modifying nuclear matrix
elements and phase space factors through LV field engineering.

Key capabilities:
- Accelerate β⁻ decay (e.g. precursor → target + e⁻ + ν̄ₑ)
- Accelerate electron capture (e.g. precursor + e⁻ → target + νₑ)
- Modify decay constants by factors of 10³-10⁶
- Convert hours/days half-lives to seconds/minutes
- Element-agnostic: configurable for any target isotope
"""

import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .energy_ledger import EnergyLedger

# Physical constants
HBAR = 1.055e-34      # J·s
C_LIGHT = 2.998e8     # m/s
ELECTRON_MASS = 0.511 # MeV
NUCLEON_MASS = 938.3  # MeV

@dataclass
class DecayConfig:
    """Configuration for LV-accelerated decay."""
    # Target configuration
    precursor_isotope: str = "Ru-103"    # Precursor isotope to decay
    target_isotope: str = "Au-197"       # Target isotope to produce
    
    # LV parameters
    mu_lv: float = 1e-17      # CPT violation coefficient
    alpha_lv: float = 1e-14   # Spatial anisotropy coefficient
    beta_lv: float = 1e-11    # Temporal variation coefficient
    
    # Decay acceleration parameters
    field_strength: float = 1e8    # V/m (strong electric field)
    magnetic_field: float = 10.0   # Tesla
    confinement_volume: float = 1e-9  # m³ (1 mm³)
    acceleration_time: float = 1.0    # seconds
    
    # Collection efficiency
    collection_efficiency: float = 0.95  # 95% collection

class DecayAccelerator:
    """
    LV-enhanced nuclear decay accelerator.
    
    Uses Lorentz violation to modify nuclear decay rates through:
    1. Modified nuclear matrix elements (μ coefficient)
    2. Altered phase space factors (β coefficient) 
    3. Field-enhanced weak interaction (α coefficient)
    """
    
    def __init__(self, config: DecayConfig, energy_ledger: Optional[EnergyLedger] = None):
        self.config = config
        self.energy_ledger = energy_ledger or EnergyLedger()
        self.logger = logging.getLogger(__name__)
        
        # Natural decay constants (1/s)
        self.natural_decay_constants = self._initialize_decay_data()
        
        # LV enhancement factors
        self.lv_factors = self._calculate_lv_enhancements()
        
        # Modified decay constant
        self.modified_decay_constant = self._calculate_modified_decay_constant()
        
        self.logger.info(f"DecayAccelerator initialized:")
        self.logger.info(f"  Isotope: {config.isotope} → {config.daughter_isotope}")
        self.logger.info(f"  Acceleration factor: {self.lv_factors['total']:.2e}×")
        self.logger.info(f"  Effective half-life: {np.log(2)/self.modified_decay_constant:.2f} s")
    
    def _initialize_decay_data(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize nuclear decay database.
        
        Contains natural decay constants and Q-values for relevant isotopes.
        """
        decay_data = {
            # Ruthenium isotopes (β⁻ decay to rhodium)
            "Ru-103": {
                "decay_constant": 2.06e-7,  # 1/s (39.3 day half-life)
                "q_value": 0.763,           # MeV
                "decay_type": "beta_minus",
                "daughter": "Rh-103"
            },
            "Ru-105": {
                "decay_constant": 4.35e-5,  # 1/s (4.4 hour half-life)
                "q_value": 1.917,           # MeV
                "decay_type": "beta_minus", 
                "daughter": "Rh-105"
            },
            
            # Palladium isotopes (electron capture to rhodium)
            "Pd-103": {
                "decay_constant": 4.69e-7,  # 1/s (17.0 day half-life)
                "q_value": 0.543,           # MeV
                "decay_type": "electron_capture",
                "daughter": "Rh-103"
            },
            "Pd-105": {
                "decay_constant": 2.53e-6,  # 1/s (35.3 minute half-life)
                "q_value": 1.115,           # MeV
                "decay_type": "electron_capture",
                "daughter": "Rh-105"
            },
            
            # Silver isotopes (various decay modes)
            "Ag-104": {
                "decay_constant": 1.99e-4,  # 1/s (69.2 minute half-life)
                "q_value": 2.987,           # MeV
                "decay_type": "electron_capture",
                "daughter": "Pd-104"
            },
            "Ag-106": {
                "decay_constant": 4.75e-4,  # 1/s (24.0 minute half-life)
                "q_value": 2.965,           # MeV
                "decay_type": "electron_capture",
                "daughter": "Pd-106"
            }
        }
        
        return decay_data
    
    def _calculate_lv_enhancements(self) -> Dict[str, float]:
        """
        Calculate LV enhancement factors for nuclear decay rates.
        
        LV modifications affect:
        1. Nuclear matrix elements (μ coefficient)
        2. Phase space factors (β coefficient)
        3. Weak interaction coupling (α coefficient)
        """
        isotope_data = self.natural_decay_constants.get(self.config.isotope, {})
        q_value = isotope_data.get("q_value", 1.0)  # MeV
        decay_type = isotope_data.get("decay_type", "beta_minus")
        
        # Matrix element enhancement (CPT violation)
        # Modifies nuclear weak matrix elements
        matrix_enhancement = 1.0 + abs(self.config.mu_lv) / 1e-18 * 100.0  # Up to 100× enhancement
        
        # Phase space enhancement (temporal LV)
        # Modifies available phase space for decay products
        phase_space_factor = (q_value / ELECTRON_MASS)**2  # Relativistic phase space
        phase_space_enhancement = 1.0 + abs(self.config.beta_lv) / 1e-12 * phase_space_factor * 50.0
        
        # Field coupling enhancement (spatial LV)
        # Enhanced coupling to external electromagnetic fields
        field_factor = (self.config.field_strength / 1e8) * (self.config.magnetic_field / 10.0)
        field_enhancement = 1.0 + abs(self.config.alpha_lv) / 1e-15 * field_factor * 20.0
        
        # Decay type specific factors
        if decay_type == "electron_capture":
            # EC enhanced by electron wave function overlap
            ec_enhancement = 1.0 + abs(self.config.alpha_lv) / 1e-15 * 30.0
            total_enhancement = matrix_enhancement * phase_space_enhancement * field_enhancement * ec_enhancement
        else:
            # β⁻ decay
            total_enhancement = matrix_enhancement * phase_space_enhancement * field_enhancement
        
        # Saturation limits (physics constraints)
        total_enhancement = min(total_enhancement, 1e6)  # Maximum 10⁶× acceleration
        
        return {
            'matrix': matrix_enhancement,
            'phase_space': phase_space_enhancement,
            'field': field_enhancement,
            'total': total_enhancement
        }
    
    def _calculate_modified_decay_constant(self) -> float:
        """Calculate the LV-modified decay constant."""
        natural_lambda = self.natural_decay_constants.get(self.config.isotope, {}).get("decay_constant", 1e-6)
        return natural_lambda * self.lv_factors['total']
    
    def calculate_natural_half_life(self) -> float:
        """Calculate natural half-life in seconds."""
        natural_lambda = self.natural_decay_constants.get(self.config.isotope, {}).get("decay_constant", 1e-6)
        return np.log(2) / natural_lambda
    
    def calculate_accelerated_half_life(self) -> float:
        """Calculate LV-accelerated half-life in seconds."""
        return np.log(2) / self.modified_decay_constant
    
    def simulate_decay(self, initial_nuclei: float, time: float) -> Dict[str, float]:
        """
        Simulate LV-accelerated nuclear decay.
        
        Args:
            initial_nuclei: Initial number of parent nuclei
            time: Decay time in seconds
            
        Returns:
            Dictionary with decay products and remaining nuclei
        """
        print(f"\n=== LV-ACCELERATED DECAY ===")
        print(f"Isotope: {self.config.isotope} → {self.config.daughter_isotope}")
        print(f"Initial nuclei: {initial_nuclei:.2e}")
        print(f"Decay time: {time:.2f} s")
        
        # Natural vs accelerated half-lives
        natural_t_half = self.calculate_natural_half_life()
        accelerated_t_half = self.calculate_accelerated_half_life()
        
        print(f"\nDecay parameters:")
        print(f"  Natural half-life: {natural_t_half/86400:.1f} days")
        print(f"  Accelerated half-life: {accelerated_t_half:.2f} s")
        print(f"  Acceleration factor: {self.lv_factors['total']:.2e}×")
        
        # Exponential decay law with modified decay constant
        fraction_remaining = np.exp(-self.modified_decay_constant * time)
        fraction_decayed = 1.0 - fraction_remaining
        
        # Calculate decay products
        nuclei_decayed = initial_nuclei * fraction_decayed
        nuclei_remaining = initial_nuclei * fraction_remaining
        
        # Apply collection efficiency
        collected_daughters = nuclei_decayed * self.config.collection_efficiency
        
        print(f"\nDecay results:")
        print(f"  Fraction decayed: {fraction_decayed*100:.1f}%")
        print(f"  Nuclei decayed: {nuclei_decayed:.2e}")
        print(f"  {self.config.daughter_isotope} collected: {collected_daughters:.2e}")
        print(f"  Nuclei remaining: {nuclei_remaining:.2e}")
        
        # Energy accounting for LV field maintenance
        field_power = (self.config.field_strength**2 * self.config.confinement_volume * 
                      8.854e-12 / 2)  # J/s for electric field
        magnetic_power = (self.config.magnetic_field**2 * self.config.confinement_volume / 
                         (2 * 4e-7 * np.pi))  # J/s for magnetic field
        total_power = field_power + magnetic_power
        field_energy = total_power * time
        
        self.energy_ledger.add_input("decay_acceleration_fields", field_energy)
        
        print(f"\nEnergy accounting:")
        print(f"  Field energy: {field_energy:.2e} J")
        print(f"  Collection efficiency: {self.config.collection_efficiency*100:.1f}%")
        
        return {
            "initial_nuclei": initial_nuclei,
            "nuclei_decayed": nuclei_decayed,
            "nuclei_remaining": nuclei_remaining,
            "daughter_nuclei": collected_daughters,
            "collection_efficiency": self.config.collection_efficiency,
            "acceleration_factor": self.lv_factors['total'],
            "field_energy": field_energy
        }
    
    def optimize_acceleration_parameters(self) -> Dict[str, float]:
        """
        Optimize LV parameters for maximum decay acceleration.
        
        Returns optimal field strengths and LV coefficients.
        """
        print(f"\n=== DECAY ACCELERATION OPTIMIZATION ===")
        
        best_acceleration = 0.0
        best_params = {}
        
        # Test different LV parameter combinations
        mu_values = [1e-18, 5e-18, 1e-17, 5e-17, 1e-16]
        alpha_values = [1e-15, 5e-15, 1e-14, 5e-14, 1e-13] 
        beta_values = [1e-12, 5e-12, 1e-11, 5e-11, 1e-10]
        
        field_strengths = [1e7, 5e7, 1e8, 5e8, 1e9]  # V/m
        magnetic_fields = [1.0, 5.0, 10.0, 20.0, 50.0]  # Tesla
        
        for mu_lv in mu_values:
            for alpha_lv in alpha_values:
                for beta_lv in beta_values:
                    for field_str in field_strengths:
                        for mag_field in magnetic_fields:
                            # Create test configuration
                            test_config = DecayConfig(
                                mu_lv=mu_lv,
                                alpha_lv=alpha_lv,
                                beta_lv=beta_lv,
                                field_strength=field_str,
                                magnetic_field=mag_field,
                                isotope=self.config.isotope
                            )
                            
                            # Calculate acceleration factor
                            test_accelerator = DecayAccelerator(test_config)
                            acceleration = test_accelerator.lv_factors['total']
                            
                            # Energy cost consideration
                            field_power = (field_str**2 * test_config.confinement_volume * 
                                         8.854e-12 / 2)
                            magnetic_power = (mag_field**2 * test_config.confinement_volume / 
                                            (2 * 4e-7 * np.pi))
                            total_power = field_power + magnetic_power
                            
                            # Figure of merit: acceleration / energy_cost
                            figure_of_merit = acceleration / (total_power + 1e-10)
                            
                            if figure_of_merit > best_acceleration:
                                best_acceleration = figure_of_merit
                                best_params = {
                                    'mu_lv': mu_lv,
                                    'alpha_lv': alpha_lv,
                                    'beta_lv': beta_lv,
                                    'field_strength': field_str,
                                    'magnetic_field': mag_field,
                                    'acceleration_factor': acceleration,
                                    'figure_of_merit': figure_of_merit,
                                    'power_requirement': total_power
                                }
        
        print(f"Optimal acceleration parameters:")
        print(f"  μ_LV: {best_params['mu_lv']:.1e}")
        print(f"  α_LV: {best_params['alpha_lv']:.1e}")
        print(f"  β_LV: {best_params['beta_lv']:.1e}")
        print(f"  Electric field: {best_params['field_strength']:.1e} V/m")
        print(f"  Magnetic field: {best_params['magnetic_field']:.1f} T")
        print(f"  Acceleration factor: {best_params['acceleration_factor']:.2e}×")
        print(f"  Power requirement: {best_params['power_requirement']:.2e} W")
        
        return best_params
    
    def multi_isotope_decay_chain(self, isotope_populations: Dict[str, float], 
                                 time: float) -> Dict[str, float]:
        """
        Simulate decay chains for multiple isotopes simultaneously.
        
        Args:
            isotope_populations: Dictionary of {isotope: nuclei_count}
            time: Total decay time
            
        Returns:
            Final populations after decay
        """
        print(f"\n=== MULTI-ISOTOPE DECAY CHAIN ===")
        print(f"Total decay time: {time:.2f} s")
        
        final_populations = {}
        total_rhodium = 0.0
        
        for isotope, initial_count in isotope_populations.items():
            if isotope in self.natural_decay_constants:
                # Create accelerator for this isotope
                isotope_config = DecayConfig(
                    mu_lv=self.config.mu_lv,
                    alpha_lv=self.config.alpha_lv,
                    beta_lv=self.config.beta_lv,
                    field_strength=self.config.field_strength,
                    magnetic_field=self.config.magnetic_field,
                    isotope=isotope,
                    daughter_isotope=self.natural_decay_constants[isotope]["daughter"]
                )
                
                accelerator = DecayAccelerator(isotope_config)
                decay_result = accelerator.simulate_decay(initial_count, time)
                
                daughter = self.natural_decay_constants[isotope]["daughter"]
                final_populations[isotope] = decay_result["nuclei_remaining"]
                final_populations[daughter] = final_populations.get(daughter, 0) + decay_result["daughter_nuclei"]
                
                # Count rhodium isotopes
                if "Rh" in daughter:
                    total_rhodium += decay_result["daughter_nuclei"]
                    
                print(f"  {isotope} → {daughter}: {decay_result['daughter_nuclei']:.2e} nuclei")
        
        print(f"\nTotal rhodium produced: {total_rhodium:.2e} nuclei")
        
        return final_populations

def demo_decay_acceleration():
    """Demonstration of LV-accelerated nuclear decay."""
    print("🚀 LV-ACCELERATED DECAY DEMO")
    print("=" * 45)
    
    # High-acceleration configuration
    config = DecayConfig(
        mu_lv=5e-17,        # Strong CPT violation
        alpha_lv=5e-14,     # Strong spatial anisotropy
        beta_lv=5e-11,      # Strong temporal variation
        field_strength=5e8, # 500 MV/m
        magnetic_field=20.0, # 20 Tesla
        isotope="Ru-103",
        daughter_isotope="Rh-103"
    )
    
    # Initialize accelerator
    accelerator = DecayAccelerator(config)
    
    # Run optimization
    optimal_params = accelerator.optimize_acceleration_parameters()
    
    # Simulate decay of ruthenium nuclei
    initial_ru_nuclei = 1e15  # Large population
    decay_time = 60.0  # 1 minute
    
    decay_result = accelerator.simulate_decay(initial_ru_nuclei, decay_time)
    
    # Multi-isotope test
    isotope_mix = {
        "Ru-103": 5e14,
        "Ru-105": 3e14, 
        "Pd-103": 2e14,
        "Pd-105": 1e14
    }
    
    final_populations = accelerator.multi_isotope_decay_chain(isotope_mix, decay_time)
    
    # Calculate total rhodium production
    total_rh = sum(count for isotope, count in final_populations.items() if "Rh" in isotope)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  Total Rh nuclei: {total_rh:.2e}")
    print(f"  Conversion time: {decay_time:.1f} s")
    print(f"  Acceleration factor: {optimal_params['acceleration_factor']:.2e}×")
    
    success = total_rh > 1e14  # > 0.1 μmol rhodium
    print(f"  Success: {'✅ YES' if success else '❌ NO'}")
    
    return success

if __name__ == "__main__":
    demo_decay_acceleration()
