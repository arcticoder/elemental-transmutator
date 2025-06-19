"""
Advanced spallation transmutation module with Lorentz violation enhancement.
Element-agnostic implementation for transmutation pipeline.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

@dataclass
class TransmutationConfig:
    """Configuration for spallation transmutation parameters."""
    beam_energy: float = 1.5e9  # eV (1.5 GeV)
    beam_type: str = "proton"
    beam_flux: float = 1e15     # particles/cm²/s
    target_isotope: str = "Au-197"
    feedstock_isotope: str = "Pt-195"
    
    # Lorentz violation parameters
    mu_lv: float = 2.5e-12      # LV mass scale (eV)
    alpha_lv: float = 0.85      # Cross-section enhancement
    beta_lv: float = 0.65       # Energy threshold reduction
    
    # Material properties
    beam_width_m: float = 0.02  # 2 cm beam width
    irradiation_time_s: float = 3600  # 1 hour
    material_density_g_cm3: float = 21.0  # Platinum-like density

class SpallationTransmuter:
    """
    Advanced spallation transmutation engine with Lorentz violation enhancement.
    """
    
    def __init__(self, config: TransmutationConfig = None):
        """Initialize the transmutation engine with optional configuration."""
        self.config = config or TransmutationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load element-specific configuration if available
        self._load_element_config()
        
        # Parse isotope information
        self.feedstock_z, self.feedstock_a = self._parse_isotope(self.config.feedstock_isotope)
        self.target_z, self.target_a = self._parse_isotope(self.config.target_isotope)
        
        # Pre-calculate cross-sections for efficiency
        self.cross_sections = self._calculate_cross_sections(
            self.feedstock_z, self.feedstock_a,
            self.target_z, self.target_a
        )
        
        # LV enhancement factors
        self.lv_factors = self._calculate_lv_enhancements()
        
        self.logger.info(f"SpallationTransmuter initialized:")
        self.logger.info(f"  Beam: {self.config.beam_energy/1e6:.1f} MeV {self.config.beam_type}")
        self.logger.info(f"  Feedstock: {self.config.feedstock_isotope}")
        self.logger.info(f"  Target: {self.config.target_isotope}")
        self.logger.info(f"  LV enhancement: {self.lv_factors['total']:.2f}×")
    
    def _load_element_config(self):
        """Load element-specific configuration from config.json if available."""
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
                self.config.target_isotope = cfg.get("target_isotope", self.config.target_isotope)
                self.config.feedstock_isotope = cfg.get("feedstock_isotope", self.config.feedstock_isotope)
                
                # Update beam parameters from config
                beam = cfg.get("beam_profile", {})
                if "energy_MeV" in beam:
                    self.config.beam_energy = beam["energy_MeV"] * 1e6  # Convert to eV
                if "type" in beam:
                    self.config.beam_type = beam["type"]
                if "flux" in beam:
                    self.config.beam_flux = beam["flux"]
                    
                # Update LV parameters
                lv = cfg.get("lv_params", {})
                self.config.mu_lv = lv.get("mu", self.config.mu_lv)
                self.config.alpha_lv = lv.get("alpha", self.config.alpha_lv)
                self.config.beta_lv = lv.get("beta", self.config.beta_lv)
                
        except FileNotFoundError:
            self.logger.info("No config.json found, using default configuration")
        except Exception as e:
            self.logger.warning(f"Error loading config.json: {e}")
    
    def _parse_isotope(self, isotope: str) -> Tuple[int, int]:
        """Parse isotope string to get atomic number (Z) and mass number (A)."""
        # Extended mapping for feedstock analysis
        element_map = {
            "Fe": 26, "Ag": 47, "Au": 79, "Pt": 78, "Pd": 46, "Rh": 45,
            "Cd": 48, "Cu": 29, "Ni": 28, "Co": 27, "Zn": 30, "Hg": 80,
            "Pb": 82, "Bi": 83, "Tl": 81, "W": 74, "Ta": 73, "Re": 75
        }
        
        parts = isotope.split("-")
        element = parts[0]
        mass_number = int(parts[1])
        atomic_number = element_map.get(element, 26)  # Default to Fe
        
        return atomic_number, mass_number
    
    def _calculate_cross_sections(self, feedstock_z: int, feedstock_a: int, 
                                target_z: int, target_a: int) -> Dict[str, Dict[str, float]]:
        """Calculate cross-sections using improved semi-empirical formulas."""
        # Base cross-section using geometric model with LV enhancement
        beam_energy_mev = self.config.beam_energy / 1e6
        
        # Improved semi-empirical formula for spallation cross-sections
        # σ = σ₀ × (A_feedstock)^α × (E_beam)^β × f_proximity × f_LV
        
        # Base cross-section depends on proximity in atomic number
        z_diff = abs(target_z - feedstock_z)
        if z_diff <= 2:
            sigma_0 = 150.0  # millibarns - same or nearby elements
        elif z_diff <= 5:
            sigma_0 = 80.0   # millibarns - moderate separation
        else:
            sigma_0 = 30.0   # millibarns - large separation
        
        # Mass and energy dependence
        alpha = 0.7     # Mass dependence
        beta = 0.4      # Energy dependence
        
        # Calculate proximity factor (favors nearby elements)
        proximity_factor = np.exp(-0.2 * z_diff)
        
        # Mass difference penalty (harder to change mass significantly)
        mass_diff = abs(target_a - feedstock_a)
        mass_penalty = np.exp(-0.05 * mass_diff)
        
        # Base cross-section calculation
        sigma_base = sigma_0 * (feedstock_a ** alpha) * (beam_energy_mev ** beta) * proximity_factor * mass_penalty
        
        # Apply LV enhancement
        lv_enhancement = self.lv_factors.get('cross_section', 1.0)
        sigma_enhanced = sigma_base * lv_enhancement
        
        # Additional reaction channels
        channels = {
            'direct': {
                'cross_section': sigma_enhanced,
                'energy_threshold': 10.0,  # MeV
                'lv_enhancement': lv_enhancement
            },
            'cascade': {
                'cross_section': sigma_enhanced * 0.6,
                'energy_threshold': 50.0,  # MeV
                'lv_enhancement': lv_enhancement * 0.8
            },
            'fragmentation': {
                'cross_section': sigma_enhanced * 0.3,
                'energy_threshold': 100.0,  # MeV
                'lv_enhancement': lv_enhancement * 0.5
            }
        }
        
        return channels
    
    def _calculate_lv_enhancements(self) -> Dict[str, float]:
        """Calculate Lorentz violation enhancement factors."""
        # Energy-dependent LV enhancement
        beam_energy_gev = self.config.beam_energy / 1e9
        
        # LV enhancement grows with energy
        energy_factor = 1.0 + self.config.alpha_lv * (beam_energy_gev / 2.0)
        
        # Mass scale effects
        mass_factor = 1.0 + self.config.beta_lv * np.log(beam_energy_gev + 1)
        
        # Composite enhancement
        total_enhancement = energy_factor * mass_factor
        
        return {
            'cross_section': total_enhancement,
            'energy_threshold': 1.0 / total_enhancement,  # Inverse for thresholds
            'total': total_enhancement
        }
    
    def transmute_sample(self, sample_mass_g: float, irradiation_time_s: float = None) -> Dict[str, Any]:
        """
        Perform spallation transmutation on a sample.
        
        Args:
            sample_mass_g: Mass of feedstock sample in grams
            irradiation_time_s: Irradiation time in seconds (optional)
            
        Returns:
            Dict containing transmutation results
        """
        irradiation_time = irradiation_time_s or self.config.irradiation_time_s
        
        # Calculate number of feedstock nuclei
        avogadro = 6.022e23
        feedstock_nuclei = (sample_mass_g / self.feedstock_a) * avogadro
        
        # Calculate beam parameters
        beam_area_cm2 = np.pi * (self.config.beam_width_m * 100 / 2) ** 2  # Convert to cm²
        total_beam_particles = self.config.beam_flux * beam_area_cm2 * irradiation_time
        
        # Calculate reaction rates for each channel
        results = {}
        total_transmuted = 0
        
        for channel_name, channel_data in self.cross_sections.items():
            sigma_mb = channel_data['cross_section']
            sigma_cm2 = sigma_mb * 1e-27  # Convert millibarns to cm²
            
            # Apply energy threshold check
            if self.config.beam_energy / 1e6 >= channel_data['energy_threshold']:
                # Reaction rate = σ × Φ × N × t
                reaction_rate = sigma_cm2 * self.config.beam_flux * feedstock_nuclei
                transmuted_nuclei = reaction_rate * irradiation_time
                
                # Convert to mass
                transmuted_mass_g = (transmuted_nuclei / avogadro) * self.target_a
                
                results[channel_name] = {
                    'cross_section_mb': sigma_mb,
                    'reaction_rate_per_s': reaction_rate,
                    'transmuted_nuclei': transmuted_nuclei,
                    'transmuted_mass_g': transmuted_mass_g,
                    'yield_fraction': transmuted_nuclei / feedstock_nuclei,
                    'lv_enhancement': channel_data['lv_enhancement']
                }
                
                total_transmuted += transmuted_nuclei
            else:
                results[channel_name] = {
                    'cross_section_mb': sigma_mb,
                    'reaction_rate_per_s': 0.0,
                    'transmuted_nuclei': 0.0,
                    'transmuted_mass_g': 0.0,
                    'yield_fraction': 0.0,
                    'lv_enhancement': channel_data['lv_enhancement']
                }
        
        # Calculate total yields
        total_yield_mass_g = sum(ch['transmuted_mass_g'] for ch in results.values())
        total_yield_fraction = total_transmuted / feedstock_nuclei if feedstock_nuclei > 0 else 0
        
        # Add summary
        results['summary'] = {
            'feedstock_isotope': self.config.feedstock_isotope,
            'target_isotope': self.config.target_isotope,
            'sample_mass_g': sample_mass_g,
            'irradiation_time_s': irradiation_time,
            'total_yield_mass_g': total_yield_mass_g,
            'total_yield_fraction': total_yield_fraction,
            'conversion_efficiency': total_yield_fraction * 100,  # Percentage
            'beam_energy_mev': self.config.beam_energy / 1e6,
            'lv_total_enhancement': self.lv_factors['total']
        }
        
        return results
    
    def optimize_beam_energy(self, sample_mass_g: float, 
                           energy_range_mev: Tuple[float, float] = (100, 3000),
                           energy_steps: int = 20) -> Dict[str, Any]:
        """
        Optimize beam energy for maximum transmutation yield.
        
        Args:
            sample_mass_g: Sample mass in grams
            energy_range_mev: Energy range to scan (min_MeV, max_MeV)
            energy_steps: Number of energy points to evaluate
            
        Returns:
            Dict containing optimization results
        """
        original_energy = self.config.beam_energy
        
        energies_mev = np.linspace(energy_range_mev[0], energy_range_mev[1], energy_steps)
        yields = []
        
        for energy_mev in energies_mev:
            # Update beam energy
            self.config.beam_energy = energy_mev * 1e6  # Convert to eV
            
            # Recalculate cross-sections and LV factors
            self.cross_sections = self._calculate_cross_sections(
                self.feedstock_z, self.feedstock_a,
                self.target_z, self.target_a
            )
            self.lv_factors = self._calculate_lv_enhancements()
            
            # Calculate yield
            result = self.transmute_sample(sample_mass_g)
            yields.append(result['summary']['total_yield_mass_g'])
        
        # Find optimal energy
        optimal_idx = np.argmax(yields)
        optimal_energy_mev = energies_mev[optimal_idx]
        optimal_yield_g = yields[optimal_idx]
        
        # Restore original energy
        self.config.beam_energy = original_energy
        self.cross_sections = self._calculate_cross_sections(
            self.feedstock_z, self.feedstock_a,
            self.target_z, self.target_a
        )
        self.lv_factors = self._calculate_lv_enhancements()
        
        return {
            'optimal_energy_mev': optimal_energy_mev,
            'optimal_yield_g': optimal_yield_g,
            'energy_scan_mev': energies_mev.tolist(),
            'yield_scan_g': yields,
            'improvement_factor': optimal_yield_g / yields[0] if yields[0] > 0 else float('inf')
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics for the transmutation setup."""
        return {
            'beam_energy_mev': self.config.beam_energy / 1e6,
            'beam_flux_per_cm2_s': self.config.beam_flux,
            'lv_enhancement_factor': self.lv_factors['total'],
            'primary_cross_section_mb': self.cross_sections['direct']['cross_section'],
            'feedstock_z': self.feedstock_z,
            'feedstock_a': self.feedstock_a,
            'target_z': self.target_z,
            'target_a': self.target_a
        }

def create_transmuter_from_config(config_file: str = "config.json") -> SpallationTransmuter:
    """Factory function to create a transmuter from a configuration file."""
    config = TransmutationConfig()
    
    try:
        with open(config_file, "r") as f:
            cfg = json.load(f)
            
        # Update configuration from file
        config.target_isotope = cfg.get("target_isotope", config.target_isotope)
        config.feedstock_isotope = cfg.get("feedstock_isotope", config.feedstock_isotope)
        
        beam = cfg.get("beam_profile", {})
        if "energy_MeV" in beam:
            config.beam_energy = beam["energy_MeV"] * 1e6
        if "type" in beam:
            config.beam_type = beam["type"]
        if "flux" in beam:
            config.beam_flux = beam["flux"]
            
        lv = cfg.get("lv_params", {})
        config.mu_lv = lv.get("mu", config.mu_lv)
        config.alpha_lv = lv.get("alpha", config.alpha_lv)
        config.beta_lv = lv.get("beta", config.beta_lv)
        
    except FileNotFoundError:
        print(f"Config file {config_file} not found, using defaults")
    except Exception as e:
        print(f"Error loading config: {e}")
    
    return SpallationTransmuter(config)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create transmuter with default configuration
    transmuter = SpallationTransmuter()
    
    # Perform transmutation on 1g sample
    results = transmuter.transmute_sample(1.0)
    
    print(f"\nTransmutation Results:")
    print(f"Feedstock: {results['summary']['feedstock_isotope']}")
    print(f"Target: {results['summary']['target_isotope']}")
    print(f"Input mass: {results['summary']['sample_mass_g']:.3f} g")
    print(f"Output mass: {results['summary']['total_yield_mass_g']:.6f} g")
    print(f"Conversion efficiency: {results['summary']['conversion_efficiency']:.4f}%")
    print(f"LV enhancement: {results['summary']['lv_total_enhancement']:.2f}×")
    
    # Show channel breakdown
    print(f"\nChannel breakdown:")
    for channel, data in results.items():
        if channel != 'summary':
            print(f"  {channel}: {data['transmuted_mass_g']:.6f} g "
                  f"(σ={data['cross_section_mb']:.2f} mb, "
                  f"LV={data['lv_enhancement']:.2f}×)")
