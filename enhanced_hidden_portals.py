#!/usr/bin/env python3
"""
Enhanced Hidden Sector Portals: Axion and Dark Photon Energy Channels
=====================================================================

This module implements enhanced hidden sector portal couplings with LV-modified
mixing angles for energy transfer through axion-like particles and dark photons.

Key Features:
1. Axion-photon coupling with running coupling constants
2. Dark photon kinetic mixing portals
3. LV-modified portal strengths and energy scales
4. Hidden sector energy reservoir tapping
5. Visible sector energy materialization

Physics:
- Axion coupling: L = (1/4) g_aγγ a F_μν F̃^μν
- Dark photon mixing: L = (ε/2) F_μν F'^μν
- LV running couplings: g(E), ε(E) with energy-dependent modifications
- Portal energy transfer rates and efficiency optimization

Author: LV Energy Converter Framework
"""

import numpy as np
from scipy.special import kv, iv, hyp2f1
from scipy import integrate, optimize, interpolate
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class HiddenPortalConfig:
    """Configuration for hidden sector portal couplings."""
    
    # Axion parameters
    axion_mass: float = 1e-5              # Axion mass (eV)
    axion_decay_constant: float = 1e12    # Axion decay constant (GeV)
    axion_photon_coupling: float = 1e-10  # g_aγγ coupling constant
    axion_abundance: float = 0.1          # Hidden sector axion abundance
    
    # Dark photon parameters
    dark_photon_mass: float = 1e-3        # Dark photon mass (eV)
    kinetic_mixing: float = 1e-6          # ε kinetic mixing parameter
    dark_sector_temperature: float = 1e-3 # Dark sector temperature (eV)
    dark_photon_abundance: float = 0.1    # Dark photon abundance
    
    # LV parameters
    mu_lv: float = 1e-17                  # CPT violation coefficient
    alpha_lv: float = 1e-14               # Lorentz violation coefficient
    beta_lv: float = 1e-11                # Gravitational LV coefficient
    
    # Portal coupling parameters
    portal_resonance_frequency: float = 1e10  # Portal resonance (Hz)
    portal_quality_factor: float = 1e6        # Portal Q factor
    portal_coupling_strength: float = 0.1     # Portal coupling strength
    coherence_time: float = 1e-3              # Portal coherence time (s)
    
    # Energy extraction parameters
    extraction_volume: float = 1e-6           # Extraction volume (m³)
    extraction_efficiency: float = 0.1        # Extraction efficiency
    extraction_time: float = 1.0              # Extraction time (s)
    
    # Computational parameters
    energy_steps: int = 1000                  # Energy discretization
    min_energy: float = 1e-6                  # Minimum energy (eV)
    max_energy: float = 1e3                   # Maximum energy (eV)

class EnhancedHiddenPortals:
    """
    Enhanced hidden sector portal system with LV modifications.
    
    This class implements energy transfer through axion-like particles
    and dark photons with LV-enhanced coupling strengths.
    """
    
    def __init__(self, config: HiddenPortalConfig):
        self.config = config
        
        # Physical constants
        self.hbar = 1.055e-34    # J⋅s
        self.c = 3e8             # m/s
        self.e = 1.602e-19       # C
        self.eV_to_J = 1.602e-19 # eV to Joules
        self.GeV_to_eV = 1e9     # GeV to eV
        
        # Energy array for calculations
        self.energy_array = np.logspace(
            np.log10(self.config.min_energy),
            np.log10(self.config.max_energy),
            self.config.energy_steps
        )  # eV
        
        # Initialize running couplings
        self._calculate_running_couplings()
        
        # Initialize portal transfer functions
        self._calculate_portal_transfer_rates()
    
    def _calculate_running_couplings(self):
        """Calculate energy-dependent coupling constants with LV modifications."""
        # Axion-photon coupling with LV running
        self.g_agg_running = np.zeros(len(self.energy_array))
        
        # Dark photon kinetic mixing with LV running
        self.epsilon_running = np.zeros(len(self.energy_array))
        
        for i, E in enumerate(self.energy_array):
            # Energy scale in GeV
            E_GeV = E / self.GeV_to_eV
            
            # LV enhancement factors
            lv_enhancement_axion = self._lv_enhancement_factor(E, 'axion')
            lv_enhancement_dark_photon = self._lv_enhancement_factor(E, 'dark_photon')
            
            # Running axion coupling
            # Base coupling with logarithmic running and LV enhancement
            beta_g = 0.1  # Simplified beta function coefficient
            g_base = self.config.axion_photon_coupling
            self.g_agg_running[i] = g_base * (1 + beta_g * np.log(E_GeV)) * lv_enhancement_axion
            
            # Running kinetic mixing
            # Base mixing with RG evolution and LV enhancement
            beta_eps = 0.05  # Simplified beta function coefficient
            eps_base = self.config.kinetic_mixing
            self.epsilon_running[i] = eps_base * (1 + beta_eps * np.log(E_GeV)) * lv_enhancement_dark_photon
    
    def _lv_enhancement_factor(self, energy: float, portal_type: str) -> float:
        """Calculate LV enhancement factor for portal coupling."""
        # Energy scale
        E_scale = energy * self.eV_to_J / self.hbar  # s^-1
        
        # LV corrections depend on portal type
        if portal_type == 'axion':
            # Axion couples to electromagnetic field
            enhancement = (1 + self.config.alpha_lv * E_scale +
                          self.config.mu_lv * E_scale**2)
        elif portal_type == 'dark_photon':
            # Dark photon kinetic mixing
            enhancement = (1 + self.config.beta_lv * np.sqrt(E_scale) +
                          self.config.alpha_lv * E_scale)
        else:
            enhancement = 1.0
        
        return enhancement
    
    def _calculate_portal_transfer_rates(self):
        """Calculate energy transfer rates through portals."""
        # Axion portal transfer rates
        self.axion_transfer_rates = np.zeros(len(self.energy_array))
        
        # Dark photon portal transfer rates
        self.dark_photon_transfer_rates = np.zeros(len(self.energy_array))
        
        for i, E in enumerate(self.energy_array):
            # Axion transfer rate
            self.axion_transfer_rates[i] = self._axion_transfer_rate(E, i)
            
            # Dark photon transfer rate
            self.dark_photon_transfer_rates[i] = self._dark_photon_transfer_rate(E, i)
    
    def _axion_transfer_rate(self, energy: float, index: int) -> float:
        """Calculate axion portal energy transfer rate."""
        # Axion-photon coupling at this energy
        g_agg = self.g_agg_running[index]
        
        # Axion mass and decay constant
        m_a = self.config.axion_mass * self.eV_to_J
        f_a = self.config.axion_decay_constant * self.GeV_to_eV * self.eV_to_J
        
        # Energy in Joules
        E_J = energy * self.eV_to_J
        
        # Phase space factor
        if E_J > m_a:
            phase_space = np.sqrt(1 - (m_a / E_J)**2)
        else:
            phase_space = 0.0
        
        # Transfer rate with resonance enhancement
        resonance_factor = self._portal_resonance_factor(energy)
        
        # Base transfer rate: Γ ∝ g²E³/f²
        base_rate = (g_agg**2 * E_J**3) / (f_a**2 * self.hbar)
        
        return base_rate * phase_space * resonance_factor * self.config.axion_abundance
    
    def _dark_photon_transfer_rate(self, energy: float, index: int) -> float:
        """Calculate dark photon portal energy transfer rate."""
        # Kinetic mixing at this energy
        epsilon = self.epsilon_running[index]
        
        # Dark photon mass
        m_dp = self.config.dark_photon_mass * self.eV_to_J
        
        # Energy in Joules
        E_J = energy * self.eV_to_J
        
        # Phase space factor
        if E_J > m_dp:
            phase_space = np.sqrt(1 - (m_dp / E_J)**2)
        else:
            phase_space = 0.0
        
        # Thermal factor from dark sector
        T_dark = self.config.dark_sector_temperature * self.eV_to_J
        thermal_factor = np.exp(-E_J / T_dark) if T_dark > 0 else 1.0
        
        # Transfer rate with resonance enhancement
        resonance_factor = self._portal_resonance_factor(energy)
        
        # Base transfer rate: Γ ∝ ε²E
        base_rate = (epsilon**2 * E_J) / self.hbar
        
        return base_rate * phase_space * thermal_factor * resonance_factor * self.config.dark_photon_abundance
    
    def _portal_resonance_factor(self, energy: float) -> float:
        """Calculate portal resonance enhancement factor."""
        # Convert to frequency
        freq = energy * self.eV_to_J / self.hbar  # Hz
        
        # Resonance enhancement
        resonance_freq = self.config.portal_resonance_frequency
        Q = self.config.portal_quality_factor
        
        # Lorentzian resonance
        detuning = abs(freq - resonance_freq) / resonance_freq
        resonance_factor = 1 + self.config.portal_coupling_strength / (1 + Q**2 * detuning**2)
        
        return resonance_factor
    
    def calculate_total_portal_power(self) -> float:
        """
        Calculate total power transfer through all portals.
        
        Returns:
        --------
        float
            Total portal power (W)
        """
        # Integrate over energy spectrum
        dE = np.diff(self.energy_array)
        dE = np.append(dE, dE[-1])  # Extend for integration
        
        # Power contributions
        axion_power = np.sum(self.axion_transfer_rates * self.energy_array * self.eV_to_J * dE)
        dark_photon_power = np.sum(self.dark_photon_transfer_rates * self.energy_array * self.eV_to_J * dE)
        
        total_power = (axion_power + dark_photon_power) * self.config.extraction_efficiency
        
        return total_power
    
    def calculate_portal_energy_spectrum(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate energy spectrum of portal energy transfer.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Energy array (eV), axion spectrum (W/eV), dark photon spectrum (W/eV)
        """
        # Power spectral density
        axion_spectrum = self.axion_transfer_rates * self.energy_array * self.eV_to_J
        dark_photon_spectrum = self.dark_photon_transfer_rates * self.energy_array * self.eV_to_J
        
        return self.energy_array, axion_spectrum, dark_photon_spectrum
    
    def optimize_portal_parameters(self, target_power: float = 1e-15) -> Dict[str, float]:
        """
        Optimize portal parameters for target power extraction.
        
        Parameters:
        -----------
        target_power : float
            Target power extraction (W)
            
        Returns:
        --------
        Dict[str, float]
            Optimized parameters
        """
        def objective(params):
            g_agg, epsilon, resonance_freq = params
            
            # Update configuration
            old_config = (self.config.axion_photon_coupling, 
                         self.config.kinetic_mixing,
                         self.config.portal_resonance_frequency)
            
            self.config.axion_photon_coupling = g_agg
            self.config.kinetic_mixing = epsilon
            self.config.portal_resonance_frequency = resonance_freq
            
            # Recalculate
            self._calculate_running_couplings()
            self._calculate_portal_transfer_rates()
            
            # Calculate power
            power = self.calculate_total_portal_power()
            
            # Restore configuration
            (self.config.axion_photon_coupling, 
             self.config.kinetic_mixing,
             self.config.portal_resonance_frequency) = old_config
            
            return abs(power - target_power)
        
        # Optimization bounds
        bounds = [
            (1e-12, 1e-8),     # g_agg
            (1e-8, 1e-4),      # epsilon
            (1e8, 1e12)        # resonance_freq
        ]
        
        # Initial guess
        x0 = [self.config.axion_photon_coupling, 
              self.config.kinetic_mixing,
              self.config.portal_resonance_frequency]
        
        # Optimize
        result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return {
            'optimal_axion_coupling': result.x[0],
            'optimal_kinetic_mixing': result.x[1],
            'optimal_resonance_frequency': result.x[2],
            'achieved_power': target_power,
            'success': result.success
        }
    
    def simulate_portal_dynamics(self, evolution_time: float = None) -> Dict[str, np.ndarray]:
        """
        Simulate time evolution of portal energy transfer.
        
        Parameters:
        -----------
        evolution_time : Optional[float]
            Evolution time (s)
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Time evolution data
        """
        if evolution_time is None:
            evolution_time = self.config.extraction_time
        
        # Time array
        time_steps = 1000
        times = np.linspace(0, evolution_time, time_steps)
        
        # Initialize arrays
        axion_power = np.zeros(time_steps)
        dark_photon_power = np.zeros(time_steps)
        total_power = np.zeros(time_steps)
        cumulative_energy = np.zeros(time_steps)
        
        # Portal coherence evolution
        coherence_decay = np.exp(-times / self.config.coherence_time)
        
        for i, t in enumerate(times):
            # Time-dependent power with coherence decay
            base_power = self.calculate_total_portal_power()
            
            # Apply coherence factor
            coherence_factor = coherence_decay[i]
            
            # Power components
            energy_spectrum = self.calculate_portal_energy_spectrum()
            
            axion_power[i] = np.trapz(energy_spectrum[1], energy_spectrum[0]) * coherence_factor
            dark_photon_power[i] = np.trapz(energy_spectrum[2], energy_spectrum[0]) * coherence_factor
            total_power[i] = axion_power[i] + dark_photon_power[i]
            
            # Cumulative energy
            if i > 0:
                dt = times[i] - times[i-1]
                cumulative_energy[i] = cumulative_energy[i-1] + total_power[i] * dt
        
        return {
            'times': times,
            'axion_power': axion_power,
            'dark_photon_power': dark_photon_power,
            'total_power': total_power,
            'cumulative_energy': cumulative_energy,
            'coherence_factor': coherence_decay
        }
    
    def visualize_portal_performance(self, save_path: Optional[str] = None):
        """
        Visualize portal energy transfer performance.
        
        Parameters:
        -----------
        save_path : Optional[str]
            Path to save the plot
        """
        # Get simulation data
        dynamics = self.simulate_portal_dynamics()
        energy_spectrum = self.calculate_portal_energy_spectrum()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Energy spectrum
        ax1.loglog(energy_spectrum[0], energy_spectrum[1] * 1e15, 'b-', linewidth=2, label='Axion')
        ax1.loglog(energy_spectrum[0], energy_spectrum[2] * 1e15, 'r-', linewidth=2, label='Dark Photon')
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Power Spectral Density (fW/eV)')
        ax1.set_title('Portal Energy Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Running couplings
        ax2.loglog(self.energy_array, self.g_agg_running, 'g-', linewidth=2, label='g_{aγγ}')
        ax2.loglog(self.energy_array, self.epsilon_running * 1e6, 'orange', linewidth=2, label='ε × 10⁶')
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Coupling Strength')
        ax2.set_title('LV-Enhanced Running Couplings')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Time evolution
        ax3.plot(dynamics['times'] * 1e3, dynamics['total_power'] * 1e15, 'k-', linewidth=2, label='Total')
        ax3.plot(dynamics['times'] * 1e3, dynamics['axion_power'] * 1e15, 'b--', label='Axion')
        ax3.plot(dynamics['times'] * 1e3, dynamics['dark_photon_power'] * 1e15, 'r--', label='Dark Photon')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Power (fW)')
        ax3.set_title('Portal Power Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Cumulative energy
        ax4.plot(dynamics['times'] * 1e3, dynamics['cumulative_energy'] * 1e18, 'purple', linewidth=2)
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Cumulative Energy (aJ)')
        ax4.set_title('Total Extracted Energy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive portal performance report."""
        dynamics = self.simulate_portal_dynamics()
        energy_spectrum = self.calculate_portal_energy_spectrum()
        total_power = self.calculate_total_portal_power()
        
        report = {
            'portal_configuration': {
                'axion_mass': self.config.axion_mass,
                'axion_coupling': self.config.axion_photon_coupling,
                'dark_photon_mass': self.config.dark_photon_mass,
                'kinetic_mixing': self.config.kinetic_mixing,
                'lv_parameters': {
                    'mu_lv': self.config.mu_lv,
                    'alpha_lv': self.config.alpha_lv,
                    'beta_lv': self.config.beta_lv
                }
            },
            'performance_metrics': {
                'total_portal_power': total_power,
                'axion_power_contribution': np.trapz(energy_spectrum[1], energy_spectrum[0]),
                'dark_photon_power_contribution': np.trapz(energy_spectrum[2], energy_spectrum[0]),
                'total_extracted_energy': dynamics['cumulative_energy'][-1],
                'extraction_efficiency': self.config.extraction_efficiency
            },
            'spectral_analysis': {
                'peak_axion_energy': energy_spectrum[0][np.argmax(energy_spectrum[1])],
                'peak_dark_photon_energy': energy_spectrum[0][np.argmax(energy_spectrum[2])],
                'energy_range': [self.config.min_energy, self.config.max_energy],
                'coupling_enhancement_factor': np.max(self.g_agg_running) / self.config.axion_photon_coupling
            },
            'temporal_dynamics': {
                'coherence_time': self.config.coherence_time,
                'extraction_time': self.config.extraction_time,
                'final_coherence_factor': dynamics['coherence_factor'][-1],
                'power_stability': np.std(dynamics['total_power']) / np.mean(dynamics['total_power'])
            }
        }
        
        return report

def demo_enhanced_hidden_portals():
    """Demonstrate enhanced hidden sector portals."""
    print("=== Enhanced Hidden Sector Portals Demo ===")
    
    # Create configuration with strong LV enhancement
    config = HiddenPortalConfig(
        axion_mass=1e-5,                    # 10 μeV axion
        axion_photon_coupling=1e-10,        # Strong axion coupling
        dark_photon_mass=1e-3,              # 1 meV dark photon
        kinetic_mixing=1e-6,                # Kinetic mixing
        mu_lv=1e-17,                       # 100× experimental bound
        alpha_lv=1e-14,                    # 100× experimental bound
        beta_lv=1e-11,                     # 100× experimental bound
        portal_resonance_frequency=1e10,    # 10 GHz resonance
        extraction_efficiency=0.1,          # 10% efficiency
        extraction_time=1.0                 # 1 second
    )
    
    # Initialize portal system
    portals = EnhancedHiddenPortals(config)
    
    # Generate report
    report = portals.generate_report()
    
    print(f"Total Portal Power: {report['performance_metrics']['total_portal_power']:.2e} W")
    print(f"Axion Contribution: {report['performance_metrics']['axion_power_contribution']:.2e} W")
    print(f"Dark Photon Contribution: {report['performance_metrics']['dark_photon_power_contribution']:.2e} W")
    print(f"Total Extracted Energy: {report['performance_metrics']['total_extracted_energy']:.2e} J")
    print(f"Coupling Enhancement: {report['spectral_analysis']['coupling_enhancement_factor']:.1f}×")
    
    # Optimization
    print("\n=== Portal Optimization ===")
    optimal = portals.optimize_portal_parameters(target_power=1e-15)
    print(f"Optimization Success: {optimal['success']}")
    print(f"Optimal Axion Coupling: {optimal['optimal_axion_coupling']:.2e}")
    print(f"Optimal Kinetic Mixing: {optimal['optimal_kinetic_mixing']:.2e}")
    print(f"Optimal Resonance Frequency: {optimal['optimal_resonance_frequency']:.2e} Hz")
    
    # Visualization
    print("\n=== Generating Visualization ===")
    portals.visualize_portal_performance('enhanced_hidden_portals.png')
    
    return portals, report

if __name__ == "__main__":
    demo_enhanced_hidden_portals()
