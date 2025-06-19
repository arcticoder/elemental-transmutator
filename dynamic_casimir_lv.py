#!/usr/bin/env python3
"""
Dynamic Casimir LV: Time-Dependent Vacuum Energy Extraction
===========================================================

This module implements dynamic Casimir effect with LV-enhanced photon production
from time-dependent boundary conditions. Enables MW-scale power extraction from
vacuum fluctuations when LV parameters exceed experimental bounds.

Key Features:
1. **Time-Dependent Boundaries**: Oscillating walls, moving mirrors
2. **LV-Enhanced Resonances**: Modified mode spectrum with broadened resonances  
3. **Photon Production Rate**: Γ_dyn(Ω,μ) ∝ Σₙ|Mₙ(μ)|²δ(Ω-2ωₙ(μ))
4. **Power Extraction**: kW→MW scaling with LV parameter enhancement
5. **Resonance Engineering**: Optimal drive frequencies for maximum yield

Physics Framework:
- Dynamic boundary conditions in LV spacetime
- Enhanced vacuum-to-photon conversion rates
- Frequency-dependent LV amplification
- Integration with Casimir LV foundation

Author: Quantum Geometry Hidden Sector Framework  
"""

import numpy as np
import scipy.special as sp
from scipy import integrate, optimize, fft
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

from casimir_lv import CasimirLVCalculator, CasimirLVConfig

@dataclass
class DynamicCasimirConfig:
    """Configuration for dynamic Casimir LV calculations."""
    # Base Casimir configuration
    casimir_config: CasimirLVConfig = None
    
    # Dynamic parameters
    drive_frequency: float = 1e10        # Drive frequency Ω (Hz)
    drive_amplitude: float = 1e-9        # Oscillation amplitude (m)
    drive_phase: float = 0.0             # Drive phase (rad)
    
    # Cavity parameters  
    cavity_length: float = 1e-4          # Base cavity length (m)
    quality_factor: float = 1e6          # Cavity Q factor
    
    # Time evolution
    time_duration: float = 1e-6          # Simulation time (s)
    time_steps: int = 1000               # Number of time steps
    
    # Power extraction
    extraction_efficiency: float = 0.1   # Power extraction efficiency
    load_resistance: float = 50.0        # Load resistance (Ω)

class DynamicCasimirLV:
    """
    Dynamic Casimir effect calculator with LV enhancements.
    """
    
    def __init__(self, config: DynamicCasimirConfig = None):
        self.config = config or DynamicCasimirConfig()
        
        if self.config.casimir_config is None:
            self.config.casimir_config = CasimirLVConfig()
        
        # Initialize base Casimir calculator
        self.casimir_calc = CasimirLVCalculator(self.config.casimir_config)
        
        # Time array
        self.time_array = np.linspace(0, self.config.time_duration, self.config.time_steps)
        self.dt = self.time_array[1] - self.time_array[0]
        
        print("⚡ Dynamic Casimir LV Calculator Initialized")
        print(f"   Drive frequency: {self.config.drive_frequency:.2e} Hz")
        print(f"   Drive amplitude: {self.config.drive_amplitude:.2e} m")
        print(f"   LV enhancement: μ×{self.config.casimir_config.mu/1e-20:.1f}")
    
    def is_pathway_active(self) -> bool:
        """Check if LV parameters exceed experimental bounds to activate pathway."""
        return self.casimir_calc.is_pathway_active()
    
    def time_dependent_cavity_length(self, t: np.ndarray) -> np.ndarray:
        """
        Compute time-dependent cavity length L(t).
        
        L(t) = L₀ + A cos(Ωt + φ)
        """
        return (self.config.cavity_length + 
                self.config.drive_amplitude * 
                np.cos(self.config.drive_frequency * t + self.config.drive_phase))
    
    def instantaneous_mode_frequencies(self, t: float) -> np.ndarray:
        """
        Compute instantaneous mode frequencies ωₙ(t) for time-dependent cavity.
        """
        L_t = self.time_dependent_cavity_length(np.array([t]))[0]
        
        # Mode numbers (limited by cutoff)
        n_max = int(self.config.casimir_config.k_max * L_t / np.pi)
        n_max = min(n_max, self.config.casimir_config.n_modes)
        n_values = np.arange(1, n_max + 1)
        
        # Perpendicular momenta
        k_perp = n_values * np.pi / L_t
        
        # LV-modified frequencies
        omega_n = self.casimir_calc.lv_dispersion_relation(k_perp)
        
        return omega_n
    
    def adiabatic_mode_evolution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute adiabatic evolution of mode frequencies.
        
        Returns:
        --------
        omega_matrix : np.ndarray
            Mode frequencies vs time [time, mode]
        mode_numbers : np.ndarray
            Mode numbers
        """
        # Determine maximum number of modes across all times
        L_min = self.config.cavity_length - self.config.drive_amplitude
        L_max = self.config.cavity_length + self.config.drive_amplitude
        
        n_max_global = int(self.config.casimir_config.k_max * L_max / np.pi)
        n_max_global = min(n_max_global, self.config.casimir_config.n_modes)
        
        mode_numbers = np.arange(1, n_max_global + 1)
        omega_matrix = np.zeros((len(self.time_array), len(mode_numbers)))
        
        for i, t in enumerate(self.time_array):
            omega_t = self.instantaneous_mode_frequencies(t)
            # Pad with zeros if needed
            omega_matrix[i, :len(omega_t)] = omega_t
        
        return omega_matrix, mode_numbers
    
    def mode_mixing_matrix(self, t: float, dt: float) -> np.ndarray:
        """
        Compute mode mixing matrix for small time step.
        """
        omega_t = self.instantaneous_mode_frequencies(t)
        omega_t_dt = self.instantaneous_mode_frequencies(t + dt)
        
        n_modes = min(len(omega_t), len(omega_t_dt))
        
        # Simplified mode mixing (Berry connection approximation)
        mixing_matrix = np.eye(n_modes)
        
        for i in range(n_modes):
            for j in range(n_modes):
                if i != j and n_modes > 1:
                    # Off-diagonal mixing
                    omega_diff = abs(omega_t[i] - omega_t[j])
                    if omega_diff > 1e-10:
                        coupling = (self.config.drive_frequency * self.config.drive_amplitude / 
                                  self.config.cavity_length) / omega_diff
                        mixing_matrix[i, j] = coupling * dt
        
        return mixing_matrix
    
    def photon_production_amplitude(self, n_initial: int, n_final: int, 
                                  omega_matrix: np.ndarray) -> complex:
        """
        Compute photon production amplitude Mₙ(μ) for mode transition.
        """
        if n_initial >= omega_matrix.shape[1] or n_final >= omega_matrix.shape[1]:
            return 0.0
        
        # Time-dependent phase accumulation
        phase_integral = 0.0
        for i in range(len(self.time_array) - 1):
            omega_avg = 0.5 * (omega_matrix[i, n_initial] + omega_matrix[i+1, n_initial])
            phase_integral += omega_avg * self.dt
        
        # LV enhancement factor
        mu_enhancement = (self.config.casimir_config.mu / 
                         self.casimir_calc.experimental_bounds['mu'])
        alpha_enhancement = (self.config.casimir_config.alpha / 
                           self.casimir_calc.experimental_bounds['alpha'])
        
        lv_factor = np.sqrt(mu_enhancement * alpha_enhancement)
        
        # Base amplitude (parametric driving)
        drive_strength = (self.config.drive_amplitude / self.config.cavity_length *
                         self.config.drive_frequency)
        
        base_amplitude = drive_strength / np.sqrt(abs(n_final - n_initial) + 1)
        
        # LV-enhanced amplitude
        amplitude = base_amplitude * lv_factor * np.exp(1j * phase_integral)
        
        return amplitude
    
    def photon_production_rate(self) -> Dict:
        """
        Compute total photon production rate Γ_dyn(Ω,μ).
        """
        omega_matrix, mode_numbers = self.adiabatic_mode_evolution()
        
        total_rate = 0.0
        mode_contributions = []
        resonance_frequencies = []
        
        for n_i in range(len(mode_numbers)):
            for n_f in range(len(mode_numbers)):
                if n_f != n_i:
                    # Production amplitude
                    amplitude = self.photon_production_amplitude(n_i, n_f, omega_matrix)
                    
                    # Resonance condition: Ω ≈ 2ωₙ
                    avg_frequency = np.mean(omega_matrix[:, n_i])
                    resonance_detuning = abs(self.config.drive_frequency - 2 * avg_frequency)
                    
                    # Lorentzian lineshape (cavity damping)
                    linewidth = avg_frequency / self.config.quality_factor
                    resonance_factor = linewidth / (resonance_detuning**2 + linewidth**2)
                    
                    # Photon production rate for this transition
                    rate_contribution = abs(amplitude)**2 * resonance_factor
                    total_rate += rate_contribution
                    
                    mode_contributions.append({
                        'n_initial': n_i,
                        'n_final': n_f,
                        'amplitude': abs(amplitude),
                        'frequency': avg_frequency,
                        'rate': rate_contribution
                    })
                    
                    # Store resonance frequencies
                    if resonance_factor > 0.1 * linewidth / linewidth**2:  # Significant resonance
                        resonance_frequencies.append(2 * avg_frequency)
        
        # Find dominant resonances
        if resonance_frequencies:
            resonance_frequencies = np.array(resonance_frequencies)
            unique_resonances = []
            
            for freq in resonance_frequencies:
                if not any(abs(freq - ur) / ur < 0.1 for ur in unique_resonances):
                    unique_resonances.append(freq)
        else:
            unique_resonances = []
        
        return {
            'total_rate': total_rate,
            'mode_contributions': mode_contributions,
            'resonance_frequencies': unique_resonances,
            'lv_enhancement': (self.config.casimir_config.mu / 
                             self.casimir_calc.experimental_bounds['mu'])
        }
    
    def power_extraction_analysis(self) -> Dict:
        """
        Analyze power extraction from dynamic Casimir effect.
        """
        production_results = self.photon_production_rate()
        
        # Average photon energy
        omega_matrix, _ = self.adiabatic_mode_evolution()
        avg_photon_energy = np.mean(omega_matrix[omega_matrix > 0]) * self.config.casimir_config.hbar
        
        # Photon flux (photons/s)
        photon_flux = production_results['total_rate']
        
        # Power generation (Watts)
        power_generated = photon_flux * avg_photon_energy
        
        # Extracted power (accounting for efficiency)
        power_extracted = power_generated * self.config.extraction_efficiency
        
        # Power scaling with LV parameters
        lv_enhancement = production_results['lv_enhancement']
        power_enhancement = lv_enhancement**2  # |amplitude|² scaling
        
        standard_power = power_extracted / power_enhancement if power_enhancement > 1 else power_extracted
        
        return {
            'photon_flux': photon_flux,
            'avg_photon_energy': avg_photon_energy,
            'power_generated': power_generated,
            'power_extracted': power_extracted,
            'power_enhancement': power_enhancement,
            'standard_power': standard_power,
            'power_density': power_extracted / (np.pi * (1e-3)**2),  # Per cm²
            'lv_scaling': f"P ∝ μ^{2:.1f}" if lv_enhancement > 1 else "Standard regime"
        }
    
    def frequency_sweep_analysis(self, frequency_range: np.ndarray) -> Dict:
        """
        Sweep drive frequency to find optimal extraction conditions.
        """
        results = {
            'frequencies': frequency_range,
            'power_extracted': np.zeros_like(frequency_range),
            'photon_rates': np.zeros_like(frequency_range),
            'resonance_strengths': np.zeros_like(frequency_range)
        }
        
        original_freq = self.config.drive_frequency
        
        print(f"🔄 Frequency sweep: {len(frequency_range)} points")
        
        for i, freq in enumerate(frequency_range):
            self.config.drive_frequency = freq
            
            # Recompute power extraction
            power_results = self.power_extraction_analysis()
            
            results['power_extracted'][i] = power_results['power_extracted']
            results['photon_rates'][i] = power_results['photon_flux']
            
            # Resonance strength (peak finding)
            production_results = self.photon_production_rate()
            resonance_strength = len(production_results['resonance_frequencies'])
            results['resonance_strengths'][i] = resonance_strength
        
        # Restore original frequency
        self.config.drive_frequency = original_freq
        
        # Find optimal frequency
        optimal_idx = np.argmax(results['power_extracted'])
        optimal_frequency = frequency_range[optimal_idx]
        
        results['optimal_frequency'] = optimal_frequency
        results['max_power'] = results['power_extracted'][optimal_idx]
        
        print("✅ Frequency sweep completed!")
        return results
    
    def visualize_dynamic_casimir_lv(self) -> None:
        """
        Create comprehensive visualization of dynamic Casimir LV effects.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Dynamic Casimir LV: Vacuum Energy Extraction', fontsize=14)
        
        # 1. Time-dependent cavity length and mode frequencies
        ax1 = axes[0, 0]
        L_t = self.time_dependent_cavity_length(self.time_array)
        ax1.plot(self.time_array * 1e9, L_t * 1e6, 'b-', linewidth=2)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Cavity length (μm)')
        ax1.set_title('Time-Dependent Cavity Length')
        ax1.grid(True, alpha=0.3)
        
        # 2. Mode frequency evolution
        ax2 = axes[0, 1]
        omega_matrix, mode_numbers = self.adiabatic_mode_evolution()
        
        # Plot first few modes
        for n in range(min(5, len(mode_numbers))):
            if np.any(omega_matrix[:, n] > 0):
                ax2.plot(self.time_array * 1e9, omega_matrix[:, n] * 1e-12, 
                        label=f'Mode {n+1}', linewidth=2)
        
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Frequency (THz)')
        ax2.set_title('Mode Frequency Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Photon production spectrum
        ax3 = axes[1, 0]
        production_results = self.photon_production_rate()
        
        # Extract mode frequencies and rates
        frequencies = [contrib['frequency'] for contrib in production_results['mode_contributions']]
        rates = [contrib['rate'] for contrib in production_results['mode_contributions']]
        
        if frequencies and rates:
            ax3.stem(np.array(frequencies) * 1e-12, rates, basefmt=" ")
            ax3.set_xlabel('Frequency (THz)')
            ax3.set_ylabel('Production rate')
            ax3.set_title('Photon Production Spectrum')
            ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Power vs drive frequency
        ax4 = axes[1, 1]
        
        # Frequency sweep around cavity resonances
        base_freq = np.pi * self.config.casimir_config.c / self.config.cavity_length
        freq_range = np.linspace(0.5 * base_freq, 2.0 * base_freq, 50)
        
        sweep_results = self.frequency_sweep_analysis(freq_range)
        
        ax4.plot(freq_range * 1e-12, sweep_results['power_extracted'] * 1e6, 'r-', linewidth=2)
        ax4.set_xlabel('Drive frequency (THz)')
        ax4.set_ylabel('Extracted power (μW)')
        ax4.set_title('Power vs Drive Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Mark optimal frequency
        if 'optimal_frequency' in sweep_results:
            ax4.axvline(sweep_results['optimal_frequency'] * 1e-12, 
                       color='g', linestyle='--', alpha=0.7, label='Optimal')
            ax4.legend()
        
        plt.tight_layout()
        plt.show()

def demo_dynamic_casimir_lv():
    """
    Demonstration of dynamic Casimir LV effects.
    """
    print("⚡ Dynamic Casimir LV Demo: Vacuum Power Extraction")
    print("=" * 55)
    
    # Test different LV enhancement scenarios
    lv_configs = [
        CasimirLVConfig(mu=1e-20, alpha=1e-15),  # Standard
        CasimirLVConfig(mu=1e-18, alpha=1e-13),  # 100x enhancement
        CasimirLVConfig(mu=1e-16, alpha=1e-11),  # 10,000x enhancement
    ]
    
    drive_frequencies = [1e10, 1e11, 1e12]  # GHz, THz ranges
    
    results = []
    
    for i, lv_config in enumerate(lv_configs):
        print(f"\n📊 LV Configuration {i+1}: μ={lv_config.mu:.2e}, α={lv_config.alpha:.2e}")
        
        for j, drive_freq in enumerate(drive_frequencies):
            print(f"   Drive frequency: {drive_freq:.1e} Hz")
            
            # Create dynamic Casimir configuration
            dyn_config = DynamicCasimirConfig(
                casimir_config=lv_config,
                drive_frequency=drive_freq,
                drive_amplitude=1e-9,  # 1 nm oscillation
                cavity_length=1e-4     # 100 μm cavity
            )
            
            calculator = DynamicCasimirLV(dyn_config)
            
            # Analyze power extraction
            power_results = calculator.power_extraction_analysis()
            
            print(f"     Photon flux: {power_results['photon_flux']:.2e} photons/s")
            print(f"     Power extracted: {power_results['power_extracted']:.2e} W")
            print(f"     Power enhancement: {power_results['power_enhancement']:.2e}")
            print(f"     Power density: {power_results['power_density']:.2e} W/cm²")
            
            results.append({
                'lv_config': lv_config,
                'drive_frequency': drive_freq,
                'power_results': power_results
            })
    
    # Find maximum power extraction case
    max_power_result = max(results, key=lambda r: r['power_results']['power_extracted'])
    
    print(f"\n🏆 Maximum Power Extraction:")
    print(f"   LV parameters: μ={max_power_result['lv_config'].mu:.2e}")
    print(f"   Drive frequency: {max_power_result['drive_frequency']:.2e} Hz")
    print(f"   Extracted power: {max_power_result['power_results']['power_extracted']:.2e} W")
    print(f"   Enhancement over standard: {max_power_result['power_results']['power_enhancement']:.2e}")
    
    # Comprehensive visualization
    print(f"\n📊 Generating visualization...")
    optimal_config = DynamicCasimirConfig(
        casimir_config=max_power_result['lv_config'],
        drive_frequency=max_power_result['drive_frequency']
    )
    
    calculator = DynamicCasimirLV(optimal_config)
    calculator.visualize_dynamic_casimir_lv()
    
    print("\n✅ Dynamic Casimir LV Demo Complete!")
    return results

if __name__ == "__main__":
    demo_dynamic_casimir_lv()
