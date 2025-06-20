#!/usr/bin/env python3
"""
Transmutation Scanner - Automated Parameter Sweeps for Rhodium Production
=========================================================================

This module implements automated parameter scanning and optimization
for the nuclear transmutation stage of the rhodium replicator system.

Key Features:
1. Multi-dimensional parameter sweeps
2. Beam energy and flux optimization
3. Seed isotope selection analysis
4. LV coefficient tuning
5. Yield vs. waste mapping
6. Real-time optimization feedback

Author: LV Energy Converter Framework
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import json
import time

try:
    from .nuclear_transmutation import NuclearTransmuter, TransmutationConfig
    from .energy_ledger import EnergyLedger, EnergyType
except ImportError:
    from nuclear_transmutation import NuclearTransmuter, TransmutationConfig
    from energy_ledger import EnergyLedger, EnergyType

@dataclass
class ScanParameters:
    """Parameters for transmutation parameter sweeps."""
    
    # Beam parameter ranges
    beam_energies: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # MeV
    beam_fluxes: List[float] = field(default_factory=lambda: [1e12, 1e13, 1e14, 1e15])  # particles/cm²/s
    
    # Seed isotope options
    seed_isotopes: List[str] = field(default_factory=lambda: ["Ru-102", "Pd-103", "Ru-104"])
    
    # LV parameter ranges
    mu_lv_range: List[float] = field(default_factory=lambda: [1e-19, 1e-18, 1e-17, 1e-16])
    alpha_lv_range: List[float] = field(default_factory=lambda: [1e-16, 1e-15, 1e-14, 1e-13])
    beta_lv_range: List[float] = field(default_factory=lambda: [1e-13, 1e-12, 1e-11, 1e-10])
    
    # Scan control
    max_scan_time: float = 300.0  # 5 minutes max scan time
    target_yield: float = 1e-15   # Target rhodium yield (kg)
    waste_limit: float = 1e6      # Radioactive waste limit (Bq)

@dataclass 
class ScanResults:
    """Results from parameter scan."""
    
    # Scan metrics
    total_combinations: int = 0
    completed_combinations: int = 0
    scan_time: float = 0.0
    
    # Best results
    best_yield: float = 0.0
    best_efficiency: float = 0.0
    best_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Result matrices
    yield_matrix: np.ndarray = None
    efficiency_matrix: np.ndarray = None
    waste_matrix: np.ndarray = None
    
    # Parameter sensitivity
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # Optimization recommendations
    recommendations: List[str] = field(default_factory=list)

class TransmutationScanner:
    """
    Automated parameter scanner for rhodium transmutation optimization.
    """
    
    def __init__(self, scan_params: ScanParameters):
        self.scan_params = scan_params
        self.energy_ledger = EnergyLedger("Transmutation_Scanner")
        
        print(f"🔍 Transmutation Scanner initialized:")
        print(f"  Beam energies: {len(scan_params.beam_energies)} values")
        print(f"  Beam fluxes: {len(scan_params.beam_fluxes)} values") 
        print(f"  Seed isotopes: {len(scan_params.seed_isotopes)} options")
        print(f"  LV parameters: {len(scan_params.mu_lv_range)} × {len(scan_params.alpha_lv_range)} × {len(scan_params.beta_lv_range)}")
        print(f"  Total combinations: {self._calculate_total_combinations()}")
    
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        return (len(self.scan_params.beam_energies) * 
                len(self.scan_params.beam_fluxes) * 
                len(self.scan_params.seed_isotopes) *
                len(self.scan_params.mu_lv_range) *
                len(self.scan_params.alpha_lv_range) *
                len(self.scan_params.beta_lv_range))
    
    def execute_parameter_sweep(self) -> ScanResults:
        """Execute complete parameter sweep."""
        start_time = time.time()
        
        print(f"\n🚀 EXECUTING TRANSMUTATION PARAMETER SWEEP")
        print(f"="*60)
        
        results = ScanResults()
        results.total_combinations = self._calculate_total_combinations()
        
        # Initialize result storage
        n_energies = len(self.scan_params.beam_energies)
        n_fluxes = len(self.scan_params.beam_fluxes)
        
        results.yield_matrix = np.zeros((n_energies, n_fluxes))
        results.efficiency_matrix = np.zeros((n_energies, n_fluxes))
        results.waste_matrix = np.zeros((n_energies, n_fluxes))
        
        best_yield = 0.0
        best_params = {}
        completed = 0
        
        print(f"Scanning {results.total_combinations:,} parameter combinations...")
        
        # Main parameter sweep loop
        for i, energy in enumerate(self.scan_params.beam_energies):
            for j, flux in enumerate(self.scan_params.beam_fluxes):
                for seed in self.scan_params.seed_isotopes:
                    for mu in self.scan_params.mu_lv_range:
                        for alpha in self.scan_params.alpha_lv_range:
                            for beta in self.scan_params.beta_lv_range:
                                
                                # Check timeout
                                if time.time() - start_time > self.scan_params.max_scan_time:
                                    print(f"\n⏰ Timeout reached after {completed:,} combinations")
                                    break
                                
                                # Run transmutation simulation
                                yield_result, efficiency, waste = self._simulate_transmutation(
                                    energy, flux, seed, mu, alpha, beta
                                )
                                
                                # Store results
                                results.yield_matrix[i, j] = max(results.yield_matrix[i, j], yield_result)
                                results.efficiency_matrix[i, j] = max(results.efficiency_matrix[i, j], efficiency)
                                results.waste_matrix[i, j] = min(results.waste_matrix[i, j], waste) if results.waste_matrix[i, j] == 0 else min(results.waste_matrix[i, j], waste)
                                
                                # Track best result
                                if yield_result > best_yield and waste < self.scan_params.waste_limit:
                                    best_yield = yield_result
                                    best_params = {
                                        'beam_energy': energy,
                                        'beam_flux': flux,
                                        'seed_isotope': seed,
                                        'mu_lv': mu,
                                        'alpha_lv': alpha,
                                        'beta_lv': beta,
                                        'yield': yield_result,
                                        'efficiency': efficiency,
                                        'waste': waste
                                    }
                                
                                completed += 1
                                
                                # Progress update
                                if completed % 100 == 0:
                                    progress = completed / results.total_combinations * 100
                                    print(f"  Progress: {progress:.1f}% ({completed:,}/{results.total_combinations:,})")
        
        # Finalize results
        results.completed_combinations = completed
        results.scan_time = time.time() - start_time
        results.best_yield = best_yield
        results.best_parameters = best_params
        
        if best_params:
            results.best_efficiency = best_params['efficiency']
        
        # Generate recommendations
        self._generate_recommendations(results)
        
        print(f"\n✅ PARAMETER SWEEP COMPLETE")
        print(f"  Combinations tested: {completed:,}")
        print(f"  Scan time: {results.scan_time:.1f}s")
        print(f"  Best yield: {best_yield*1e18:.1f} ag")
        
        return results
    
    def _simulate_transmutation(self, energy: float, flux: float, seed: str, 
                              mu: float, alpha: float, beta: float) -> Tuple[float, float, float]:
        """Simulate transmutation for given parameters."""
        
        # Create configuration
        config = TransmutationConfig(
            target_isotope="Rh-103",
            seed_isotope=seed,
            transmutation_pathway="neutron",
            beam_energy=energy,
            beam_flux=flux,
            beam_duration=3600.0,  # 1 hour
            mu_lv=mu,
            alpha_lv=alpha,
            beta_lv=beta,
            collection_efficiency=0.9
        )
        
        try:
            # Run simplified transmutation calculation
            transmuter = NuclearTransmuter(config, self.energy_ledger)
            
            # Simulate with small seed mass
            seed_mass = 1e-12  # 1 picogram
            results = transmuter.simulate_transmutation_run(seed_mass)
            
            return results.rhodium_yield, results.conversion_efficiency, results.waste_activity
            
        except Exception as e:
            # Return zero results on error
            return 0.0, 0.0, 1e12  # High waste penalty
    
    def _generate_recommendations(self, results: ScanResults):
        """Generate optimization recommendations based on scan results."""
        
        if not results.best_parameters:
            results.recommendations.append("No valid parameter combinations found within constraints")
            return
        
        best = results.best_parameters
        
        # Energy recommendations
        if best['beam_energy'] <= 1.0:
            results.recommendations.append("Low energy (~1 MeV) optimal for neutron capture")
        elif best['beam_energy'] >= 2.5:
            results.recommendations.append("High energy optimal - consider spallation reactions")
        else:
            results.recommendations.append("Intermediate energy optimal - balanced efficiency")
        
        # Flux recommendations
        if best['beam_flux'] >= 1e14:
            results.recommendations.append("High flux beneficial - invest in beam intensity")
        else:
            results.recommendations.append("Moderate flux sufficient - focus on beam quality")
        
        # LV parameter recommendations
        if best['mu_lv'] >= 1e-17:
            results.recommendations.append("Strong CPT violation beneficial for Gamow enhancement")
        
        if best['alpha_lv'] >= 1e-14:
            results.recommendations.append("Significant Lorentz violation improves cross-sections")
        
        # Isotope recommendations
        results.recommendations.append(f"Optimal seed isotope: {best['seed_isotope']}")
        
        # Yield assessment
        if best['yield'] >= self.scan_params.target_yield:
            results.recommendations.append("Target yield achievable with optimal parameters")
        else:
            results.recommendations.append("Target yield challenging - consider multi-stage processing")
    
    def export_scan_results(self, results: ScanResults, filename: str = "transmutation_scan_results.json"):
        """Export scan results to JSON file."""
        
        export_data = {
            "scan_summary": {
                "total_combinations": results.total_combinations,
                "completed_combinations": results.completed_combinations,
                "scan_time": results.scan_time,
                "best_yield": results.best_yield,
                "best_efficiency": results.best_efficiency
            },
            "best_parameters": results.best_parameters,
            "recommendations": results.recommendations,
            "scan_parameters": {
                "beam_energies": self.scan_params.beam_energies,
                "beam_fluxes": self.scan_params.beam_fluxes,
                "seed_isotopes": self.scan_params.seed_isotopes,
                "target_yield": self.scan_params.target_yield,
                "waste_limit": self.scan_params.waste_limit
            },
            "yield_matrix": results.yield_matrix.tolist() if results.yield_matrix is not None else None,
            "efficiency_matrix": results.efficiency_matrix.tolist() if results.efficiency_matrix is not None else None
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Scan results exported to {filename}")
        return filename

def demo_transmutation_scanning():
    """Demonstrate transmutation parameter scanning."""
    
    print("🔬 TRANSMUTATION PARAMETER SCANNING DEMO")
    print("="*50)
    print("🎯 Objective: Find optimal parameters for rhodium yield")
    print("⚡ Method: Multi-dimensional parameter sweep with LV enhancement")
    
    # Create scan parameters (reduced for demo)
    scan_params = ScanParameters(
        beam_energies=[0.5, 1.0, 1.5, 2.0],      # 4 energies
        beam_fluxes=[1e13, 1e14],                  # 2 fluxes
        seed_isotopes=["Ru-102", "Pd-103"],        # 2 isotopes
        mu_lv_range=[1e-18, 1e-17],               # 2 LV values
        alpha_lv_range=[1e-15, 1e-14],            # 2 LV values
        beta_lv_range=[1e-12, 1e-11],             # 2 LV values
        max_scan_time=60.0,                        # 1 minute timeout
        target_yield=1e-15                         # 1 fg target
    )
    
    # Execute scan
    scanner = TransmutationScanner(scan_params)
    results = scanner.execute_parameter_sweep()
    
    # Display results
    print(f"\n📊 SCAN RESULTS SUMMARY:")
    print(f"  Best yield: {results.best_yield*1e18:.1f} attograms rhodium")
    print(f"  Best efficiency: {results.best_efficiency:.2e}")
    print(f"  Scan completion: {results.completed_combinations}/{results.total_combinations}")
    
    if results.best_parameters:
        print(f"\n⚙️  OPTIMAL PARAMETERS:")
        for key, value in results.best_parameters.items():
            if isinstance(value, float):
                if value > 1e-6:
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value:.2e}")
            else:
                print(f"    {key}: {value}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(results.recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Export results
    filename = scanner.export_scan_results(results)
    
    print(f"\n✅ TRANSMUTATION SCANNING COMPLETE")
    print(f"📈 Optimization data available for experimental planning")
    print(f"🎯 Ready for laboratory parameter validation")
    
    return scanner, results

if __name__ == "__main__":
    demo_transmutation_scanning()
