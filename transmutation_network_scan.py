#!/usr/bin/env python3
"""
Transmutation Network Scanner
=============================

Advanced parameter scanning for the complete nuclear reaction network including:
1. Spallation transmutation pathways
2. LV-accelerated decay chains  
3. Multi-target optimization
4. Energy efficiency analysis
5. Experimental feasibility assessment

This scanner identifies optimal parameter combinations across the full
spallation → decay → collection pipeline for maximum rhodium yield.
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import matplotlib.pyplot as plt

from spallation_transmutation import SpallationTransmuter, SpallationConfig
from decay_accelerator import DecayAccelerator, DecayConfig  
from energy_ledger import EnergyLedger

@dataclass
class NetworkScanConfig:
    """Configuration for network parameter scanning."""
    # Beam parameter ranges
    beam_energies: List[float] = None  # MeV
    beam_types: List[str] = None
    beam_fluxes: List[float] = None    # particles/cm²/s
    
    # Target parameter ranges
    target_isotopes: List[str] = None
    target_masses: List[float] = None  # kg
    
    # LV parameter ranges
    mu_lv_range: Tuple[float, float] = (1e-18, 1e-16)
    alpha_lv_range: Tuple[float, float] = (1e-15, 1e-13)
    beta_lv_range: Tuple[float, float] = (1e-12, 1e-10)
    
    # Decay acceleration ranges
    field_strength_range: Tuple[float, float] = (1e7, 1e9)  # V/m
    magnetic_field_range: Tuple[float, float] = (1.0, 50.0)  # Tesla
    
    # Timing parameters
    spallation_time: float = 60.0     # seconds
    decay_time: float = 60.0          # seconds
    
    # Scan resolution
    num_samples: int = 100
    parallel_workers: int = 4
    
    def __post_init__(self):
        """Set default parameter ranges if not provided."""
        if self.beam_energies is None:
            self.beam_energies = [20, 30, 50, 80, 100, 150, 200]  # MeV
        if self.beam_types is None:
            self.beam_types = ["proton", "deuteron"]
        if self.beam_fluxes is None:
            self.beam_fluxes = [1e13, 5e13, 1e14, 5e14, 1e15]  # particles/cm²/s
        if self.target_isotopes is None:
            self.target_isotopes = ["Ag-107", "Ag-109", "Cd-110", "Cd-112"]
        if self.target_masses is None:
            self.target_masses = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]  # kg

@dataclass
class ScanResult:
    """Results from a single parameter combination."""
    # Input parameters
    beam_energy: float
    beam_type: str
    beam_flux: float
    target_isotope: str
    target_mass: float
    mu_lv: float
    alpha_lv: float
    beta_lv: float
    field_strength: float
    magnetic_field: float
    
    # Spallation results
    spallation_yields: Dict[str, float]
    spallation_energy: float
    
    # Decay results
    decay_yields: Dict[str, float]
    decay_energy: float
    
    # Final metrics
    total_rhodium_nuclei: float
    total_rhodium_mass: float
    total_energy_input: float
    energy_efficiency: float
    rhodium_per_joule: float
    
    # Success flags
    spallation_success: bool
    decay_success: bool
    overall_success: bool
    
    # Experimental feasibility
    feasibility_score: float

class NetworkScanner:
    """
    Advanced parameter scanner for the complete nuclear transmutation network.
    
    Optimizes across all system parameters to find maximum rhodium yield
    configurations with realistic energy requirements and experimental feasibility.
    """
    
    def __init__(self, config: NetworkScanConfig):
        self.config = config
        self.results: List[ScanResult] = []
        self.best_result: Optional[ScanResult] = None
        
    def evaluate_parameter_set(self, params: Dict[str, float]) -> ScanResult:
        """
        Evaluate a single parameter combination.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            ScanResult with all metrics calculated
        """
        # Create energy ledger for this run
        energy_ledger = EnergyLedger()
        
        # Configure spallation transmuter
        spallation_config = SpallationConfig(
            beam_type=params["beam_type"],
            beam_energy=params["beam_energy"] * 1e6,  # Convert MeV to eV
            beam_flux=params["beam_flux"],
            beam_duration=self.config.spallation_time,
            target_isotope=params["target_isotope"],
            target_mass=params["target_mass"],
            mu_lv=params["mu_lv"],
            alpha_lv=params["alpha_lv"],
            beta_lv=params["beta_lv"]
        )
        
        # Run spallation simulation
        try:
            spallator = SpallationTransmuter(spallation_config, energy_ledger)
            spallation_yields = spallator.simulate_spallation()
            spallation_success = sum(spallation_yields.values()) > 1e10  # Minimum yield threshold
        except Exception as e:
            spallation_yields = {}
            spallation_success = False
            print(f"Spallation simulation failed: {e}")
        
        # Configure decay accelerator for multiple isotopes
        decay_results = {}
        decay_success = False
        
        # Map spallation products to decay pathways
        decay_pathways = {
            "Rh-103": "Rh-103",  # Direct production (no decay needed)
            "Rh-105": "Rh-105",  # Direct production
            "Rh-104": "Rh-104",  # Direct production
        }
        
        # Also consider precursor isotopes that decay to rhodium
        precursor_yields = {}
        if "Ru" in params["target_isotope"] or True:  # Can produce Ru from spallation
            # Estimate Ru production from spallation
            precursor_yields["Ru-103"] = spallation_yields.get("Rh-103", 0) * 0.5  # 50% Ru precursor
            precursor_yields["Pd-103"] = spallation_yields.get("Rh-103", 0) * 0.3  # 30% Pd precursor
        
        total_decay_energy = 0.0
        
        for precursor, nuclei_count in precursor_yields.items():
            if nuclei_count > 1e8:  # Only process significant populations
                # Configure decay accelerator
                decay_config = DecayConfig(
                    mu_lv=params["mu_lv"],
                    alpha_lv=params["alpha_lv"],
                    beta_lv=params["beta_lv"],
                    field_strength=params["field_strength"],
                    magnetic_field=params["magnetic_field"],
                    isotope=precursor,
                    acceleration_time=self.config.decay_time
                )
                
                try:
                    accelerator = DecayAccelerator(decay_config, energy_ledger)
                    decay_result = accelerator.simulate_decay(nuclei_count, self.config.decay_time)
                    
                    daughter_isotope = decay_result.get("daughter_nuclei", 0)
                    if daughter_isotope > 0:
                        decay_results[f"{precursor}_daughter"] = daughter_isotope
                        decay_success = True
                        total_decay_energy += decay_result.get("field_energy", 0)
                        
                except Exception as e:
                    print(f"Decay simulation failed for {precursor}: {e}")
        
        # Calculate total rhodium production
        total_rh_nuclei = 0.0
        for isotope, count in spallation_yields.items():
            if "Rh" in isotope:
                total_rh_nuclei += count
        
        for isotope, count in decay_results.items():
            if "daughter" in isotope:
                total_rh_nuclei += count
        
        # Calculate rhodium mass (using Rh-103 atomic mass)
        rh_atomic_mass = 102.906 * 1.66054e-27  # kg
        total_rh_mass = total_rh_nuclei * rh_atomic_mass
        
        # Energy accounting
        total_energy_input = energy_ledger.get_total_input() + total_decay_energy
        
        # Efficiency metrics
        energy_efficiency = 0.0
        rhodium_per_joule = 0.0
        if total_energy_input > 0:
            energy_efficiency = (total_rh_mass * 9e16) / total_energy_input  # E=mc²
            rhodium_per_joule = total_rh_mass / total_energy_input
        
        # Feasibility assessment
        feasibility_score = self.calculate_feasibility_score(params, total_rh_mass, total_energy_input)
        
        # Overall success criteria
        overall_success = (
            total_rh_nuclei > 1e12 and          # Minimum rhodium yield
            energy_efficiency > 1e-12 and       # Reasonable efficiency
            feasibility_score > 0.3             # Experimentally feasible
        )
        
        return ScanResult(
            beam_energy=params["beam_energy"],
            beam_type=params["beam_type"],
            beam_flux=params["beam_flux"],
            target_isotope=params["target_isotope"],
            target_mass=params["target_mass"],
            mu_lv=params["mu_lv"],
            alpha_lv=params["alpha_lv"],
            beta_lv=params["beta_lv"],
            field_strength=params["field_strength"],
            magnetic_field=params["magnetic_field"],
            spallation_yields=spallation_yields,
            spallation_energy=energy_ledger.get_total_input(),
            decay_yields=decay_results,
            decay_energy=total_decay_energy,
            total_rhodium_nuclei=total_rh_nuclei,
            total_rhodium_mass=total_rh_mass,
            total_energy_input=total_energy_input,
            energy_efficiency=energy_efficiency,
            rhodium_per_joule=rhodium_per_joule,
            spallation_success=spallation_success,
            decay_success=decay_success,
            overall_success=overall_success,
            feasibility_score=feasibility_score
        )
    
    def calculate_feasibility_score(self, params: Dict[str, float], 
                                  rhodium_mass: float, energy_input: float) -> float:
        """
        Calculate experimental feasibility score (0-1 scale).
        
        Considers beam requirements, field strengths, and practical constraints.
        """
        score = 1.0
        
        # Beam energy feasibility (prefer 20-100 MeV range)
        beam_energy = params["beam_energy"]
        if beam_energy < 20 or beam_energy > 200:
            score *= 0.5
        elif beam_energy > 100:
            score *= 0.8
        
        # Beam flux feasibility (cyclotron limits)
        beam_flux = params["beam_flux"]
        if beam_flux > 1e15:
            score *= 0.3  # Very challenging
        elif beam_flux > 5e14:
            score *= 0.7
        
        # Field strength feasibility
        field_strength = params["field_strength"]
        if field_strength > 1e9:
            score *= 0.2  # Breakdown limit
        elif field_strength > 5e8:
            score *= 0.6
        
        # Magnetic field feasibility
        magnetic_field = params["magnetic_field"]
        if magnetic_field > 20:
            score *= 0.4  # Superconducting limit
        elif magnetic_field > 10:
            score *= 0.8
        
        # LV parameter feasibility (experimental bounds)
        if params["mu_lv"] > 1e-16:
            score *= 0.1  # Far beyond current bounds
        if params["alpha_lv"] > 1e-13:
            score *= 0.1
        if params["beta_lv"] > 1e-10:
            score *= 0.1
        
        # Yield feasibility (minimum detectable amounts)
        if rhodium_mass < 1e-18:  # < 1 ag
            score *= 0.1
        elif rhodium_mass < 1e-15:  # < 1 fg
            score *= 0.5
        
        return max(0.0, min(1.0, score))
    
    def generate_parameter_combinations(self) -> List[Dict[str, float]]:
        """Generate all parameter combinations to test."""
        combinations = []
        
        # Use Latin Hypercube Sampling for better coverage
        n_samples = self.config.num_samples
        
        # Generate LV parameter samples
        mu_samples = np.logspace(np.log10(self.config.mu_lv_range[0]), 
                                np.log10(self.config.mu_lv_range[1]), 
                                max(5, n_samples//20))
        alpha_samples = np.logspace(np.log10(self.config.alpha_lv_range[0]), 
                                   np.log10(self.config.alpha_lv_range[1]), 
                                   max(5, n_samples//20))
        beta_samples = np.logspace(np.log10(self.config.beta_lv_range[0]), 
                                  np.log10(self.config.beta_lv_range[1]), 
                                  max(5, n_samples//20))
        
        # Generate field parameter samples
        field_samples = np.logspace(np.log10(self.config.field_strength_range[0]),
                                   np.log10(self.config.field_strength_range[1]),
                                   max(5, n_samples//20))
        mag_samples = np.linspace(self.config.magnetic_field_range[0],
                                 self.config.magnetic_field_range[1],
                                 max(5, n_samples//20))
        
        # Create combinations
        count = 0
        for beam_energy in self.config.beam_energies:
            for beam_type in self.config.beam_types:
                for beam_flux in self.config.beam_fluxes:
                    for target_isotope in self.config.target_isotopes:
                        for target_mass in self.config.target_masses:
                            for mu_lv in mu_samples[:3]:  # Limit combinations
                                for alpha_lv in alpha_samples[:3]:
                                    for beta_lv in beta_samples[:3]:
                                        for field_strength in field_samples[:2]:
                                            for magnetic_field in mag_samples[:2]:
                                                combinations.append({
                                                    "beam_energy": beam_energy,
                                                    "beam_type": beam_type,
                                                    "beam_flux": beam_flux,
                                                    "target_isotope": target_isotope,
                                                    "target_mass": target_mass,
                                                    "mu_lv": mu_lv,
                                                    "alpha_lv": alpha_lv,
                                                    "beta_lv": beta_lv,
                                                    "field_strength": field_strength,
                                                    "magnetic_field": magnetic_field
                                                })
                                                count += 1
                                                if count >= n_samples:
                                                    return combinations
        
        return combinations
    
    def run_scan(self) -> List[ScanResult]:
        """
        Run the complete parameter scan.
        
        Returns:
            List of all scan results sorted by rhodium yield
        """
        print("🔬 TRANSMUTATION NETWORK PARAMETER SCAN")
        print("=" * 50)
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations()
        total_combinations = len(param_combinations)
        
        print(f"Testing {total_combinations} parameter combinations...")
        print(f"Beam energies: {self.config.beam_energies} MeV")
        print(f"Beam types: {self.config.beam_types}")
        print(f"Target isotopes: {self.config.target_isotopes}")
        print(f"LV parameter ranges:")
        print(f"  μ: {self.config.mu_lv_range[0]:.1e} - {self.config.mu_lv_range[1]:.1e}")
        print(f"  α: {self.config.alpha_lv_range[0]:.1e} - {self.config.alpha_lv_range[1]:.1e}")
        print(f"  β: {self.config.beta_lv_range[0]:.1e} - {self.config.beta_lv_range[1]:.1e}")
        
        # Run evaluations with progress tracking
        start_time = time.time()
        completed = 0
        
        if self.config.parallel_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                future_to_params = {
                    executor.submit(self.evaluate_parameter_set, params): params
                    for params in param_combinations
                }
                
                for future in as_completed(future_to_params):
                    try:
                        result = future.result()
                        self.results.append(result)
                        completed += 1
                        
                        if completed % 10 == 0:
                            elapsed = time.time() - start_time
                            progress = completed / total_combinations * 100
                            print(f"Progress: {progress:.1f}% ({completed}/{total_combinations}) - "
                                  f"Elapsed: {elapsed:.1f}s")
                            
                    except Exception as e:
                        print(f"Evaluation failed: {e}")
                        completed += 1
        else:
            # Serial execution
            for i, params in enumerate(param_combinations):
                try:
                    result = self.evaluate_parameter_set(params)
                    self.results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        progress = (i + 1) / total_combinations * 100
                        print(f"Progress: {progress:.1f}% ({i+1}/{total_combinations}) - "
                              f"Elapsed: {elapsed:.1f}s")
                        
                except Exception as e:
                    print(f"Evaluation {i+1} failed: {e}")
        
        # Sort results by rhodium yield
        self.results.sort(key=lambda r: r.total_rhodium_mass, reverse=True)
        
        # Find best result
        if self.results:
            self.best_result = self.results[0]
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Scan completed in {elapsed_time:.1f} seconds")
        print(f"   {len(self.results)} results generated")
        
        return self.results
    
    def analyze_results(self) -> Dict[str, any]:
        """Analyze scan results and generate summary statistics."""
        if not self.results:
            return {}
        
        # Success rate
        successful_results = [r for r in self.results if r.overall_success]
        success_rate = len(successful_results) / len(self.results) * 100
        
        # Best results
        top_10 = self.results[:10]
        
        # Parameter correlations
        rhodium_yields = [r.total_rhodium_mass for r in self.results]
        feasibility_scores = [r.feasibility_score for r in self.results]
        
        analysis = {
            "total_combinations": len(self.results),
            "success_rate": success_rate,
            "successful_count": len(successful_results),
            "best_rhodium_mass": self.best_result.total_rhodium_mass if self.best_result else 0,
            "best_efficiency": self.best_result.energy_efficiency if self.best_result else 0,
            "best_feasibility": self.best_result.feasibility_score if self.best_result else 0,
            "average_yield": np.mean(rhodium_yields),
            "median_yield": np.median(rhodium_yields),
            "yield_std": np.std(rhodium_yields),
            "average_feasibility": np.mean(feasibility_scores),
            "top_10_results": [asdict(r) for r in top_10]
        }
        
        return analysis
    
    def export_results(self, filename: str = "network_scan_results.json"):
        """Export scan results to JSON file."""
        export_data = {
            "scan_config": asdict(self.config),
            "scan_timestamp": datetime.now().isoformat(),
            "analysis": self.analyze_results(),
            "all_results": [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📁 Results exported to {filename}")
    
    def print_summary(self):
        """Print a comprehensive summary of scan results."""
        if not self.results:
            print("❌ No results to summarize")
            return
        
        analysis = self.analyze_results()
        
        print(f"\n🎯 TRANSMUTATION NETWORK SCAN SUMMARY")
        print("=" * 50)
        print(f"Total combinations tested: {analysis['total_combinations']}")
        print(f"Successful combinations: {analysis['successful_count']} ({analysis['success_rate']:.1f}%)")
        
        if self.best_result:
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"  Beam: {self.best_result.beam_energy:.1f} MeV {self.best_result.beam_type}")
            print(f"  Target: {self.best_result.target_isotope}")
            print(f"  Flux: {self.best_result.beam_flux:.2e} particles/cm²/s")
            print(f"  LV params: μ={self.best_result.mu_lv:.1e}, α={self.best_result.alpha_lv:.1e}, β={self.best_result.beta_lv:.1e}")
            print(f"  Fields: E={self.best_result.field_strength:.1e} V/m, B={self.best_result.magnetic_field:.1f} T")
            print(f"\n📊 PERFORMANCE:")
            print(f"  Rhodium mass: {self.best_result.total_rhodium_mass*1e15:.2f} fg")
            print(f"  Rhodium nuclei: {self.best_result.total_rhodium_nuclei:.2e}")
            print(f"  Energy efficiency: {self.best_result.energy_efficiency:.2e}")
            print(f"  Feasibility score: {self.best_result.feasibility_score:.2f}")
            print(f"  Energy input: {self.best_result.total_energy_input:.2e} J")
        
        print(f"\n📈 STATISTICS:")
        print(f"  Average yield: {analysis['average_yield']*1e15:.2f} fg")
        print(f"  Median yield: {analysis['median_yield']*1e15:.2f} fg")
        print(f"  Yield std dev: {analysis['yield_std']*1e15:.2f} fg")
        print(f"  Average feasibility: {analysis['average_feasibility']:.2f}")

def demo_network_scan():
    """Demonstration of advanced transmutation network scanning."""
    print("🚀 TRANSMUTATION NETWORK SCANNER DEMO")
    print("=" * 50)
    
    # Configure scan with focused parameters
    config = NetworkScanConfig(
        beam_energies=[50, 80, 100],  # MeV  
        beam_types=["proton", "deuteron"],
        beam_fluxes=[1e14, 5e14],     # particles/cm²/s
        target_isotopes=["Ag-109", "Cd-112"],
        target_masses=[1e-5, 5e-5],   # kg
        mu_lv_range=(1e-17, 5e-17),
        alpha_lv_range=(5e-15, 5e-14),
        beta_lv_range=(5e-12, 5e-11),
        num_samples=50,               # Reduced for demo
        parallel_workers=2
    )
    
    # Run scan
    scanner = NetworkScanner(config)
    results = scanner.run_scan()
    
    # Analyze and report
    scanner.print_summary()
    scanner.export_results("demo_network_scan.json")
    
    # Success criteria
    success = (len(results) > 0 and 
              scanner.best_result and 
              scanner.best_result.total_rhodium_mass > 1e-15)  # > 1 fg
    
    print(f"\n🎯 DEMO SUCCESS: {'✅ YES' if success else '❌ NO'}")
    
    return success

if __name__ == "__main__":
    demo_network_scan()
