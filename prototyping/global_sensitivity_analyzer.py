#!/usr/bin/env python3
"""
Global Sensitivity Analysis for Photonuclear Gold Production
===========================================================

Uses Sobol and Morris methods to identify leverage points in the parameter space.
Helps find "hidden" optimization opportunities before expensive experimental runs.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import time

# SALib for global sensitivity analysis
try:
    from SALib.sample import saltelli, morris
    from SALib.analyze import sobol, morris as morris_analyze
    from SALib.util import read_param_file
    SALIB_AVAILABLE = True
except ImportError:
    logging.warning("SALib not available - install with: pip install SALib")
    SALIB_AVAILABLE = False

from atomic_binder import AtomicDataBinder
from simulations import Geant4Interface, SimulationParameters

@dataclass
class SensitivityParameter:
    """Parameter definition for sensitivity analysis."""
    name: str
    bounds: Tuple[float, float]  # (min, max)
    description: str
    units: str

@dataclass
class SensitivityResults:
    """Results from global sensitivity analysis."""
    method: str
    total_samples: int
    parameters: List[str]
    parameter_names: List[str]  # Alias for parameters
    first_order_indices: Dict[str, float]
    total_order_indices: Dict[str, float]
    parameter_rankings: List[Tuple[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    analysis_time_s: float
    summary_statistics: Dict[str, Any]  # Summary statistics

class GlobalSensitivityAnalyzer:
    """Global sensitivity analysis for photonuclear transmutation optimization."""
    
    def __init__(self, output_dir: str = "sensitivity_analysis"):
        """Initialize the sensitivity analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize physics engines
        self.atomic_data = AtomicDataBinder()
        self.simulator = Geant4Interface()
        
        # Define parameter space for sensitivity analysis
        self.parameters = self._define_parameter_space()
        
        self.logger.info("Global sensitivity analyzer initialized")
    
    def _define_parameter_space(self) -> List[SensitivityParameter]:
        """Define the full parameter space for sensitivity analysis."""
        
        return [
            # Beam parameters
            SensitivityParameter("beam_energy_mev", (8.0, 30.0), "Photon beam energy", "MeV"),
            SensitivityParameter("beam_intensity", (1e10, 1e15), "Photon flux", "photons/s/cm²"),
            SensitivityParameter("pulse_width_ns", (0.001, 1000.0), "Pulse duration", "ns"),
            SensitivityParameter("pulse_frequency_hz", (1.0, 1e6), "Pulse repetition rate", "Hz"),
            
            # Target composition (fractions)
            SensitivityParameter("pb208_fraction", (0.0, 1.0), "Pb-208 fraction", "dimensionless"),
            SensitivityParameter("bi209_fraction", (0.0, 1.0), "Bi-209 fraction", "dimensionless"),
            SensitivityParameter("tl203_fraction", (0.0, 1.0), "Tl-203 fraction", "dimensionless"),
            SensitivityParameter("tl205_fraction", (0.0, 1.0), "Tl-205 fraction", "dimensionless"),
            SensitivityParameter("hg202_fraction", (0.0, 1.0), "Hg-202 fraction", "dimensionless"),
            SensitivityParameter("hg200_fraction", (0.0, 1.0), "Hg-200 fraction", "dimensionless"),
            
            # Neutron source fractions (for two-step processes)
            SensitivityParameter("be9_fraction", (0.0, 0.5), "Be-9 neutron source fraction", "dimensionless"),
            SensitivityParameter("d2_fraction", (0.0, 0.3), "D2O neutron source fraction", "dimensionless"),
            
            # Target geometry and physics
            SensitivityParameter("target_mass_g", (0.1, 50.0), "Total target mass", "g"),
            SensitivityParameter("target_density_g_cm3", (1.0, 15.0), "Target density", "g/cm³"),
            SensitivityParameter("target_thickness_cm", (0.1, 5.0), "Target thickness", "cm"),
            SensitivityParameter("moderator_thickness_cm", (0.0, 10.0), "Neutron moderator thickness", "cm"),
            
            # Environmental conditions
            SensitivityParameter("temperature_k", (200.0, 400.0), "Target temperature", "K"),
            SensitivityParameter("pressure_atm", (0.1, 10.0), "Ambient pressure", "atm"),
            
            # Irradiation parameters
            SensitivityParameter("total_dose_kgy", (1.0, 1000.0), "Total absorbed dose", "kGy"),
            SensitivityParameter("dose_rate_kgy_h", (0.1, 100.0), "Dose rate", "kGy/h"),
        ]
    
    def create_salib_problem(self, selected_params: List[str] = None) -> Dict[str, Any]:
        """Create SALib problem definition."""
        
        if not SALIB_AVAILABLE:
            raise ImportError("SALib is required for sensitivity analysis")
        
        # Use all parameters if none specified
        if selected_params is None:
            selected_params = [p.name for p in self.parameters]
        
        # Filter parameters
        active_params = [p for p in self.parameters if p.name in selected_params]
        
        problem = {
            'num_vars': len(active_params),
            'names': [p.name for p in active_params],
            'bounds': [list(p.bounds) for p in active_params],
            'groups': None
        }
        
        return problem, active_params
    
    def normalize_target_composition(self, params: Dict[str, float]) -> Dict[str, float]:
        """Normalize target composition fractions to sum to 1."""
        
        # Extract composition fractions
        composition_keys = [k for k in params.keys() if k.endswith('_fraction')]
        total_fraction = sum(params[k] for k in composition_keys)
        
        # Normalize to sum to 1
        normalized_composition = {}
        if total_fraction > 0:
            for key in composition_keys:
                isotope = key.replace('_fraction', '').replace('pb208', 'Pb-208').replace('bi209', 'Bi-209')\
                           .replace('tl203', 'Tl-203').replace('tl205', 'Tl-205')\
                           .replace('hg202', 'Hg-202').replace('hg200', 'Hg-200')\
                           .replace('be9', 'Be-9').replace('d2', 'D-2')
                normalized_composition[isotope] = params[key] / total_fraction
        
        return normalized_composition
    
    def evaluate_yield_function(self, param_values: np.ndarray, param_names: List[str]) -> float:
        """Evaluate gold yield for given parameter values."""
        
        # Convert array to parameter dictionary
        params = dict(zip(param_names, param_values))
        
        # Normalize target composition
        target_composition = self.normalize_target_composition(params)
        
        # Skip if no meaningful composition
        if not target_composition:
            return 0.0
        
        # Extract key parameters
        beam_energy = params.get('beam_energy_mev', 13.5)
        beam_intensity = params.get('beam_intensity', 1e12)
        total_dose = params.get('total_dose_kgy', 50.0)
        target_mass = params.get('target_mass_g', 5.0)
        pulse_width = params.get('pulse_width_ns', 1000.0)  # Default to CW
        
        # Determine beam profile
        if pulse_width < 1.0:
            beam_profile = "pulsed_ps"
        elif pulse_width < 100.0:
            beam_profile = "pulsed_ns"
        else:
            beam_profile = "continuous"
        
        # Calculate photon fluence
        dose_rate = params.get('dose_rate_kgy_h', 1.0)
        irradiation_time_s = (total_dose / dose_rate) * 3600
        fluence_per_cm2 = beam_intensity * irradiation_time_s
        
        # Calculate gold production efficiency
        try:
            efficiency = self.atomic_data.calculate_gold_production_efficiency(
                target_composition, beam_energy, fluence_per_cm2, beam_profile
            )
            
            # Convert to gold atoms produced
            target_atoms = (target_mass / 200.0) * 6.022e23  # Approximate atomic mass
            gold_atoms = efficiency * target_atoms
            
            # Convert to mg of gold
            gold_mass_mg = (gold_atoms / 6.022e23) * 197.0 * 1000  # Au-197 atomic mass
            
            return gold_mass_mg
            
        except Exception as e:
            self.logger.warning(f"Error evaluating yield: {e}")
            return 0.0
    
    def run_sobol_analysis(self, n_samples: int = 1024, 
                          selected_params: List[str] = None) -> SensitivityResults:
        """Run Sobol global sensitivity analysis."""
        
        if not SALIB_AVAILABLE:
            raise ImportError("SALib is required for Sobol analysis")
        
        start_time = time.time()
        
        # Create problem definition
        problem, active_params = self.create_salib_problem(selected_params)
        param_names = [p.name for p in active_params]
        
        self.logger.info(f"Running Sobol analysis with {len(param_names)} parameters, {n_samples} samples")
        
        # Generate samples
        param_values = saltelli.sample(problem, n_samples)
        total_samples = param_values.shape[0]
        
        self.logger.info(f"Generated {total_samples} parameter combinations")
        
        # Evaluate model for all samples
        y_values = np.array([
            self.evaluate_yield_function(params, param_names) 
            for params in param_values
        ])
        
        # Run Sobol analysis
        sobol_results = sobol.analyze(problem, y_values)
        
        # Extract results
        first_order = dict(zip(param_names, sobol_results['S1']))
        total_order = dict(zip(param_names, sobol_results['ST']))
        
        # Confidence intervals
        confidence_intervals = {}
        for i, param in enumerate(param_names):
            conf_s1 = sobol_results['S1_conf'][i]
            conf_st = sobol_results['ST_conf'][i]
            confidence_intervals[param] = (
                (first_order[param] - conf_s1, first_order[param] + conf_s1),
                (total_order[param] - conf_st, total_order[param] + conf_st)
            )
          # Rank parameters by total-order sensitivity
        rankings = sorted(total_order.items(), key=lambda x: x[1], reverse=True)
        
        analysis_time = time.time() - start_time
        
        results = SensitivityResults(
            method="Sobol",
            total_samples=total_samples,
            parameters=param_names,
            parameter_names=param_names,  # Alias
            first_order_indices=first_order,
            total_order_indices=total_order,
            parameter_rankings=rankings,
            confidence_intervals=confidence_intervals,
            analysis_time_s=analysis_time,
            summary_statistics={'mean_yield': np.mean(y_values), 'std_yield': np.std(y_values)}
        )
        
        # Save results
        self.save_sensitivity_results(results, "sobol_analysis.json")
        
        return results
    
    def run_morris_screening(self, n_trajectories: int = 100,
                           selected_params: List[str] = None) -> SensitivityResults:
        """Run Morris method for parameter screening."""
        
        if not SALIB_AVAILABLE:
            raise ImportError("SALib is required for Morris analysis")
        
        start_time = time.time()
        
        # Create problem definition
        problem, active_params = self.create_salib_problem(selected_params)
        param_names = [p.name for p in active_params]
        
        self.logger.info(f"Running Morris screening with {len(param_names)} parameters, {n_trajectories} trajectories")
        
        # Generate samples
        param_values = morris.sample(problem, n_trajectories)
        total_samples = param_values.shape[0]
        
        # Evaluate model
        y_values = np.array([
            self.evaluate_yield_function(params, param_names) 
            for params in param_values
        ])
        
        # Run Morris analysis
        morris_results = morris_analyze.analyze(problem, param_values, y_values)
        
        # Extract results (Morris uses mu_star as main sensitivity measure)
        total_order = dict(zip(param_names, morris_results['mu_star']))
        first_order = dict(zip(param_names, morris_results['mu']))  # Raw mean
        
        # Rank parameters
        rankings = sorted(total_order.items(), key=lambda x: x[1], reverse=True)
        
        analysis_time = time.time() - start_time
        
        results = SensitivityResults(
            method="Morris",
            total_samples=total_samples,
            parameters=param_names,
            first_order_indices=first_order,
            total_order_indices=total_order,
            parameter_rankings=rankings,
            confidence_intervals={},  # Morris doesn't provide confidence intervals
            analysis_time_s=analysis_time
        )
        
        # Save results
        self.save_sensitivity_results(results, "morris_screening.json")
        
        return results
    
    def save_sensitivity_results(self, results: SensitivityResults, filename: str):
        """Save sensitivity analysis results to JSON."""
        
        output_file = self.output_dir / filename
        
        data = {
            'method': results.method,
            'total_samples': results.total_samples,
            'parameters': results.parameters,
            'first_order_indices': results.first_order_indices,
            'total_order_indices': results.total_order_indices,
            'parameter_rankings': results.parameter_rankings,
            'confidence_intervals': results.confidence_intervals,
            'analysis_time_s': results.analysis_time_s,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved sensitivity results to {output_file}")
    
    def identify_leverage_points(self, results: SensitivityResults, 
                               threshold: float = 0.1) -> List[str]:
        """Identify high-leverage parameters for focused optimization."""
        
        leverage_points = []
        
        for param, sensitivity in results.parameter_rankings:
            if sensitivity >= threshold:
                leverage_points.append(param)
        
        return leverage_points
    
    def generate_optimization_recommendations(self, results: SensitivityResults) -> Dict[str, Any]:
        """Generate optimization recommendations based on sensitivity analysis."""
        
        # Identify top sensitivity parameters
        top_params = results.parameter_rankings[:5]
        leverage_points = self.identify_leverage_points(results)
        
        recommendations = {
            'high_impact_parameters': [p[0] for p in top_params],
            'leverage_points': leverage_points,
            'optimization_strategy': self._generate_strategy(top_params),
            'parameter_bounds_to_tighten': self._suggest_bounds_refinement(results),
            'two_step_opportunities': self._identify_two_step_opportunities(results)
        }
        
        return recommendations
    
    def _generate_strategy(self, top_params: List[Tuple[str, float]]) -> List[str]:
        """Generate optimization strategy based on top parameters."""
        
        strategies = []
        
        for param, sensitivity in top_params:
            if 'beam_energy' in param:
                strategies.append("Focus on beam energy optimization around GDR peaks (13-16 MeV)")
            elif 'fraction' in param:
                strategies.append(f"Optimize {param} - consider pure or enriched targets")
            elif 'pulse' in param:
                strategies.append("Investigate pulsed beam advantages for nonlinear enhancement")
            elif 'dose' in param:
                strategies.append("Optimize dose and dose rate for maximum yield/cost ratio")
            elif 'target' in param:
                strategies.append("Optimize target geometry and density for interaction efficiency")
        
        return strategies
    
    def _suggest_bounds_refinement(self, results: SensitivityResults) -> Dict[str, Tuple[float, float]]:
        """Suggest refined parameter bounds based on sensitivity."""
        
        refined_bounds = {}
        
        # Focus search space around high-sensitivity parameters
        for param, sensitivity in results.parameter_rankings[:3]:
            if sensitivity > 0.2:  # High sensitivity
                original_param = next(p for p in self.parameters if p.name == param)
                min_val, max_val = original_param.bounds
                
                # Narrow the bounds by 50% around the center
                center = (min_val + max_val) / 2
                range_width = (max_val - min_val) * 0.5
                
                refined_bounds[param] = (
                    max(min_val, center - range_width/2),
                    min(max_val, center + range_width/2)
                )
        
        return refined_bounds
    
    def _identify_two_step_opportunities(self, results: SensitivityResults) -> List[str]:
        """Identify opportunities for two-step photoneutron processes."""
        
        opportunities = []
        
        # Check if neutron source parameters are sensitive
        neutron_params = ['be9_fraction', 'd2_fraction']
        target_params = ['pb208_fraction', 'bi209_fraction', 'tl203_fraction', 'hg202_fraction']
        
        neutron_sensitivities = {p: results.total_order_indices.get(p, 0.0) for p in neutron_params}
        target_sensitivities = {p: results.total_order_indices.get(p, 0.0) for p in target_params}
        
        if any(s > 0.05 for s in neutron_sensitivities.values()):
            opportunities.append("Two-step photoneutron processes show promise")
            
            if neutron_sensitivities['be9_fraction'] > 0.1:
                opportunities.append("Be-9(γ,n) neutron source highly effective")
            
            if neutron_sensitivities['d2_fraction'] > 0.1:
                opportunities.append("D2O photodisintegration neutron source effective")
        
        return opportunities

    def sobol_sensitivity_analysis(self, param_ranges: Dict[str, Tuple[float, float]], 
                                   samples: int, pathway_name: str) -> Dict:
        """Wrapper for sobol analysis to match comprehensive analyzer interface."""
        # Convert param_ranges to selected_params format
        selected_params = list(param_ranges.keys())
        
        # Run the actual Sobol analysis
        results = self.run_sobol_analysis(n_samples=samples, selected_params=selected_params)
          # Return in format expected by comprehensive analyzer
        return {
            'pathway_name': pathway_name,
            'first_order': results.first_order_indices,
            'total_order': results.total_order_indices,
            'second_order': {},  # Not available in our SensitivityResults
            'parameters': results.parameter_names if hasattr(results, 'parameter_names') else results.parameters,
            'summary': getattr(results, 'summary_statistics', {})
        }
    
    def variance_based_analysis(self, param_ranges: Dict[str, Tuple[float, float]], 
                               samples: int, pathway_name: str) -> Dict:
        """Placeholder variance-based analysis for compatibility."""
        # For now, return a minimal result
        return {
            'pathway_name': pathway_name,
            'variance_contribution': {},
            'parameters': list(param_ranges.keys())
        }

def main():
    """Test global sensitivity analysis."""
    logging.basicConfig(level=logging.INFO)
    
    analyzer = GlobalSensitivityAnalyzer()
    
    # Run quick Morris screening first
    print("Running Morris screening...")
    morris_results = analyzer.run_morris_screening(n_trajectories=50)
    
    print(f"\nTop 5 parameters (Morris):")
    for param, sensitivity in morris_results.parameter_rankings[:5]:
        print(f"  {param}: {sensitivity:.4f}")
    
    # Generate recommendations
    recommendations = analyzer.generate_optimization_recommendations(morris_results)
    print(f"\nOptimization recommendations:")
    for strategy in recommendations['optimization_strategy']:
        print(f"  - {strategy}")
    
    # Optionally run full Sobol analysis on top parameters
    top_params = [p[0] for p in morris_results.parameter_rankings[:8]]
    
    print(f"\nRunning focused Sobol analysis on top {len(top_params)} parameters...")
    sobol_results = analyzer.run_sobol_analysis(n_samples=256, selected_params=top_params)
    
    print(f"\nSobol first-order indices:")
    for param in top_params:
        s1 = sobol_results.first_order_indices.get(param, 0.0)
        st = sobol_results.total_order_indices.get(param, 0.0)
        print(f"  {param}: S1={s1:.4f}, ST={st:.4f}")
    
    return 0

if __name__ == '__main__':
    exit(main())
