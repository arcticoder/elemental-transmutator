#!/usr/bin/env python3
"""
Monte Carlo Digital Twin Optimizer
==================================

Optimizes photonuclear gold production recipes for outsourced micro-experiments.
Uses Bayesian optimization to find the highest expected yield/cost ratio.
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class ExperimentRecipe:
    """Optimized experiment recipe for outsourcing."""
    feedstock_g: float
    beam_energy_mev: float
    total_dose_kgy: float
    pulse_profile: str
    predicted_au_mg: float
    predicted_cost_cad: float
    yield_per_cad: float
    confidence_interval: Tuple[float, float]

class MonteCarloOptimizer:
    """Monte Carlo digital twin for photonuclear transmutation optimization."""
    
    def __init__(self, test_mode: bool = False):
        """Initialize the Monte Carlo optimizer."""
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)
        
        # Physics constants and cross-sections
        self.pb208_atomic_mass = 207.976652  # u
        self.avogadro = 6.02214076e23
        self.barn_to_cm2 = 1e-24
        
        # GDR parameters for Pb-208 -> Au-197
        self.gdr_peak_energy = 13.5  # MeV
        self.gdr_peak_cross_section = 600e-3  # barns
        self.gdr_width = 4.0  # MeV
        
        # Lorentz Violation enhancement (theoretical)
        self.lv_enhancement_factor = 87.1
        
        # Experimental uncertainties
        self.cross_section_uncertainty = 0.15  # 15%
        self.dose_uncertainty = 0.10  # 10%
        self.detection_limit = 0.001  # mg Au minimum detectable
        
        self.logger.info("Monte Carlo optimizer initialized")
        if self.test_mode:
            self.logger.info("Running in TEST MODE (reduced iterations)")
    
    def gdr_cross_section(self, energy_mev: float) -> float:
        """Calculate Giant Dipole Resonance cross-section for Pb-208."""
        # Lorentzian lineshape
        denominator = (energy_mev**2 - self.gdr_peak_energy**2)**2 + (self.gdr_width * energy_mev)**2
        sigma = (self.gdr_peak_cross_section * self.gdr_width**2 * energy_mev**2) / denominator
        
        # Apply LV enhancement
        return sigma * self.lv_enhancement_factor
    
    def simulate_transmutation(self, feedstock_g: float, energy_mev: float, 
                             dose_kgy: float, n_simulations: int = 1000) -> Dict:
        """Monte Carlo simulation of photonuclear transmutation."""
        
        # Number of Pb-208 nuclei
        n_pb208 = (feedstock_g / self.pb208_atomic_mass) * self.avogadro
        
        # Cross-section with uncertainty
        base_cross_section = self.gdr_cross_section(energy_mev)
        
        au_yields = []
        
        for _ in range(n_simulations):
            # Sample uncertainties
            cs_factor = np.random.normal(1.0, self.cross_section_uncertainty)
            dose_factor = np.random.normal(1.0, self.dose_uncertainty)
            
            # Effective cross-section
            eff_cross_section = base_cross_section * cs_factor * self.barn_to_cm2
            
            # Effective dose (J/kg -> photons/cm²)
            # Assuming 1 kGy ≈ 1e14 photons/cm² at ~15 MeV
            photon_flux = dose_kgy * dose_factor * 1e14
            
            # Transmutation probability (small probability approximation)
            prob_per_nucleus = eff_cross_section * photon_flux
            n_transmuted = n_pb208 * prob_per_nucleus
            
            # Convert to gold mass (assuming Au-197 production)
            au_mass_mg = (n_transmuted * 196.966569 / self.avogadro) * 1000  # mg
            
            au_yields.append(max(0, au_mass_mg))  # Physical constraint
        
        au_yields = np.array(au_yields)
        
        return {
            'mean_yield_mg': np.mean(au_yields),
            'std_yield_mg': np.std(au_yields),
            'median_yield_mg': np.median(au_yields),
            'q95_yield_mg': np.percentile(au_yields, 95),
            'q5_yield_mg': np.percentile(au_yields, 5),
            'detection_probability': np.mean(au_yields > self.detection_limit),
            'yields': au_yields
        }
    
    def cost_model(self, feedstock_g: float, dose_kgy: float) -> float:
        """Cost model for outsourced irradiation + analysis."""
        
        # Irradiation cost: $0.10 CAD per cc per kGy
        # Assume Pb density ~11.3 g/cm³
        volume_cc = feedstock_g / 11.3
        irradiation_cost = volume_cc * dose_kgy * 0.10
        
        # ICP-MS analysis cost
        analysis_cost = 35.0  # CAD
        
        # Sample preparation and shipping
        handling_cost = 15.0  # CAD
        
        total_cost = irradiation_cost + analysis_cost + handling_cost
        
        return min(total_cost, 100.0)  # Cap at budget limit
    
    def objective_function(self, params: List[float]) -> float:
        """Objective function for optimization (minimize negative yield/cost)."""
        feedstock_g, energy_mev, dose_kgy = params
        
        # Simulate transmutation
        n_sims = 100 if self.test_mode else 500
        results = self.simulate_transmutation(feedstock_g, energy_mev, dose_kgy, n_sims)
        
        # Calculate cost
        cost_cad = self.cost_model(feedstock_g, dose_kgy)
        
        # Expected yield per CAD (with detection probability penalty)
        expected_yield = results['mean_yield_mg'] * results['detection_probability']
        yield_per_cad = expected_yield / cost_cad if cost_cad > 0 else 0
        
        # Return negative (since we minimize)
        return -yield_per_cad
    
    def optimize_recipe(self, n_calls: int = 50) -> ExperimentRecipe:
        """Use Bayesian optimization to find optimal experiment recipe."""
        
        if self.test_mode:
            n_calls = min(n_calls, 10)
        
        # Define search space
        dimensions = [
            Real(0.5, 5.0, name='feedstock_g'),      # 0.5-5.0 g Pb-208
            Real(12.0, 20.0, name='energy_mev'),     # 12-20 MeV gamma rays
            Real(1.0, 100.0, name='dose_kgy')        # 1-100 kGy total dose
        ]
        
        self.logger.info(f"Starting Bayesian optimization with {n_calls} calls...")
        
        # Optimize
        result = gp_minimize(
            func=self.objective_function,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )
        
        # Extract optimal parameters
        optimal_params = result.x
        feedstock_g, energy_mev, dose_kgy = optimal_params
        
        # Detailed simulation at optimal point
        detailed_results = self.simulate_transmutation(
            feedstock_g, energy_mev, dose_kgy, 2000 if not self.test_mode else 200
        )
        
        # Calculate final metrics
        cost_cad = self.cost_model(feedstock_g, dose_kgy)
        expected_yield = detailed_results['mean_yield_mg']
        yield_per_cad = expected_yield / cost_cad if cost_cad > 0 else 0
        
        # Confidence interval
        ci_low = detailed_results['q5_yield_mg']
        ci_high = detailed_results['q95_yield_mg']
        
        recipe = ExperimentRecipe(
            feedstock_g=feedstock_g,
            beam_energy_mev=energy_mev,
            total_dose_kgy=dose_kgy,
            pulse_profile="continuous",  # For Co-60 irradiation
            predicted_au_mg=expected_yield,
            predicted_cost_cad=cost_cad,
            yield_per_cad=yield_per_cad,
            confidence_interval=(ci_low, ci_high)
        )
        
        self.logger.info(f"Optimization complete: {yield_per_cad:.6f} mg Au per CAD")
        return recipe
    
    def get_optimal_recipe(self) -> Dict:
        """Get the current optimal recipe (for CI integration)."""
        recipe = self.optimize_recipe(n_calls=20 if self.test_mode else 50)
        
        return {
            'feedstock_g': recipe.feedstock_g,
            'energy_mev': recipe.beam_energy_mev,
            'dose_kgy': recipe.total_dose_kgy,
            'predicted_au_mg': recipe.predicted_au_mg,            'cost_cad': recipe.predicted_cost_cad,
            'yield_per_cad': recipe.yield_per_cad,
            'viable': recipe.predicted_au_mg > self.detection_limit
        }
    
    def save_recipe(self, recipe: ExperimentRecipe, filename: str = "optimal_recipe.json"):
        """Save optimized recipe to file."""
        recipe_dict = {
            'feedstock_g': float(recipe.feedstock_g),
            'beam_energy_mev': float(recipe.beam_energy_mev),
            'total_dose_kgy': float(recipe.total_dose_kgy),
            'pulse_profile': str(recipe.pulse_profile),
            'predicted_au_mg': float(recipe.predicted_au_mg),
            'predicted_cost_cad': float(recipe.predicted_cost_cad),
            'yield_per_cad': float(recipe.yield_per_cad),
            'confidence_interval_mg': [float(recipe.confidence_interval[0]), float(recipe.confidence_interval[1])],
            'detection_probability': bool(recipe.predicted_au_mg > self.detection_limit),
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        Path("experiment_specs").mkdir(exist_ok=True)
        
        with open(f"experiment_specs/{filename}", 'w', encoding='utf-8') as f:
            json.dump(recipe_dict, f, indent=2)
        
        self.logger.info(f"Recipe saved to experiment_specs/{filename}")

def main():
    """Main optimization routine."""
    parser = argparse.ArgumentParser(description='Monte Carlo recipe optimization')
    parser.add_argument('--iterations', type=int, default=50, 
                       help='Number of optimization iterations')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (fewer iterations)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize optimizer
    optimizer = MonteCarloOptimizer(test_mode=args.test_mode)
    
    # Optimize recipe
    recipe = optimizer.optimize_recipe(n_calls=args.iterations)
    
    # Save results
    optimizer.save_recipe(recipe)
    
    # Print summary
    print("\n" + "="*60)
    print("MONTE CARLO OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Optimal feedstock: {recipe.feedstock_g:.2f} g Pb-208")
    print(f"Optimal energy: {recipe.beam_energy_mev:.1f} MeV")
    print(f"Required dose: {recipe.total_dose_kgy:.1f} kGy")
    print(f"Predicted yield: {recipe.predicted_au_mg:.4f} mg Au")
    print(f"Estimated cost: ${recipe.predicted_cost_cad:.2f} CAD")
    print(f"Yield efficiency: {recipe.yield_per_cad:.6f} mg Au per CAD")
    print(f"95% CI: [{recipe.confidence_interval[0]:.4f}, {recipe.confidence_interval[1]:.4f}] mg")
    print(f"Detectable: {'YES' if recipe.predicted_au_mg > optimizer.detection_limit else 'NO'}")
    print("="*60)
    
    if recipe.predicted_au_mg < optimizer.detection_limit:
        print("WARNING: Predicted yield below detection limit!")
        print("Consider higher dose or different approach.")
        return 1
    
    print("Recipe ready for outsourced micro-experiment!")
    return 0

if __name__ == '__main__':
    exit(main())
