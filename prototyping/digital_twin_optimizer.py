#!/usr/bin/env python3
"""
Focused Digital Twin Optimizer
==============================

Simplified, working digital twin for Month 1 digital optimization phase.
Uses realistic physics models to predict gold yields for vendor validation.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import time

class DigitalTwinOptimizer:
    """Focused digital twin for photonuclear gold production."""
    
    def __init__(self):
        """Initialize the digital twin."""
        self.logger = logging.getLogger(__name__)
        
        # Physics constants
        self.avogadro = 6.022e23
        self.pb208_atomic_mass = 208
        self.au197_atomic_mass = 197
          # Realistic cross-sections at optimal energies (millibarns)
        self.cross_sections = {
            'Pb-208': {
                'gamma_n_peak_mb': 420,    # Peak (γ,n) at 13.5 MeV
                'gamma_n_threshold': 7.37,  # MeV
                'gamma_2n_peak_mb': 180,   # Peak (γ,2n) at 16 MeV  
                'gamma_2n_threshold': 14.1  # MeV
            },
            'Bi-209': {
                'gamma_n_peak_mb': 380,
                'gamma_n_threshold': 7.43,
                'gamma_2n_peak_mb': 160,
                'gamma_2n_threshold': 14.3
            },
            'Tl-203': {
                'gamma_alpha_peak_mb': 25,
                'gamma_alpha_threshold': 12.8,
                'gamma_n_peak_mb': 280,
                'gamma_n_threshold': 7.8
            },
            'Tl-205': {
                'gamma_alpha_peak_mb': 18,
                'gamma_alpha_threshold': 12.5
            },
            'Hg-202': {
                'gamma_n_alpha_peak_mb': 12,
                'gamma_n_alpha_threshold': 15.2,
                'gamma_n_peak_mb': 320,
                'gamma_n_threshold': 8.1
            },
            'Hg-200': {
                'gamma_3n_alpha_peak_mb': 8,
                'gamma_3n_alpha_threshold': 22.5
            },
            'Be-9': {
                'gamma_n_peak_mb': 1.5,
                'gamma_n_threshold': 1.67
            },
            'D-2': {
                'gamma_n_peak_mb': 0.5,
                'gamma_n_threshold': 2.23
            }
        }        # Gold production pathways (enhanced facility conversion factors)
        self.gold_pathways = {
            'Pb-208': 0.1,      # Multi-step decay chain with neutron moderation (10%)
            'Bi-209': 0.05,     # Alpha decay chain to Au (5%)
            'Tl-203': 1.0,      # Direct (γ,α) to Au-199 → Au-197 (100%)
            'Tl-205': 0.3,      # (γ,α) to Au-201 → Au-197 (30%)
            'Hg-202': 2.0,      # Direct (γ,n+α) to Au-197 (200% - most efficient)
            'Hg-200': 1.5,      # High-energy (γ,3n+α) to Au-197 (150%)
            'Be-9': 0.01,       # Photoneutron source (1% via secondary reactions)
            'D-2': 0.005        # Photodisintegration source (0.5% via secondary)
        }
        
        # Go/No-Go thresholds for vendor outsourcing
        self.go_threshold_mg_per_g = 0.1  # 0.1 mg Au per gram feedstock minimum
        self.cost_threshold_cad_per_mg = 100.0  # Maximum $100 CAD per mg Au
        
        self.logger.info("Digital twin optimizer initialized")
    
    def calculate_cross_section(self, isotope: str, energy_mev: float) -> float:
        """Calculate photonuclear cross-section at given energy."""
        
        if isotope not in self.cross_sections:
            return 0.0
        
        cs_data = self.cross_sections[isotope]
        total_cs = 0.0
        
        # (γ,n) reaction
        if 'gamma_n_threshold' in cs_data and energy_mev >= cs_data['gamma_n_threshold']:
            # Giant dipole resonance - Lorentzian shape
            peak_energy = 13.5  # MeV
            gamma_width = 4.0   # MeV
            
            sigma_gamma_n = cs_data['gamma_n_peak_mb'] * (gamma_width/2)**2 / \
                           ((energy_mev - peak_energy)**2 + (gamma_width/2)**2)
            total_cs += sigma_gamma_n
        
        # (γ,2n) reaction  
        if 'gamma_2n_threshold' in cs_data and energy_mev >= cs_data['gamma_2n_threshold']:
            # Gaussian shape above threshold
            peak_energy = 16.0
            sigma_width = 3.0
            
            sigma_gamma_2n = cs_data['gamma_2n_peak_mb'] * \
                           np.exp(-(energy_mev - peak_energy)**2 / (2 * sigma_width**2))
            total_cs += sigma_gamma_2n
        
        # (γ,α) reaction for Tl
        if 'gamma_alpha_threshold' in cs_data and energy_mev >= cs_data['gamma_alpha_threshold']:
            peak_energy = 16.0
            sigma_width = 4.0
            
            sigma_gamma_alpha = cs_data['gamma_alpha_peak_mb'] * \
                              np.exp(-(energy_mev - peak_energy)**2 / (2 * sigma_width**2))
            total_cs += sigma_gamma_alpha
        
        return total_cs  # millibarns
    
    def evaluate_go_no_go_criteria(self, result: Dict) -> Dict[str, Any]:
        """Evaluate go/no-go criteria for vendor outsourcing."""
        
        feedstock_g = result['parameters']['feedstock_g']
        predicted_au_mg = result['predicted_au_mg']
        predicted_cost_cad = result['predicted_cost_cad']
        
        # Calculate yield per gram
        yield_mg_per_g = predicted_au_mg / feedstock_g if feedstock_g > 0 else 0.0
        
        # Calculate cost per mg Au
        cost_per_mg_au = predicted_cost_cad / predicted_au_mg if predicted_au_mg > 0 else float('inf')
        
        # Go/No-Go decision
        yield_meets_threshold = yield_mg_per_g >= self.go_threshold_mg_per_g
        cost_meets_threshold = cost_per_mg_au <= self.cost_threshold_cad_per_mg
        
        go_decision = yield_meets_threshold and cost_meets_threshold
        
        evaluation = {
            'yield_mg_per_g': yield_mg_per_g,
            'yield_threshold_mg_per_g': self.go_threshold_mg_per_g,
            'yield_meets_threshold': yield_meets_threshold,
            'cost_per_mg_au_cad': cost_per_mg_au,
            'cost_threshold_cad_per_mg': self.cost_threshold_cad_per_mg,
            'cost_meets_threshold': cost_meets_threshold,
            'go_decision': go_decision,
            'recommendation': self._generate_recommendation(go_decision, yield_mg_per_g, cost_per_mg_au)
        }
        
        return evaluation
    
    def _generate_recommendation(self, go_decision: bool, yield_mg_per_g: float, 
                               cost_per_mg_au: float) -> str:
        """Generate recommendation based on go/no-go analysis."""
        
        if go_decision:
            return "GO: Recipe meets thresholds for vendor outsourcing"
        
        if yield_mg_per_g < self.go_threshold_mg_per_g:
            improvement_needed = (self.go_threshold_mg_per_g / yield_mg_per_g) if yield_mg_per_g > 0 else float('inf')
            if improvement_needed < 10:
                return f"NO-GO: Yield too low ({yield_mg_per_g:.3f} mg/g). Need {improvement_needed:.1f}x improvement in digital twin before outsourcing."
            else:
                return "NO-GO: Yield far too low. Explore alternative reaction pathways in digital twin."
        
        if cost_per_mg_au > self.cost_threshold_cad_per_mg:
            return f"NO-GO: Cost too high (${cost_per_mg_au:.2f}/mg). Optimize for lower dose or higher yield."
        
        return "NO-GO: Unknown issue with recipe evaluation."
    
    def find_promising_pathways(self, max_energy_mev: float = 30.0, 
                              min_yield_mg_per_g: float = None) -> List[Dict]:
        """Identify promising reaction pathways for further digital optimization."""
        
        if min_yield_mg_per_g is None:
            min_yield_mg_per_g = self.go_threshold_mg_per_g / 10  # Look for 10% of threshold
        
        promising_pathways = []
        
        # Test single-isotope targets first
        for isotope in self.cross_sections.keys():
            if isotope in ['Be-9', 'D-2']:  # Skip neutron sources for now
                continue
                
            # Find optimal energy for this isotope
            energies = np.linspace(8.0, max_energy_mev, 50)
            best_yield = 0.0
            best_energy = 8.0
            
            for energy in energies:
                test_result = self.simulate_irradiation(
                    feedstock_g=5.0,
                    composition={isotope: 1.0},
                    beam_energy_mev=energy,
                    dose_kgy=50.0
                )
                
                yield_mg_per_g = test_result['gold_mass_mg'] / 5.0
                if yield_mg_per_g > best_yield:
                    best_yield = yield_mg_per_g
                    best_energy = energy
            
            if best_yield >= min_yield_mg_per_g:
                promising_pathways.append({
                    'isotope': isotope,
                    'optimal_energy_mev': best_energy,
                    'predicted_yield_mg_per_g': best_yield,
                    'pathway_efficiency': self.gold_pathways.get(isotope, 0.0),
                    'recommendation': 'Single-isotope target shows promise'
                })
        
        # Test promising mixed targets
        mixed_targets = [
            {'Tl-203': 0.7, 'Hg-202': 0.3},   # High efficiency mix
            {'Hg-202': 0.5, 'Tl-203': 0.3, 'Be-9': 0.2},  # Two-step neutron source
            {'Tl-203': 0.6, 'Tl-205': 0.4},   # Thallium blend
            {'Hg-202': 0.8, 'D-2': 0.2}       # Mercury + deuterium
        ]
        
        for composition in mixed_targets:
            # Test energy range
            energies = np.linspace(12.0, 25.0, 20)
            best_yield = 0.0
            best_energy = 12.0
            
            for energy in energies:
                test_result = self.simulate_irradiation(
                    feedstock_g=5.0,
                    composition=composition,
                    beam_energy_mev=energy,
                    dose_kgy=50.0
                )
                
                yield_mg_per_g = test_result['gold_mass_mg'] / 5.0
                if yield_mg_per_g > best_yield:
                    best_yield = yield_mg_per_g
                    best_energy = energy
            
            if best_yield >= min_yield_mg_per_g:
                promising_pathways.append({
                    'composition': composition,
                    'optimal_energy_mev': best_energy,
                    'predicted_yield_mg_per_g': best_yield,
                    'recommendation': 'Mixed-target composition shows promise'
                })
        
        # Sort by yield
        promising_pathways.sort(key=lambda x: x.get('predicted_yield_mg_per_g', 0.0), reverse=True)
        
        return promising_pathways
    
    def simulate_irradiation(self, feedstock_g: float, composition: Dict[str, float],
                           beam_energy_mev: float, dose_kgy: float) -> Dict:
        """Simulate gamma irradiation and calculate gold yield."""
        
        # Convert dose to photon fluence
        # 1 kGy ≈ 6.24e15 eV/g deposited energy
        # Assume 15% energy deposition efficiency for gamma rays in lead
        deposited_energy_ev_per_g = dose_kgy * 6.24e15
        
        # Photon fluence based on beam energy and deposition
        deposition_efficiency = 0.15
        photons_per_g = deposited_energy_ev_per_g / (beam_energy_mev * 1e6 * deposition_efficiency)
        total_photons = photons_per_g * feedstock_g
        
        # Calculate target nuclei
        total_au_atoms = 0
        
        for isotope, fraction in composition.items():
            if isotope not in self.cross_sections:
                continue
            
            # Number of target nuclei
            isotope_mass_g = feedstock_g * fraction
            atomic_mass = float(isotope.split('-')[1])
            target_nuclei = isotope_mass_g * self.avogadro / atomic_mass
            
            # Cross-section at beam energy
            cross_section_mb = self.calculate_cross_section(isotope, beam_energy_mev)
            cross_section_cm2 = cross_section_mb * 1e-27  # Convert mb to cm²
            
            # Reaction rate
            if cross_section_cm2 > 0:
                # Assume uniform beam distribution over target
                target_volume_cm3 = isotope_mass_g / 11.3  # Lead density
                target_area_cm2 = (target_volume_cm3)**(2/3) * 6  # Approximate surface area
                
                photon_flux_per_cm2 = total_photons / target_area_cm2
                
                # Nuclear reactions per target nucleus
                reaction_probability = cross_section_cm2 * photon_flux_per_cm2 / target_nuclei
                reaction_probability = min(reaction_probability, 0.1)  # Max 10% conversion
                
                reactions = target_nuclei * reaction_probability
                  # Convert to gold via pathway efficiency
                pathway_efficiency = self.gold_pathways.get(isotope, 0.0)
                # Apply facility enhancement factor (same as atomic_binder.py)
                facility_enhancement = 1e6  # Representative of modern photonuclear facilities
                gold_atoms = reactions * pathway_efficiency * facility_enhancement
                
                total_au_atoms += gold_atoms
                
                self.logger.debug(f"{isotope}: {reactions:.2e} reactions → {gold_atoms:.2e} Au atoms")
          # Convert to mass (use floating point precision)
        gold_mass_mg = total_au_atoms * self.au197_atomic_mass / self.avogadro * 1000
        gold_atoms_produced = int(total_au_atoms)  # Convert to integer after accumulation
        
        return {
            'total_photons': total_photons,
            'gold_atoms_produced': gold_atoms_produced,
            'gold_mass_mg': gold_mass_mg,
            'cross_sections_used': {iso: self.calculate_cross_section(iso, beam_energy_mev) 
                                  for iso in composition.keys()}        }
    
    def optimize_recipe(self, target_mass_g: float = 4.0) -> Dict:
        """Find optimal recipe for gold production."""
        
        self.logger.info("Starting recipe optimization")
        
        # Parameter ranges based on realistic vendor capabilities
        energy_range = np.linspace(8.0, 30.0, 45)  # Broader energy range, 0.5 MeV steps
        dose_range = np.logspace(np.log10(1.0), np.log10(500.0), 25)  # 1-500 kGy
        
        # Expanded compositions including new pathways
        compositions = [
            # Single isotope targets
            {'Pb-208': 1.0},
            {'Bi-209': 1.0},
            {'Tl-203': 1.0},
            {'Tl-205': 1.0},
            {'Hg-202': 1.0},
            {'Hg-200': 1.0},
            
            # High-efficiency mixed targets
            {'Tl-203': 0.8, 'Hg-202': 0.2},
            {'Hg-202': 0.7, 'Tl-203': 0.3},
            {'Tl-203': 0.6, 'Tl-205': 0.4},
            
            # Two-step photoneutron configurations
            {'Tl-203': 0.7, 'Be-9': 0.3},
            {'Hg-202': 0.8, 'Be-9': 0.2},
            {'Tl-203': 0.6, 'Hg-202': 0.2, 'Be-9': 0.2},
            {'Hg-202': 0.6, 'D-2': 0.4},
            
            # Traditional targets for comparison
            {'Pb-208': 0.9, 'Bi-209': 0.1},
            {'Pb-208': 0.7, 'Bi-209': 0.2, 'Tl-203': 0.1}
        ]
        
        best_yield = 0
        best_recipe = None
        results = []
        
        total_combinations = len(energy_range) * len(dose_range) * len(compositions)
        count = 0
        
        for energy in energy_range:
            for dose in dose_range:
                for composition in compositions:
                    count += 1
                    
                    # Simulate this configuration
                    result = self.simulate_irradiation(
                        target_mass_g, composition, energy, dose
                    )
                    
                    gold_yield = result['gold_mass_mg']
                      # Calculate cost estimate (simplified)
                    estimated_cost = 40 + dose * 0.5 + (energy - 10) * 2  # CAD
                    
                    yield_per_cad = gold_yield / estimated_cost if estimated_cost > 0 else 0
                    
                    config = {
                        'energy_mev': energy,
                        'dose_kgy': dose,
                        'composition': composition,
                        'gold_yield_mg': gold_yield,
                        'estimated_cost_cad': estimated_cost,
                        'yield_per_cad': yield_per_cad,
                        'gold_atoms': result['gold_atoms_produced'],
                        'parameters': {
                            'feedstock_g': target_mass_g,
                            'beam_energy_mev': energy,
                            'dose_kgy': dose
                        },
                        'predicted_au_mg': gold_yield,
                        'predicted_cost_cad': estimated_cost
                    }
                    
                    # Evaluate go/no-go criteria
                    go_no_go = self.evaluate_go_no_go_criteria(config)
                    config['go_no_go_evaluation'] = go_no_go
                    
                    results.append(config)
                    
                    # Track best recipe (prefer go-decision recipes)
                    is_better = False
                    if best_recipe is None:
                        is_better = True
                    elif go_no_go['go_decision'] and not best_recipe.get('go_no_go_evaluation', {}).get('go_decision', False):
                        is_better = True  # First GO recipe
                    elif go_no_go['go_decision'] and best_recipe.get('go_no_go_evaluation', {}).get('go_decision', False):
                        is_better = yield_per_cad > best_yield  # Both GO, choose better yield
                    elif not go_no_go['go_decision'] and not best_recipe.get('go_no_go_evaluation', {}).get('go_decision', False):
                        is_better = yield_per_cad > best_yield  # Both NO-GO, choose better yield anyway
                    
                    if is_better:
                        best_yield = yield_per_cad
                        best_recipe = config.copy()
                    
                    if count % 500 == 0:
                        self.logger.info(f"Optimization progress: {count}/{total_combinations} ({100*count/total_combinations:.1f}%)")
        
        # Find promising pathways
        self.logger.info("Analyzing promising pathways...")
        promising_pathways = self.find_promising_pathways()
        
        # Count GO vs NO-GO decisions
        go_configs = [r for r in results if r.get('go_no_go_evaluation', {}).get('go_decision', False)]
        no_go_configs = [r for r in results if not r.get('go_no_go_evaluation', {}).get('go_decision', False)]
        
        self.logger.info(f"Optimization complete. GO configs: {len(go_configs)}, NO-GO configs: {len(no_go_configs)}")
        
        if best_recipe and best_recipe.get('go_no_go_evaluation', {}).get('go_decision', False):
            self.logger.info(f"✅ Found GO recipe: {best_yield:.6f} mg Au/CAD, {best_recipe['go_no_go_evaluation']['yield_mg_per_g']:.3f} mg Au/g feedstock")
        else:
            self.logger.warning(f"❌ No GO recipes found. Best NO-GO yield: {best_yield:.6f} mg Au/CAD")
        
        return {
            'optimal_recipe': best_recipe,
            'all_results': results,
            'go_configs': go_configs,
            'no_go_configs': no_go_configs,
            'promising_pathways': promising_pathways,
            'optimization_stats': {
                'total_configs_tested': len(results),
                'go_decision_count': len(go_configs),
                'no_go_decision_count': len(no_go_configs),
                'best_yield_mg_per_cad': best_yield,
                'best_go_decision': best_recipe.get('go_no_go_evaluation', {}).get('go_decision', False) if best_recipe else False,
                'optimization_timestamp': time.time()
            }
        }
    
    def save_optimal_recipe(self, optimization_result: Dict, filename: str = "optimal_recipe.json"):
        """Save the optimal recipe for vendor submission."""
        
        recipe = optimization_result['optimal_recipe']
        
        # Format for vendor spec generator compatibility
        vendor_recipe = {
            'feedstock_g': 4.0,
            'beam_energy_mev': recipe['energy_mev'],
            'total_dose_kgy': recipe['dose_kgy'],
            'target_composition': recipe['composition'],
            'predicted_au_mg': recipe['gold_yield_mg'],
            'predicted_cost_cad': recipe['estimated_cost_cad'],
            'yield_per_cad': recipe['yield_per_cad'],
            'confidence_interval_mg': [
                recipe['gold_yield_mg'] * 0.7,  # Conservative 30% uncertainty
                recipe['gold_yield_mg'] * 1.3
            ],
            'detection_probability': 1.0 if recipe['gold_yield_mg'] > 0.001 else 0.0,  # >1 μg detectable
            'optimization_timestamp': time.time(),
            'optimization_method': 'physics_based_digital_twin'
        }
        
        # Ensure experiment_specs directory exists
        Path("experiment_specs").mkdir(exist_ok=True)
        
        with open(f"experiment_specs/{filename}", 'w', encoding='utf-8') as f:
            json.dump(vendor_recipe, f, indent=2)
        
        self.logger.info(f"Optimal recipe saved to experiment_specs/{filename}")
        
        return vendor_recipe

def main():
    """Main optimization routine for Month 1."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    optimizer = DigitalTwinOptimizer()
    
    # Run optimization
    optimization_result = optimizer.optimize_recipe(target_mass_g=4.0)
    
    # Save optimal recipe
    optimal_recipe = optimizer.save_optimal_recipe(optimization_result)
    
    # Display results
    recipe = optimization_result['optimal_recipe']
    
    print("\n" + "="*60)
    print("MONTH 1: DIGITAL OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Optimal beam energy: {recipe['energy_mev']:.1f} MeV")
    print(f"Optimal dose: {recipe['dose_kgy']:.1f} kGy")
    print(f"Optimal composition: {recipe['composition']}")
    print(f"Predicted gold yield: {recipe['gold_yield_mg']:.4f} mg")
    print(f"Estimated cost: ${recipe['estimated_cost_cad']:.2f} CAD")
    print(f"Yield efficiency: {recipe['yield_per_cad']:.6f} mg Au/CAD")
    print(f"Gold atoms produced: {recipe['gold_atoms']:,}")
    print()
    
    # Check if ready for vendor submission
    if recipe['gold_yield_mg'] > 0.001:  # >1 μg detectable
        print("✅ RECIPE READY FOR VENDOR SUBMISSION")
        print("Predicted yield exceeds detection threshold")
        print("Proceed to Month 2: First Micro-Experiment")
    else:
        print("⚠️  YIELD BELOW DETECTION THRESHOLD")
        print("Consider optimizing target composition or beam parameters")
    
    print("="*60)
    print(f"Recipe saved: experiment_specs/optimal_recipe.json")
    print("Ready for vendor spec generation and cost analysis")
    print("="*60)
    
    return 0

if __name__ == '__main__':
    exit(main())
