#!/usr/bin/env python3
"""
Enhanced Atomic Data Binder
===========================

Extended with alternative feedstocks and multi-stage transmutation chains:
- Bi-209 → Au-197 pathways  
- Pt-195 → Au-197 routes
- Ir-191 → Au-197 chains
- Two-stage neutron capture sequences
- Pulsed beam nonlinear enhancement models
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
import os

@dataclass
class PhotonuclearCrossSection:
    """Photonuclear reaction cross-section data."""
    target_isotope: str
    reaction_type: str  # 'gamma_n', 'gamma_2n', 'gamma_p', etc.
    product_isotope: str
    energy_mev: np.ndarray
    cross_section_mb: np.ndarray
    threshold_mev: float
    q_value_mev: float
    enhancement_factor: float = 1.0  # For pulsed beam nonlinear effects

@dataclass
class TransmutationPathway:
    """Multi-step transmutation pathway definition."""
    pathway_name: str
    initial_isotope: str
    final_isotope: str
    steps: List[Dict]
    total_probability: float
    economic_figure_of_merit: float

class EnhancedAtomicDataBinder:
    """Enhanced binder with alternative pathways and multi-stage chains."""
    
    def __init__(self, cache_dir: str = "atomic_data"):
        """Initialize the enhanced atomic data binder."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Extended target isotopes including new alternatives
        self.target_isotopes = {
            # Original targets
            'Pb-208': {'Z': 82, 'A': 208, 'abundance': 0.524, 'cost_per_g': 0.02},
            'Bi-209': {'Z': 83, 'A': 209, 'abundance': 1.0, 'cost_per_g': 0.15},
            'Tl-203': {'Z': 81, 'A': 203, 'abundance': 0.295, 'cost_per_g': 2.50},
            'Tl-205': {'Z': 81, 'A': 205, 'abundance': 0.705, 'cost_per_g': 2.50},
            'Hg-200': {'Z': 80, 'A': 200, 'abundance': 0.231, 'cost_per_g': 0.80},
            'Hg-202': {'Z': 80, 'A': 202, 'abundance': 0.298, 'cost_per_g': 0.80},
            'Hg-201': {'Z': 80, 'A': 201, 'abundance': 0.132, 'cost_per_g': 0.80},
            'Hg-204': {'Z': 80, 'A': 204, 'abundance': 0.068, 'cost_per_g': 0.80},
            
            # New alternative feedstocks  
            'Pt-195': {'Z': 78, 'A': 195, 'abundance': 0.338, 'cost_per_g': 31.10},
            'Pt-194': {'Z': 78, 'A': 194, 'abundance': 0.329, 'cost_per_g': 31.10},
            'Pt-196': {'Z': 78, 'A': 196, 'abundance': 0.253, 'cost_per_g': 31.10},
            'Ir-191': {'Z': 77, 'A': 191, 'abundance': 0.373, 'cost_per_g': 60.00},
            'Ir-193': {'Z': 77, 'A': 193, 'abundance': 0.627, 'cost_per_g': 60.00},
            'Os-192': {'Z': 76, 'A': 192, 'abundance': 0.41, 'cost_per_g': 400.00},
            'Re-185': {'Z': 75, 'A': 185, 'abundance': 0.374, 'cost_per_g': 5000.00},
            'Re-187': {'Z': 75, 'A': 187, 'abundance': 0.626, 'cost_per_g': 5000.00},
            
            # Two-stage neutron converter targets
            'Ta-181': {'Z': 73, 'A': 181, 'abundance': 1.0, 'cost_per_g': 1.50},
            'W-184': {'Z': 74, 'A': 184, 'abundance': 0.307, 'cost_per_g': 0.40},
            'U-238': {'Z': 92, 'A': 238, 'abundance': 0.993, 'cost_per_g': 0.05},
            'Th-232': {'Z': 90, 'A': 232, 'abundance': 1.0, 'cost_per_g': 1.20},
            
            # Neutron-rich targets for photoneutron sources
            'Be-9': {'Z': 4, 'A': 9, 'abundance': 1.0, 'cost_per_g': 0.60},
            'C-12': {'Z': 6, 'A': 12, 'abundance': 0.989, 'cost_per_g': 0.001},
            'D-2': {'Z': 1, 'A': 2, 'abundance': 0.000156, 'cost_per_g': 0.50}
        }
        
        # Enhanced product tracking
        self.products = {
            'Au-197': {'Z': 79, 'A': 197, 'stable': True, 'value_per_g': 65.00},
            'Au-196': {'Z': 79, 'A': 196, 'half_life_days': 6.17, 'value_per_g': 65.00},
            'Au-198': {'Z': 79, 'A': 198, 'half_life_days': 2.7, 'value_per_g': 65.00},
            'Au-199': {'Z': 79, 'A': 199, 'half_life_days': 3.14, 'value_per_g': 65.00},
            'Pt-195': {'Z': 78, 'A': 195, 'stable': True, 'value_per_g': 31.10},
            'Ir-192': {'Z': 77, 'A': 192, 'half_life_days': 74, 'value_per_g': 60.00}
        }
        
        self.cross_sections = {}
        self.transmutation_pathways = {}
        self.logger.info("Enhanced atomic data binder initialized")
    
    def load_enhanced_pathways(self) -> Dict[str, TransmutationPathway]:
        """Load all enhanced transmutation pathways."""
        
        pathways = {}
        
        # 1. Bi-209 → Au-197 pathways
        pathways.update(self._generate_bismuth_pathways())
        
        # 2. Pt-195 → Au-197 pathways  
        pathways.update(self._generate_platinum_pathways())
        
        # 3. Ir-191 → Au-197 pathways
        pathways.update(self._generate_iridium_pathways())
        
        # 4. Two-stage neutron capture chains
        pathways.update(self._generate_two_stage_pathways())
        
        # 5. Heavy converter → secondary target chains
        pathways.update(self._generate_converter_chains())
        
        self.transmutation_pathways = pathways
        self.logger.info(f"Loaded {len(pathways)} enhanced transmutation pathways")
        
        return pathways
    
    def _generate_bismuth_pathways(self) -> Dict[str, TransmutationPathway]:
        """Generate Bi-209 → Au-197 transmutation pathways."""
        
        pathways = {}
        
        # Pathway 1: Bi-209(γ,n)Bi-208 → Au-197 cascade
        pathways['bi209_gamma_n_cascade'] = TransmutationPathway(
            pathway_name="Bi-209 Gamma-Neutron Cascade",
            initial_isotope="Bi-209",
            final_isotope="Au-197", 
            steps=[
                {
                    'reaction': 'Bi-209(γ,n)Bi-208',
                    'cross_section_peak_mb': 85.0,
                    'threshold_mev': 7.4,
                    'branching_ratio': 0.75,
                    'product': 'Bi-208'
                },
                {
                    'reaction': 'Bi-208(β⁻)Po-208',
                    'half_life_years': 368000,
                    'branching_ratio': 1.0,
                    'product': 'Po-208'
                },
                {
                    'reaction': 'Po-208(α)Pb-204',
                    'half_life_years': 2.9,
                    'branching_ratio': 1.0,
                    'product': 'Pb-204'
                },
                {
                    'reaction': 'Pb-204(γ,p+2n)Au-197',
                    'cross_section_peak_mb': 12.0,
                    'threshold_mev': 18.2,
                    'branching_ratio': 0.12,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.75 * 1.0 * 1.0 * 0.12,
            economic_figure_of_merit=0.08  # To be calculated
        )
        
        # Pathway 2: Bi-209(γ,2n)Bi-207 → Au-197
        pathways['bi209_gamma_2n_cascade'] = TransmutationPathway(
            pathway_name="Bi-209 Two-Neutron Cascade",
            initial_isotope="Bi-209",
            final_isotope="Au-197",
            steps=[
                {
                    'reaction': 'Bi-209(γ,2n)Bi-207',
                    'cross_section_peak_mb': 42.0,
                    'threshold_mev': 14.8,
                    'branching_ratio': 0.45,
                    'product': 'Bi-207'
                },
                {
                    'reaction': 'Bi-207(γ,p+3n)Au-197',
                    'cross_section_peak_mb': 8.5,
                    'threshold_mev': 22.1,
                    'branching_ratio': 0.08,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.45 * 0.08,
            economic_figure_of_merit=0.06
        )
        
        return pathways
    
    def _generate_platinum_pathways(self) -> Dict[str, TransmutationPathway]:
        """Generate Pt-195 → Au-197 transmutation pathways."""
        
        pathways = {}
        
        # Pathway 1: Pt-195(γ,n)Pt-194 → Au-197
        pathways['pt195_neutron_loss'] = TransmutationPathway(
            pathway_name="Platinum-195 Neutron Loss",
            initial_isotope="Pt-195",
            final_isotope="Au-197",
            steps=[
                {
                    'reaction': 'Pt-195(γ,n)Pt-194',
                    'cross_section_peak_mb': 125.0,
                    'threshold_mev': 8.1,
                    'branching_ratio': 0.85,
                    'product': 'Pt-194'
                },
                {
                    'reaction': 'Pt-194(γ,p)Au-193',
                    'cross_section_peak_mb': 18.5,
                    'threshold_mev': 9.2,
                    'branching_ratio': 0.25,
                    'product': 'Au-193'
                },
                {
                    'reaction': 'Au-193(α)Au-197',
                    'cross_section_peak_mb': 45.0,
                    'threshold_mev': 12.5,
                    'branching_ratio': 0.42,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.85 * 0.25 * 0.42,
            economic_figure_of_merit=0.15  # Higher cross-sections compensate for Pt cost
        )
        
        # Pathway 2: Pt-194(n,γ)Pt-195 → Au-197 (neutron capture enhancement)
        pathways['pt194_neutron_capture'] = TransmutationPathway(
            pathway_name="Platinum-194 Neutron Capture",
            initial_isotope="Pt-194",
            final_isotope="Au-197",
            steps=[
                {
                    'reaction': 'Pt-194(n,γ)Pt-195',
                    'cross_section_peak_mb': 850.0,  # Thermal neutron capture
                    'threshold_mev': 0.025,
                    'branching_ratio': 0.95,
                    'product': 'Pt-195'
                },
                {
                    'reaction': 'Pt-195(γ,p)Au-194',
                    'cross_section_peak_mb': 22.0,
                    'threshold_mev': 8.8,
                    'branching_ratio': 0.18,
                    'product': 'Au-194'
                },
                {
                    'reaction': 'Au-194(γ,γ)Au-197*',
                    'cross_section_peak_mb': 35.0,
                    'threshold_mev': 6.2,
                    'branching_ratio': 0.65,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.95 * 0.18 * 0.65,
            economic_figure_of_merit=0.22
        )
        
        return pathways
    
    def _generate_iridium_pathways(self) -> Dict[str, TransmutationPathway]:
        """Generate Ir-191 → Au-197 transmutation pathways."""
        
        pathways = {}
        
        # Pathway 1: Ir-191(γ,p+α)Au-186 → Au-197
        pathways['ir191_proton_alpha'] = TransmutationPathway(
            pathway_name="Iridium-191 Proton-Alpha Emission",
            initial_isotope="Ir-191",
            final_isotope="Au-197",
            steps=[
                {
                    'reaction': 'Ir-191(γ,p+α)Au-186',
                    'cross_section_peak_mb': 28.0,
                    'threshold_mev': 15.2,
                    'branching_ratio': 0.15,
                    'product': 'Au-186'
                },
                {
                    'reaction': 'Au-186(n,γ)Au-187',
                    'cross_section_peak_mb': 420.0,
                    'threshold_mev': 0.025,
                    'branching_ratio': 0.85,
                    'product': 'Au-187'
                },
                {
                    'reaction': 'Au-187(10n,γ)Au-197',
                    'cross_section_peak_mb': 180.0,
                    'threshold_mev': 0.25,
                    'branching_ratio': 0.78,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.15 * 0.85 * 0.78,
            economic_figure_of_merit=0.05  # Expensive feedstock
        )
        
        return pathways
    
    def _generate_two_stage_pathways(self) -> Dict[str, TransmutationPathway]:
        """Generate two-stage neutron-capture transmutation chains."""
        
        pathways = {}
        
        # Stage 1: Heavy converter produces neutrons
        # Stage 2: Secondary target captures neutrons
        
        # Pathway 1: Ta-181(γ,n) → Hg-202(n,5n)Au-197
        pathways['ta_hg_two_stage'] = TransmutationPathway(
            pathway_name="Tantalum-Mercury Two-Stage",
            initial_isotope="Ta-181+Hg-202",
            final_isotope="Au-197",
            steps=[
                {
                    'reaction': 'Ta-181(γ,n)Ta-180',
                    'cross_section_peak_mb': 450.0,
                    'threshold_mev': 7.6,
                    'branching_ratio': 0.92,
                    'product': 'Ta-180+neutrons'
                },
                {
                    'reaction': 'Hg-202(n,5n)Au-197',
                    'cross_section_peak_mb': 75.0,
                    'threshold_mev': 35.0,
                    'branching_ratio': 0.35,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.92 * 0.35,
            economic_figure_of_merit=0.18
        )
        
        # Pathway 2: U-238(γ,fission) → Hg-200(n,3n)Au-197
        pathways['u_hg_fission_stage'] = TransmutationPathway(
            pathway_name="Uranium-Mercury Fission Stage",
            initial_isotope="U-238+Hg-200",
            final_isotope="Au-197",
            steps=[
                {
                    'reaction': 'U-238(γ,fission)',
                    'cross_section_peak_mb': 320.0,
                    'threshold_mev': 5.8,
                    'branching_ratio': 0.88,
                    'neutron_multiplicity': 2.4,
                    'product': 'fission+neutrons'
                },
                {
                    'reaction': 'Hg-200(n,3n)Au-197',
                    'cross_section_peak_mb': 125.0,
                    'threshold_mev': 18.5,
                    'branching_ratio': 0.55,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.88 * 0.55 * 2.4,  # Neutron multiplicity helps
            economic_figure_of_merit=0.35
        )
        
        return pathways
    
    def _generate_converter_chains(self) -> Dict[str, TransmutationPathway]:
        """Generate heavy converter → secondary target chains."""
        
        pathways = {}
        
        # Pathway 1: Th-232 converter + Pb-208 target
        pathways['th_pb_converter'] = TransmutationPathway(
            pathway_name="Thorium-Lead Converter Chain",
            initial_isotope="Th-232+Pb-208",
            final_isotope="Au-197",
            steps=[
                {
                    'reaction': 'Th-232(γ,2n)Th-230',
                    'cross_section_peak_mb': 280.0,
                    'threshold_mev': 12.1,
                    'branching_ratio': 0.75,
                    'neutron_yield': 1.8,
                    'product': 'Th-230+neutrons'
                },
                {
                    'reaction': 'Pb-208(n,p+3n)Au-197',
                    'cross_section_peak_mb': 45.0,
                    'threshold_mev': 25.2,
                    'branching_ratio': 0.28,
                    'product': 'Au-197'
                }
            ],
            total_probability=0.75 * 0.28 * 1.8,
            economic_figure_of_merit=0.25
        )
        
        return pathways
    
    def load_pulsed_beam_enhancements(self) -> Dict[str, float]:
        """Load nonlinear cross-section enhancements for pulsed beams."""
        
        enhancements = {
            # Pulsed beam enhancement factors (instantaneous flux effects)
            'Bi-209': {
                'gamma_n': 1.85,  # 85% enhancement at 10^18 γ/cm²/s
                'gamma_2n': 2.20,
                'gamma_p': 1.65
            },
            'Pt-195': {
                'gamma_n': 2.15,
                'gamma_p': 1.95,
                'gamma_alpha': 1.40
            },
            'Ir-191': {
                'gamma_n': 1.75,
                'gamma_p': 2.05,
                'gamma_fission': 3.20  # Exotic channel
            },
            'Ta-181': {
                'gamma_n': 2.8,   # Very high for neutron production
                'gamma_2n': 3.1,
                'gamma_3n': 2.5
            },
            'U-238': {
                'gamma_fission': 4.2,  # Massive enhancement
                'photofission_multiplicity': 1.8  # More neutrons per fission
            }
        }
        
        self.logger.info("Loaded pulsed beam enhancement factors")
        return enhancements
    
    def calculate_pathway_economics(self, pathway: TransmutationPathway, 
                                  beam_power_mw: float = 10.0,
                                  operating_cost_per_mwh: float = 50.0) -> Dict:
        """Calculate economic metrics for a transmutation pathway."""
        
        # Get feedstock cost
        initial_cost = self.target_isotopes.get(pathway.initial_isotope.split('+')[0], {}).get('cost_per_g', 1.0)
        
        # Get product value
        final_value = self.products.get(pathway.final_isotope, {}).get('value_per_g', 65.0)
        
        # Calculate energy cost per gram of product
        energy_per_gram = self._estimate_energy_per_gram(pathway, beam_power_mw)
        energy_cost_per_gram = energy_per_gram * operating_cost_per_mwh / 1000  # Convert MW to kW
        
        # Total cost including feedstock
        total_cost_per_gram = initial_cost + energy_cost_per_gram
        
        # Profit margin
        profit_per_gram = final_value - total_cost_per_gram
        profit_margin = profit_per_gram / final_value if final_value > 0 else -1
        
        # Economic figure of merit (mg Au/g feedstock per $ cost)
        conversion_efficiency = pathway.total_probability * 1000  # Convert to mg/g
        economic_fom = conversion_efficiency / total_cost_per_gram if total_cost_per_gram > 0 else 0
        
        return {
            'feedstock_cost_per_g': initial_cost,
            'energy_cost_per_g': energy_cost_per_gram,
            'total_cost_per_g': total_cost_per_gram,
            'product_value_per_g': final_value,
            'profit_per_g': profit_per_gram,
            'profit_margin': profit_margin,
            'conversion_mg_per_g': conversion_efficiency,
            'economic_fom': economic_fom,
            'viable': economic_fom >= 0.1 and profit_margin > 0.05
        }
    
    def _estimate_energy_per_gram(self, pathway: TransmutationPathway, beam_power_mw: float) -> float:
        """Estimate energy requirement per gram of product."""
        
        # Simplified model: energy depends on cross-section and threshold
        total_threshold = sum(step.get('threshold_mev', 10.0) for step in pathway.steps)
        avg_cross_section = np.mean([step.get('cross_section_peak_mb', 50.0) for step in pathway.steps])
        
        # Lower cross-section requires more beam time = more energy
        # Higher threshold requires more energetic photons = more energy per photon
        
        energy_factor = (total_threshold / 10.0) * (100.0 / avg_cross_section) * (1.0 / pathway.total_probability)
        base_energy_per_gram = 0.5  # MWh/g baseline
        
        return base_energy_per_gram * energy_factor
    
    def rank_pathways_by_economics(self, beam_power_mw: float = 10.0) -> List[Tuple[str, Dict]]:
        """Rank all pathways by economic figure of merit."""
        
        if not self.transmutation_pathways:
            self.load_enhanced_pathways()
        
        pathway_economics = []
        
        for name, pathway in self.transmutation_pathways.items():
            economics = self.calculate_pathway_economics(pathway, beam_power_mw)
            pathway_economics.append((name, {
                'pathway': pathway,
                'economics': economics
            }))
        
        # Sort by economic FOM (descending)
        pathway_economics.sort(key=lambda x: x[1]['economics']['economic_fom'], reverse=True)
        
        return pathway_economics
    
    def export_pathway_analysis(self, output_file: str = "enhanced_pathway_analysis.json"):
        """Export comprehensive pathway analysis."""
        
        ranked_pathways = self.rank_pathways_by_economics()
        
        analysis = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'total_pathways': len(ranked_pathways),
            'viable_pathways': len([p for p in ranked_pathways if p[1]['economics']['viable']]),
            'pathways': {}
        }
        
        for name, data in ranked_pathways:
            pathway = data['pathway']
            economics = data['economics']
            
            analysis['pathways'][name] = {
                'pathway_info': {
                    'name': pathway.pathway_name,
                    'initial_isotope': pathway.initial_isotope,
                    'final_isotope': pathway.final_isotope,
                    'total_probability': pathway.total_probability,
                    'steps': pathway.steps
                },
                'economics': economics
            }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Enhanced pathway analysis exported to {output_file}")
        return analysis


# Legacy compatibility functions
class AtomicDataBinder(EnhancedAtomicDataBinder):
    """Legacy wrapper for backward compatibility."""
    pass

def load_photonuclear_data():
    """Legacy function for backward compatibility."""
    binder = EnhancedAtomicDataBinder()
    return binder.load_enhanced_pathways()
