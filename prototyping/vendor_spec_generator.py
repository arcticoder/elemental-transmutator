#!/usr/bin/env python3
"""
Vendor Specification Generator
==============================

Generates detailed specifications for outsourced micro-experiments,
including sample preparation, irradiation parameters, and analysis requirements.
"""

import argparse
import json
import logging
from typing import Dict, List
from pathlib import Path
from datetime import datetime, timedelta

class VendorSpecGenerator:
    """Generator for vendor-ready experiment specifications."""
    
    def __init__(self, test_mode: bool = False):
        """Initialize the specification generator."""
        self.test_mode = test_mode
        self.logger = logging.getLogger(__name__)
        
        # Sample preparation standards
        self.pb_density = 11.34  # g/cm³
        self.pb208_purity_required = 0.99  # 99% minimum
          # Safety and regulatory requirements
        self.max_activity_exemption = 10e6  # Bq, typical exemption limit
        
        self.logger.info("Vendor specification generator initialized")
    
    def generate_irradiation_spec(self, recipe: Dict) -> Dict:
        """Generate irradiation service specification."""
        
        feedstock_g = recipe['feedstock_g']
        dose_kgy = recipe.get('total_dose_kgy', recipe.get('dose_kgy', 10.0))
        energy_mev = recipe.get('beam_energy_mev', recipe.get('energy_mev', 15.0))
        
        # Calculate sample dimensions
        volume_cm3 = feedstock_g / self.pb_density
        
        # Assume cylindrical pellet with height = diameter
        radius_cm = (volume_cm3 / (2 * 3.14159))**(1/3)
        height_cm = 2 * radius_cm
        
        spec = {
            "service_type": "gamma_irradiation",
            "irradiation_parameters": {
                "source_type": "Co-60",
                "target_dose_kgy": dose_kgy,
                "dose_rate_kgy_h": 1.0,  # Typical for Co-60 irradiator
                "estimated_time_hours": dose_kgy / 1.0,
                "temperature_c": 25,  # Room temperature
                "atmosphere": "air"
            },
            "sample_specifications": {
                "material": "Lead-208 metal",
                "mass_g": feedstock_g,
                "purity_minimum": self.pb208_purity_required,
                "form": "pressed_pellet",
                "dimensions": {
                    "shape": "cylinder",
                    "radius_cm": radius_cm,
                    "height_cm": height_cm,
                    "volume_cm3": volume_cm3
                },
                "container": "polyethylene_vial",
                "labeling": "Pb-208 sample, research use"
            },
            "dosimetry_requirements": {
                "dose_mapping": True,
                "dosimeter_type": "alanine",
                "calibration_standard": "NPL_UK",
                "uncertainty_target_percent": 5.0
            },
            "post_irradiation": {
                "cooling_time_hours": 24,
                "activity_measurement_required": True,
                "expected_isotopes": ["Pb-207", "Tl-207", "Au-197"],
                "shipping_classification": "radioactive_material_excepted"
            },
            "quality_assurance": {
                "chain_of_custody": True,
                "photo_documentation": True,
                "temperature_logging": True
            }
        }
        
        return spec
    
    def generate_analysis_spec(self, recipe: Dict) -> Dict:
        """Generate analytical service specification."""
        
        feedstock_g = recipe['feedstock_g']
        predicted_yield_mg = recipe.get('predicted_au_mg', 0.01)
        
        # Detection limit requirement (10x better than predicted yield)
        required_detection_limit = predicted_yield_mg / 10.0
        
        spec = {
            "service_type": "precious_metals_analysis",
            "analytical_method": "ICP-MS",
            "target_elements": {
                "primary": ["Au"],
                "secondary": ["Pt", "Pd", "Ag"],  # Other precious metals
                "matrix": ["Pb", "Tl"]  # Matrix elements
            },
            "sample_preparation": {
                "dissolution_method": "aqua_regia",
                "dilution_factor": 100,
                "internal_standards": ["Ir-193", "Re-185"],
                "pre_concentration": "fire_assay_optional"
            },
            "analytical_requirements": {
                "detection_limit_mg_kg": required_detection_limit * 1000 / feedstock_g,  # mg/kg
                "precision_rsd_percent": 10.0,
                "accuracy_bias_percent": 15.0,
                "calibration_range_mg_kg": [0.01, 1000],
                "reference_materials": ["NIST_SRM_87a", "CANMET_TDB-1"]
            },
            "quality_control": {
                "method_blanks": 2,
                "duplicate_analysis": True,
                "spike_recovery": True,
                "reference_material_frequency": "every_10_samples"
            },
            "reporting_requirements": {
                "units": "mg/kg_and_total_mg",
                "significant_figures": 4,
                "uncertainty_reporting": True,
                "detection_limit_statement": True,
                "raw_data_included": True
            },
            "sample_handling": {
                "radioactive_sample": True,
                "license_required": "Type_II_nuclear_substance_license",
                "waste_disposal": "included",
                "return_sample": False
            }
        }
        
        return spec
    
    def generate_shipping_spec(self, recipe: Dict) -> Dict:
        """Generate shipping and logistics specification."""
        
        feedstock_g = recipe['feedstock_g']
        
        # Estimate post-irradiation activity
        dose_kgy = recipe.get('dose_kgy', 10.0)
        estimated_activity_bq = feedstock_g * dose_kgy * 1e3  # Rough estimate
        
        spec = {
            "shipping_classification": {
                "un_number": "UN2910" if estimated_activity_bq < self.max_activity_exemption else "UN2982",
                "proper_shipping_name": "Radioactive material, excepted package",
                "hazard_class": "7",
                "packaging_group": "N/A",
                "excepted_quantity": estimated_activity_bq < self.max_activity_exemption
            },
            "packaging_requirements": {
                "primary_container": "polyethylene_vial_5ml",
                "secondary_container": "lead_lined_box",
                "outer_packaging": "UN_spec_7A_drum",
                "cushioning": "vermiculite",
                "radiation_shielding": "2mm_lead_equivalent"
            },
            "documentation": {
                "radiation_survey": True,
                "contamination_survey": True,
                "activity_declaration": True,
                "material_safety_data_sheet": True,
                "chain_of_custody": True
            },
            "logistics": {
                "carrier": "FedEx_dangerous_goods",
                "service_level": "next_day",
                "insurance_value_cad": 500,
                "tracking_required": True,
                "signature_required": True
            },
            "regulatory_compliance": {
                "transport_canada_approval": True,
                "receiving_license_verified": True,
                "import_permit_if_required": True
            }
        }
        
        return spec
    
    def generate_complete_specification(self) -> Dict:
        """Generate complete experiment specification package."""
          # Load optimal recipe
        try:
            with open("experiment_specs/optimal_recipe.json", 'r') as f:
                recipe = json.load(f)
        except FileNotFoundError:
            # Default recipe for testing
            recipe = {
                'feedstock_g': 1.0,
                'total_dose_kgy': 10.0,
                'beam_energy_mev': 15.0,
                'predicted_au_mg': 0.01,
                'predicted_cost_cad': 60.0
            }
            self.logger.warning("No optimal recipe found, using defaults")
        
        # Generate specifications
        irradiation_spec = self.generate_irradiation_spec(recipe)
        analysis_spec = self.generate_analysis_spec(recipe)
        shipping_spec = self.generate_shipping_spec(recipe)
        
        # Timeline
        start_date = datetime.now() + timedelta(days=7)  # One week prep time
        irradiation_complete = start_date + timedelta(days=5)
        analysis_complete = irradiation_complete + timedelta(days=7)
        
        complete_spec = {
            "experiment_metadata": {
                "experiment_id": f"PGPR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "project_name": "Photonuclear Gold Production Research",
                "principal_investigator": "Digital Researcher",
                "institution": "Independent Research",
                "experiment_date": start_date.isoformat(),
                "specification_version": "1.0"
            },
            "experiment_overview": {
                "objective": "Measure gold production from Co-60 irradiation of Pb-208",
                "hypothesis": "Photonuclear transmutation will produce detectable Au-197",
                "success_criteria": {
                    "minimum_detection": f"{recipe.get('predicted_au_mg', 0.01):.4f} mg Au",
                    "measurement_uncertainty": "< 20%",
                    "contamination_control": "< 0.001 mg Au background"
                }
            },
            "experimental_parameters": recipe,
            "service_specifications": {
                "irradiation": irradiation_spec,
                "analysis": analysis_spec,
                "shipping": shipping_spec
            },
            "timeline": {
                "sample_preparation": start_date.strftime('%Y-%m-%d'),
                "irradiation_start": start_date.strftime('%Y-%m-%d'),
                "irradiation_complete": irradiation_complete.strftime('%Y-%m-%d'),
                "analysis_complete": analysis_complete.strftime('%Y-%m-%d'),
                "total_duration_days": (analysis_complete - start_date).days
            },
            "budget": {
                "estimated_total_cad": recipe.get('cost_cad', 60.0),
                "payment_terms": "Net 30",
                "currency": "CAD"
            },
            "risk_assessment": {
                "radiation_exposure": "Minimal, ALARA principles",
                "sample_loss": "Low, chain of custody",
                "contamination": "Medium, requires controls",
                "detection_failure": "Medium, detection limits adequate"
            }
        }
        
        return complete_spec
    
    def save_specifications(self, specs: Dict):
        """Save specifications to files."""
        
        # Ensure directory exists
        Path("experiment_specs").mkdir(exist_ok=True)
        
        # Save complete specification
        with open("experiment_specs/complete_specification.json", 'w', encoding='utf-8') as f:
            json.dump(specs, f, indent=2)
        
        # Generate vendor-specific RFQ (Request for Quote) documents
        self._generate_irradiation_rfq(specs)
        self._generate_analysis_rfq(specs)
        
        self.logger.info("Specifications saved to experiment_specs/")
    
    def _generate_irradiation_rfq(self, specs: Dict):
        """Generate RFQ document for irradiation services."""
        
        irrad_spec = specs['service_specifications']['irradiation']
        
        rfq = f"""
REQUEST FOR QUOTE - GAMMA IRRADIATION SERVICES
============================================

Project: {specs['experiment_metadata']['project_name']}
Experiment ID: {specs['experiment_metadata']['experiment_id']}
Date: {datetime.now().strftime('%Y-%m-%d')}

SAMPLE SPECIFICATIONS:
- Material: {irrad_spec['sample_specifications']['material']}
- Mass: {irrad_spec['sample_specifications']['mass_g']:.2f} g
- Form: {irrad_spec['sample_specifications']['form']}
- Container: {irrad_spec['sample_specifications']['container']}

IRRADIATION REQUIREMENTS:
- Source: {irrad_spec['irradiation_parameters']['source_type']}
- Target Dose: {irrad_spec['irradiation_parameters']['target_dose_kgy']:.1f} kGy
- Dose Rate: {irrad_spec['irradiation_parameters']['dose_rate_kgy_h']:.1f} kGy/h
- Temperature: {irrad_spec['irradiation_parameters']['temperature_c']}°C
- Atmosphere: {irrad_spec['irradiation_parameters']['atmosphere']}

DOSIMETRY:
- Dose mapping required: {irrad_spec['dosimetry_requirements']['dose_mapping']}
- Dosimeter type: {irrad_spec['dosimetry_requirements']['dosimeter_type']}
- Uncertainty target: {irrad_spec['dosimetry_requirements']['uncertainty_target_percent']}%

DELIVERABLES:
- Irradiated sample
- Dosimetry report
- Chain of custody documentation
- Activity measurement

Please provide quote including:
1. Service cost (CAD)
2. Turnaround time
3. Quality assurance procedures
4. Shipping arrangements

Contact: [Your contact information]
"""
        
        with open("experiment_specs/irradiation_rfq.txt", 'w', encoding='utf-8') as f:
            f.write(rfq)
    
    def _generate_analysis_rfq(self, specs: Dict):
        """Generate RFQ document for analytical services."""
        
        analysis_spec = specs['service_specifications']['analysis']
        
        rfq = f"""
REQUEST FOR QUOTE - PRECIOUS METALS ANALYSIS
==========================================

Project: {specs['experiment_metadata']['project_name']}
Experiment ID: {specs['experiment_metadata']['experiment_id']}
Date: {datetime.now().strftime('%Y-%m-%d')}

SAMPLE TYPE:
- Material: Post-irradiation Lead-208
- Expected mass: {specs['experimental_parameters'].get('feedstock_g', 1.0):.1f} g
- Radioactive: YES (low-level, exempted)
- Matrix: Lead metal

ANALYTICAL REQUIREMENTS:
- Method: {analysis_spec['analytical_method']}
- Target elements: {', '.join(analysis_spec['target_elements']['primary'])}
- Detection limit: {analysis_spec['analytical_requirements']['detection_limit_mg_kg']:.3f} mg/kg
- Precision: {analysis_spec['analytical_requirements']['precision_rsd_percent']}% RSD
- Accuracy: {analysis_spec['analytical_requirements']['accuracy_bias_percent']}% bias

SAMPLE PREPARATION:
- Dissolution: {analysis_spec['sample_preparation']['dissolution_method']}
- Internal standards: {', '.join(analysis_spec['sample_preparation']['internal_standards'])}

QUALITY CONTROL:
- Method blanks: {analysis_spec['quality_control']['method_blanks']}
- Duplicate analysis: {analysis_spec['quality_control']['duplicate_analysis']}
- Spike recovery: {analysis_spec['quality_control']['spike_recovery']}

SPECIAL REQUIREMENTS:
- Radioactive sample handling license required
- Waste disposal included
- Raw data and uncertainty reporting required

Please provide quote including:
1. Analysis cost (CAD)
2. Turnaround time
3. Detection limits achieved
4. Quality assurance procedures

Contact: [Your contact information]
"""
        
        with open("experiment_specs/analysis_rfq.txt", 'w', encoding='utf-8') as f:
            f.write(rfq)

def main():
    """Main specification generation routine."""
    parser = argparse.ArgumentParser(description='Generate vendor specifications')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize generator
    generator = VendorSpecGenerator(test_mode=args.test_mode)
    
    # Generate specifications
    specs = generator.generate_complete_specification()
    
    # Save to files
    generator.save_specifications(specs)
    
    # Print summary
    print("\n" + "="*60)
    print("VENDOR SPECIFICATIONS GENERATED")
    print("="*60)
    print(f"Experiment ID: {specs['experiment_metadata']['experiment_id']}")
    print(f"Feedstock: {specs['experimental_parameters'].get('feedstock_g', 1.0):.1f} g Pb-208")
    print(f"Target dose: {specs['experimental_parameters'].get('dose_kgy', 10.0):.1f} kGy")
    print(f"Predicted yield: {specs['experimental_parameters'].get('predicted_au_mg', 0.01):.4f} mg Au")
    print(f"Estimated cost: ${specs['budget']['estimated_total_cad']:.2f} CAD")
    print(f"Timeline: {specs['timeline']['total_duration_days']} days")
    print()
    print("Files generated:")
    print("- experiment_specs/complete_specification.json")
    print("- experiment_specs/irradiation_rfq.txt")
    print("- experiment_specs/analysis_rfq.txt")
    print("="*60)
    
    return 0

if __name__ == '__main__':
    exit(main())
