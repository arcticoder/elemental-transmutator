#!/usr/bin/env python3
"""
Safety and Regulatory Compliance Simulator
==========================================

Comprehensive safety analysis and regulatory compliance framework for
pilot-scale rhodium replicator operations. Calculates shielding requirements,
activation products, waste streams, and regulatory documentation.
"""

import numpy as np
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import sys
import os

@dataclass
class RadiationSource:
    """Radiation source characterization."""
    isotope: str
    activity_bq: float
    energy_mev: float
    half_life_hours: float
    emission_type: str  # 'gamma', 'neutron', 'beta', 'alpha'

@dataclass
class ShieldingMaterial:
    """Shielding material properties."""
    name: str
    density_g_cm3: float
    z_effective: float
    mass_attenuation_coeff: float  # cm²/g at 1 MeV
    neutron_removal_cross_section: float  # cm⁻¹
    cost_per_kg: float

@dataclass
class SafetyParameters:
    """Safety and regulatory parameters."""
    
    # Dose limits (mSv/year)
    worker_dose_limit: float = 20.0     # Radiation worker limit
    public_dose_limit: float = 1.0      # General public limit
    
    # Facility parameters
    beam_power_mw: float = 0.1          # 100 kW beam
    operating_hours_per_year: float = 4000  # 50% uptime
    facility_volume_m3: float = 1000    # 1000 m³ hot cell facility
    
    # Activation thresholds (Bq/kg)
    clearance_level: float = 1e4        # Below regulatory concern
    low_level_waste: float = 1e7        # LLW classification
    intermediate_waste: float = 1e10     # ILW classification
    
    # Transport limits
    type_a_activity_limit: float = 1e12  # Bq - Type A package limit
    
class SafetyAnalyzer:
    """Comprehensive safety and regulatory analysis."""
    
    def __init__(self):
        self.shielding_materials = self._initialize_shielding_materials()
        self.activation_products = {}
        self.dose_calculations = {}
        
    def _initialize_shielding_materials(self) -> Dict[str, ShieldingMaterial]:
        """Initialize database of shielding materials."""
        materials = {
            'concrete': ShieldingMaterial(
                name='Ordinary Concrete',
                density_g_cm3=2.3,
                z_effective=11.0,
                mass_attenuation_coeff=0.0636,  # cm²/g at 1 MeV
                neutron_removal_cross_section=0.089,  # cm⁻¹
                cost_per_kg=0.10
            ),
            'lead': ShieldingMaterial(
                name='Lead',
                density_g_cm3=11.34,
                z_effective=82.0,
                mass_attenuation_coeff=0.0706,
                neutron_removal_cross_section=0.120,
                cost_per_kg=2.50
            ),
            'steel': ShieldingMaterial(
                name='Steel',
                density_g_cm3=7.87,
                z_effective=26.0,
                mass_attenuation_coeff=0.0595,
                neutron_removal_cross_section=0.162,
                cost_per_kg=0.80
            ),
            'polyethylene': ShieldingMaterial(
                name='Polyethylene',
                density_g_cm3=0.92,
                z_effective=5.5,
                mass_attenuation_coeff=0.0857,
                neutron_removal_cross_section=0.116,
                cost_per_kg=1.20
            ),
            'water': ShieldingMaterial(
                name='Water',
                density_g_cm3=1.0,
                z_effective=7.4,
                mass_attenuation_coeff=0.0706,
                neutron_removal_cross_section=0.103,
                cost_per_kg=0.001
            )
        }
        return materials
    
    def calculate_prompt_radiation(self, beam_energy_mev: float, 
                                 beam_current_ma: float,
                                 target_mass_kg: float) -> Dict[str, RadiationSource]:
        """Calculate prompt radiation from spallation reactions."""
        print(f"\n☢️ PROMPT RADIATION ANALYSIS")
        print("=" * 40)
        print(f"Beam: {beam_energy_mev:.1f} MeV, {beam_current_ma:.1f} mA")
        print(f"Target: {target_mass_kg*1000:.1f} g")
          # Prompt neutron production (empirical formula)
        neutron_yield_per_proton = 0.024 * (beam_energy_mev / 100.0)**0.8  # Rudstam formula
        proton_rate = beam_current_ma * 1e-3 / 1.602e-19  # protons/s
        neutron_rate = neutron_yield_per_proton * proton_rate
        
        # Prompt gamma production
        gamma_multiplicity = 3.5  # Average gammas per spallation
        gamma_rate = neutron_rate * gamma_multiplicity
        gamma_energy_avg = 1.2  # MeV average
        
        # Secondary neutrons from (n,f) and (n,2n)
        secondary_neutron_fraction = 0.15
        secondary_neutron_rate = neutron_rate * secondary_neutron_fraction
        
        # Delayed neutron precursors
        delayed_neutron_fraction = 0.0065
        delayed_neutron_rate = neutron_rate * delayed_neutron_fraction
        
        prompt_sources = {
            'prompt_neutrons': RadiationSource(
                isotope='n-fast',
                activity_bq=neutron_rate,
                energy_mev=2.0,  # Average fast neutron energy
                half_life_hours=0,  # Instantaneous
                emission_type='neutron'
            ),
            'prompt_gammas': RadiationSource(
                isotope='gamma-prompt',
                activity_bq=gamma_rate,
                energy_mev=gamma_energy_avg,
                half_life_hours=0,
                emission_type='gamma'
            ),
            'secondary_neutrons': RadiationSource(
                isotope='n-secondary',
                activity_bq=secondary_neutron_rate,
                energy_mev=1.5,
                half_life_hours=0,
                emission_type='neutron'
            ),
            'delayed_neutrons': RadiationSource(
                isotope='n-delayed',
                activity_bq=delayed_neutron_rate,
                energy_mev=0.5,
                half_life_hours=0.02,  # ~1 minute half-life
                emission_type='neutron'
            )
        }
        
        print(f"  Prompt neutron rate: {neutron_rate:.2e} n/s")
        print(f"  Prompt gamma rate: {gamma_rate:.2e} γ/s")
        print(f"  Secondary neutron rate: {secondary_neutron_rate:.2e} n/s")
        print(f"  Delayed neutron rate: {delayed_neutron_rate:.2e} n/s")
        
        return prompt_sources
    
    def calculate_activation_products(self, beam_energy_mev: float,
                                    beam_current_ma: float,
                                    irradiation_hours: float,
                                    structural_materials: Dict[str, float]) -> Dict[str, RadiationSource]:
        """Calculate activation products in structural materials."""
        print(f"\n🔴 ACTIVATION PRODUCT ANALYSIS")
        print("=" * 40)
        print(f"Irradiation time: {irradiation_hours:.1f} hours")
        
        # Neutron flux in structural materials (simplified)
        proton_rate = beam_current_ma * 1e-3 / 1.602e-19
        neutron_production_rate = proton_rate * 0.024 * (beam_energy_mev / 100.0)**0.8
        
        # Assume 1% of neutrons reach structural materials
        structural_neutron_flux = neutron_production_rate * 0.01 / (4 * np.pi * 2.0**2)  # n/cm²/s at 2m
        
        activation_products = {}
        
        # Common activation reactions in structural materials
        activation_data = {
            'steel': {
                'Fe-54(n,p)Mn-54': {'cross_section': 0.082, 'half_life_hours': 7.4e4, 'target_fraction': 0.058},
                'Fe-56(n,p)Mn-56': {'cross_section': 0.00109, 'half_life_hours': 2.58, 'target_fraction': 0.918},
                'Ni-58(n,p)Co-58': {'cross_section': 0.113, 'half_life_hours': 1.7e4, 'target_fraction': 0.02},
                'Cr-50(n,p)V-50': {'cross_section': 0.016, 'half_life_hours': 3.4e12, 'target_fraction': 0.001}
            },
            'concrete': {
                'Al-27(n,p)Mg-27': {'cross_section': 0.0032, 'half_life_hours': 9.46/60, 'target_fraction': 0.05},
                'Si-28(n,p)Al-28': {'cross_section': 0.00025, 'half_life_hours': 2.24/60, 'target_fraction': 0.25},
                'Ca-40(n,p)K-40': {'cross_section': 1e-6, 'half_life_hours': 1.1e13, 'target_fraction': 0.20}
            }
        }
        
        for material, mass_kg in structural_materials.items():
            if material in activation_data:
                print(f"\n  {material.title()}: {mass_kg:.1f} kg")
                
                for reaction, data in activation_data[material].items():
                    # Calculate number of target nuclei
                    atomic_mass = float(reaction.split('(')[0].split('-')[1])
                    target_nuclei = (mass_kg * 1000 * 6.022e23 * data['target_fraction']) / atomic_mass
                    
                    # Activation rate
                    cross_section_cm2 = data['cross_section'] * 1e-24  # barns to cm²
                    production_rate = structural_neutron_flux * target_nuclei * cross_section_cm2
                    
                    # Activity after irradiation (Bateman equation)
                    decay_constant = 0.693 / data['half_life_hours']
                    activity_bq = production_rate * (1 - np.exp(-decay_constant * irradiation_hours))
                    
                    if activity_bq > 1e6:  # Only track significant activities
                        product_isotope = reaction.split(')')[1]
                        activation_products[f"{material}_{product_isotope}"] = RadiationSource(
                            isotope=product_isotope,
                            activity_bq=activity_bq,
                            energy_mev=1.0,  # Typical gamma energy
                            half_life_hours=data['half_life_hours'],
                            emission_type='gamma'
                        )
                        
                        print(f"    {reaction}: {activity_bq:.2e} Bq")
        
        return activation_products
    
    def calculate_shielding_requirements(self, radiation_sources: Dict[str, RadiationSource],
                                       target_dose_rate_usv_h: float = 2.5) -> Dict[str, Dict]:
        """Calculate shielding requirements for dose rate limits."""
        print(f"\n🛡️ SHIELDING REQUIREMENTS ANALYSIS")
        print("=" * 45)
        print(f"Target dose rate: {target_dose_rate_usv_h:.1f} µSv/h")
        
        shielding_designs = {}
        
        for material_name, material in self.shielding_materials.items():
            total_thickness_cm = 0
            total_mass_kg = 0
            total_cost_usd = 0
            
            print(f"\n  {material.name} shielding:")
            
            for source_name, source in radiation_sources.items():
                if source.emission_type == 'gamma':
                    # Gamma shielding calculation
                    # Dose rate (µSv/h) = Activity × Gamma constant × e^(-µt) / distance²
                    distance_m = 2.0  # Assume 2m working distance
                    gamma_constant = 1.3e-13  # µSv⋅m²/(h⋅Bq) for 1 MeV gammas
                    
                    unshielded_dose_rate = (source.activity_bq * gamma_constant * source.energy_mev) / distance_m**2
                    
                    if unshielded_dose_rate > target_dose_rate_usv_h:
                        # Required attenuation factor
                        attenuation_factor = unshielded_dose_rate / target_dose_rate_usv_h
                        
                        # Calculate thickness using Beer-Lambert law
                        mu_total = material.mass_attenuation_coeff * material.density_g_cm3  # cm⁻¹
                        required_thickness_cm = np.log(attenuation_factor) / mu_total
                        
                        total_thickness_cm = max(total_thickness_cm, required_thickness_cm)
                        
                        print(f"    {source_name}: {required_thickness_cm:.1f} cm")
                
                elif source.emission_type == 'neutron':
                    # Neutron shielding calculation
                    neutron_flux = source.activity_bq / (4 * np.pi * (200)**2)  # n/cm²/s at 2m
                    dose_rate_usv_h = neutron_flux * 3.6e-6 * source.energy_mev  # Simplified dose conversion
                    
                    if dose_rate_usv_h > target_dose_rate_usv_h:
                        # Neutron removal calculation
                        attenuation_factor = dose_rate_usv_h / target_dose_rate_usv_h
                        required_thickness_cm = np.log(attenuation_factor) / material.neutron_removal_cross_section
                        
                        total_thickness_cm = max(total_thickness_cm, required_thickness_cm)
                        
                        print(f"    {source_name}: {required_thickness_cm:.1f} cm")
            
            # Calculate mass and cost
            if total_thickness_cm > 0:
                # Assume cylindrical shield 3m diameter x 3m height
                shield_volume_m3 = np.pi * 1.5**2 * 3.0 - np.pi * 1.0**2 * 3.0  # Approximate
                shield_volume_cm3 = shield_volume_m3 * 1e6
                total_mass_kg = total_thickness_cm * shield_volume_cm3 * material.density_g_cm3 / 1000
                total_cost_usd = total_mass_kg * material.cost_per_kg
            
            shielding_designs[material_name] = {
                'thickness_cm': total_thickness_cm,
                'mass_kg': total_mass_kg,
                'cost_usd': total_cost_usd,
                'material': material.name
            }
            
            print(f"    Total thickness: {total_thickness_cm:.1f} cm")
            print(f"    Total mass: {total_mass_kg/1000:.1f} tonnes")
            print(f"    Total cost: ${total_cost_usd:,.0f}")
        
        return shielding_designs
    
    def analyze_waste_streams(self, activation_products: Dict[str, RadiationSource],
                            process_waste_kg_per_batch: float = 10.0) -> Dict:
        """Analyze radioactive waste streams and classification."""
        print(f"\n🗑️ RADIOACTIVE WASTE ANALYSIS")
        print("=" * 40)
        
        waste_analysis = {
            'solid_waste': {'mass_kg': 0, 'activity_bq': 0, 'classification': 'exempt'},
            'liquid_waste': {'volume_l': 0, 'activity_bq': 0, 'classification': 'exempt'},
            'gaseous_waste': {'volume_m3': 0, 'activity_bq': 0, 'classification': 'exempt'}
        }
        
        # Solid waste (activated structural materials)
        total_solid_activity = sum(source.activity_bq for source in activation_products.values())
        solid_waste_mass = 1000  # kg of potentially activated steel/concrete
        
        if total_solid_activity > 0:
            specific_activity_bq_kg = total_solid_activity / solid_waste_mass
            
            # Classify based on specific activity
            if specific_activity_bq_kg < 1e4:
                classification = 'exempt'
            elif specific_activity_bq_kg < 1e7:
                classification = 'low_level'
            elif specific_activity_bq_kg < 1e10:
                classification = 'intermediate_level'
            else:
                classification = 'high_level'
            
            waste_analysis['solid_waste'] = {
                'mass_kg': solid_waste_mass,
                'activity_bq': total_solid_activity,
                'specific_activity_bq_kg': specific_activity_bq_kg,
                'classification': classification
            }
        
        # Liquid waste (process chemicals, cooling water)
        liquid_waste_volume = process_waste_kg_per_batch * 5  # Assume 5L per kg processed
        liquid_contamination_fraction = 1e-6  # 1 ppm contamination
        liquid_activity = total_solid_activity * liquid_contamination_fraction
        
        if liquid_activity > 0:
            specific_activity_bq_l = liquid_activity / liquid_waste_volume
            
            if specific_activity_bq_l < 1e2:
                classification = 'exempt'
            elif specific_activity_bq_l < 1e5:
                classification = 'low_level'
            else:
                classification = 'intermediate_level'
            
            waste_analysis['liquid_waste'] = {
                'volume_l': liquid_waste_volume,
                'activity_bq': liquid_activity,
                'specific_activity_bq_l': specific_activity_bq_l,
                'classification': classification
            }
        
        # Gaseous waste (ventilation, off-gas)
        ventilation_rate_m3_h = 1000  # m³/h for hot cell
        operating_hours = 8
        gas_volume = ventilation_rate_m3_h * operating_hours
        gas_contamination_fraction = 1e-9  # Very low airborne contamination
        gas_activity = total_solid_activity * gas_contamination_fraction
        
        if gas_activity > 0:
            specific_activity_bq_m3 = gas_activity / gas_volume
            
            if specific_activity_bq_m3 < 1e1:
                classification = 'exempt'
            elif specific_activity_bq_m3 < 1e3:
                classification = 'low_level'
            else:
                classification = 'intermediate_level'
            
            waste_analysis['gaseous_waste'] = {
                'volume_m3': gas_volume,
                'activity_bq': gas_activity,
                'specific_activity_bq_m3': specific_activity_bq_m3,
                'classification': classification
            }
        
        # Print waste summary
        print(f"  Solid waste: {waste_analysis['solid_waste']['mass_kg']:.0f} kg, "
              f"{waste_analysis['solid_waste']['activity_bq']:.2e} Bq, "
              f"Class: {waste_analysis['solid_waste']['classification']}")
        
        print(f"  Liquid waste: {waste_analysis['liquid_waste']['volume_l']:.0f} L, "
              f"{waste_analysis['liquid_waste']['activity_bq']:.2e} Bq, "
              f"Class: {waste_analysis['liquid_waste']['classification']}")
        
        print(f"  Gaseous waste: {waste_analysis['gaseous_waste']['volume_m3']:.0f} m³, "
              f"{waste_analysis['gaseous_waste']['activity_bq']:.2e} Bq, "
              f"Class: {waste_analysis['gaseous_waste']['classification']}")
        
        return waste_analysis
    
    def generate_regulatory_dossier(self, shielding_designs: Dict, 
                                  waste_analysis: Dict,
                                  radiation_sources: Dict[str, RadiationSource]) -> Dict:
        """Generate regulatory compliance documentation."""
        print(f"\n📋 REGULATORY COMPLIANCE DOSSIER")
        print("=" * 45)
        
        dossier = {
            'facility_classification': self._determine_facility_classification(radiation_sources),
            'required_licenses': self._identify_required_licenses(waste_analysis),
            'safety_systems': self._specify_safety_systems(shielding_designs),
            'monitoring_requirements': self._define_monitoring_requirements(),
            'emergency_procedures': self._outline_emergency_procedures(),
            'transport_requirements': self._specify_transport_requirements(waste_analysis),
            'disposal_pathways': self._identify_disposal_pathways(waste_analysis)
        }
        
        # Print key regulatory findings
        print(f"  Facility class: {dossier['facility_classification']}")
        print(f"  Required licenses: {len(dossier['required_licenses'])} types")
        print(f"  Safety systems: {len(dossier['safety_systems'])} required")
        print(f"  Monitoring points: {len(dossier['monitoring_requirements'])} locations")
        
        return dossier
    
    def _determine_facility_classification(self, sources: Dict[str, RadiationSource]) -> str:
        """Determine regulatory facility classification."""
        max_activity = max(source.activity_bq for source in sources.values())
        
        if max_activity > 1e15:
            return "Category I - High Activity"
        elif max_activity > 1e12:
            return "Category II - Intermediate Activity"
        elif max_activity > 1e9:
            return "Category III - Low Activity"
        else:
            return "Category IV - Very Low Activity"
    
    def _identify_required_licenses(self, waste_analysis: Dict) -> List[str]:
        """Identify required regulatory licenses."""
        licenses = ['Radioactive Materials License']
        
        # Check waste classifications
        classifications = {waste['classification'] for waste in waste_analysis.values()}
        
        if 'low_level' in classifications or 'intermediate_level' in classifications:
            licenses.append('Radioactive Waste Management License')
        
        if any(waste.get('activity_bq', 0) > 1e10 for waste in waste_analysis.values()):
            licenses.append('Nuclear Facility Operating License')
        
        licenses.extend([
            'Accelerator Operation Permit',
            'Environmental Discharge Permit',
            'Occupational Health & Safety Certificate'
        ])
        
        return licenses
    
    def _specify_safety_systems(self, shielding_designs: Dict) -> List[str]:
        """Specify required safety systems."""
        systems = [
            'Radiation Area Monitoring System (RAMS)',
            'Emergency Shutdown System',
            'Ventilation and Filtration System',
            'Fire Suppression System',
            'Access Control System'
        ]
        
        # Add based on shielding requirements
        if any(design['thickness_cm'] > 50 for design in shielding_designs.values()):
            systems.append('Remote Handling Equipment')
        
        if any(design['mass_kg'] > 10000 for design in shielding_designs.values()):
            systems.append('Structural Integrity Monitoring')
        
        return systems
    
    def _define_monitoring_requirements(self) -> List[str]:
        """Define radiation monitoring requirements."""
        return [
            'Beam Line Area Monitors',
            'Target Area Gamma Detectors',
            'Neutron Detection System',
            'Stack Monitoring for Airborne Activity',
            'Workplace Air Sampling',
            'Liquid Effluent Monitoring',
            'Personal Dosimetry Program',
            'Environmental Monitoring Network'
        ]
    
    def _outline_emergency_procedures(self) -> List[str]:
        """Outline emergency response procedures."""
        return [
            'Radiation Emergency Response Plan',
            'Beam Trip/SCRAM Procedures',
            'Contamination Control Procedures',
            'Medical Emergency Procedures',
            'Fire Emergency Procedures',
            'Evacuation Procedures',
            'Notification Procedures (Regulatory)',
            'Recovery and Restoration Procedures'
        ]
    
    def _specify_transport_requirements(self, waste_analysis: Dict) -> Dict:
        """Specify radioactive material transport requirements."""
        transport_reqs = {}
        
        for waste_type, waste_data in waste_analysis.items():
            activity = waste_data.get('activity_bq', 0)
            
            if activity < 1e12:
                package_type = 'Type A'
            elif activity < 1e15:
                package_type = 'Type B'
            else:
                package_type = 'Type C'
            
            transport_reqs[waste_type] = {
                'package_type': package_type,
                'special_requirements': self._get_transport_special_reqs(package_type),
                'estimated_cost_per_shipment': self._estimate_transport_cost(package_type)
            }
        
        return transport_reqs
    
    def _get_transport_special_reqs(self, package_type: str) -> List[str]:
        """Get special transport requirements by package type."""
        base_reqs = ['DOT Hazmat Training', 'Radiation Protection Program']
        
        if package_type == 'Type B':
            base_reqs.extend(['NRC Certificate of Compliance', 'Special Nuclear Material Accounting'])
        elif package_type == 'Type C':
            base_reqs.extend(['Aircraft Transport Approval', 'International Atomic Energy Agency Approval'])
        
        return base_reqs
    
    def _estimate_transport_cost(self, package_type: str) -> float:
        """Estimate transport cost per shipment (USD)."""
        cost_map = {
            'Type A': 5000,
            'Type B': 25000,
            'Type C': 100000
        }
        return cost_map.get(package_type, 5000)
    
    def _identify_disposal_pathways(self, waste_analysis: Dict) -> Dict:
        """Identify waste disposal pathways and costs."""
        disposal_paths = {}
        
        for waste_type, waste_data in waste_analysis.items():
            classification = waste_data.get('classification', 'exempt')
            
            if classification == 'exempt':
                disposal_paths[waste_type] = {
                    'pathway': 'Conventional Waste Disposal',
                    'cost_per_kg': 0.50,
                    'facility': 'Municipal/Industrial Landfill'
                }
            elif classification == 'low_level':
                disposal_paths[waste_type] = {
                    'pathway': 'Low-Level Radioactive Waste Disposal',
                    'cost_per_kg': 500.0,
                    'facility': 'Licensed LLW Disposal Facility'
                }
            elif classification == 'intermediate_level':
                disposal_paths[waste_type] = {
                    'pathway': 'Intermediate-Level Waste Storage',
                    'cost_per_kg': 2000.0,
                    'facility': 'Licensed ILW Storage Facility'
                }
            else:
                disposal_paths[waste_type] = {
                    'pathway': 'High-Level Waste Repository',
                    'cost_per_kg': 10000.0,
                    'facility': 'Deep Geological Repository'
                }
        
        return disposal_paths

def main():
    """Main safety and regulatory analysis."""
    print("☢️ CHEAP FEEDSTOCK RHODIUM REPLICATOR")
    print("   SAFETY & REGULATORY COMPLIANCE ANALYSIS")
    print("=" * 55)
    
    # Initialize analyzer
    analyzer = SafetyAnalyzer()
    
    # Simulate pilot plant parameters
    beam_energy = 120.0  # MeV
    beam_current = 1.0   # mA
    target_mass = 0.010  # kg (10g Fe-56 target)
    irradiation_time = 8.0  # hours per day
    
    # Structural materials in facility
    structural_materials = {
        'steel': 5000,     # kg structural steel
        'concrete': 50000  # kg concrete shielding
    }
    
    # Calculate radiation sources
    print("🔥 Calculating radiation sources...")
    prompt_sources = analyzer.calculate_prompt_radiation(beam_energy, beam_current, target_mass)
    activation_products = analyzer.calculate_activation_products(
        beam_energy, beam_current, irradiation_time, structural_materials
    )
    
    # Combine all sources
    all_sources = {**prompt_sources, **activation_products}
    
    # Calculate shielding requirements
    shielding_designs = analyzer.calculate_shielding_requirements(all_sources)
    
    # Analyze waste streams
    waste_analysis = analyzer.analyze_waste_streams(activation_products)
    
    # Generate regulatory dossier
    regulatory_dossier = analyzer.generate_regulatory_dossier(
        shielding_designs, waste_analysis, all_sources
    )
    
    # Compile comprehensive safety report
    safety_report = {
        'analysis_date': datetime.now().isoformat(),
        'facility_parameters': {
            'beam_energy_mev': beam_energy,
            'beam_current_ma': beam_current,
            'target_mass_kg': target_mass,
            'irradiation_hours': irradiation_time
        },
        'radiation_sources': {name: {
            'isotope': source.isotope,
            'activity_bq': source.activity_bq,
            'energy_mev': source.energy_mev,
            'emission_type': source.emission_type
        } for name, source in all_sources.items()},
        'shielding_designs': shielding_designs,
        'waste_analysis': waste_analysis,
        'regulatory_dossier': regulatory_dossier,
        'cost_summary': analyzer._calculate_total_costs(shielding_designs, waste_analysis)
    }
    
    # Save comprehensive report
    with open('safety_regulatory_analysis.json', 'w') as f:
        json.dump(safety_report, f, indent=2, default=str)
    
    # Print summary conclusions
    print(f"\n🎯 SAFETY & REGULATORY SUMMARY")
    print("=" * 40)
    
    # Optimal shielding recommendation
    best_shielding = min(shielding_designs.items(), key=lambda x: x[1]['cost_usd'])
    print(f"Recommended shielding: {best_shielding[1]['material']}")
    print(f"  Thickness: {best_shielding[1]['thickness_cm']:.1f} cm")
    print(f"  Cost: ${best_shielding[1]['cost_usd']:,.0f}")
    
    # Waste classification summary
    waste_classes = {waste['classification'] for waste in waste_analysis.values()}
    print(f"Waste classifications: {', '.join(waste_classes)}")
    
    # Regulatory complexity
    n_licenses = len(regulatory_dossier['required_licenses'])
    if n_licenses > 5:
        complexity = "HIGH"
    elif n_licenses > 3:
        complexity = "MODERATE"
    else:
        complexity = "LOW"
    
    print(f"Regulatory complexity: {complexity} ({n_licenses} licenses required)")
    
    # Total compliance cost estimate
    total_cost = safety_report['cost_summary']['total_usd']
    print(f"Total safety/compliance cost: ${total_cost:,.0f}")
    
    if total_cost < 1e6:
        print("✅ FEASIBLE: Safety costs manageable for pilot scale")
    elif total_cost < 5e6:
        print("⚠️ EXPENSIVE: Significant safety investment required")
    else:
        print("❌ PROHIBITIVE: Safety costs may prevent pilot development")
    
    print(f"\n💾 Detailed safety analysis saved to safety_regulatory_analysis.json")
    print(f"🚀 Ready for pilot plant engineering design!")
    
    return safety_report

if __name__ == "__main__":
    # Add cost calculation method to analyzer
    def _calculate_total_costs(self, shielding_designs: Dict, waste_analysis: Dict) -> Dict:
        """Calculate total safety and compliance costs."""
        
        # Shielding costs
        min_shielding_cost = min(design['cost_usd'] for design in shielding_designs.values())
        
        # Waste disposal costs (annual estimate)
        disposal_cost_annual = 0
        disposal_paths = self._identify_disposal_pathways(waste_analysis)
        
        for waste_type, waste_data in waste_analysis.items():
            if waste_type in disposal_paths:
                mass_kg = waste_data.get('mass_kg', waste_data.get('volume_l', waste_data.get('volume_m3', 1)))
                cost_per_kg = disposal_paths[waste_type]['cost_per_kg']
                disposal_cost_annual += mass_kg * cost_per_kg * 50  # 50 batches/year
        
        # Regulatory compliance costs
        regulatory_cost = len(self.regulatory_dossier.get('required_licenses', [])) * 50000  # $50k per license
        
        # Monitoring and safety systems
        monitoring_cost = len(self.regulatory_dossier.get('monitoring_requirements', [])) * 25000  # $25k per system
        
        total_cost = min_shielding_cost + disposal_cost_annual + regulatory_cost + monitoring_cost
        
        return {
            'shielding_usd': min_shielding_cost,
            'waste_disposal_annual_usd': disposal_cost_annual,
            'regulatory_compliance_usd': regulatory_cost,
            'monitoring_systems_usd': monitoring_cost,
            'total_usd': total_cost
        }
    
    # Add method to SafetyAnalyzer class
    SafetyAnalyzer._calculate_total_costs = _calculate_total_costs
    
    main()
