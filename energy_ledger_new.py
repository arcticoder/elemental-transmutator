#!/usr/bin/env python3
"""
Enhanced Energy Ledger: Comprehensive Multi-Pathway Energy Accounting
=====================================================================

This module provides detailed energy accounting for the LV energy converter,
tracking all energy flows across multiple pathways to verify net positive 
energy extraction and ensure conservation laws are respected.

Enhanced Features:
1. Complete multi-pathway energy flow tracking
2. Higher-dimension LV operator accounting
3. Dynamic vacuum extraction monitoring
4. Macroscopic negative energy tracking
5. Hidden sector portal coupling effects
6. LQG coherence and graviton entanglement
7. Pathway synergy and cross-coupling analysis
8. Vacuum stability metrics
9. Real-time conservation verification
10. Advanced visualization and reporting

Author: LV Energy Converter Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import json
import logging
import time
from datetime import datetime
from collections import defaultdict

class EnergyType(Enum):
    """Energy flow types for comprehensive multi-pathway tracking."""
    # Core energy types
    INPUT_DRIVE = "input_drive"                    # Energy to drive actuators
    INPUT_LV_FIELD = "input_lv_field"             # Energy to create LV fields
    NEGATIVE_RESERVOIR = "negative_reservoir"      # Negative energy created
    POSITIVE_EXTRACTION = "positive_extraction"   # Positive energy extracted
    PORTAL_TRANSFER = "portal_transfer"            # Energy through portals
    COHERENCE_MAINTENANCE = "coherence_maintenance" # Energy for coherence
    LOSSES_DISSIPATION = "losses_dissipation"     # Thermal/resistive losses
    LOSSES_LEAKAGE = "losses_leakage"             # Radiation/field leakage
    LOSSES_DECOHERENCE = "losses_decoherence"     # Quantum decoherence losses
    STORAGE_TEMPORARY = "storage_temporary"        # Temporary energy storage
    OUTPUT_USEFUL = "output_useful"                # Net useful energy output
    FEEDBACK_RECYCLE = "feedback_recycle"          # Energy recycled to inputs
    
    # Multi-pathway energy types
    LV_OPERATOR_HIGHER_DIM = "lv_operator_higher_dim"  # Higher-dimension LV operator effects
    DYNAMIC_VACUUM_CASIMIR = "dynamic_vacuum_casimir"  # Dynamical Casimir effect extraction
    NEGATIVE_ENERGY_CAVITY = "negative_energy_cavity"  # Macroscopic negative energy regions
    AXION_PORTAL_COUPLING = "axion_portal_coupling"    # Axion portal energy transfer
    DARK_PHOTON_PORTAL = "dark_photon_portal"          # Dark photon portal coupling
    GRAVITON_ENTANGLEMENT = "graviton_entanglement"    # Graviton-mediated energy transfer
    LQG_SPIN_NETWORK = "lqg_spin_network"              # LQG spin network coherence energy
    PATHWAY_SYNERGY = "pathway_synergy"                # Multi-pathway synergistic effects
    VACUUM_INSTABILITY = "vacuum_instability"          # Vacuum instability driven extraction
    METAMATERIAL_RESONANCE = "metamaterial_resonance"  # Metamaterial enhancement effects

@dataclass
class EnergyTransaction:
    """Single energy transaction record with enhanced metadata."""
    timestamp: float                    # Simulation time
    energy_type: EnergyType            # Type of energy flow
    amount: float                      # Energy amount (J)
    location: str                      # Where in the system
    pathway: str                       # Which pathway (casimir, portal, etc.)
    details: Dict = field(default_factory=dict)  # Additional metadata
    operator_coeffs: Dict = field(default_factory=dict)  # LV operator coefficients
    coupling_strengths: Dict = field(default_factory=dict)  # Portal/coupling parameters

@dataclass
class PathwayInteraction:
    """Record of interactions between different energy pathways."""
    timestamp: float
    pathway_1: str
    pathway_2: str
    interaction_type: str  # "enhancement", "interference", "resonance", etc.
    coupling_strength: float
    energy_transfer: float
    stability_impact: float

class EnergyLedger:
    """
    Comprehensive multi-pathway energy accounting system for LV energy converter.
    
    Tracks all energy flows across multiple pathways with joule-level precision
    to verify net positive energy extraction and thermodynamic consistency.
    """
    
    def __init__(self, system_id: str = "LV_Energy_System"):
        """Initialize the comprehensive energy ledger."""
        self.system_id = system_id
        self.transactions: List[EnergyTransaction] = []
        self.pathway_interactions: List[PathwayInteraction] = []
        self.start_time = time.time()
        self.simulation_time = 0.0
        
        # Running totals by category
        self.totals = {energy_type: 0.0 for energy_type in EnergyType}
        self.pathway_totals = defaultdict(float)
        
        # Conservation tracking
        self.total_input = 0.0
        self.total_output = 0.0
        self.total_stored = 0.0
        self.total_losses = 0.0
        
        # Enhanced multi-pathway tracking
        self.lv_operator_effects = defaultdict(list)
        self.vacuum_stability_metrics = defaultdict(float)
        self.portal_coupling_strengths = defaultdict(float)
        self.coherence_tracking = defaultdict(float)
        self.synergy_effects = defaultdict(float)
        
        # LV operator tracking (higher-dimension SME)
        self.lv_operator_coefficients = {}
        self.operator_energy_contributions = defaultdict(float)
        
        # Performance metrics
        self.efficiency_history = []
        self.net_gain_history = []
        self.peak_power_pathways = {}
        self.efficiency_by_pathway = {}
        self.stability_windows = {}
        
        # Pathway-specific metrics
        self.pathway_metrics = {
            'casimir': {'power': [], 'efficiency': [], 'stability': []},
            'negative_cavity': {'power': [], 'efficiency': [], 'stability': []},
            'axion_portal': {'power': [], 'efficiency': [], 'stability': []},
            'dark_photon': {'power': [], 'efficiency': [], 'stability': []},
            'graviton': {'power': [], 'efficiency': [], 'stability': []},
            'lqg': {'power': [], 'efficiency': [], 'stability': []},
            'vacuum_instability': {'power': [], 'efficiency': [], 'stability': []},
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized comprehensive energy ledger for {system_id}")
        
        # Create summary tables for different energy categories
        self._initialize_tracking_tables()
    
    def _initialize_tracking_tables(self):
        """Initialize tracking tables for organized data collection."""
        self.operator_tracking = {
            'dimension_4': defaultdict(float),
            'dimension_5': defaultdict(float),
            'dimension_6': defaultdict(float),
            'dimension_higher': defaultdict(float),
        }
        
        self.portal_tracking = {
            'axion': {'coupling': 0.0, 'energy_flow': [], 'resonances': []},
            'dark_photon': {'mixing_angle': 0.0, 'energy_flow': [], 'resonances': []},
            'graviton': {'coherence': 0.0, 'entanglement': [], 'energy_flow': []},
        }
        
        self.vacuum_tracking = {
            'casimir_pressure': [],
            'energy_density': [],
            'instability_parameter': [],
            'stability_window': [],
        }
    
    def log_transaction(self, 
                       energy_type: EnergyType, 
                       amount: float, 
                       location: str = "", 
                       pathway: str = "", 
                       details: Dict = None,
                       operator_coeffs: Dict = None,
                       coupling_strengths: Dict = None) -> None:
        """
        Log a single energy transaction with enhanced metadata.
        
        Parameters:
        -----------
        energy_type : EnergyType
            Type of energy flow
        amount : float
            Energy amount in Joules (positive for inputs/gains, negative for losses)
        location : str
            Location in the system
        pathway : str
            Which pathway generated this transaction
        details : Dict
            Additional metadata
        operator_coeffs : Dict
            LV operator coefficients if applicable
        coupling_strengths : Dict
            Portal/coupling parameters if applicable
        """
        transaction = EnergyTransaction(
            timestamp=self.simulation_time,
            energy_type=energy_type,
            amount=amount,
            location=location,
            pathway=pathway,
            details=details or {},
            operator_coeffs=operator_coeffs or {},
            coupling_strengths=coupling_strengths or {}
        )
        
        self.transactions.append(transaction)
        self.totals[energy_type] += amount
        self.pathway_totals[pathway] += amount
        
        # Update conservation totals
        if energy_type in [EnergyType.INPUT_DRIVE, EnergyType.INPUT_LV_FIELD]:
            self.total_input += amount
        elif energy_type in [EnergyType.OUTPUT_USEFUL]:
            self.total_output += amount
        elif energy_type in [EnergyType.STORAGE_TEMPORARY]:
            self.total_stored += amount
        elif energy_type in [EnergyType.LOSSES_DISSIPATION, EnergyType.LOSSES_LEAKAGE, 
                           EnergyType.LOSSES_DECOHERENCE]:
            self.total_losses += abs(amount)
        
        # Update pathway-specific tracking
        self._update_pathway_metrics(pathway, energy_type, amount)
    
    def log_pathway_interaction(self, 
                              pathway_1: str, 
                              pathway_2: str,
                              interaction_type: str,
                              coupling_strength: float,
                              energy_transfer: float,
                              stability_impact: float = 0.0) -> None:
        """Log interaction between energy pathways."""
        interaction = PathwayInteraction(
            timestamp=self.simulation_time,
            pathway_1=pathway_1,
            pathway_2=pathway_2,
            interaction_type=interaction_type,
            coupling_strength=coupling_strength,
            energy_transfer=energy_transfer,
            stability_impact=stability_impact
        )
        
        self.pathway_interactions.append(interaction)
        
        # Update synergy tracking
        synergy_key = f"{pathway_1}_{pathway_2}"
        self.synergy_effects[synergy_key] += energy_transfer
    
    def log_lv_operator_effect(self, 
                             operator_dimension: int,
                             operator_type: str,
                             coefficient: float,
                             energy_contribution: float) -> None:
        """Log effects from higher-dimension LV operators."""
        self.lv_operator_effects[operator_type].append({
            'timestamp': self.simulation_time,
            'dimension': operator_dimension,
            'coefficient': coefficient,
            'energy': energy_contribution
        })
        
        self.operator_energy_contributions[operator_type] += energy_contribution
        
        # Update operator tracking by dimension
        dim_key = f"dimension_{operator_dimension}" if operator_dimension <= 6 else "dimension_higher"
        self.operator_tracking[dim_key][operator_type] += energy_contribution
    
    def log_vacuum_stability(self, 
                           stability_parameter: float,
                           energy_density: float,
                           casimir_pressure: float = 0.0,
                           instability_growth: float = 0.0) -> None:
        """Log vacuum stability metrics."""
        self.vacuum_stability_metrics.update({
            'stability_parameter': stability_parameter,
            'energy_density': energy_density,
            'casimir_pressure': casimir_pressure,
            'instability_growth': instability_growth,
            'timestamp': self.simulation_time
        })
        
        # Update vacuum tracking
        self.vacuum_tracking['casimir_pressure'].append(casimir_pressure)
        self.vacuum_tracking['energy_density'].append(energy_density)
        self.vacuum_tracking['instability_parameter'].append(instability_growth)
        self.vacuum_tracking['stability_window'].append(stability_parameter)
    
    def _update_pathway_metrics(self, pathway: str, energy_type: EnergyType, amount: float):
        """Update pathway-specific performance metrics."""
        if pathway in self.pathway_metrics:
            if energy_type == EnergyType.OUTPUT_USEFUL:
                self.pathway_metrics[pathway]['power'].append(amount)
            
            # Calculate running efficiency for this pathway
            pathway_input = sum(t.amount for t in self.transactions 
                              if t.pathway == pathway and 
                              t.energy_type in [EnergyType.INPUT_DRIVE, EnergyType.INPUT_LV_FIELD])
            pathway_output = sum(t.amount for t in self.transactions 
                               if t.pathway == pathway and 
                               t.energy_type == EnergyType.OUTPUT_USEFUL)
            
            if pathway_input > 0:
                efficiency = pathway_output / pathway_input
                self.pathway_metrics[pathway]['efficiency'].append(efficiency)
    
    def advance_time(self, dt: float) -> None:
        """Advance simulation time."""
        self.simulation_time += dt
    
    def calculate_net_energy_gain(self) -> float:
        """
        Calculate net energy gain for current cycle.
        
        Returns:
        --------
        float
            Net energy gain (J) - positive means energy extraction
        """
        total_useful_output = self.totals[EnergyType.OUTPUT_USEFUL]
        total_input_energy = (self.totals[EnergyType.INPUT_DRIVE] + 
                             self.totals[EnergyType.INPUT_LV_FIELD])
        
        return total_useful_output - total_input_energy
    
    def calculate_conversion_efficiency(self) -> float:
        """
        Calculate overall conversion efficiency.
        
        Returns:
        --------
        float
            Efficiency (0-1) = useful_output / total_input
        """
        if self.total_input == 0:
            return 0.0
        
        return self.totals[EnergyType.OUTPUT_USEFUL] / self.total_input
    
    def calculate_pathway_efficiency(self, pathway: str) -> float:
        """Calculate efficiency for specific pathway."""
        pathway_input = sum(t.amount for t in self.transactions 
                          if t.pathway == pathway and 
                          t.energy_type in [EnergyType.INPUT_DRIVE, EnergyType.INPUT_LV_FIELD])
        pathway_output = sum(t.amount for t in self.transactions 
                           if t.pathway == pathway and 
                           t.energy_type == EnergyType.OUTPUT_USEFUL)
        
        if pathway_input == 0:
            return 0.0
        
        return pathway_output / pathway_input
    
    def verify_conservation(self, tolerance: float = 1e-12) -> Tuple[bool, float]:
        """
        Verify energy conservation within tolerance.
        
        Parameters:
        -----------
        tolerance : float
            Maximum allowed conservation violation (J)
            
        Returns:
        --------
        Tuple[bool, float]
            (conservation_satisfied, violation_amount)
        """
        # Calculate total energy in system
        total_in = self.total_input
        total_out = self.total_output + self.total_losses + self.total_stored
        
        violation = abs(total_in - total_out)
        return violation <= tolerance, violation
    
    def get_pathway_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each pathway."""
        summary = {}
        
        for pathway in set(t.pathway for t in self.transactions if t.pathway):
            pathway_transactions = [t for t in self.transactions if t.pathway == pathway]
            
            total_energy = sum(t.amount for t in pathway_transactions)
            input_energy = sum(t.amount for t in pathway_transactions 
                             if t.energy_type in [EnergyType.INPUT_DRIVE, EnergyType.INPUT_LV_FIELD])
            output_energy = sum(t.amount for t in pathway_transactions 
                              if t.energy_type == EnergyType.OUTPUT_USEFUL)
            
            efficiency = output_energy / input_energy if input_energy > 0 else 0.0
            
            summary[pathway] = {
                'total_energy': total_energy,
                'input_energy': input_energy,
                'output_energy': output_energy,
                'efficiency': efficiency,
                'transaction_count': len(pathway_transactions)
            }
        
        return summary
    
    def get_synergy_analysis(self) -> Dict[str, float]:
        """Analyze synergy effects between pathways."""
        synergy_analysis = {}
        
        for synergy_key, total_transfer in self.synergy_effects.items():
            pathway_1, pathway_2 = synergy_key.split('_', 1)
            
            # Calculate synergy strength
            individual_1 = self.pathway_totals.get(pathway_1, 0.0)
            individual_2 = self.pathway_totals.get(pathway_2, 0.0)
            combined_expected = individual_1 + individual_2
            
            if combined_expected > 0:
                synergy_factor = total_transfer / combined_expected
                synergy_analysis[synergy_key] = synergy_factor
        
        return synergy_analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive energy accounting report."""
        net_gain = self.calculate_net_energy_gain()
        efficiency = self.calculate_conversion_efficiency()
        conservation_ok, violation = self.verify_conservation()
        pathway_summary = self.get_pathway_summary()
        synergy_analysis = self.get_synergy_analysis()
        
        report = {
            'system_id': self.system_id,
            'simulation_time': self.simulation_time,
            'timestamp': datetime.now().isoformat(),
            
            'energy_balance': {
                'total_input': self.total_input,
                'total_output': self.total_output,
                'total_losses': self.total_losses,
                'total_stored': self.total_stored,
                'net_gain': net_gain,
                'efficiency': efficiency
            },
            
            'conservation': {
                'satisfied': conservation_ok,
                'violation': violation
            },
            
            'pathway_analysis': pathway_summary,
            'synergy_effects': synergy_analysis,
            
            'lv_operators': {
                'total_contributions': dict(self.operator_energy_contributions),
                'by_dimension': dict(self.operator_tracking)
            },
            
            'vacuum_stability': dict(self.vacuum_stability_metrics),
            
            'portal_couplings': dict(self.portal_coupling_strengths),
            
            'transaction_count': len(self.transactions),
            'interaction_count': len(self.pathway_interactions)
        }
        
        return report
    
    def visualize_energy_flows(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of energy flows."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Comprehensive Energy Flow Analysis - {self.system_id}', fontsize=16)
        
        # 1. Energy balance pie chart
        ax1 = axes[0, 0]
        energy_categories = ['Input', 'Output', 'Losses', 'Stored']
        energy_values = [self.total_input, self.total_output, self.total_losses, self.total_stored]
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        
        ax1.pie(energy_values, labels=energy_categories, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Energy Balance Distribution')
        
        # 2. Pathway comparison
        ax2 = axes[0, 1]
        pathway_summary = self.get_pathway_summary()
        pathways = list(pathway_summary.keys())
        efficiencies = [pathway_summary[p]['efficiency'] for p in pathways]
        
        bars = ax2.bar(pathways, efficiencies, color='skyblue')
        ax2.set_title('Pathway Efficiencies')
        ax2.set_ylabel('Efficiency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, eff in zip(bars, efficiencies):
            height = bar.get_height()
            ax2.annotate(f'{eff:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # 3. Energy flow over time
        ax3 = axes[1, 0]
        times = [t.timestamp for t in self.transactions]
        cumulative_net = []
        running_net = 0.0
        
        for t in self.transactions:
            if t.energy_type == EnergyType.OUTPUT_USEFUL:
                running_net += t.amount
            elif t.energy_type in [EnergyType.INPUT_DRIVE, EnergyType.INPUT_LV_FIELD]:
                running_net -= t.amount
            cumulative_net.append(running_net)
        
        ax3.plot(times, cumulative_net, 'g-', linewidth=2)
        ax3.set_title('Cumulative Net Energy Gain')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Net Energy (J)')
        ax3.grid(True, alpha=0.3)
        
        # 4. LV operator contributions
        ax4 = axes[1, 1]
        if self.operator_energy_contributions:
            operators = list(self.operator_energy_contributions.keys())
            contributions = list(self.operator_energy_contributions.values())
            
            ax4.barh(operators, contributions, color='lightcoral')
            ax4.set_title('LV Operator Energy Contributions')
            ax4.set_xlabel('Energy Contribution (J)')
        else:
            ax4.text(0.5, 0.5, 'No LV operator data', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes)
            ax4.set_title('LV Operator Energy Contributions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Energy flow visualization saved to {save_path}")
        
        plt.show()
    
    def export_data(self, filepath: str) -> None:
        """Export ledger data to JSON file."""
        export_data = {
            'metadata': {
                'system_id': self.system_id,
                'simulation_time': self.simulation_time,
                'export_timestamp': datetime.now().isoformat()
            },
            'transactions': [
                {
                    'timestamp': t.timestamp,
                    'energy_type': t.energy_type.value,
                    'amount': t.amount,
                    'location': t.location,
                    'pathway': t.pathway,
                    'details': t.details,
                    'operator_coeffs': t.operator_coeffs,
                    'coupling_strengths': t.coupling_strengths
                }
                for t in self.transactions
            ],
            'pathway_interactions': [
                {
                    'timestamp': i.timestamp,
                    'pathway_1': i.pathway_1,
                    'pathway_2': i.pathway_2,
                    'interaction_type': i.interaction_type,
                    'coupling_strength': i.coupling_strength,
                    'energy_transfer': i.energy_transfer,
                    'stability_impact': i.stability_impact
                }
                for i in self.pathway_interactions
            ],
            'summary': self.generate_comprehensive_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Ledger data exported to {filepath}")

def demo_enhanced_energy_ledger():
    """Demonstrate enhanced energy ledger capabilities."""
    print("=" * 80)
    print("Enhanced Multi-Pathway Energy Ledger Demonstration")
    print("=" * 80)
    
    # Initialize ledger
    ledger = EnergyLedger("Multi_Pathway_LV_System")
    
    # Simulate multi-pathway operation
    pathways = ['casimir', 'negative_cavity', 'axion_portal', 'dark_photon', 'graviton']
    
    for i in range(10):
        ledger.advance_time(0.1)
        
        for pathway in pathways:
            # Input energy
            input_energy = np.random.uniform(10, 50)
            ledger.log_transaction(
                EnergyType.INPUT_DRIVE, input_energy, 
                location=f"{pathway}_drive", pathway=pathway,
                details={'frequency': np.random.uniform(1e9, 1e12)}
            )
            
            # Output energy (with different efficiencies)
            efficiency = 0.8 + 0.4 * np.random.random()  # 80-120% efficiency
            output_energy = input_energy * efficiency
            ledger.log_transaction(
                EnergyType.OUTPUT_USEFUL, output_energy,
                location=f"{pathway}_output", pathway=pathway
            )
            
            # Losses
            loss_energy = input_energy * np.random.uniform(0.05, 0.15)
            ledger.log_transaction(
                EnergyType.LOSSES_DISSIPATION, -loss_energy,
                location=f"{pathway}_dissipation", pathway=pathway
            )
        
        # Log pathway interactions
        if i > 2:
            ledger.log_pathway_interaction(
                'casimir', 'negative_cavity', 'resonance',
                coupling_strength=0.1, energy_transfer=5.0
            )
            
            ledger.log_pathway_interaction(
                'axion_portal', 'dark_photon', 'interference',
                coupling_strength=0.05, energy_transfer=-2.0
            )
        
        # Log LV operator effects
        if i % 3 == 0:
            ledger.log_lv_operator_effect(
                operator_dimension=5,
                operator_type='CPT_violation',
                coefficient=1e-18,
                energy_contribution=np.random.uniform(1, 10)
            )
    
    # Generate and display report
    report = ledger.generate_comprehensive_report()
    
    print(f"\nSystem: {report['system_id']}")
    print(f"Simulation Time: {report['simulation_time']:.1f} s")
    print(f"Net Energy Gain: {report['energy_balance']['net_gain']:.2f} J")
    print(f"Overall Efficiency: {report['energy_balance']['efficiency']:.1%}")
    print(f"Conservation Satisfied: {report['conservation']['satisfied']}")
    
    print("\nPathway Analysis:")
    for pathway, metrics in report['pathway_analysis'].items():
        print(f"  {pathway}: {metrics['efficiency']:.1%} efficiency, "
              f"{metrics['output_energy']:.1f} J output")
    
    print("\nSynergy Effects:")
    for synergy, factor in report['synergy_effects'].items():
        print(f"  {synergy}: {factor:.3f} synergy factor")
    
    print("\nLV Operator Contributions:")
    for operator, contribution in report['lv_operators']['total_contributions'].items():
        print(f"  {operator}: {contribution:.2f} J")
    
    # Create visualization
    ledger.visualize_energy_flows("enhanced_energy_flow_demo.png")
    
    print(f"\nTotal Transactions: {report['transaction_count']}")
    print(f"Pathway Interactions: {report['interaction_count']}")
    print("\nDemonstration complete!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstration
    demo_enhanced_energy_ledger()
