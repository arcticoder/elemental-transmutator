#!/usr/bin/env python3
"""
Outsourcing Cost Analyzer
=========================

Analyzes the cost structure and economic viability of outsourced 
micro-experiments for photonuclear gold production.
"""

import argparse
import json
import numpy as np
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VendorQuote:
    """Vendor quote for irradiation or analysis services."""
    vendor_name: str
    service_type: str  # 'irradiation' or 'analysis'
    base_cost_cad: float
    per_unit_cost_cad: float
    unit_description: str
    turnaround_days: int
    minimum_order: float
    location: str

@dataclass
class ExperimentCost:
    """Cost breakdown for a single micro-experiment."""
    irradiation_cost: float
    analysis_cost: float
    shipping_cost: float
    preparation_cost: float
    total_cost: float
    cost_per_mg_target: float

class OutsourcingAnalyzer:
    """Analyzer for outsourced micro-experiment costs and viability."""
    
    def __init__(self):
        """Initialize the outsourcing analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Current gold price (CAD per gram)
        self.gold_price_cad_per_g = 85.0  # ~$2700 USD/oz at 1.35 CAD/USD
        
        # Vendor database (realistic Canadian vendors)
        self.vendors = self._load_vendor_database()
        
        self.logger.info("Outsourcing cost analyzer initialized")
    
    def _load_vendor_database(self) -> List[VendorQuote]:
        """Load database of Canadian vendors for irradiation and analysis."""
        
        vendors = [
            # Irradiation services
            VendorQuote(
                vendor_name="Nordion/Sotera Health",
                service_type="irradiation",
                base_cost_cad=25.0,
                per_unit_cost_cad=0.08,  # CAD per cc per kGy
                unit_description="cc·kGy",
                turnaround_days=5,
                minimum_order=1.0,
                location="Ottawa, ON"
            ),
            VendorQuote(
                vendor_name="McMaster Nuclear Reactor",
                service_type="irradiation",
                base_cost_cad=50.0,
                per_unit_cost_cad=0.15,
                unit_description="cc·kGy",
                turnaround_days=14,
                minimum_order=0.5,
                location="Hamilton, ON"
            ),
            
            # Analysis services
            VendorQuote(
                vendor_name="ALS Minerals",
                service_type="analysis",
                base_cost_cad=15.0,
                per_unit_cost_cad=25.0,  # Per sample
                unit_description="sample",
                turnaround_days=7,
                minimum_order=1.0,
                location="Vancouver, BC"
            ),
            VendorQuote(
                vendor_name="AGAT Laboratories",
                service_type="analysis", 
                base_cost_cad=20.0,
                per_unit_cost_cad=30.0,
                unit_description="sample",
                turnaround_days=5,
                minimum_order=1.0,
                location="Multiple locations"
            ),
            VendorQuote(
                vendor_name="SGS Canada",
                service_type="analysis",
                base_cost_cad=25.0,
                per_unit_cost_cad=35.0,
                unit_description="sample",
                turnaround_days=10,
                minimum_order=1.0,
                location="Multiple locations"
            )
        ]
        
        return vendors
    
    def calculate_experiment_cost(self, feedstock_g: float, dose_kgy: float,
                                irradiation_vendor: str = "Nordion/Sotera Health",
                                analysis_vendor: str = "ALS Minerals") -> ExperimentCost:
        """Calculate total cost for a single micro-experiment."""
        
        # Find vendor quotes
        irrad_vendor = next((v for v in self.vendors 
                           if v.vendor_name == irradiation_vendor and v.service_type == "irradiation"), None)
        analysis_vendor_obj = next((v for v in self.vendors 
                                  if v.vendor_name == analysis_vendor and v.service_type == "analysis"), None)
        
        if not irrad_vendor or not analysis_vendor_obj:
            raise ValueError("Vendor not found in database")
        
        # Calculate irradiation cost
        # Pb density ≈ 11.3 g/cm³
        volume_cc = feedstock_g / 11.3
        irradiation_units = volume_cc * dose_kgy
        irradiation_cost = irrad_vendor.base_cost_cad + (irradiation_units * irrad_vendor.per_unit_cost_cad)
        
        # Analysis cost (ICP-MS for gold content)
        analysis_cost = analysis_vendor_obj.base_cost_cad + analysis_vendor_obj.per_unit_cost_cad
        
        # Shipping and handling
        shipping_cost = 15.0  # CAD, typical Canada Post
        
        # Sample preparation (pellet forming, sealing)
        preparation_cost = 10.0  # CAD
        
        total_cost = irradiation_cost + analysis_cost + shipping_cost + preparation_cost
        cost_per_mg_target = total_cost / (feedstock_g * 1000)  # CAD per mg of feedstock
        
        return ExperimentCost(
            irradiation_cost=irradiation_cost,
            analysis_cost=analysis_cost,
            shipping_cost=shipping_cost,
            preparation_cost=preparation_cost,
            total_cost=total_cost,
            cost_per_mg_target=cost_per_mg_target
        )
    
    def is_viable_for_outsourcing(self, predicted_yield_mg: float, cost_per_run: float,
                                target_roi: float = 2.0) -> bool:
        """Check if predicted yield makes outsourcing economically viable."""
        
        # Value of predicted gold yield
        gold_value_cad = predicted_yield_mg * self.gold_price_cad_per_g / 1000
        
        # ROI calculation
        roi = gold_value_cad / cost_per_run if cost_per_run > 0 else 0
        
        return roi >= target_roi
    
    def vendor_comparison(self, feedstock_g: float, dose_kgy: float) -> Dict:
        """Compare costs across different vendor combinations."""
        
        irradiation_vendors = [v for v in self.vendors if v.service_type == "irradiation"]
        analysis_vendors = [v for v in self.vendors if v.service_type == "analysis"]
        
        comparisons = []
        
        for irrad_v in irradiation_vendors:
            for analysis_v in analysis_vendors:
                try:
                    cost = self.calculate_experiment_cost(
                        feedstock_g, dose_kgy, 
                        irrad_v.vendor_name, analysis_v.vendor_name
                    )
                    
                    total_turnaround = irrad_v.turnaround_days + analysis_v.turnaround_days + 3  # Shipping
                    
                    comparisons.append({
                        'irradiation_vendor': irrad_v.vendor_name,
                        'analysis_vendor': analysis_v.vendor_name,
                        'total_cost': cost.total_cost,
                        'turnaround_days': total_turnaround,
                        'cost_breakdown': {
                            'irradiation': cost.irradiation_cost,
                            'analysis': cost.analysis_cost,
                            'shipping': cost.shipping_cost,
                            'preparation': cost.preparation_cost
                        }
                    })
                except Exception as e:
                    self.logger.warning(f"Error calculating cost for {irrad_v.vendor_name} + {analysis_v.vendor_name}: {e}")
        
        # Sort by total cost
        comparisons.sort(key=lambda x: x['total_cost'])
        
        return {
            'feedstock_g': feedstock_g,
            'dose_kgy': dose_kgy,
            'vendor_combinations': comparisons,
            'cheapest_option': comparisons[0] if comparisons else None,
            'most_expensive': comparisons[-1] if comparisons else None,
            'cost_range_cad': (comparisons[0]['total_cost'], comparisons[-1]['total_cost']) if comparisons else (0, 0)
        }
    
    def monthly_budget_analysis(self, monthly_budget_cad: float, 
                              feedstock_g: float, dose_kgy: float) -> Dict:
        """Analyze how many experiments can be run within monthly budget."""
        
        # Get cheapest vendor combination
        comparison = self.vendor_comparison(feedstock_g, dose_kgy)
        cheapest_cost = comparison['cheapest_option']['total_cost']
        
        # Calculate experiments per month
        experiments_per_month = int(monthly_budget_cad // cheapest_cost)
        remaining_budget = monthly_budget_cad - (experiments_per_month * cheapest_cost)
        
        # Annual projection
        annual_experiments = experiments_per_month * 12
        annual_feedstock_g = annual_experiments * feedstock_g
        
        return {
            'monthly_budget_cad': monthly_budget_cad,
            'cost_per_experiment': cheapest_cost,
            'experiments_per_month': experiments_per_month,
            'remaining_budget': remaining_budget,
            'annual_experiments': annual_experiments,
            'annual_feedstock_g': annual_feedstock_g,
            'cheapest_vendor_combo': comparison['cheapest_option'],
            'budget_utilization_percent': ((experiments_per_month * cheapest_cost) / monthly_budget_cad) * 100
        }
    
    def break_even_analysis(self, feedstock_g: float, dose_kgy: float,
                          predicted_yield_mg: float) -> Dict:
        """Calculate break-even point for gold production."""
        
        comparison = self.vendor_comparison(feedstock_g, dose_kgy)
        cost_per_experiment = comparison['cheapest_option']['total_cost']
        
        # Value per experiment
        gold_value_per_experiment = predicted_yield_mg * self.gold_price_cad_per_g / 1000
        
        # Break-even calculation
        net_per_experiment = gold_value_per_experiment - cost_per_experiment
        
        if net_per_experiment > 0:
            # Profitable
            roi_percent = (net_per_experiment / cost_per_experiment) * 100
            break_even_experiments = 1
        else:
            # Loss per experiment
            roi_percent = -((cost_per_experiment - gold_value_per_experiment) / cost_per_experiment) * 100
            break_even_experiments = float('inf')  # Never profitable
        
        # Minimum yield for break-even
        break_even_yield_mg = (cost_per_experiment / self.gold_price_cad_per_g) * 1000
        
        return {
            'cost_per_experiment': cost_per_experiment,
            'predicted_yield_mg': predicted_yield_mg,
            'gold_value_per_experiment': gold_value_per_experiment,
            'net_per_experiment': net_per_experiment,
            'roi_percent': roi_percent,
            'break_even_experiments': break_even_experiments,
            'break_even_yield_mg': break_even_yield_mg,
            'currently_profitable': net_per_experiment > 0,
            'yield_gap_mg': break_even_yield_mg - predicted_yield_mg
        }

def main():
    """Main cost analysis routine."""
    parser = argparse.ArgumentParser(description='Outsourcing cost analysis')
    parser.add_argument('--budget', type=float, default=100.0,
                       help='Monthly budget in CAD')
    parser.add_argument('--target-roi', type=float, default=2.0,
                       help='Target return on investment')
    parser.add_argument('--feedstock', type=float, default=1.0,
                       help='Feedstock mass in grams')
    parser.add_argument('--dose', type=float, default=10.0,
                       help='Irradiation dose in kGy')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize analyzer
    analyzer = OutsourcingAnalyzer()
    
    # Run analysis
    vendor_comp = analyzer.vendor_comparison(args.feedstock, args.dose)
    budget_analysis = analyzer.monthly_budget_analysis(args.budget, args.feedstock, args.dose)
    
    # Try to load predicted yield from optimization results
    try:
        with open("experiment_specs/optimal_recipe.json", 'r') as f:
            recipe = json.load(f)
            predicted_yield = recipe['predicted_au_mg']
    except:
        predicted_yield = 0.01  # Default assumption
    
    break_even = analyzer.break_even_analysis(args.feedstock, args.dose, predicted_yield)
    
    # Print results
    print("\n" + "="*60)
    print("OUTSOURCING COST ANALYSIS")
    print("="*60)
    print(f"Feedstock: {args.feedstock:.1f} g Pb-208")
    print(f"Dose: {args.dose:.1f} kGy")
    print(f"Monthly budget: ${args.budget:.2f} CAD")
    print()
    
    print("VENDOR COMPARISON:")
    cheapest = vendor_comp['cheapest_option']
    print(f"  Cheapest option: ${cheapest['total_cost']:.2f}")
    print(f"    Irradiation: {cheapest['irradiation_vendor']}")
    print(f"    Analysis: {cheapest['analysis_vendor']}")
    print(f"    Turnaround: {cheapest['turnaround_days']} days")
    print()
    
    print("BUDGET ANALYSIS:")
    print(f"  Experiments per month: {budget_analysis['experiments_per_month']}")
    print(f"  Remaining budget: ${budget_analysis['remaining_budget']:.2f}")
    print(f"  Annual experiments: {budget_analysis['annual_experiments']}")
    print(f"  Budget utilization: {budget_analysis['budget_utilization_percent']:.1f}%")
    print()
    
    print("BREAK-EVEN ANALYSIS:")
    print(f"  Predicted yield: {break_even['predicted_yield_mg']:.4f} mg Au")
    print(f"  Cost per experiment: ${break_even['cost_per_experiment']:.2f}")
    print(f"  Gold value per experiment: ${break_even['gold_value_per_experiment']:.2f}")
    print(f"  Net per experiment: ${break_even['net_per_experiment']:.2f}")
    print(f"  ROI: {break_even['roi_percent']:.1f}%")
    print(f"  Currently profitable: {'YES' if break_even['currently_profitable'] else 'NO'}")
    
    if not break_even['currently_profitable']:
        print(f"  Break-even yield needed: {break_even['break_even_yield_mg']:.4f} mg Au")
        print(f"  Yield gap: {break_even['yield_gap_mg']:.4f} mg Au")
    
    print("="*60)
    
    # Viability check
    viable = analyzer.is_viable_for_outsourcing(predicted_yield, cheapest['total_cost'], args.target_roi)
    print(f"ECONOMIC VIABILITY: {'VIABLE' if viable else 'NOT VIABLE'}")
    
    if not viable:
        print("Recommendations:")
        print("1. Optimize recipe for higher yield")
        print("2. Find cheaper vendor options")
        print("3. Increase feedstock mass per experiment")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
