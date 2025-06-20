#!/usr/bin/env python3
"""
Cost-Efficiency Analysis for Photonuclear Gold Production
=========================================================

Comprehensive cost analysis including:
- Energy costs (gamma beam generation, cooling, controls)
- Feedstock costs (Pb-208 procurement)
- Equipment amortization
- Operating expenses
- Revenue calculations
- Break-even analysis
- LV energy recovery optimization

Cost-reduction R&D analysis:
- Co-60 vs linac gamma sources
- LV energy harvester integration
- Process optimization opportunities
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt

@dataclass
class CostParameters:
    """Economic parameters for cost analysis."""
    # Energy costs (CAD/kWh)
    electricity_cost_cad_kwh: float = 0.12  # Ontario average
    
    # Gold market price (CAD/g)
    gold_price_cad_g: float = 85.0  # Current market price
    
    # Feedstock costs (CAD/kg)
    pb208_cost_cad_kg: float = 50000.0  # Enriched Pb-208
    
    # Equipment costs (CAD)
    linac_system_cost_cad: float = 2000000.0  # 2M CAD for 30kW system
    co60_source_cost_cad: float = 200000.0    # 200k CAD for sealed source
    facility_cost_cad: float = 1000000.0      # Lab facility
    
    # Operating parameters
    equipment_lifetime_years: float = 10.0
    operation_hours_per_year: float = 6000.0  # 250 days * 24 hours
    maintenance_cost_percent: float = 0.05    # 5% per year
    
    # Staff costs (CAD/year)
    operator_salary_cad: float = 75000.0
    technician_salary_cad: float = 65000.0
    
    # LV enhancement factors
    lv_energy_recovery_factor: float = 50.0   # 50x energy reduction potential
    lv_yield_enhancement: float = 100.0       # 100x yield improvement

@dataclass
class ProductionMetrics:
    """Production performance metrics."""
    target_mass_g: float = 1.0
    irradiation_time_h: float = 6.0
    conversion_efficiency_percent: float = 0.001
    gold_yield_mg: float = 0.01
    energy_consumption_kwh: float = 180.0  # 30kW * 6h
    
    # LV-enhanced metrics
    lv_enhanced_yield_mg: float = 1.0      # 100x improvement
    lv_energy_consumption_kwh: float = 3.6  # 50x reduction

class CostEfficiencyAnalyzer:
    """Comprehensive cost-efficiency analysis system."""
    
    def __init__(self, cost_params: Optional[CostParameters] = None):
        """Initialize the cost analyzer."""
        self.cost_params = cost_params or CostParameters()
        self.logger = logging.getLogger(__name__)
        
    def calculate_energy_cost_per_gram_gold(self, metrics: ProductionMetrics, 
                                          use_lv_enhancement: bool = False) -> Dict[str, float]:
        """Calculate energy cost per gram of gold produced."""
        if use_lv_enhancement:
            energy_kwh = metrics.lv_energy_consumption_kwh
            gold_yield_mg = metrics.lv_enhanced_yield_mg
        else:
            energy_kwh = metrics.energy_consumption_kwh
            gold_yield_mg = metrics.gold_yield_mg
        
        # Convert mg to g
        gold_yield_g = gold_yield_mg / 1000.0
        
        if gold_yield_g > 0:
            energy_cost_cad = energy_kwh * self.cost_params.electricity_cost_cad_kwh
            energy_cost_per_g_gold = energy_cost_cad / gold_yield_g
        else:
            energy_cost_per_g_gold = float('inf')
        
        return {
            'total_energy_kwh': energy_kwh,
            'energy_cost_cad': energy_cost_cad,
            'gold_yield_g': gold_yield_g,
            'energy_cost_per_g_gold_cad': energy_cost_per_g_gold,
            'lv_enhanced': use_lv_enhancement
        }
    
    def calculate_feedstock_cost_per_gram_gold(self, metrics: ProductionMetrics) -> Dict[str, float]:
        """Calculate feedstock cost per gram of gold."""
        # Convert target mass to kg
        target_mass_kg = metrics.target_mass_g / 1000.0
        feedstock_cost_cad = target_mass_kg * self.cost_params.pb208_cost_cad_kg
        
        # Gold yield in grams
        gold_yield_g = metrics.gold_yield_mg / 1000.0
        
        if gold_yield_g > 0:
            feedstock_cost_per_g_gold = feedstock_cost_cad / gold_yield_g
        else:
            feedstock_cost_per_g_gold = float('inf')
        
        return {
            'target_mass_kg': target_mass_kg,
            'feedstock_cost_cad': feedstock_cost_cad,
            'gold_yield_g': gold_yield_g,
            'feedstock_cost_per_g_gold_cad': feedstock_cost_per_g_gold
        }
    
    def calculate_equipment_amortization(self, use_co60: bool = False) -> Dict[str, float]:
        """Calculate equipment amortization costs."""
        if use_co60:
            equipment_cost = self.cost_params.co60_source_cost_cad
        else:
            equipment_cost = self.cost_params.linac_system_cost_cad
        
        total_cost = equipment_cost + self.cost_params.facility_cost_cad
        
        # Annual amortization
        annual_amortization = total_cost / self.cost_params.equipment_lifetime_years
        
        # Cost per hour
        hourly_amortization = annual_amortization / self.cost_params.operation_hours_per_year
        
        # Maintenance costs
        annual_maintenance = total_cost * self.cost_params.maintenance_cost_percent
        hourly_maintenance = annual_maintenance / self.cost_params.operation_hours_per_year
        
        return {
            'equipment_cost_cad': equipment_cost,
            'total_capital_cost_cad': total_cost,
            'annual_amortization_cad': annual_amortization,
            'hourly_amortization_cad': hourly_amortization,
            'annual_maintenance_cad': annual_maintenance,
            'hourly_maintenance_cad': hourly_maintenance,
            'total_hourly_equipment_cost_cad': hourly_amortization + hourly_maintenance,
            'gamma_source': 'Co-60' if use_co60 else 'Linac'
        }
    
    def calculate_operating_costs(self, metrics: ProductionMetrics) -> Dict[str, float]:
        """Calculate operating costs per production run."""
        # Staff costs per hour
        total_annual_salary = (self.cost_params.operator_salary_cad + 
                              self.cost_params.technician_salary_cad)
        hourly_staff_cost = total_annual_salary / self.cost_params.operation_hours_per_year
        
        # Total operating cost for this run
        staff_cost_run = hourly_staff_cost * metrics.irradiation_time_h
        
        return {
            'hourly_staff_cost_cad': hourly_staff_cost,
            'staff_cost_run_cad': staff_cost_run,
            'irradiation_time_h': metrics.irradiation_time_h
        }
    
    def comprehensive_cost_analysis(self, metrics: ProductionMetrics, 
                                  use_co60: bool = False,
                                  use_lv_enhancement: bool = False) -> Dict[str, Any]:
        """Perform comprehensive cost-benefit analysis."""
        
        # Calculate individual cost components
        energy_costs = self.calculate_energy_cost_per_gram_gold(metrics, use_lv_enhancement)
        feedstock_costs = self.calculate_feedstock_cost_per_gram_gold(metrics)
        equipment_costs = self.calculate_equipment_amortization(use_co60)
        operating_costs = self.calculate_operating_costs(metrics)
        
        # Equipment costs for this run
        equipment_cost_run = (equipment_costs['total_hourly_equipment_cost_cad'] * 
                             metrics.irradiation_time_h)
        
        # Total production costs
        total_cost_run = (energy_costs['energy_cost_cad'] + 
                         feedstock_costs['feedstock_cost_cad'] +
                         equipment_cost_run +
                         operating_costs['staff_cost_run_cad'])
        
        # Gold production and revenue
        gold_yield_g = energy_costs['gold_yield_g']
        revenue_cad = gold_yield_g * self.cost_params.gold_price_cad_g
        
        # Profit/loss
        profit_loss_cad = revenue_cad - total_cost_run
        
        # Cost per gram of gold
        if gold_yield_g > 0:
            total_cost_per_g_gold = total_cost_run / gold_yield_g
            profit_margin_percent = (profit_loss_cad / revenue_cad) * 100
        else:
            total_cost_per_g_gold = float('inf')
            profit_margin_percent = -100.0
        
        # Break-even analysis
        breakeven_gold_price = total_cost_per_g_gold
        price_advantage_percent = ((self.cost_params.gold_price_cad_g - breakeven_gold_price) / 
                                  self.cost_params.gold_price_cad_g * 100)
        
        return {
            'scenario': {
                'gamma_source': equipment_costs['gamma_source'],
                'lv_enhanced': use_lv_enhancement,
                'target_mass_g': metrics.target_mass_g,
                'irradiation_time_h': metrics.irradiation_time_h
            },
            'production': {
                'gold_yield_g': gold_yield_g,
                'conversion_efficiency_percent': metrics.conversion_efficiency_percent
            },
            'costs': {
                'energy_cost_cad': energy_costs['energy_cost_cad'],
                'feedstock_cost_cad': feedstock_costs['feedstock_cost_cad'],
                'equipment_cost_run_cad': equipment_cost_run,
                'operating_cost_run_cad': operating_costs['staff_cost_run_cad'],
                'total_cost_run_cad': total_cost_run,
                'total_cost_per_g_gold_cad': total_cost_per_g_gold
            },
            'revenue': {
                'gold_price_cad_g': self.cost_params.gold_price_cad_g,
                'revenue_cad': revenue_cad,
                'profit_loss_cad': profit_loss_cad,
                'profit_margin_percent': profit_margin_percent
            },
            'breakeven': {
                'breakeven_gold_price_cad_g': breakeven_gold_price,
                'price_advantage_percent': price_advantage_percent,
                'economically_viable': profit_loss_cad > 0
            }
        }
    
    def scenario_comparison(self, metrics: ProductionMetrics) -> Dict[str, Any]:
        """Compare different technology scenarios."""
        scenarios = {}
        
        # Scenario 1: Linac, no LV enhancement
        scenarios['linac_baseline'] = self.comprehensive_cost_analysis(
            metrics, use_co60=False, use_lv_enhancement=False
        )
        
        # Scenario 2: Co-60, no LV enhancement  
        scenarios['co60_baseline'] = self.comprehensive_cost_analysis(
            metrics, use_co60=True, use_lv_enhancement=False
        )
        
        # Scenario 3: Linac with LV enhancement
        scenarios['linac_lv_enhanced'] = self.comprehensive_cost_analysis(
            metrics, use_co60=False, use_lv_enhancement=True
        )
        
        # Scenario 4: Co-60 with LV enhancement
        scenarios['co60_lv_enhanced'] = self.comprehensive_cost_analysis(
            metrics, use_co60=True, use_lv_enhancement=True
        )
        
        # Find best scenario
        best_scenario = min(scenarios.keys(), 
                           key=lambda x: scenarios[x]['costs']['total_cost_per_g_gold_cad'])
        
        return {
            'scenarios': scenarios,
            'best_scenario': best_scenario,
            'best_profit_cad': scenarios[best_scenario]['revenue']['profit_loss_cad'],
            'best_cost_per_g_cad': scenarios[best_scenario]['costs']['total_cost_per_g_gold_cad'],
            'comparison_date': datetime.now().isoformat()
        }
    
    def generate_cost_report(self, metrics: ProductionMetrics) -> str:
        """Generate a comprehensive cost analysis report."""
        comparison = self.scenario_comparison(metrics)
        
        report = []
        report.append("PHOTONUCLEAR GOLD PRODUCTION - COST ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Gold Market Price: ${self.cost_params.gold_price_cad_g:.2f} CAD/g")
        report.append(f"Electricity Cost: ${self.cost_params.electricity_cost_cad_kwh:.3f} CAD/kWh")
        report.append("")
        
        report.append("SCENARIO COMPARISON")
        report.append("-" * 30)
        
        for scenario_name, data in comparison['scenarios'].items():
            scenario_title = scenario_name.replace('_', ' ').title()
            report.append(f"\n{scenario_title}:")
            report.append(f"  Gamma Source: {data['scenario']['gamma_source']}")
            report.append(f"  LV Enhanced: {data['scenario']['lv_enhanced']}")
            report.append(f"  Gold Yield: {data['production']['gold_yield_g']:.6f} g")
            report.append(f"  Total Cost: ${data['costs']['total_cost_run_cad']:.2f} CAD")
            report.append(f"  Cost per g Au: ${data['costs']['total_cost_per_g_gold_cad']:.2f} CAD/g")
            report.append(f"  Revenue: ${data['revenue']['revenue_cad']:.2f} CAD")
            report.append(f"  Profit/Loss: ${data['revenue']['profit_loss_cad']:.2f} CAD")
            report.append(f"  Economically Viable: {data['breakeven']['economically_viable']}")
        
        report.append(f"\nBEST SCENARIO: {comparison['best_scenario'].replace('_', ' ').title()}")
        best = comparison['scenarios'][comparison['best_scenario']]
        report.append(f"Profit: ${best['revenue']['profit_loss_cad']:.2f} CAD")
        report.append(f"Margin: {best['revenue']['profit_margin_percent']:.1f}%")
        
        report.append("\nCOST REDUCTION OPPORTUNITIES")
        report.append("-" * 35)
        report.append("1. LV Energy Recovery: Up to 50x energy cost reduction")
        report.append("2. LV Yield Enhancement: Up to 100x yield improvement")  
        report.append("3. Co-60 Source: Lower capital costs vs linac")
        report.append("4. Process Optimization: Improved conversion efficiency")
        report.append("5. Scale Effects: Reduced per-unit costs at higher volumes")
        
        return "\n".join(report)

def demo_cost_analysis():
    """Demonstrate the cost analysis capabilities."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("PHOTONUCLEAR GOLD PRODUCTION - COST ANALYSIS DEMO")
    print("=" * 55)
    
    # Create analyzer
    analyzer = CostEfficiencyAnalyzer()
    
    # Define production metrics for current system
    current_metrics = ProductionMetrics(
        target_mass_g=1.0,
        irradiation_time_h=6.0,
        conversion_efficiency_percent=0.001,
        gold_yield_mg=0.01,
        energy_consumption_kwh=180.0
    )
    
    # Generate comprehensive report
    report = analyzer.generate_cost_report(current_metrics)
    print(report)
    
    # Energy cost breakdown
    print("\n" + "=" * 55)
    print("DETAILED ENERGY COST ANALYSIS")
    print("=" * 55)
    
    energy_baseline = analyzer.calculate_energy_cost_per_gram_gold(current_metrics, False)
    energy_lv = analyzer.calculate_energy_cost_per_gram_gold(current_metrics, True)
    
    print(f"Baseline Energy Cost: ${energy_baseline['energy_cost_per_g_gold_cad']:.2f} CAD/g Au")
    print(f"LV Enhanced Energy Cost: ${energy_lv['energy_cost_per_g_gold_cad']:.2f} CAD/g Au")
    print(f"Energy Cost Reduction: {energy_baseline['energy_cost_per_g_gold_cad'] / energy_lv['energy_cost_per_g_gold_cad']:.0f}x")
    
    return analyzer

if __name__ == "__main__":
    demo_cost_analysis()
