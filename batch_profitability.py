#!/usr/bin/env python3
"""
Batch Profitability Analysis - Small-Scale Gold Conversion Economics
===================================================================

Analyzes ROI for nanogram to microgram scale gold conversion batches,
including consumables, overhead, and market spread considerations.

Author: Advanced Energy Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import argparse

@dataclass
class BatchEconomics:
    """Economic parameters for batch conversion"""
    # Input costs
    feedstock_cost_per_kg: float    # $/kg for Hg/Pt feedstock
    beam_energy_cost_per_j: float   # $/J for accelerator operation
    consumables_cost_per_run: float # $ for chemicals, maintenance per run
    equipment_amortization_per_run: float # $ equipment cost per run
    labor_cost_per_run: float       # $ operator time per run
    
    # Market parameters
    gold_spot_price_per_g: float    # $/g current gold price
    market_premium_factor: float    # multiplier for small quantity sales
    transaction_cost_per_sale: float # $ fixed cost per sale
    
    # Operational parameters
    batch_size_kg: float            # kg feedstock per batch
    energy_per_batch_j: float       # J energy consumption per batch
    conversion_efficiency: float    # fraction converted to gold

class BatchProfitabilityAnalyzer:
    """Analyzes profitability across different batch sizes and market conditions"""
    
    def __init__(self):
        self.default_economics = BatchEconomics(
            # Conservative cost estimates
            feedstock_cost_per_kg=50.0,        # $50/kg for Hg-202 (expensive isotope)
            beam_energy_cost_per_j=1e-6,       # $1/MJ for electricity + overhead
            consumables_cost_per_run=5.0,      # $5 per run (chemicals, etc.)
            equipment_amortization_per_run=25.0, # $25 per run (equipment amortization)
            labor_cost_per_run=50.0,           # $50 per run (30 min @ $100/hr)
            
            # Gold market parameters (2025 estimates)
            gold_spot_price_per_g=65.0,        # $65/g spot price
            market_premium_factor=2.0,         # 2x premium for small quantities
            transaction_cost_per_sale=15.0,    # $15 transaction cost
            
            # Performance estimates
            batch_size_kg=1e-9,                # 1 ng default batch
            energy_per_batch_j=1e3,            # 1 kJ per batch
            conversion_efficiency=0.15          # 15% Hg → Au efficiency
        )
    
    def calculate_batch_costs(self, economics: BatchEconomics) -> Dict:
        """Calculate all costs for a single batch"""
        
        # Material costs
        feedstock_cost = economics.feedstock_cost_per_kg * economics.batch_size_kg
        energy_cost = economics.beam_energy_cost_per_j * economics.energy_per_batch_j
        
        # Fixed costs
        fixed_costs = (economics.consumables_cost_per_run +
                      economics.equipment_amortization_per_run +
                      economics.labor_cost_per_run)
        
        # Transaction costs
        transaction_cost = economics.transaction_cost_per_sale
        
        total_cost = feedstock_cost + energy_cost + fixed_costs + transaction_cost
        
        return {
            'feedstock_cost': feedstock_cost,
            'energy_cost': energy_cost,
            'fixed_costs': fixed_costs,
            'transaction_cost': transaction_cost,
            'total_cost': total_cost
        }
    
    def calculate_batch_revenue(self, economics: BatchEconomics) -> Dict:
        """Calculate revenue for a single batch"""
        
        # Gold yield calculation
        # Assume atomic weight conversion: Hg-202 → Au-197 (simplified)
        molar_mass_hg = 0.202  # kg/mol
        molar_mass_au = 0.197  # kg/mol
        
        # Theoretical maximum yield (if 100% conversion)
        max_gold_kg = economics.batch_size_kg * (molar_mass_au / molar_mass_hg)
        
        # Actual yield with efficiency
        actual_gold_kg = max_gold_kg * economics.conversion_efficiency
        actual_gold_g = actual_gold_kg * 1000
        
        # Market value
        spot_value = actual_gold_g * economics.gold_spot_price_per_g
        market_value = spot_value * economics.market_premium_factor
        
        return {
            'gold_yield_g': actual_gold_g,
            'spot_value': spot_value,
            'market_value': market_value,
            'premium_value': market_value - spot_value
        }
    
    def analyze_batch_profitability(self, economics: BatchEconomics) -> Dict:
        """Complete profitability analysis for a batch"""
        
        costs = self.calculate_batch_costs(economics)
        revenue = self.calculate_batch_revenue(economics)
        
        profit = revenue['market_value'] - costs['total_cost']
        roi = (profit / costs['total_cost'] * 100) if costs['total_cost'] > 0 else 0
        profit_margin = (profit / revenue['market_value'] * 100) if revenue['market_value'] > 0 else 0
        
        # Break-even analysis
        breakeven_batch_size = costs['total_cost'] / (
            economics.gold_spot_price_per_g * 
            economics.market_premium_factor * 
            economics.conversion_efficiency * 
            (0.197 / 0.202) * 1000  # kg to g conversion
        )
        
        return {
            'costs': costs,
            'revenue': revenue,
            'profit': profit,
            'roi_percent': roi,
            'profit_margin_percent': profit_margin,
            'breakeven_batch_size_kg': breakeven_batch_size,
            'profitable': profit > 0
        }
    
    def batch_size_sweep(self, batch_sizes_kg: List[float], 
                        base_economics: BatchEconomics = None) -> pd.DataFrame:
        """Analyze profitability across different batch sizes"""
        
        if base_economics is None:
            base_economics = self.default_economics
        
        results = []
        
        for batch_size in batch_sizes_kg:
            # Create economics for this batch size
            economics = BatchEconomics(
                feedstock_cost_per_kg=base_economics.feedstock_cost_per_kg,
                beam_energy_cost_per_j=base_economics.beam_energy_cost_per_j,
                consumables_cost_per_run=base_economics.consumables_cost_per_run,
                equipment_amortization_per_run=base_economics.equipment_amortization_per_run,
                labor_cost_per_run=base_economics.labor_cost_per_run,
                gold_spot_price_per_g=base_economics.gold_spot_price_per_g,
                market_premium_factor=base_economics.market_premium_factor,
                transaction_cost_per_sale=base_economics.transaction_cost_per_sale,
                batch_size_kg=batch_size,
                energy_per_batch_j=base_economics.energy_per_batch_j,
                conversion_efficiency=base_economics.conversion_efficiency
            )
            
            analysis = self.analyze_batch_profitability(economics)
            
            results.append({
                'batch_size_kg': batch_size,
                'batch_size_ng': batch_size * 1e12,
                'gold_yield_ug': analysis['revenue']['gold_yield_g'] * 1e6,
                'total_cost': analysis['costs']['total_cost'],
                'market_value': analysis['revenue']['market_value'],
                'profit': analysis['profit'],
                'roi_percent': analysis['roi_percent'],
                'profit_margin_percent': analysis['profit_margin_percent'],
                'profitable': analysis['profitable']
            })
        
        return pd.DataFrame(results)
    
    def market_sensitivity_analysis(self, gold_prices: List[float], 
                                  premium_factors: List[float]) -> pd.DataFrame:
        """Analyze sensitivity to market conditions"""
        
        results = []
        base_economics = self.default_economics
        
        for gold_price in gold_prices:
            for premium_factor in premium_factors:
                economics = BatchEconomics(
                    feedstock_cost_per_kg=base_economics.feedstock_cost_per_kg,
                    beam_energy_cost_per_j=base_economics.beam_energy_cost_per_j,
                    consumables_cost_per_run=base_economics.consumables_cost_per_run,
                    equipment_amortization_per_run=base_economics.equipment_amortization_per_run,
                    labor_cost_per_run=base_economics.labor_cost_per_run,
                    gold_spot_price_per_g=gold_price,
                    market_premium_factor=premium_factor,
                    transaction_cost_per_sale=base_economics.transaction_cost_per_sale,
                    batch_size_kg=base_economics.batch_size_kg,
                    energy_per_batch_j=base_economics.energy_per_batch_j,
                    conversion_efficiency=base_economics.conversion_efficiency
                )
                
                analysis = self.analyze_batch_profitability(economics)
                
                results.append({
                    'gold_price': gold_price,
                    'premium_factor': premium_factor,
                    'profit': analysis['profit'],
                    'roi_percent': analysis['roi_percent'],
                    'profitable': analysis['profitable'],
                    'breakeven_batch_size_kg': analysis['breakeven_batch_size_kg']
                })
        
        return pd.DataFrame(results)
    
    def plot_profitability_analysis(self, df: pd.DataFrame, title: str = "Batch Profitability Analysis"):
        """Create visualization of profitability analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Profit vs Batch Size
        ax1.semilogx(df['batch_size_ng'], df['profit'], 'b-', linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Batch Size (ng)')
        ax1.set_ylabel('Profit ($)')
        ax1.set_title('Profit vs Batch Size')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ROI vs Batch Size
        ax2.semilogx(df['batch_size_ng'], df['roi_percent'], 'g-', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Batch Size (ng)')
        ax2.set_ylabel('ROI (%)')
        ax2.set_title('Return on Investment vs Batch Size')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cost Breakdown
        profitable_batches = df[df['profitable']]
        if len(profitable_batches) > 0:
            ax3.semilogx(df['batch_size_ng'], df['total_cost'], 'r-', label='Total Cost', linewidth=2)
            ax3.semilogx(df['batch_size_ng'], df['market_value'], 'g-', label='Market Value', linewidth=2)
            ax3.set_xlabel('Batch Size (ng)')
            ax3.set_ylabel('Value ($)')
            ax3.set_title('Cost vs Revenue')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Gold Yield
        ax4.loglog(df['batch_size_ng'], df['gold_yield_ug'], 'orange', linewidth=2)
        ax4.set_xlabel('Batch Size (ng)')
        ax4.set_ylabel('Gold Yield (μg)')
        ax4.set_title('Gold Yield vs Input Mass')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_file: str = "batch_profitability_report.json"):
        """Generate comprehensive profitability report"""
        
        # Batch size sweep (1 ng to 1 mg)
        batch_sizes = np.logspace(-12, -6, 25)  # 1 ng to 1 mg
        batch_df = self.batch_size_sweep(batch_sizes)
        
        # Market sensitivity
        gold_prices = [50, 60, 70, 80, 90]  # $/g
        premium_factors = [1.2, 1.5, 2.0, 2.5, 3.0]
        market_df = self.market_sensitivity_analysis(gold_prices, premium_factors)
        
        # Find optimal operating points
        profitable_batches = batch_df[batch_df['profitable']]
        
        if len(profitable_batches) > 0:
            best_roi_idx = profitable_batches['roi_percent'].idxmax()
            best_roi_batch = profitable_batches.loc[best_roi_idx]
            
            min_profitable_size = profitable_batches['batch_size_kg'].min()
            max_profitable_size = profitable_batches['batch_size_kg'].max()
        else:
            best_roi_batch = None
            min_profitable_size = None
            max_profitable_size = None
        
        # Compile report
        report = {
            'analysis_summary': {
                'total_batch_sizes_analyzed': len(batch_df),
                'profitable_batch_count': len(profitable_batches),
                'profitability_ratio': len(profitable_batches) / len(batch_df),
                'min_profitable_batch_size_kg': min_profitable_size,
                'max_profitable_batch_size_kg': max_profitable_size,
                'best_roi_batch': best_roi_batch.to_dict() if best_roi_batch is not None else None
            },
            'economic_assumptions': {
                'feedstock_cost_per_kg': self.default_economics.feedstock_cost_per_kg,
                'gold_spot_price_per_g': self.default_economics.gold_spot_price_per_g,
                'market_premium_factor': self.default_economics.market_premium_factor,
                'conversion_efficiency': self.default_economics.conversion_efficiency,
                'fixed_cost_per_run': (self.default_economics.consumables_cost_per_run +
                                     self.default_economics.equipment_amortization_per_run +
                                     self.default_economics.labor_cost_per_run)
            },
            'batch_analysis': batch_df.to_dict('records'),
            'market_sensitivity': market_df.to_dict('records')
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Profitability report saved to {output_file}")
        
        # Create visualization
        fig = self.plot_profitability_analysis(batch_df)
        plot_file = output_file.replace('.json', '_plots.png')
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Profitability plots saved to {plot_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Batch Gold Conversion Profitability Analysis')
    parser.add_argument('--output', default='batch_profitability_report.json',
                       help='Output report filename')
    parser.add_argument('--batch-size', type=float,
                       help='Analyze specific batch size (kg)')
    parser.add_argument('--gold-price', type=float, default=65.0,
                       help='Gold spot price ($/g)')
    parser.add_argument('--efficiency', type=float, default=0.15,
                       help='Conversion efficiency (fraction)')
    
    args = parser.parse_args()
    
    analyzer = BatchProfitabilityAnalyzer()
    
    # Update default economics if specified
    if args.gold_price != 65.0:
        analyzer.default_economics.gold_spot_price_per_g = args.gold_price
    if args.efficiency != 0.15:
        analyzer.default_economics.conversion_efficiency = args.efficiency
    
    if args.batch_size:
        # Analyze specific batch size
        economics = analyzer.default_economics
        economics.batch_size_kg = args.batch_size
        
        analysis = analyzer.analyze_batch_profitability(economics)
        
        print(f"\nBATCH PROFITABILITY ANALYSIS")
        print(f"===========================")
        print(f"Batch size: {args.batch_size*1e12:.1f} ng")
        print(f"Gold yield: {analysis['revenue']['gold_yield_g']*1e6:.2f} μg")
        print(f"Total cost: ${analysis['costs']['total_cost']:.2f}")
        print(f"Market value: ${analysis['revenue']['market_value']:.2f}")
        print(f"Profit: ${analysis['profit']:.2f}")
        print(f"ROI: {analysis['roi_percent']:.1f}%")
        print(f"Profitable: {'YES' if analysis['profitable'] else 'NO'}")
        
    else:
        # Generate full report
        report = analyzer.generate_report(args.output)
        
        print(f"\nPROFITABILITY SUMMARY")
        print(f"====================")
        print(f"Profitable batches: {report['analysis_summary']['profitable_batch_count']}")
        print(f"Profitability ratio: {report['analysis_summary']['profitability_ratio']:.1%}")
        
        if report['analysis_summary']['best_roi_batch']:
            best = report['analysis_summary']['best_roi_batch']
            print(f"Best ROI batch: {best['batch_size_ng']:.1f} ng → {best['roi_percent']:.1f}% ROI")

if __name__ == "__main__":
    main()
