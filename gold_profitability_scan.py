#!/usr/bin/env python3
"""
Gold Production Profitability Scanner
====================================

Systematic analysis of different feedstock options for Au-197 production.
Compares Fe-56, Hg-202, Pt-197, and Pb-208 feedstocks to determine optimal ROI.
"""

import json
import subprocess
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import shutil

class GoldProfitabilityScanner:
    """Scans multiple feedstock options for gold production profitability."""
    
    def __init__(self):
        self.feedstock_configs = {
            "Fe-56": "config_gold_fe.json",
            "Hg-202": "config_gold_hg.json", 
            "Pt-197": "config_gold_pt.json",
            "Pb-208": "config_gold_pb.json"
        }
          # Add photonuclear configs for comparison
        self.photonuclear_configs = {
            "Pt-195-GDR": "config_gold_photonuclear.json",
            "Hg-202-GDR-24h": "config_gold_hg_photonuclear.json", 
            "Pb-208-GDR-48h": "config_gold_pb_photonuclear.json"
        }
        
        self.results = []
        
        # Current gold price ($/g) from Reuters
        self.gold_price_per_g = 108.36
        
    def run_single_analysis(self, feedstock: str, config_file: str) -> Dict:
        """Run transmutation analysis for a single feedstock."""
        print(f"\n{'='*60}")
        print(f"ANALYZING FEEDSTOCK: {feedstock}")
        print(f"{'='*60}")
        
        # Copy config file to main config
        shutil.copy(config_file, "config.json")
        
        # Run the transmutation
        try:
            result = subprocess.run(
                ["python", "__main__.py"], 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"Error running transmutation for {feedstock}:")
                print(result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Timeout running transmutation for {feedstock}")
            return None
        
        # Load results
        try:
            with open("transmutation_results.json", "r") as f:
                trans_results = json.load(f)
        except FileNotFoundError:
            print(f"No results file found for {feedstock}")
            return None
          # Load config for economic parameters
        with open(config_file, "r") as f:
            config = json.load(f)
        
        # Calculate economics
        output_mass_g = trans_results["output_mass_g"]
        conversion_efficiency = trans_results["conversion_efficiency"]
        
        # Revenue
        revenue = output_mass_g * self.gold_price_per_g
        
        # Costs
        feedstock_cost_per_kg = config["economic_params"]["feedstock_cost_per_kg"]
        feedstock_mass_kg = trans_results["sample_mass_g"] / 1000  # Convert g to kg
        material_cost = feedstock_mass_kg * feedstock_cost_per_kg
        
        energy_cost_kwh = config["economic_params"]["energy_cost_per_kwh"]
        energy_used_kwh = trans_results.get("energy_used_kwh", 0)
        energy_cost = energy_used_kwh * energy_cost_kwh
        
        overhead_per_hour = config["economic_params"]["facility_overhead_per_hour"]
        duration_hours = config["duration_s"] / 3600
        overhead_cost = overhead_per_hour * duration_hours
        
        total_cost = material_cost + energy_cost + overhead_cost
        
        # Profit and ROI
        profit = revenue - total_cost
        roi = (profit / total_cost * 100) if total_cost > 0 else float('inf')
        
        # Yield ratio
        feedstock_mass_g = feedstock_mass_kg * 1000
        yield_ratio = output_mass_g / feedstock_mass_g if feedstock_mass_g > 0 else 0
        
        return {
            "feedstock": feedstock,
            "feedstock_cost_per_g": feedstock_cost_per_kg / 1000,
            "gold_revenue_per_g": self.gold_price_per_g,
            "yield_g_au_per_g_feed": yield_ratio,
            "mass_produced_g": output_mass_g,
            "feedstock_mass_g": feedstock_mass_g,
            "conversion_efficiency": conversion_efficiency,
            "gross_revenue": revenue,
            "feed_cost": material_cost,
            "energy_cost": energy_cost,
            "overhead_cost": overhead_cost,
            "total_cost": total_cost,
            "profit": profit,
            "roi_percent": roi,
            "lv_enhancement": trans_results.get("lv_enhancement", 1.0)
        }
    
    def run_full_scan(self) -> pd.DataFrame:
        """Run profitability analysis for all feedstock options."""
        print(f"\n{'='*80}")
        print("GOLD PRODUCTION PROFITABILITY SCAN")
        print(f"Target: Au-197 | Gold Price: ${self.gold_price_per_g:.2f}/g")
        print(f"{'='*80}")
        
        # Test conventional spallation feedstocks
        for feedstock, config_file in self.feedstock_configs.items():
            result = self.run_single_analysis(feedstock, config_file)
            if result:
                self.results.append(result)
        
        # Test photonuclear (GDR) options
        for feedstock, config_file in self.photonuclear_configs.items():
            result = self.run_single_analysis(feedstock, config_file)
            if result:
                self.results.append(result)
        
        # Create results DataFrame
        df = pd.DataFrame(self.results)
        
        if not df.empty:
            # Sort by ROI
            df = df.sort_values('roi_percent', ascending=False)
            
            # Display results table
            self.display_results_table(df)
            
            # Save to CSV
            df.to_csv("gold_profitability_scan.csv", index=False)
            print(f"\nDetailed results saved to: gold_profitability_scan.csv")
        
        return df
    
    def display_results_table(self, df: pd.DataFrame):
        """Display formatted results table."""
        print(f"\n{'='*120}")
        print("GOLD PRODUCTION PROFITABILITY ANALYSIS")
        print(f"{'='*120}")
        
        print(f"{'Feedstock':<10} {'Cost/g':<10} {'Au Rev/g':<10} {'Yield':<12} {'Gross Rev':<12} {'Feed Cost':<10} {'Profit':<12} {'ROI %':<8}")
        print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*8}")
        
        for _, row in df.iterrows():
            print(f"{row['feedstock']:<10} "
                  f"${row['feedstock_cost_per_g']:<9.2f} "
                  f"${row['gold_revenue_per_g']:<9.2f} "
                  f"{row['yield_g_au_per_g_feed']:<12.2e} "
                  f"${row['gross_revenue']:<11.2f} "
                  f"${row['feed_cost']:<9.2f} "
                  f"${row['profit']:<11.2f} "
                  f"{row['roi_percent']:<8.1f}")
        
        print(f"\n{'='*120}")
        
        # Highlight best option
        best = df.iloc[0]
        print(f"🏆 BEST FEEDSTOCK: {best['feedstock']}")
        print(f"   ROI: {best['roi_percent']:.1f}%")
        print(f"   Profit per mg feedstock: ${best['profit']:.2f}")
        print(f"   Yield: {best['yield_g_au_per_g_feed']:.2e} g Au / g feedstock")
        
        # Analysis insights
        print(f"\n💡 INSIGHTS:")
        cheapest = df.loc[df['feedstock_cost_per_g'].idxmin()]
        highest_yield = df.loc[df['yield_g_au_per_g_feed'].idxmax()] 
        
        print(f"   • Cheapest feedstock: {cheapest['feedstock']} (${cheapest['feedstock_cost_per_g']:.4f}/g)")
        print(f"   • Highest yield: {highest_yield['feedstock']} ({highest_yield['yield_g_au_per_g_feed']:.2e} g Au/g feed)")
        print(f"   • Economic sweet spot: {best['feedstock']} balances cost vs yield optimally")

def main():
    """Run the gold profitability scan."""
    scanner = GoldProfitabilityScanner()
    results_df = scanner.run_full_scan()
    
    if not results_df.empty:
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   • Number of feedstocks analyzed: {len(results_df)}")
        print(f"   • Profitable options: {len(results_df[results_df['roi_percent'] > 0])}")
        print(f"   • Best ROI: {results_df['roi_percent'].max():.1f}%")
        print(f"   • Worst ROI: {results_df['roi_percent'].min():.1f}%")
        
        # Recommendations
        profitable = results_df[results_df['roi_percent'] > 0]
        if len(profitable) > 0:
            print(f"\n✅ RECOMMENDATION: Use {profitable.iloc[0]['feedstock']} for optimal gold production")
        else:
            print(f"\n⚠️  WARNING: No feedstock options currently profitable at these parameters")
            print(f"   Consider optimizing beam energy, duration, or LV parameters")

if __name__ == "__main__":
    main()
