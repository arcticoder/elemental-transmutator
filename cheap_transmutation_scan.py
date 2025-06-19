#!/usr/bin/env python3
"""
Cheap Transmutation Economic Scanner
===================================

Economic optimization and parameter sweeping for the cheap feedstock
rhodium replicator. Maximizes profit by optimizing feedstock selection,
beam parameters, and LV coefficients.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime

# Import our feedstock modules  
from feedstock_selector import FeedstockSelector, FeedstockCandidate
from cheap_feedstock_network import FeedstockNetworkTransmuter

@dataclass
class EconomicResult:
    """Economic analysis result for a parameter combination."""
    feedstock: str
    beam_type: str
    beam_energy: float
    lv_params: Dict[str, float]
    input_mass_kg: float
    input_cost: float
    rhodium_mass_kg: float
    rhodium_value: float
    energy_cost: float
    net_profit: float
    profit_ratio: float
    roi_percent: float
    payback_time_hours: float

class CheapTransmutationScanner:
    """Economic scanner for cheap feedstock rhodium production."""
    
    def __init__(self):
        self.feedstock_selector = FeedstockSelector()
        self.results = []
        
        # Economic parameters
        self.rhodium_price = 25000.0    # $/kg
        self.electricity_cost = 0.12    # $/kWh
        self.equipment_cost = 500000.0  # $ (cyclotron + facilities)
        self.operating_cost_per_hour = 50.0  # $/hr
        
    def calculate_energy_cost(self, energy_joules: float) -> float:
        """Calculate electricity cost for energy consumption."""
        energy_kwh = energy_joules / 3.6e6  # J to kWh
        return energy_kwh * self.electricity_cost
    
    def run_economic_analysis(self, feedstock: str, beam_config: Dict, 
                            lv_params: Dict, input_mass: float) -> EconomicResult:
        """Run complete economic analysis for given parameters."""
        
        # Create transmuter
        transmuter = FeedstockNetworkTransmuter(
            lv_params=lv_params,
            feedstock_isotope=feedstock,
            beam_profile=beam_config
        )
        
        # Run transmutation
        results = transmuter.full_chain(mass_kg=input_mass)
        
        # Get feedstock data
        candidate = self.feedstock_selector.candidates[feedstock]
        
        # Calculate costs
        input_cost = input_mass * candidate.market_price
        energy_cost = self.calculate_energy_cost(results["total_energy_J"])
        
        # Calculate total operation time (sum of all stages)
        total_time = sum(stage.duration for stage in transmuter.stages)
        operating_cost = (total_time / 3600.0) * self.operating_cost_per_hour
        
        total_cost = input_cost + energy_cost + operating_cost
        
        # Calculate revenue
        rhodium_value = results["rhodium_mass_kg"] * self.rhodium_price
        
        # Calculate profit metrics
        net_profit = rhodium_value - total_cost
        profit_ratio = rhodium_value / (total_cost + 1e-10)
        roi_percent = (net_profit / total_cost) * 100
        
        # Payback time (how long to recoup equipment cost)
        if net_profit > 0:
            runs_per_hour = 3600.0 / total_time
            profit_per_hour = net_profit * runs_per_hour
            payback_time_hours = self.equipment_cost / profit_per_hour
        else:
            payback_time_hours = float('inf')
        
        return EconomicResult(
            feedstock=feedstock,
            beam_type=beam_config["type"],
            beam_energy=beam_config["energy"],
            lv_params=lv_params.copy(),
            input_mass_kg=input_mass,
            input_cost=input_cost,
            rhodium_mass_kg=results["rhodium_mass_kg"],
            rhodium_value=rhodium_value,
            energy_cost=energy_cost + operating_cost,
            net_profit=net_profit,
            profit_ratio=profit_ratio,
            roi_percent=roi_percent,
            payback_time_hours=payback_time_hours
        )
    
    def scan_feedstock_economics(self, input_mass: float = 1e-3) -> List[EconomicResult]:
        """Scan all feedstock options for economic viability."""
        print(f"💰 ECONOMIC VIABILITY SCAN")
        print(f"Input mass: {input_mass*1000:.1f} g per batch")
        print("=" * 40)
        
        # Standard beam and LV configurations
        beam_configs = [
            {"type": "proton", "energy": 100e6},
            {"type": "proton", "energy": 120e6},
            {"type": "proton", "energy": 150e6},
            {"type": "deuteron", "energy": 100e6},
            {"type": "deuteron", "energy": 120e6}
        ]
        
        lv_configs = [
            {"mu_lv": 1e-16, "alpha_lv": 1e-13, "beta_lv": 1e-10},   # Strong
            {"mu_lv": 5e-16, "alpha_lv": 5e-13, "beta_lv": 5e-10},   # Very strong  
            {"mu_lv": 1e-15, "alpha_lv": 1e-12, "beta_lv": 1e-9}     # Extreme
        ]
        
        economic_results = []
        
        for feedstock in self.feedstock_selector.candidates.keys():
            print(f"\n📊 Analyzing {feedstock}...")
            
            best_result = None
            best_profit = -float('inf')
            
            for beam_config in beam_configs:
                for lv_config in lv_configs:
                    try:
                        result = self.run_economic_analysis(
                            feedstock=feedstock,
                            beam_config=beam_config,
                            lv_params=lv_config,
                            input_mass=input_mass
                        )
                        
                        if result.net_profit > best_profit:
                            best_profit = result.net_profit
                            best_result = result
                            
                    except Exception as e:
                        print(f"    Error with {beam_config['type']} {beam_config['energy']/1e6:.0f}MeV: {e}")
                        continue
            
            if best_result:
                economic_results.append(best_result)
                print(f"  Best profit: ${best_result.net_profit:.2f}")
                print(f"  ROI: {best_result.roi_percent:.1f}%")
                print(f"  Payback: {best_result.payback_time_hours:.1f} hours")
        
        return economic_results
    
    def optimize_parameters(self, feedstock: str, sample_count: int = 50) -> Dict[str, Any]:
        """Optimize beam and LV parameters for maximum profit."""
        print(f"\n🔬 PARAMETER OPTIMIZATION FOR {feedstock}")
        print("=" * 40)
        
        # Parameter ranges
        beam_energies = np.linspace(80e6, 200e6, 10)    # 80-200 MeV
        beam_types = ["proton", "deuteron"]
        
        # LV parameter ranges (log space)
        mu_range = np.logspace(-17, -14, 8)   # 1e-17 to 1e-14
        alpha_range = np.logspace(-15, -11, 8) # 1e-15 to 1e-11  
        beta_range = np.logspace(-12, -8, 8)   # 1e-12 to 1e-8
        
        best_result = None
        best_profit = -float('inf')
        optimization_results = []
        
        # Random sampling for efficiency
        for i in range(sample_count):
            # Random parameter selection
            beam_energy = np.random.choice(beam_energies)
            beam_type = np.random.choice(beam_types)
            mu_lv = np.random.choice(mu_range)
            alpha_lv = np.random.choice(alpha_range)
            beta_lv = np.random.choice(beta_range)
            
            beam_config = {"type": beam_type, "energy": beam_energy}
            lv_params = {"mu_lv": mu_lv, "alpha_lv": alpha_lv, "beta_lv": beta_lv}
            
            try:
                result = self.run_economic_analysis(
                    feedstock=feedstock,
                    beam_config=beam_config,
                    lv_params=lv_params,
                    input_mass=1e-3  # 1 g
                )
                
                optimization_results.append(result)
                
                if result.net_profit > best_profit:
                    best_profit = result.net_profit
                    best_result = result
                    
            except Exception as e:
                continue
        
        # Analysis
        profitable_results = [r for r in optimization_results if r.net_profit > 0]
        
        optimization_summary = {
            "feedstock": feedstock,
            "total_samples": sample_count,
            "successful_runs": len(optimization_results),
            "profitable_runs": len(profitable_results),
            "success_rate": len(optimization_results) / sample_count,
            "profitability_rate": len(profitable_results) / len(optimization_results) if optimization_results else 0,
            "best_result": best_result,
            "all_results": optimization_results
        }
        
        if best_result:
            print(f"✅ Best configuration found:")
            print(f"   Beam: {best_result.beam_energy/1e6:.0f} MeV {best_result.beam_type}")
            print(f"   LV params: μ={best_result.lv_params['mu_lv']:.1e}, α={best_result.lv_params['alpha_lv']:.1e}")
            print(f"   Net profit: ${best_result.net_profit:.2f} per gram")
            print(f"   ROI: {best_result.roi_percent:.1f}%")
            print(f"   Payback time: {best_result.payback_time_hours:.1f} hours")
        else:
            print("❌ No profitable configuration found")
        
        return optimization_summary
    
    def generate_profit_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive profit analysis report."""
        print(f"\n📈 COMPREHENSIVE PROFIT ANALYSIS")
        print("=" * 45)
        
        # Run economic scan
        economic_results = self.scan_feedstock_economics(input_mass=1e-3)
        
        # Sort by profitability
        profitable_feedstocks = [r for r in economic_results if r.net_profit > 0]
        profitable_feedstocks.sort(key=lambda x: x.net_profit, reverse=True)
        
        # Optimize parameters for top 3 feedstocks
        optimization_results = {}
        for result in profitable_feedstocks[:3]:
            optimization_results[result.feedstock] = self.optimize_parameters(
                feedstock=result.feedstock,
                sample_count=30
            )
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "cheap_feedstock_profit_optimization",
            "economic_parameters": {
                "rhodium_price_per_kg": self.rhodium_price,
                "electricity_cost_per_kwh": self.electricity_cost,
                "equipment_cost": self.equipment_cost
            },
            "feedstock_economic_ranking": [
                {
                    "feedstock": r.feedstock,
                    "net_profit_per_gram": r.net_profit,
                    "roi_percent": r.roi_percent,
                    "payback_time_hours": r.payback_time_hours,
                    "rhodium_yield_mg_per_g": r.rhodium_mass_kg * 1e6
                }
                for r in profitable_feedstocks
            ],
            "optimization_results": optimization_results,
            "market_viability": len(profitable_feedstocks) > 0
        }
        
        # Print summary
        print(f"\n💎 MARKET VIABILITY SUMMARY")
        print("=" * 35)
        
        if profitable_feedstocks:
            print(f"✅ {len(profitable_feedstocks)} profitable feedstock options found")
            
            top_result = profitable_feedstocks[0]
            print(f"\n🏆 MOST PROFITABLE: {top_result.feedstock}")
            print(f"   Profit: ${top_result.net_profit:.2f} per gram feedstock")
            print(f"   ROI: {top_result.roi_percent:.1f}%")
            print(f"   Rhodium yield: {top_result.rhodium_mass_kg*1e6:.3f} mg/g")
            print(f"   Equipment payback: {top_result.payback_time_hours:.1f} hours")
            
            # Economic projections
            daily_batches = 24 * 3600 / (sum(stage.duration for stage in 
                FeedstockNetworkTransmuter({}, top_result.feedstock, {}).stages))
            daily_profit = top_result.net_profit * daily_batches
            annual_profit = daily_profit * 365
            
            print(f"\n📊 SCALING PROJECTIONS:")
            print(f"   Daily profit potential: ${daily_profit:.2f}")
            print(f"   Annual profit potential: ${annual_profit:.2f}")
            
        else:
            print("❌ No profitable configurations found with current parameters")
            print("💡 Consider: Higher LV enhancement, lower energy costs, or equipment optimization")
        
        return report

def main():
    """Main cheap transmutation economic analysis."""
    print("💰 CHEAP FEEDSTOCK ECONOMIC SCANNER")
    print("=" * 45)
    print("📊 Optimizing profit from low-cost materials → rhodium")
    print("")
    
    scanner = CheapTransmutationScanner()
    
    # Run comprehensive analysis
    report = scanner.generate_profit_analysis_report()
    
    # Save results
    with open("cheap_feedstock_economic_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: cheap_feedstock_economic_analysis.json")
    print("\n✅ ECONOMIC ANALYSIS COMPLETE")
    print("🚀 Ready for experimental implementation planning")
    
    return report

if __name__ == "__main__":
    main()
