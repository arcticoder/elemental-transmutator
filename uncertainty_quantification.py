#!/usr/bin/env python3
"""
Uncertainty Quantification for Cheap Feedstock Rhodium Replicator
================================================================

Monte Carlo analysis of parameter uncertainties and their impact on
profitability, yield, and operational risk. Essential for pilot-scale
engineering and investment risk assessment.
"""

import numpy as np
import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cheap_feedstock_network import FeedstockNetworkTransmuter
from feedstock_selector import FeedstockSelector

@dataclass
class UncertaintyParams:
    """Parameter uncertainty ranges for Monte Carlo analysis."""
    
    # Cross-section uncertainties (±%)
    fragmentation_cs_error: float = 30.0  # Literature spread ±30%
    precursor_cs_error: float = 40.0      # Mid-mass region ±40%
    rhodium_cs_error: float = 25.0        # Well-known region ±25%
    
    # LV parameter uncertainties
    mu_lv_factor_range: Tuple[float, float] = (0.1, 10.0)    # Order of magnitude
    alpha_lv_factor_range: Tuple[float, float] = (0.2, 5.0)  # Theory uncertainty
    beta_lv_factor_range: Tuple[float, float] = (0.5, 2.0)   # Constrained by tests
    
    # Beam parameter uncertainties
    beam_energy_stability: float = 2.0     # ±2% energy stability
    beam_current_stability: float = 5.0    # ±5% current fluctuations
    beam_uptime: float = 0.15              # ±15% downtime variations
    
    # Target and separation uncertainties
    target_purity: float = 0.05            # ±5% feedstock purity
    separation_efficiency_error: float = 0.10  # ±10% chemical yield
    
    # Economic uncertainties
    rhodium_price_volatility: float = 0.30  # ±30% price swings
    feedstock_cost_variation: float = 0.20  # ±20% iron price variation
    energy_cost_variation: float = 0.15     # ±15% electricity cost

class UncertaintyAnalyzer:
    """Monte Carlo uncertainty and sensitivity analysis."""
    
    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples
        self.params = UncertaintyParams()
        self.results = []
        
    def generate_parameter_samples(self) -> List[Dict]:
        """Generate Monte Carlo parameter samples."""
        print(f"🎲 Generating {self.n_samples} Monte Carlo samples...")
        
        samples = []
        
        for i in range(self.n_samples):
            # Cross-section variations (log-normal distribution)
            cs_frag_factor = np.random.lognormal(0, self.params.fragmentation_cs_error/100)
            cs_prec_factor = np.random.lognormal(0, self.params.precursor_cs_error/100)
            cs_rh_factor = np.random.lognormal(0, self.params.rhodium_cs_error/100)
            
            # LV parameter variations (uniform in log space)
            mu_factor = np.random.uniform(*self.params.mu_lv_factor_range)
            alpha_factor = np.random.uniform(*self.params.alpha_lv_factor_range)
            beta_factor = np.random.uniform(*self.params.beta_lv_factor_range)
            
            # Beam parameter variations (normal distribution)
            energy_factor = np.random.normal(1.0, self.params.beam_energy_stability/100)
            current_factor = np.random.normal(1.0, self.params.beam_current_stability/100)
            uptime_factor = np.random.normal(1.0, self.params.beam_uptime)
            
            # Target and separation variations
            purity_factor = np.random.normal(1.0, self.params.target_purity)
            separation_factor = np.random.normal(1.0, self.params.separation_efficiency_error)
            
            # Economic variations
            rh_price_factor = np.random.lognormal(0, self.params.rhodium_price_volatility)
            feedstock_cost_factor = np.random.lognormal(0, self.params.feedstock_cost_variation)
            energy_cost_factor = np.random.lognormal(0, self.params.energy_cost_variation)
            
            sample = {
                'cs_fragmentation': cs_frag_factor,
                'cs_precursor': cs_prec_factor,
                'cs_rhodium': cs_rh_factor,
                'mu_lv_factor': mu_factor,
                'alpha_lv_factor': alpha_factor,
                'beta_lv_factor': beta_factor,
                'beam_energy_factor': energy_factor,
                'beam_current_factor': current_factor,
                'beam_uptime_factor': uptime_factor,
                'target_purity_factor': purity_factor,
                'separation_efficiency_factor': separation_factor,
                'rhodium_price_factor': rh_price_factor,
                'feedstock_cost_factor': feedstock_cost_factor,
                'energy_cost_factor': energy_cost_factor
            }
            
            samples.append(sample)
            
        return samples
    
    def run_monte_carlo_simulation(self, feedstock: str = "Fe-56") -> pd.DataFrame:
        """Run Monte Carlo simulation with parameter variations."""
        print(f"🔬 Running Monte Carlo simulation for {feedstock}...")
        
        samples = self.generate_parameter_samples()
        results = []
        
        # Baseline parameters
        baseline_lv = {"mu_lv": 1e-15, "alpha_lv": 1e-12, "beta_lv": 1e-9}
        baseline_beam = {
            "stage_a": {"type": "proton", "energy": 120e6},
            "stage_b": {"type": "deuteron", "energy": 100e6},
            "stage_c": {"type": "deuteron", "energy": 80e6}
        }
        
        for i, sample in enumerate(samples):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{self.n_samples} ({100*i/self.n_samples:.1f}%)")
            
            try:
                # Apply parameter variations
                varied_lv = {
                    "mu_lv": baseline_lv["mu_lv"] * sample["mu_lv_factor"],
                    "alpha_lv": baseline_lv["alpha_lv"] * sample["alpha_lv_factor"],
                    "beta_lv": baseline_lv["beta_lv"] * sample["beta_lv_factor"]
                }
                
                varied_beam = {}
                for stage, params in baseline_beam.items():
                    varied_beam[stage] = {
                        "type": params["type"],
                        "energy": params["energy"] * sample["beam_energy_factor"]
                    }
                
                # Run transmutation simulation
                transmuter = FeedstockNetworkTransmuter(varied_lv, feedstock, varied_beam)
                input_mass = 0.001 * sample["target_purity_factor"]  # 1 mg varied by purity
                
                result = transmuter.full_chain(input_mass)
                
                # Apply separation efficiency and beam variations
                rhodium_yield = (result["rhodium_mass_kg"] * 
                               sample["separation_efficiency_factor"] * 
                               sample["beam_current_factor"] * 
                               sample["beam_uptime_factor"])
                
                # Economic calculations
                rhodium_price = 145000 * sample["rhodium_price_factor"]  # $/kg
                feedstock_cost = 0.12 * sample["feedstock_cost_factor"]  # $/kg Fe-56
                energy_cost = result.get("energy_cost", 0.001) * sample["energy_cost_factor"]  # $/kWh
                
                revenue = rhodium_yield * rhodium_price
                input_cost = input_mass * feedstock_cost
                operating_cost = energy_cost
                profit = revenue - input_cost - operating_cost
                
                roi = (profit / (input_cost + operating_cost + 1e-10)) * 100 if (input_cost + operating_cost) > 0 else 0
                
                # Store results
                result_entry = {
                    'sample_id': i,
                    'rhodium_yield_kg': rhodium_yield,
                    'rhodium_yield_mg': rhodium_yield * 1e6,
                    'revenue_usd': revenue,
                    'input_cost_usd': input_cost,
                    'operating_cost_usd': operating_cost,
                    'profit_usd': profit,
                    'roi_percent': roi,
                    'lv_enhancement': result.get("lv_enhancement", 1.0),
                    'energy_efficiency': result.get("energy_efficiency", 0),
                    **sample  # Include all parameter variations
                }
                
                results.append(result_entry)
                
            except Exception as e:
                # Handle simulation failures gracefully
                results.append({
                    'sample_id': i,
                    'rhodium_yield_kg': 0,
                    'profit_usd': -1000,  # Large loss for failed runs
                    'roi_percent': -100,
                    'error': str(e),
                    **sample
                })
        
        print(f"✅ Monte Carlo simulation complete!")
        return pd.DataFrame(results)
    
    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """Analyze Monte Carlo results and generate statistics."""
        print(f"\n📊 UNCERTAINTY ANALYSIS RESULTS")
        print("=" * 50)
        
        # Filter out failed runs
        valid_runs = df[df['rhodium_yield_kg'] > 0]
        failure_rate = (len(df) - len(valid_runs)) / len(df) * 100
        
        print(f"Success rate: {100-failure_rate:.1f}% ({len(valid_runs)}/{len(df)} runs)")
        
        if len(valid_runs) == 0:
            print("❌ All simulations failed - check parameter ranges!")
            return {}
        
        # Statistical analysis
        stats_results = {}
        
        for metric in ['rhodium_yield_mg', 'profit_usd', 'roi_percent']:
            data = valid_runs[metric]
            stats_results[metric] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'p5': data.quantile(0.05),
                'p95': data.quantile(0.95),
                'cv': data.std() / data.mean() if data.mean() != 0 else 0
            }
        
        # Print key statistics
        print(f"\n💎 RHODIUM YIELD (mg per mg Fe-56):")
        rh_stats = stats_results['rhodium_yield_mg']
        print(f"  Mean: {rh_stats['mean']:.2e} ± {rh_stats['std']:.2e}")
        print(f"  Median: {rh_stats['median']:.2e}")
        print(f"  90% CI: [{rh_stats['p5']:.2e}, {rh_stats['p95']:.2e}]")
        print(f"  Coefficient of variation: {rh_stats['cv']:.1%}")
        
        print(f"\n💰 PROFIT (USD per mg Fe-56):")
        profit_stats = stats_results['profit_usd']
        print(f"  Mean: ${profit_stats['mean']:.2e} ± ${profit_stats['std']:.2e}")
        print(f"  Median: ${profit_stats['median']:.2e}")
        print(f"  90% CI: [${profit_stats['p5']:.2e}, ${profit_stats['p95']:.2e}]")
        
        print(f"\n📈 ROI (%):")
        roi_stats = stats_results['roi_percent']
        print(f"  Mean: {roi_stats['mean']:.1e}% ± {roi_stats['std']:.1e}%")
        print(f"  Median: {roi_stats['median']:.1e}%")
        print(f"  90% CI: [{roi_stats['p5']:.1e}%, {roi_stats['p95']:.1e}%]")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        profit_positive = (valid_runs['profit_usd'] > 0).mean() * 100
        high_yield = (valid_runs['rhodium_yield_mg'] > 1e20).mean() * 100  # Arbitrary high threshold
        
        print(f"  Probability of profit: {profit_positive:.1f}%")
        print(f"  Probability of high yield: {high_yield:.1f}%")
        print(f"  Parameter sensitivity needed: {'HIGH' if rh_stats['cv'] > 1.0 else 'MODERATE' if rh_stats['cv'] > 0.5 else 'LOW'}")
        
        return stats_results
    
    def sensitivity_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform sensitivity analysis to identify critical parameters."""
        print(f"\n🎯 SENSITIVITY ANALYSIS")
        print("=" * 40)
        
        # Parameter columns to analyze
        param_cols = [col for col in df.columns if '_factor' in col or 'cs_' in col]
        
        # Calculate correlations with key outputs
        target_metrics = ['rhodium_yield_mg', 'profit_usd', 'roi_percent']
        sensitivities = {}
        
        for metric in target_metrics:
            sensitivities[metric] = {}
            
            for param in param_cols:
                if param in df.columns and metric in df.columns:
                    correlation = df[param].corr(df[metric])
                    sensitivities[metric][param] = abs(correlation)
        
        # Print top sensitivities
        for metric in target_metrics:
            print(f"\n📊 {metric.upper()} sensitivity:")
            sorted_params = sorted(sensitivities[metric].items(), 
                                 key=lambda x: x[1], reverse=True)
            
            for param, sensitivity in sorted_params[:5]:
                param_name = param.replace('_factor', '').replace('_', ' ').title()
                print(f"  {param_name}: {sensitivity:.3f}")
        
        return sensitivities
    
    def generate_report(self, df: pd.DataFrame, output_file: str = "uncertainty_analysis_report.json"):
        """Generate comprehensive uncertainty analysis report."""
        stats_results = self.analyze_results(df)
        sensitivities = self.sensitivity_analysis(df)
        
        # Compile report
        report = {
            "analysis_date": pd.Timestamp.now().isoformat(),
            "n_samples": self.n_samples,
            "success_rate": (df['rhodium_yield_kg'] > 0).mean() * 100,
            "statistics": stats_results,
            "sensitivities": sensitivities,
            "critical_parameters": self._identify_critical_parameters(sensitivities),
            "risk_assessment": self._assess_risks(df),
            "recommendations": self._generate_recommendations(stats_results, sensitivities)
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Report saved to: {output_file}")
        return report
    
    def _identify_critical_parameters(self, sensitivities: Dict) -> List[str]:
        """Identify parameters that most strongly affect outcomes."""
        all_sensitivities = {}
        
        for metric, params in sensitivities.items():
            for param, sensitivity in params.items():
                if param not in all_sensitivities:
                    all_sensitivities[param] = []
                all_sensitivities[param].append(sensitivity)
        
        # Average sensitivity across all metrics
        avg_sensitivities = {param: np.mean(vals) 
                           for param, vals in all_sensitivities.items()}
        
        # Return top 5 most critical parameters
        critical = sorted(avg_sensitivities.items(), 
                         key=lambda x: x[1], reverse=True)[:5]
        
        return [param for param, _ in critical]
    
    def _assess_risks(self, df: pd.DataFrame) -> Dict:
        """Assess key operational and financial risks."""
        valid_runs = df[df['rhodium_yield_kg'] > 0]
        
        if len(valid_runs) == 0:
            return {"status": "CRITICAL", "message": "No successful runs"}
        
        risks = {
            "technical_failure_rate": (len(df) - len(valid_runs)) / len(df),
            "yield_variability": valid_runs['rhodium_yield_mg'].std() / valid_runs['rhodium_yield_mg'].mean(),
            "profit_at_risk_5pct": valid_runs['profit_usd'].quantile(0.05),
            "break_even_probability": (valid_runs['profit_usd'] > 0).mean(),
            "extreme_loss_probability": (valid_runs['profit_usd'] < -1000).mean()
        }
        
        return risks
    
    def _generate_recommendations(self, stats: Dict, sensitivities: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Yield variability check
        if 'rhodium_yield_mg' in stats:
            cv = stats['rhodium_yield_mg']['cv']
            if cv > 1.0:
                recommendations.append("HIGH PRIORITY: Implement tight process control - yield variability >100%")
            elif cv > 0.5:
                recommendations.append("MEDIUM PRIORITY: Enhanced monitoring needed - yield variability >50%")
        
        # Critical parameter identification
        critical_params = self._identify_critical_parameters(sensitivities)
        if 'mu_lv_factor' in critical_params[:3]:
            recommendations.append("CRITICAL: LV field control must be ±1% or better")
        if 'beam_energy_factor' in critical_params[:3]:
            recommendations.append("CRITICAL: Beam energy stability must be ±0.5% or better")
        if 'cs_rhodium' in critical_params[:3]:
            recommendations.append("URGENT: Validate rhodium production cross-sections experimentally")
        
        # Economic stability
        if 'rhodium_price_factor' in critical_params[:2]:
            recommendations.append("STRATEGY: Consider rhodium futures hedging to manage price risk")
        
        return recommendations

def main():
    """Main uncertainty quantification analysis."""
    print("🎲 CHEAP FEEDSTOCK RHODIUM REPLICATOR")
    print("    UNCERTAINTY QUANTIFICATION ANALYSIS")
    print("=" * 55)
    
    # Initialize analyzer
    analyzer = UncertaintyAnalyzer(n_samples=5000)  # Reduced for speed
    
    # Run Monte Carlo simulation
    results_df = analyzer.run_monte_carlo_simulation("Fe-56")
    
    # Generate comprehensive report
    report = analyzer.generate_report(results_df)
    
    # Print summary conclusions
    print(f"\n🎯 UNCERTAINTY ANALYSIS CONCLUSIONS:")
    print("=" * 45)
    
    if report.get("success_rate", 0) > 90:
        print("✅ ROBUST: Process shows >90% success rate under uncertainty")
    elif report.get("success_rate", 0) > 70:
        print("⚠️ MODERATE: Process needs optimization for reliability")
    else:
        print("❌ HIGH RISK: Process requires major improvements")
    
    # Critical parameter warnings
    critical = report.get("critical_parameters", [])
    if critical:
        print(f"\n🚨 CRITICAL CONTROL PARAMETERS:")
        for param in critical[:3]:
            param_name = param.replace('_factor', '').replace('_', ' ').title()
            print(f"  • {param_name}")
    
    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\n📋 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n💾 Detailed results saved to uncertainty_analysis_report.json")
    print(f"🚀 Ready for pilot plant risk assessment!")
    
    return report

if __name__ == "__main__":
    main()
