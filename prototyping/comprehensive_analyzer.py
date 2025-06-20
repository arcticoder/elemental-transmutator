#!/usr/bin/env python3
"""
Comprehensive Digital Twin Analysis
==================================

Runs sensitivity analysis, pathway optimization, and go/no-go evaluation
to find the most promising photonuclear gold production routes before outsourcing.
"""

import numpy as np
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Any

from digital_twin_optimizer import DigitalTwinOptimizer
from global_sensitivity_analyzer import GlobalSensitivityAnalyzer
from atomic_binder import AtomicDataBinder

class ComprehensiveAnalyzer:
    """Comprehensive analysis combining sensitivity, optimization, and pathway analysis."""
    
    def __init__(self, output_dir: str = "comprehensive_analysis"):
        """Initialize the comprehensive analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis components
        self.optimizer = DigitalTwinOptimizer()
        self.sensitivity_analyzer = GlobalSensitivityAnalyzer()
        self.atomic_data = AtomicDataBinder()
        
        self.logger.info("Comprehensive analyzer initialized")
    
    def run_full_analysis(self, 
                         sensitivity_samples: int = 512,
                         optimization_detailed: bool = True) -> Dict[str, Any]:
        """Run complete digital twin analysis pipeline."""
        
        start_time = time.time()
        self.logger.info("Starting comprehensive digital twin analysis...")
        
        results = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_start_time': start_time
        }
        
        # Step 1: Run sensitivity analysis to identify leverage points
        self.logger.info("Step 1: Global sensitivity analysis...")
        try:
            morris_results = self.sensitivity_analyzer.run_morris_screening(
                n_trajectories=sensitivity_samples // 4
            )
            results['morris_screening'] = self._format_sensitivity_results(morris_results)
            
            # Focus Sobol analysis on top parameters
            top_params = [p[0] for p in morris_results.parameter_rankings[:8]]
            sobol_results = self.sensitivity_analyzer.run_sobol_analysis(
                n_samples=sensitivity_samples, 
                selected_params=top_params
            )
            results['sobol_analysis'] = self._format_sensitivity_results(sobol_results)
            
            # Generate optimization recommendations
            recommendations = self.sensitivity_analyzer.generate_optimization_recommendations(sobol_results)
            results['sensitivity_recommendations'] = recommendations
            
        except ImportError:
            self.logger.warning("SALib not available - skipping sensitivity analysis")
            results['sensitivity_analysis'] = "Skipped - SALib not installed"
        
        # Step 2: Run pathway optimization
        self.logger.info("Step 2: Pathway optimization...")
        optimization_results = self.optimizer.optimize_recipe()
        results['optimization'] = optimization_results
        
        # Step 3: Analyze promising pathways
        self.logger.info("Step 3: Promising pathway analysis...")
        promising_pathways = self.optimizer.find_promising_pathways()
        results['promising_pathways'] = promising_pathways
        
        # Step 4: Go/No-Go decision analysis
        self.logger.info("Step 4: Go/No-Go decision analysis...")
        go_no_go_analysis = self._analyze_go_no_go_decisions(optimization_results)
        results['go_no_go_analysis'] = go_no_go_analysis
        
        # Step 5: Generate comprehensive recommendations
        self.logger.info("Step 5: Generating comprehensive recommendations...")
        final_recommendations = self._generate_final_recommendations(results)
        results['final_recommendations'] = final_recommendations
        
        # Calculate analysis time
        total_time = time.time() - start_time
        results['analysis_duration_s'] = total_time
        
        # Save comprehensive results
        self._save_comprehensive_results(results)
        
        self.logger.info(f"Comprehensive analysis complete in {total_time:.1f}s")
        
        return results
    
    def _format_sensitivity_results(self, sensitivity_results) -> Dict:
        """Format sensitivity results for JSON serialization."""
        return {
            'method': sensitivity_results.method,
            'total_samples': sensitivity_results.total_samples,
            'top_5_parameters': sensitivity_results.parameter_rankings[:5],
            'analysis_time_s': sensitivity_results.analysis_time_s,
            'first_order_indices': sensitivity_results.first_order_indices,
            'total_order_indices': sensitivity_results.total_order_indices
        }
    
    def _analyze_go_no_go_decisions(self, optimization_results: Dict) -> Dict:
        """Analyze go/no-go decisions from optimization results."""
        
        go_configs = optimization_results.get('go_configs', [])
        no_go_configs = optimization_results.get('no_go_configs', [])
        
        analysis = {
            'total_configurations': len(optimization_results.get('all_results', [])),
            'go_decision_count': len(go_configs),
            'no_go_decision_count': len(no_go_configs),
            'go_decision_percentage': (len(go_configs) / len(optimization_results.get('all_results', [1]))) * 100
        }
        
        if go_configs:
            # Analyze GO configurations
            go_yields = [config['go_no_go_evaluation']['yield_mg_per_g'] for config in go_configs]
            go_costs = [config['go_no_go_evaluation']['cost_per_mg_au_cad'] for config in go_configs]
            
            analysis['go_statistics'] = {
                'best_yield_mg_per_g': max(go_yields),
                'median_yield_mg_per_g': np.median(go_yields),
                'min_cost_per_mg_au': min(go_costs),
                'median_cost_per_mg_au': np.median(go_costs),
                'recommended_for_outsourcing': True
            }
            
            # Find best GO configuration
            best_go = max(go_configs, key=lambda x: x['yield_per_cad'])
            analysis['best_go_config'] = {
                'energy_mev': best_go['energy_mev'],
                'dose_kgy': best_go['dose_kgy'],
                'composition': best_go['composition'],
                'yield_mg_per_g': best_go['go_no_go_evaluation']['yield_mg_per_g'],
                'cost_per_mg_au': best_go['go_no_go_evaluation']['cost_per_mg_au_cad']
            }
        else:
            analysis['go_statistics'] = {
                'recommended_for_outsourcing': False,
                'reason': 'No configurations meet go/no-go thresholds'
            }
        
        # Analyze why NO-GO configs failed
        if no_go_configs:
            yield_failures = [c for c in no_go_configs if not c['go_no_go_evaluation']['yield_meets_threshold']]
            cost_failures = [c for c in no_go_configs if not c['go_no_go_evaluation']['cost_meets_threshold']]
            
            analysis['no_go_analysis'] = {
                'yield_threshold_failures': len(yield_failures),
                'cost_threshold_failures': len(cost_failures),
                'improvement_strategies': self._suggest_improvement_strategies(yield_failures, cost_failures)
            }
        
        return analysis
    
    def _suggest_improvement_strategies(self, yield_failures: List, cost_failures: List) -> List[str]:
        """Suggest strategies to improve failed configurations."""
        
        strategies = []
        
        if yield_failures:
            # Analyze yield failure patterns
            low_yield_compositions = {}
            for config in yield_failures[:20]:  # Sample to avoid overwhelming analysis
                for isotope, fraction in config['composition'].items():
                    if isotope not in low_yield_compositions:
                        low_yield_compositions[isotope] = []
                    low_yield_compositions[isotope].append(fraction)
            
            # Find isotopes associated with low yields
            problematic_isotopes = []
            for isotope, fractions in low_yield_compositions.items():
                if len(fractions) > 5 and np.mean(fractions) > 0.5:  # High fraction, low yield
                    problematic_isotopes.append(isotope)
            
            if 'Pb-208' in problematic_isotopes:
                strategies.append("Pb-208 shows poor yield - consider alternative heavy targets")
            
            if not any(iso in ['Tl-203', 'Hg-202'] for iso in low_yield_compositions):
                strategies.append("Add high-efficiency isotopes (Tl-203, Hg-202) to target mix")
            
            strategies.append("Explore two-step photoneutron processes with Be-9 or D-2")
            strategies.append("Optimize beam energy for specific isotope giant dipole resonances")
        
        if cost_failures:
            strategies.append("Reduce dose requirements through higher cross-section isotopes")
            strategies.append("Optimize beam energy to minimize facility costs")
            strategies.append("Consider pulsed beam modes for enhanced efficiency")
        
        return strategies
    
    def _generate_final_recommendations(self, results: Dict) -> Dict:
        """Generate final recommendations based on comprehensive analysis."""
        
        recommendations = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_assessment': '',
            'recommended_action': '',
            'next_steps': [],
            'risk_factors': [],
            'optimization_opportunities': []
        }
        
        # Assess overall results
        go_no_go = results.get('go_no_go_analysis', {})
        has_go_configs = go_no_go.get('go_decision_count', 0) > 0
        promising_pathways = results.get('promising_pathways', [])
        
        if has_go_configs:
            recommendations['overall_assessment'] = "✅ PROMISING - Digital twin identified viable pathways"
            recommendations['recommended_action'] = "PROCEED with vendor engagement using optimized recipes"
            
            best_config = go_no_go.get('best_go_config', {})
            recommendations['next_steps'] = [
                f"Generate vendor specs for: {best_config.get('energy_mev', '?')} MeV, {best_config.get('dose_kgy', '?')} kGy",
                "Request quotes from 3-5 photonuclear facilities",
                "Plan micro-batch validation (1-5g scale)",
                f"Target composition: {best_config.get('composition', {})}"
            ]
            
            recommendations['optimization_opportunities'] = [
                "Fine-tune beam energy around GDR peaks",
                "Explore pulsed beam modes for enhanced efficiency",
                "Optimize target geometry and moderation"
            ]
            
        elif promising_pathways:
            recommendations['overall_assessment'] = "⚠️ MARGINAL - Some pathways show promise below threshold"
            recommendations['recommended_action'] = "EXPAND digital twin before outsourcing"
            
            recommendations['next_steps'] = [
                "Focus optimization on top 3 promising pathways",
                "Run detailed Geant4 simulations for validation",
                "Explore alternative isotope enrichment",
                "Consider two-step photoneutron enhancement"
            ]
            
            recommendations['optimization_opportunities'] = [
                "Two-step processes with Be-9 neutron sources",
                "Isotope-enriched targets (Tl-203, Hg-202)",
                "Advanced beam profiles (femtosecond pulses)"
            ]
            
        else:
            recommendations['overall_assessment'] = "❌ NOT VIABLE - No pathways meet economic thresholds"
            recommendations['recommended_action'] = "DO NOT OUTSOURCE - Fundamental limitations identified"
            
            recommendations['next_steps'] = [
                "Explore alternative transmutation approaches",
                "Consider different target elements (Pt, Ir)",
                "Investigate laser-driven acceleration",
                "Re-evaluate economic assumptions"
            ]
            
            recommendations['risk_factors'] = [
                "Cross-sections may be overestimated",
                "Facility enhancement factors uncertain",
                "Competing processes not fully modeled"
            ]
        
        # Add sensitivity-based recommendations if available
        if 'sensitivity_recommendations' in results:
            sens_rec = results['sensitivity_recommendations']
            if 'optimization_strategy' in sens_rec:
                recommendations['optimization_opportunities'].extend(sens_rec['optimization_strategy'])
        
        return recommendations
    
    def _save_comprehensive_results(self, results: Dict):
        """Save comprehensive analysis results."""
        
        # Save full results as JSON
        output_file = self.output_dir / "comprehensive_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save executive summary
        self._generate_executive_summary(results)
        
        # Generate visualization if possible
        try:
            self._generate_analysis_plots(results)
        except Exception as e:
            self.logger.warning(f"Could not generate plots: {e}")
        
        self.logger.info(f"Results saved to {output_file}")
    
    def _generate_executive_summary(self, results: Dict):
        """Generate executive summary document."""
        
        summary_file = self.output_dir / "executive_summary.md"
        
        final_rec = results.get('final_recommendations', {})
        go_no_go = results.get('go_no_go_analysis', {})
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Photonuclear Gold Production - Digital Twin Analysis\n\n")
            f.write(f"**Analysis Date:** {results.get('analysis_timestamp', 'Unknown')}\n\n")
            f.write(f"**Analysis Duration:** {results.get('analysis_duration_s', 0):.1f} seconds\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"**Overall Assessment:** {final_rec.get('overall_assessment', 'Unknown')}\n\n")
            f.write(f"**Recommended Action:** {final_rec.get('recommended_action', 'Unknown')}\n\n")
            
            f.write("## Key Findings\n\n")
            f.write(f"- **Total Configurations Tested:** {go_no_go.get('total_configurations', 0):,}\n")
            f.write(f"- **GO Decisions:** {go_no_go.get('go_decision_count', 0)} ({go_no_go.get('go_decision_percentage', 0):.1f}%)\n")
            f.write(f"- **NO-GO Decisions:** {go_no_go.get('no_go_decision_count', 0)}\n\n")
            
            if 'best_go_config' in go_no_go:
                best = go_no_go['best_go_config']
                f.write("## Best Configuration (If Proceeding)\n\n")
                f.write(f"- **Beam Energy:** {best.get('energy_mev', '?')} MeV\n")
                f.write(f"- **Dose:** {best.get('dose_kgy', '?')} kGy\n")
                f.write(f"- **Target Composition:** {best.get('composition', {})}\n")
                f.write(f"- **Predicted Yield:** {best.get('yield_mg_per_g', '?')} mg Au per g feedstock\n")
                f.write(f"- **Estimated Cost:** ${best.get('cost_per_mg_au', '?'):.2f} CAD per mg Au\n\n")
            
            f.write("## Next Steps\n\n")
            for step in final_rec.get('next_steps', []):
                f.write(f"1. {step}\n")
            
            f.write("\n## Risk Factors\n\n")
            for risk in final_rec.get('risk_factors', ['Standard technical and economic risks']):
                f.write(f"- {risk}\n")
            
            f.write("\n## Optimization Opportunities\n\n")
            for opp in final_rec.get('optimization_opportunities', []):
                f.write(f"- {opp}\n")
        
        self.logger.info(f"Executive summary saved to {summary_file}")
    
    def _generate_analysis_plots(self, results: Dict):
        """Generate analysis visualization plots."""
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Digital Twin Analysis Results', fontsize=16)
        
        # Plot 1: Go/No-Go distribution
        go_count = results.get('go_no_go_analysis', {}).get('go_decision_count', 0)
        no_go_count = results.get('go_no_go_analysis', {}).get('no_go_decision_count', 0)
        
        axes[0, 0].pie([go_count, no_go_count], labels=['GO', 'NO-GO'], 
                      colors=['green', 'red'], autopct='%1.1f%%')
        axes[0, 0].set_title('Go/No-Go Decision Distribution')
        
        # Plot 2: Promising pathways
        pathways = results.get('promising_pathways', [])
        if pathways:
            pathway_names = [p.get('isotope', str(p.get('composition', {}))) for p in pathways[:5]]
            pathway_yields = [p.get('predicted_yield_mg_per_g', 0) for p in pathways[:5]]
            
            axes[0, 1].bar(range(len(pathway_names)), pathway_yields)
            axes[0, 1].set_xticks(range(len(pathway_names)))
            axes[0, 1].set_xticklabels(pathway_names, rotation=45)
            axes[0, 1].set_ylabel('Yield (mg Au/g)')
            axes[0, 1].set_title('Top Promising Pathways')
        else:
            axes[0, 1].text(0.5, 0.5, 'No promising\npathways found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Promising Pathways')
        
        # Plot 3: Sensitivity analysis (if available)
        if 'sobol_analysis' in results:
            sobol = results['sobol_analysis']
            if 'top_5_parameters' in sobol:
                params = [p[0] for p in sobol['top_5_parameters']]
                sensitivities = [p[1] for p in sobol['top_5_parameters']]
                
                axes[1, 0].barh(range(len(params)), sensitivities)
                axes[1, 0].set_yticks(range(len(params)))
                axes[1, 0].set_yticklabels(params)
                axes[1, 0].set_xlabel('Sensitivity Index')
                axes[1, 0].set_title('Top Parameter Sensitivities')
            else:
                axes[1, 0].text(0.5, 0.5, 'Sensitivity analysis\nnot available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            axes[1, 0].text(0.5, 0.5, 'Sensitivity analysis\nskipped', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Plot 4: Cost vs Yield scatter (if GO configs available)
        optimization = results.get('optimization', {})
        go_configs = optimization.get('go_configs', [])
        
        if go_configs:
            yields = [c['go_no_go_evaluation']['yield_mg_per_g'] for c in go_configs]
            costs = [c['go_no_go_evaluation']['cost_per_mg_au_cad'] for c in go_configs]
            
            axes[1, 1].scatter(yields, costs, alpha=0.6, c='green')
            axes[1, 1].set_xlabel('Yield (mg Au/g)')
            axes[1, 1].set_ylabel('Cost (CAD/mg Au)')
            axes[1, 1].set_title('GO Configurations: Cost vs Yield')
            
            # Add threshold lines
            axes[1, 1].axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Yield threshold')
            axes[1, 1].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Cost threshold')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'No GO configurations\nto plot', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Cost vs Yield Analysis')
        
        plt.tight_layout()
        plot_file = self.output_dir / "analysis_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Analysis plots saved to {plot_file}")

def main():
    """Run comprehensive digital twin analysis."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    analyzer = ComprehensiveAnalyzer()
    
    print("🔬 Starting Comprehensive Digital Twin Analysis...")
    print("=" * 60)
    
    # Run full analysis
    results = analyzer.run_full_analysis(
        sensitivity_samples=1024,  # Moderate sample size for balance of speed vs accuracy
        optimization_detailed=True
    )
    
    # Print executive summary
    final_rec = results.get('final_recommendations', {})
    print(f"\n📊 ANALYSIS COMPLETE")
    print(f"Overall Assessment: {final_rec.get('overall_assessment', 'Unknown')}")
    print(f"Recommended Action: {final_rec.get('recommended_action', 'Unknown')}")
    
    go_no_go = results.get('go_no_go_analysis', {})
    print(f"\n📈 KEY METRICS:")
    print(f"  GO Decisions: {go_no_go.get('go_decision_count', 0)}")
    print(f"  NO-GO Decisions: {go_no_go.get('no_go_decision_count', 0)}")
    print(f"  Promising Pathways: {len(results.get('promising_pathways', []))}")
    
    if 'best_go_config' in go_no_go:
        best = go_no_go['best_go_config']
        print(f"\n🎯 BEST CONFIGURATION:")
        print(f"  Energy: {best.get('energy_mev', '?')} MeV")
        print(f"  Dose: {best.get('dose_kgy', '?')} kGy")
        print(f"  Yield: {best.get('yield_mg_per_g', '?')} mg Au/g")
        print(f"  Cost: ${best.get('cost_per_mg_au', '?'):.2f} CAD/mg Au")
    
    print(f"\n📋 NEXT STEPS:")
    for step in final_rec.get('next_steps', []):
        print(f"  • {step}")
    
    print(f"\n📁 Results saved to: comprehensive_analysis/")
    print(f"   - comprehensive_analysis_results.json")
    print(f"   - executive_summary.md")
    print(f"   - analysis_plots.png")
    
    return 0

if __name__ == '__main__':
    exit(main())
