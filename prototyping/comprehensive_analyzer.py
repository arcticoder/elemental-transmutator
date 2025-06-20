#!/usr/bin/env python3
"""
Enhanced Comprehensive Digital Twin Analysis
============================================

Extended analysis with new transmutation pathways including:
- Bi-209 → Au-197 routes
- Pt-195 → Au-197 chains  
- Ir-191 → Au-197 pathways
- Two-stage neutron capture sequences
- Pulsed beam nonlinear enhancement modeling
"""

import numpy as np
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Dict, List, Any, Tuple
import seaborn as sns

from digital_twin_optimizer import DigitalTwinOptimizer
from global_sensitivity_analyzer import GlobalSensitivityAnalyzer
from atomic_binder import EnhancedAtomicDataBinder

class EnhancedComprehensiveAnalyzer:
    """Enhanced comprehensive analysis with new transmutation pathways."""
    
    def __init__(self, output_dir: str = "enhanced_comprehensive_analysis"):
        """Initialize the enhanced comprehensive analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced analysis components
        self.optimizer = DigitalTwinOptimizer()
        self.sensitivity_analyzer = GlobalSensitivityAnalyzer()
        self.atomic_data = EnhancedAtomicDataBinder()
        
        # Economic thresholds for viability
        self.economic_thresholds = {
            'min_conversion_mg_per_g': 0.1,
            'max_cost_per_g_cad': 0.01,
            'min_profit_margin': 0.05,
            'min_economic_fom': 0.1
        }
        
        self.logger.info("Enhanced comprehensive analyzer initialized")
    
    def run_enhanced_analysis(self, 
                            sensitivity_samples: int = 1024,
                            optimization_detailed: bool = True,
                            include_pulsed_beams: bool = True) -> Dict[str, Any]:
        """Run complete enhanced digital twin analysis pipeline."""
        
        start_time = time.time()
        self.logger.info("Starting enhanced comprehensive digital twin analysis...")
        
        results = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_start_time': start_time,
            'economic_thresholds': self.economic_thresholds
        }
        
        # Step 1: Load and analyze all enhanced pathways
        self.logger.info("Step 1: Loading enhanced transmutation pathways...")
        pathways = self.atomic_data.load_enhanced_pathways()
        results['total_pathways_analyzed'] = len(pathways)
        
        # Step 2: Rank pathways by economic viability
        self.logger.info("Step 2: Ranking pathways by economic metrics...")
        ranked_pathways = self.atomic_data.rank_pathways_by_economics()
        results['pathway_rankings'] = self._format_pathway_rankings(ranked_pathways)
        
        # Step 3: Identify viable candidates above threshold
        viable_pathways = self._identify_viable_pathways(ranked_pathways)
        results['viable_pathways'] = viable_pathways
        results['viable_count'] = len(viable_pathways)
        
        self.logger.info(f"Found {len(viable_pathways)} viable pathways above economic thresholds")
        
        # Step 4: Enhanced sensitivity analysis on top candidates
        if viable_pathways:
            self.logger.info("Step 4: Running enhanced sensitivity analysis on viable pathways...")
            sensitivity_results = self._run_enhanced_sensitivity_analysis(
                viable_pathways, sensitivity_samples, include_pulsed_beams
            )
            results['sensitivity_analysis'] = sensitivity_results
        else:
            self.logger.warning("No viable pathways found - running sensitivity on top 3 candidates")
            top_3_pathways = {name: data for name, data in ranked_pathways[:3]}
            sensitivity_results = self._run_enhanced_sensitivity_analysis(
                top_3_pathways, sensitivity_samples, include_pulsed_beams
            )
            results['sensitivity_analysis'] = sensitivity_results
        
        # Step 5: Multi-pathway optimization
        self.logger.info("Step 5: Running multi-pathway optimization...")
        optimization_results = self._run_multi_pathway_optimization(ranked_pathways[:10])
        results['optimization_results'] = optimization_results
        
        # Step 6: Pulsed beam enhancement analysis
        if include_pulsed_beams:
            self.logger.info("Step 6: Analyzing pulsed beam enhancements...")
            pulsed_beam_analysis = self._analyze_pulsed_beam_enhancements(ranked_pathways[:5])
            results['pulsed_beam_analysis'] = pulsed_beam_analysis
        
        # Step 7: Generate go/no-go recommendation
        self.logger.info("Step 7: Generating go/no-go recommendations...")
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations
        
        # Step 8: Create visualization suite
        self.logger.info("Step 8: Creating enhanced visualizations...")
        self._create_enhanced_visualizations(results)
        
        # Step 9: Export comprehensive report
        results['analysis_duration_s'] = time.time() - start_time
        self._export_comprehensive_report(results)
        
        self.logger.info(f"Enhanced comprehensive analysis completed in {results['analysis_duration_s']:.1f}s")
        return results
    
    def _identify_viable_pathways(self, ranked_pathways: List[Tuple[str, Dict]]) -> Dict[str, Dict]:
        """Identify pathways that meet economic viability thresholds."""
        
        viable = {}
        
        for name, data in ranked_pathways:
            economics = data['economics']
            
            # Check all viability criteria
            meets_conversion = economics['conversion_mg_per_g'] >= self.economic_thresholds['min_conversion_mg_per_g']
            meets_cost = economics['total_cost_per_g'] <= self.economic_thresholds['max_cost_per_g_cad']
            meets_margin = economics['profit_margin'] >= self.economic_thresholds['min_profit_margin']
            meets_fom = economics['economic_fom'] >= self.economic_thresholds['min_economic_fom']
            
            if meets_conversion and meets_cost and meets_margin and meets_fom:
                viable[name] = data
                viable[name]['viability_score'] = self._calculate_viability_score(economics)
        
        return viable
    
    def _calculate_viability_score(self, economics: Dict) -> float:
        """Calculate composite viability score (0-100)."""
        
        # Normalize metrics to 0-100 scale
        conversion_score = min(100, economics['conversion_mg_per_g'] / 1.0 * 100)  # 1 mg/g = 100
        cost_score = max(0, 100 - economics['total_cost_per_g'] * 1000)  # Lower cost = higher score
        margin_score = min(100, economics['profit_margin'] * 100)
        fom_score = min(100, economics['economic_fom'] / 10.0 * 100)  # 10 FOM = 100
        
        # Weighted composite score
        weights = {'conversion': 0.3, 'cost': 0.3, 'margin': 0.2, 'fom': 0.2}
        
        composite_score = (
            conversion_score * weights['conversion'] +
            cost_score * weights['cost'] +
            margin_score * weights['margin'] +
            fom_score * weights['fom']
        )
        
        return composite_score
    
    def _run_enhanced_sensitivity_analysis(self, pathways: Dict[str, Dict], 
                                         samples: int, include_pulsed: bool) -> Dict:
        """Run enhanced sensitivity analysis on selected pathways."""
        
        sensitivity_results = {}
        
        for pathway_name, pathway_data in pathways.items():
            self.logger.info(f"Running sensitivity analysis for {pathway_name}...")
            
            # Define parameter ranges for this pathway
            param_ranges = self._get_pathway_parameter_ranges(pathway_data['pathway'], include_pulsed)
            
            # Run Sobol sensitivity analysis
            sobol_results = self.sensitivity_analyzer.sobol_sensitivity_analysis(
                param_ranges, samples, pathway_name
            )
            
            # Run variance-based analysis
            variance_results = self.sensitivity_analyzer.variance_based_analysis(
                param_ranges, samples
            )
            
            sensitivity_results[pathway_name] = {
                'sobol_indices': sobol_results,
                'variance_analysis': variance_results,
                'parameter_ranges': param_ranges,
                'critical_parameters': self._identify_critical_parameters(sobol_results)
            }
        
        return sensitivity_results
    
    def _get_pathway_parameter_ranges(self, pathway, include_pulsed: bool) -> Dict:
        """Get parameter ranges for sensitivity analysis."""
        
        ranges = {
            # Beam parameters
            'beam_energy_mev': (10.0, 25.0),
            'beam_power_mw': (1.0, 50.0),
            'beam_flux_per_cm2_s': (1e12, 1e15),
            'beam_pulse_width_ns': (1.0, 1000.0) if include_pulsed else (1000.0, 1000.0),
            
            # Target parameters
            'target_mass_g': (1.0, 1000.0),
            'target_density_g_cm3': (8.0, 22.0),
            'target_thickness_cm': (0.1, 10.0),
            
            # Cross-section uncertainties
            'cross_section_uncertainty': (0.8, 1.2),
            'branching_ratio_uncertainty': (0.9, 1.1),
            
            # Economic parameters
            'feedstock_cost_multiplier': (0.5, 2.0),
            'energy_cost_per_mwh': (30.0, 100.0),
            'gold_price_per_g': (50.0, 80.0),
            
            # Operational parameters
            'collection_efficiency': (0.7, 0.99),
            'purification_efficiency': (0.8, 0.98),
            'overall_uptime': (0.6, 0.95)
        }
        
        # Add pulsed beam enhancement factors if enabled
        if include_pulsed:
            ranges.update({
                'pulsed_enhancement_factor': (1.0, 5.0),
                'instantaneous_flux_multiplier': (1.0, 1000.0),
                'coherent_enhancement': (1.0, 3.0)
            })
        
        return ranges
    
    def _identify_critical_parameters(self, sobol_results: Dict) -> List[str]:
        """Identify parameters with highest sensitivity indices."""
        
        if 'first_order' not in sobol_results:
            return []
        
        # Sort parameters by first-order Sobol index
        sorted_params = sorted(
            sobol_results['first_order'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top 5 most critical parameters
        return [param for param, index in sorted_params[:5]]
    
    def _run_multi_pathway_optimization(self, top_pathways: List[Tuple[str, Dict]]) -> Dict:
        """Run optimization across multiple pathways."""
        
        optimization_results = {}
        
        for name, pathway_data in top_pathways:
            self.logger.info(f"Optimizing pathway: {name}")
            
            # Run single-pathway optimization
            single_result = self.optimizer.optimize_pathway(
                pathway_data['pathway'],
                beam_power_range=(1.0, 50.0),
                energy_range=(10.0, 25.0)
            )
            
            optimization_results[name] = single_result
        
        # Find globally optimal pathway
        best_pathway = max(
            optimization_results.items(),
            key=lambda x: x[1].get('optimized_economics', {}).get('economic_fom', 0)
        )
        
        optimization_results['global_optimum'] = {
            'best_pathway': best_pathway[0],
            'best_economics': best_pathway[1].get('optimized_economics', {}),
            'optimization_summary': self._create_optimization_summary(optimization_results)
        }
        
        return optimization_results
    
    def _create_optimization_summary(self, results: Dict) -> Dict:
        """Create summary of optimization results."""
        
        pathways_above_threshold = []
        best_fom = 0
        best_pathway = None
        
        for pathway_name, result in results.items():
            if pathway_name == 'global_optimum':
                continue
                
            economics = result.get('optimized_economics', {})
            fom = economics.get('economic_fom', 0)
            
            if fom >= self.economic_thresholds['min_economic_fom']:
                pathways_above_threshold.append(pathway_name)
            
            if fom > best_fom:
                best_fom = fom
                best_pathway = pathway_name
        
        return {
            'total_pathways_optimized': len(results) - 1,  # Exclude global_optimum
            'pathways_above_threshold': pathways_above_threshold,
            'count_above_threshold': len(pathways_above_threshold),
            'best_pathway': best_pathway,
            'best_fom': best_fom,
            'threshold_met': best_fom >= self.economic_thresholds['min_economic_fom']
        }
    
    def _analyze_pulsed_beam_enhancements(self, top_pathways: List[Tuple[str, Dict]]) -> Dict:
        """Analyze benefits of pulsed beam operation."""
        
        pulsed_analysis = {}
        
        # Load pulsed beam enhancement factors
        enhancements = self.atomic_data.load_pulsed_beam_enhancements()
        
        for name, pathway_data in top_pathways:
            pathway = pathway_data['pathway']
            initial_isotope = pathway.initial_isotope.split('+')[0]  # Handle multi-target pathways
            
            if initial_isotope in enhancements:
                enhancement_factors = enhancements[initial_isotope]
                
                # Calculate enhanced economics
                enhanced_economics = self._calculate_enhanced_economics(
                    pathway_data['economics'], enhancement_factors
                )
                
                pulsed_analysis[name] = {
                    'enhancement_factors': enhancement_factors,
                    'baseline_economics': pathway_data['economics'],
                    'enhanced_economics': enhanced_economics,
                    'improvement_ratio': enhanced_economics['economic_fom'] / pathway_data['economics']['economic_fom'],
                    'becomes_viable': enhanced_economics['economic_fom'] >= self.economic_thresholds['min_economic_fom']
                }
        
        return pulsed_analysis
    
    def _calculate_enhanced_economics(self, baseline_economics: Dict, enhancement_factors: Dict) -> Dict:
        """Calculate economics with pulsed beam enhancements."""
        
        enhanced = baseline_economics.copy()
        
        # Average enhancement across reaction types
        avg_enhancement = np.mean(list(enhancement_factors.values()))
        
        # Enhanced conversion efficiency
        enhanced['conversion_mg_per_g'] *= avg_enhancement
        
        # Energy cost may decrease due to higher efficiency
        enhanced['energy_cost_per_g'] /= avg_enhancement
        
        # Recalculate total cost and economics
        enhanced['total_cost_per_g'] = enhanced['feedstock_cost_per_g'] + enhanced['energy_cost_per_g']
        enhanced['profit_per_g'] = enhanced['product_value_per_g'] - enhanced['total_cost_per_g']
        enhanced['profit_margin'] = enhanced['profit_per_g'] / enhanced['product_value_per_g']
        enhanced['economic_fom'] = enhanced['conversion_mg_per_g'] / enhanced['total_cost_per_g']
        enhanced['viable'] = enhanced['economic_fom'] >= self.economic_thresholds['min_economic_fom']
        
        return enhanced
    
    def _generate_recommendations(self, analysis_results: Dict) -> Dict:
        """Generate go/no-go recommendations based on analysis."""
        
        recommendations = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_recommendation': 'ANALYZE',
            'confidence_level': 'MEDIUM',
            'reasoning': [],
            'next_steps': [],
            'risk_factors': []
        }
        
        viable_count = analysis_results.get('viable_count', 0)
        
        if viable_count > 0:
            # We have viable pathways - recommend proceeding
            recommendations['overall_recommendation'] = 'GO'
            recommendations['confidence_level'] = 'HIGH'
            recommendations['reasoning'].append(f"Found {viable_count} economically viable pathways")
            
            # Get best pathway
            best_pathway = None
            best_fom = 0
            
            for name, data in analysis_results.get('viable_pathways', {}).items():
                fom = data['economics']['economic_fom']
                if fom > best_fom:
                    best_fom = fom
                    best_pathway = name
            
            recommendations['reasoning'].append(f"Best pathway: {best_pathway} with FOM = {best_fom:.2f}")
            recommendations['next_steps'].extend([
                f"Prepare RFQ for {best_pathway} micro-run experiment",
                "Focus on top 3 viable pathways for initial testing",
                "Develop detailed experimental protocols",
                "Secure feedstock materials and beam time"
            ])
            
        else:
            # No viable pathways - check if any are close
            pulsed_viable = 0
            if 'pulsed_beam_analysis' in analysis_results:
                for data in analysis_results['pulsed_beam_analysis'].values():
                    if data.get('becomes_viable', False):
                        pulsed_viable += 1
            
            if pulsed_viable > 0:
                recommendations['overall_recommendation'] = 'CONDITIONAL_GO'
                recommendations['confidence_level'] = 'MEDIUM'
                recommendations['reasoning'].append(f"{pulsed_viable} pathways become viable with pulsed beam enhancement")
                recommendations['next_steps'].extend([
                    "Investigate pulsed beam / linac options",
                    "Study nonlinear photonuclear cross-section enhancements",
                    "Consider partnership with accelerator facilities"
                ])
            else:
                recommendations['overall_recommendation'] = 'NO_GO'
                recommendations['confidence_level'] = 'HIGH'
                recommendations['reasoning'].append("No pathways meet economic viability thresholds")
                recommendations['next_steps'].extend([
                    "Explore additional feedstock isotopes",
                    "Investigate alternative photon sources",
                    "Consider different target products (Pt, Pd, etc.)"
                ])
        
        # Add risk factors
        recommendations['risk_factors'].extend([
            "Cross-section data uncertainties (±20%)",
            "Feedstock cost volatility",
            "Gold price fluctuations",
            "Beam availability and reliability",
            "Scale-up challenges"
        ])
        
        return recommendations
    
    def _create_enhanced_visualizations(self, results: Dict):
        """Create comprehensive visualization suite."""
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create main comparison plot
        self._plot_pathway_comparison(results)
        
        # Create economic viability scatter plot
        self._plot_economic_viability(results)
        
        # Create sensitivity analysis heatmap
        if 'sensitivity_analysis' in results:
            self._plot_sensitivity_heatmap(results['sensitivity_analysis'])
        
        # Create pulsed beam enhancement comparison
        if 'pulsed_beam_analysis' in results:
            self._plot_pulsed_beam_comparison(results['pulsed_beam_analysis'])
        
        # Create optimization convergence plots
        if 'optimization_results' in results:
            self._plot_optimization_results(results['optimization_results'])
    
    def _plot_pathway_comparison(self, results: Dict):
        """Create comprehensive pathway comparison plot."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract pathway data
        pathway_names = []
        conversion_rates = []
        economic_foms = []
        profit_margins = []
        total_costs = []
        
        for name, data in results.get('pathway_rankings', {}).items():
            pathway_names.append(name.replace('_', ' ').title())
            conversion_rates.append(data['economics']['conversion_mg_per_g'])
            economic_foms.append(data['economics']['economic_fom'])
            profit_margins.append(data['economics']['profit_margin'] * 100)
            total_costs.append(data['economics']['total_cost_per_g'])
        
        # Conversion rate comparison
        bars1 = ax1.bar(pathway_names, conversion_rates, alpha=0.7)
        ax1.axhline(y=self.economic_thresholds['min_conversion_mg_per_g'], 
                   color='red', linestyle='--', label='Viability Threshold')
        ax1.set_ylabel('Conversion Rate (mg Au/g feedstock)')
        ax1.set_title('Gold Conversion Efficiency by Pathway')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # Economic figure of merit
        bars2 = ax2.bar(pathway_names, economic_foms, alpha=0.7, color='green')
        ax2.axhline(y=self.economic_thresholds['min_economic_fom'], 
                   color='red', linestyle='--', label='Viability Threshold')
        ax2.set_ylabel('Economic FOM (mg Au/g per $CAD)')
        ax2.set_title('Economic Figure of Merit')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # Profit margins
        bars3 = ax3.bar(pathway_names, profit_margins, alpha=0.7, color='orange')
        ax3.axhline(y=self.economic_thresholds['min_profit_margin'] * 100, 
                   color='red', linestyle='--', label='Viability Threshold')
        ax3.set_ylabel('Profit Margin (%)')
        ax3.set_title('Profit Margins')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        
        # Total costs
        bars4 = ax4.bar(pathway_names, total_costs, alpha=0.7, color='purple')
        ax4.axhline(y=self.economic_thresholds['max_cost_per_g_cad'], 
                   color='red', linestyle='--', label='Viability Threshold')
        ax4.set_ylabel('Total Cost ($CAD/g feedstock)')
        ax4.set_title('Total Production Costs')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pathway_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_economic_viability(self, results: Dict):
        """Create economic viability scatter plot."""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Extract data for scatter plot
        foms = []
        conversions = []
        colors = []
        labels = []
        
        for name, data in results.get('pathway_rankings', {}).items():
            foms.append(data['economics']['economic_fom'])
            conversions.append(data['economics']['conversion_mg_per_g'])
            
            # Color by viability
            if data['economics']['viable']:
                colors.append('green')
            elif data['economics']['economic_fom'] >= self.economic_thresholds['min_economic_fom'] * 0.5:
                colors.append('orange')
            else:
                colors.append('red')
            
            labels.append(name.replace('_', ' ').title())
        
        # Create scatter plot
        scatter = ax.scatter(foms, conversions, c=colors, s=100, alpha=0.7)
        
        # Add threshold lines
        ax.axvline(x=self.economic_thresholds['min_economic_fom'], 
                  color='red', linestyle='--', alpha=0.5, label='FOM Threshold')
        ax.axhline(y=self.economic_thresholds['min_conversion_mg_per_g'], 
                  color='red', linestyle='--', alpha=0.5, label='Conversion Threshold')
        
        # Add pathway labels
        for i, label in enumerate(labels):
            ax.annotate(label, (foms[i], conversions[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Economic FOM (mg Au/g per $CAD)')
        ax.set_ylabel('Conversion Rate (mg Au/g feedstock)')
        ax.set_title('Economic Viability Analysis\n(Green=Viable, Orange=Marginal, Red=Unviable)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'economic_viability.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensitivity_heatmap(self, sensitivity_results: Dict):
        """Create sensitivity analysis heatmap."""
        
        if not sensitivity_results:
            return
        
        # Prepare data for heatmap
        pathways = list(sensitivity_results.keys())
        all_params = set()
        
        for pathway_data in sensitivity_results.values():
            if 'sobol_indices' in pathway_data and 'first_order' in pathway_data['sobol_indices']:
                all_params.update(pathway_data['sobol_indices']['first_order'].keys())
        
        # Create sensitivity matrix
        sensitivity_matrix = np.zeros((len(pathways), len(all_params)))
        param_list = list(all_params)
        
        for i, pathway in enumerate(pathways):
            sobol_data = sensitivity_results[pathway].get('sobol_indices', {}).get('first_order', {})
            for j, param in enumerate(param_list):
                sensitivity_matrix[i, j] = sobol_data.get(param, 0)
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        sns.heatmap(sensitivity_matrix, 
                   xticklabels=[p.replace('_', ' ').title() for p in param_list],
                   yticklabels=[p.replace('_', ' ').title() for p in pathways],
                   annot=True, fmt='.3f', cmap='viridis', ax=ax)
        
        ax.set_title('Sensitivity Analysis Heatmap\n(First-Order Sobol Indices)')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Pathways')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pulsed_beam_comparison(self, pulsed_results: Dict):
        """Create pulsed beam enhancement comparison."""
        
        pathways = list(pulsed_results.keys())
        baseline_foms = [data['baseline_economics']['economic_fom'] for data in pulsed_results.values()]
        enhanced_foms = [data['enhanced_economics']['economic_fom'] for data in pulsed_results.values()]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        x = np.arange(len(pathways))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_foms, width, label='Baseline', alpha=0.7)
        bars2 = ax.bar(x + width/2, enhanced_foms, width, label='Pulsed Enhanced', alpha=0.7)
        
        # Add threshold line
        ax.axhline(y=self.economic_thresholds['min_economic_fom'], 
                  color='red', linestyle='--', label='Viability Threshold')
        
        ax.set_xlabel('Pathways')
        ax.set_ylabel('Economic FOM (mg Au/g per $CAD)')
        ax.set_title('Pulsed Beam Enhancement Benefits')
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in pathways], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pulsed_beam_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_results(self, optimization_results: Dict):
        """Create optimization results visualization."""
        
        if 'global_optimum' not in optimization_results:
            return
        
        summary = optimization_results['global_optimum'].get('optimization_summary', {})
        
        # Create summary bar chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics = ['Total Optimized', 'Above Threshold', 'Best FOM']
        values = [
            summary.get('total_pathways_optimized', 0),
            summary.get('count_above_threshold', 0),
            summary.get('best_fom', 0) * 10  # Scale for visibility
        ]
        
        bars = ax.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'orange'])
        
        ax.set_title('Optimization Results Summary')
        ax.set_ylabel('Count / Scaled Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimization_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _format_pathway_rankings(self, ranked_pathways: List[Tuple[str, Dict]]) -> Dict:
        """Format pathway rankings for output."""
        
        formatted = {}
        
        for rank, (name, data) in enumerate(ranked_pathways, 1):
            formatted[name] = {
                'rank': rank,
                'pathway_info': {
                    'name': data['pathway'].pathway_name,
                    'initial_isotope': data['pathway'].initial_isotope,
                    'final_isotope': data['pathway'].final_isotope,
                    'total_probability': data['pathway'].total_probability
                },
                'economics': data['economics']
            }
        
        return formatted
    
    def _export_comprehensive_report(self, results: Dict):
        """Export comprehensive analysis report."""
        
        # Export main results as JSON
        output_file = self.output_dir / 'enhanced_comprehensive_analysis.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Export CSV summary for easy viewing
        self._export_csv_summary(results)
        
        # Export text recommendations
        self._export_text_recommendations(results)
        
        self.logger.info(f"Comprehensive report exported to {self.output_dir}")
    
    def _export_csv_summary(self, results: Dict):
        """Export CSV summary of pathway rankings."""
        
        csv_data = []
        
        for name, data in results.get('pathway_rankings', {}).items():
            csv_data.append({
                'Pathway': name,
                'Initial_Isotope': data['pathway_info']['initial_isotope'],
                'Final_Isotope': data['pathway_info']['final_isotope'],
                'Conversion_mg_per_g': data['economics']['conversion_mg_per_g'],
                'Economic_FOM': data['economics']['economic_fom'],
                'Profit_Margin_%': data['economics']['profit_margin'] * 100,
                'Total_Cost_CAD_per_g': data['economics']['total_cost_per_g'],
                'Viable': data['economics']['viable'],
                'Rank': data['rank']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.output_dir / 'pathway_rankings.csv', index=False)
    
    def _export_text_recommendations(self, results: Dict):
        """Export human-readable recommendations."""
        
        recommendations = results.get('recommendations', {})
        
        with open(self.output_dir / 'recommendations.txt', 'w') as f:
            f.write("ENHANCED TRANSMUTATION PATHWAY ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {recommendations.get('timestamp', 'Unknown')}\n")
            f.write(f"Overall Recommendation: {recommendations.get('overall_recommendation', 'UNKNOWN')}\n")
            f.write(f"Confidence Level: {recommendations.get('confidence_level', 'UNKNOWN')}\n\n")
            
            f.write("REASONING:\n")
            for reason in recommendations.get('reasoning', []):
                f.write(f"• {reason}\n")
            f.write("\n")
            
            f.write("NEXT STEPS:\n")
            for step in recommendations.get('next_steps', []):
                f.write(f"• {step}\n")
            f.write("\n")
            
            f.write("RISK FACTORS:\n")
            for risk in recommendations.get('risk_factors', []):
                f.write(f"• {risk}\n")
            f.write("\n")
            
            # Add pathway summary
            viable_count = results.get('viable_count', 0)
            total_count = results.get('total_pathways_analyzed', 0)
            
            f.write("PATHWAY SUMMARY:\n")
            f.write(f"• Total pathways analyzed: {total_count}\n")
            f.write(f"• Economically viable pathways: {viable_count}\n")
            f.write(f"• Viability rate: {viable_count/total_count*100:.1f}%\n\n")
            
            if viable_count > 0:
                f.write("TOP VIABLE PATHWAYS:\n")
                for name, data in list(results.get('viable_pathways', {}).items())[:3]:
                    fom = data['economics']['economic_fom']
                    conversion = data['economics']['conversion_mg_per_g']
                    f.write(f"• {name}: FOM={fom:.2f}, Conversion={conversion:.3f} mg/g\n")


# Legacy compatibility
class ComprehensiveAnalyzer(EnhancedComprehensiveAnalyzer):
    """Legacy wrapper for backward compatibility."""
    
    def run_full_analysis(self, *args, **kwargs):
        """Legacy method wrapper."""
        return self.run_enhanced_analysis(*args, **kwargs)


def main():
    """Run enhanced comprehensive analysis as standalone script."""
    logging.basicConfig(level=logging.INFO)
    
    analyzer = EnhancedComprehensiveAnalyzer()
    results = analyzer.run_enhanced_analysis(
        sensitivity_samples=1024,
        optimization_detailed=True,
        include_pulsed_beams=True
    )
    
    print(f"\nAnalysis completed! Results saved to {analyzer.output_dir}")
    print(f"Overall recommendation: {results['recommendations']['overall_recommendation']}")
    print(f"Viable pathways found: {results['viable_count']}")


if __name__ == "__main__":
    main()
