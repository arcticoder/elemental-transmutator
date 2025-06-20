#!/usr/bin/env python3
"""
Enhanced Pathway Analysis Runner
================================

Script to run the enhanced comprehensive analysis with new transmutation pathways
and generate recommendations for economically viable routes to gold production.
"""

import sys
import logging
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from comprehensive_analyzer import EnhancedComprehensiveAnalyzer
from atomic_binder import EnhancedAtomicDataBinder

def main():
    """Run enhanced pathway analysis and display results."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("ENHANCED TRANSMUTATION PATHWAY ANALYSIS")
    print("Evaluating new isotope routes and multi-stage chains")
    print("=" * 80)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedComprehensiveAnalyzer()
    
    # Run comprehensive analysis
    logger.info("Starting enhanced comprehensive analysis...")
    results = analyzer.run_enhanced_analysis(
        sensitivity_samples=512,  # Reduced for faster execution
        optimization_detailed=True,
        include_pulsed_beams=True
    )
    
    # Display key results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"Total pathways analyzed: {results['total_pathways_analyzed']}")
    print(f"Economically viable pathways: {results['viable_count']}")
    
    if results['viable_count'] > 0:
        print(f"\n✅ SUCCESS: Found {results['viable_count']} viable pathway(s)!")
        
        print("\nTOP VIABLE PATHWAYS:")
        for name, data in list(results['viable_pathways'].items())[:5]:
            economics = data['economics']
            print(f"  • {name}:")
            print(f"    - Conversion: {economics['conversion_mg_per_g']:.3f} mg Au/g feedstock")
            print(f"    - Economic FOM: {economics['economic_fom']:.2f}")
            print(f"    - Profit margin: {economics['profit_margin']*100:.1f}%")
            print(f"    - Total cost: ${economics['total_cost_per_g']:.4f} CAD/g")
    else:
        print("\n❌ No pathways meet economic viability thresholds")
        
        # Check pulsed beam potential
        pulsed_viable = 0
        if 'pulsed_beam_analysis' in results:
            for data in results['pulsed_beam_analysis'].values():
                if data.get('becomes_viable', False):
                    pulsed_viable += 1
        
        if pulsed_viable > 0:
            print(f"⚡ However, {pulsed_viable} pathway(s) become viable with pulsed beams!")
    
    # Display recommendations
    recommendations = results['recommendations']
    print(f"\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    print(f"Overall recommendation: {recommendations['overall_recommendation']}")
    print(f"Confidence level: {recommendations['confidence_level']}")
    
    print("\nReasoning:")
    for reason in recommendations['reasoning']:
        print(f"  • {reason}")
    
    print("\nNext steps:")
    for step in recommendations['next_steps']:
        print(f"  • {step}")
    
    # Show pathway rankings
    print(f"\n" + "=" * 50)
    print("TOP 10 PATHWAY RANKINGS")
    print("=" * 50)
    
    rankings = results['pathway_rankings']
    for i, (name, data) in enumerate(list(rankings.items())[:10], 1):
        economics = data['economics']
        viable_status = "✅ VIABLE" if economics['viable'] else "❌ Unviable"
        
        print(f"{i:2d}. {name} - {viable_status}")
        print(f"     FOM: {economics['economic_fom']:.3f} | Conv: {economics['conversion_mg_per_g']:.3f} mg/g")
    
    # Pulsed beam analysis results
    if 'pulsed_beam_analysis' in results:
        print(f"\n" + "=" * 50)
        print("PULSED BEAM ENHANCEMENT ANALYSIS")
        print("=" * 50)
        
        for name, data in results['pulsed_beam_analysis'].items():
            improvement = data['improvement_ratio']
            becomes_viable = data['becomes_viable']
            
            status = "🚀 BECOMES VIABLE" if becomes_viable else "📈 Improved"
            print(f"{name}: {status}")
            print(f"  Enhancement: {improvement:.2f}x improvement")
            print(f"  New FOM: {data['enhanced_economics']['economic_fom']:.3f}")
    
    # Analysis files created
    print(f"\n" + "=" * 50)
    print("FILES CREATED")
    print("=" * 50)
    
    output_dir = analyzer.output_dir
    print(f"Analysis results saved to: {output_dir}")
    print("Key files:")
    print(f"  • {output_dir}/enhanced_comprehensive_analysis.json - Full results")
    print(f"  • {output_dir}/pathway_rankings.csv - Pathway comparison")
    print(f"  • {output_dir}/recommendations.txt - Human-readable summary")
    print(f"  • {output_dir}/pathway_comparison.png - Visual comparison")
    print(f"  • {output_dir}/economic_viability.png - Scatter plot analysis")
    
    # Final assessment
    print(f"\n" + "=" * 50)
    print("ECONOMIC VIABILITY ASSESSMENT")
    print("=" * 50)
    
    if results['viable_count'] > 0:
        print("🎯 RECOMMENDATION: PROCEED TO OUTSOURCE MICRO-RUNS")
        print("   Found economically viable pathways meeting all thresholds:")
        print("   • ≥0.1 mg Au/g conversion")
        print("   • ≤$0.01 CAD/g total cost")
        print("   • ≥5% profit margin")
        print("   • ≥0.1 economic figure of merit")
        
        print("\n   Next actions:")
        print("   1. Draft RFQ for top viable pathway")
        print("   2. Secure feedstock materials")
        print("   3. Contact accelerator facilities")
        print("   4. Plan experimental protocols")
        
    elif pulsed_viable > 0:
        print("⚡ RECOMMENDATION: INVESTIGATE PULSED BEAM OPTIONS")
        print("   Some pathways become viable with pulsed enhancement.")
        print("   Consider partnering with linac facilities.")
        
    else:
        print("🔄 RECOMMENDATION: EXPAND PATHWAY SEARCH")
        print("   Current pathways don't meet economic thresholds.")
        print("   Suggested extensions:")
        print("   • Additional rare earth targets")
        print("   • Alternative products (Pt, Pd, Rh)")
        print("   • Multi-stage fusion reactions")
        print("   • Exotic particle beam sources")
    
    print(f"\nAnalysis completed in {results['analysis_duration_s']:.1f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
