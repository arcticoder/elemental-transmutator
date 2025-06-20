#!/usr/bin/env python3
"""
Quick Enhanced Pathway Demo
===========================

Quick demonstration of the new transmutation pathways without full analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from atomic_binder import EnhancedAtomicDataBinder

def main():
    """Demo the enhanced pathways."""
    
    print("=" * 70)
    print("ENHANCED TRANSMUTATION PATHWAY DEMONSTRATION")
    print("=" * 70)
    
    # Initialize enhanced binder
    binder = EnhancedAtomicDataBinder()
    
    print("Loading enhanced transmutation pathways...")
    pathways = binder.load_enhanced_pathways()
    
    print(f"\n✅ Successfully loaded {len(pathways)} enhanced pathways!")
    
    print("\n" + "=" * 50)
    print("NEW PATHWAY SUMMARY")
    print("=" * 50)
    
    for i, (name, pathway) in enumerate(pathways.items(), 1):
        print(f"\n{i}. {pathway.pathway_name}")
        print(f"   Route: {pathway.initial_isotope} → {pathway.final_isotope}")
        print(f"   Conversion probability: {pathway.total_probability:.4f}")
        print(f"   Number of steps: {len(pathway.steps)}")
        
        # Show first step as example
        if pathway.steps:
            first_step = pathway.steps[0]
            print(f"   First step: {first_step.get('reaction', 'N/A')}")
    
    print("\n" + "=" * 50)
    print("ECONOMIC ANALYSIS")
    print("=" * 50)
    
    # Calculate economics for each pathway
    economics_results = []
    
    for name, pathway in pathways.items():
        try:
            economics = binder.calculate_pathway_economics(pathway)
            economics_results.append((name, pathway, economics))
        except Exception as e:
            print(f"Warning: Could not calculate economics for {name}: {e}")
    
    # Sort by economic figure of merit
    economics_results.sort(key=lambda x: x[2]['economic_fom'], reverse=True)
    
    print(f"\nTop pathways by economic figure of merit:")
    
    viable_count = 0
    for i, (name, pathway, economics) in enumerate(economics_results[:5], 1):
        
        viable_status = "✅ VIABLE" if economics['economic_fom'] >= 0.1 else "❌ Unviable"
        if economics['economic_fom'] >= 0.1:
            viable_count += 1
        
        print(f"\n{i}. {pathway.pathway_name} - {viable_status}")
        print(f"   Conversion: {economics['conversion_mg_per_g']:.3f} mg Au/g feedstock")
        print(f"   Economic FOM: {economics['economic_fom']:.3f}")
        print(f"   Total cost: ${economics['total_cost_per_g']:.4f} CAD/g")
        print(f"   Profit margin: {economics['profit_margin']*100:.1f}%")
    
    # Load pulsed beam enhancements
    print("\n" + "=" * 50)
    print("PULSED BEAM ENHANCEMENTS")
    print("=" * 50)
    
    try:
        enhancements = binder.load_pulsed_beam_enhancements()
        print("\nPulsed beam enhancement factors loaded:")
        
        for isotope, factors in enhancements.items():
            print(f"\n{isotope}:")
            for reaction, factor in factors.items():
                print(f"  {reaction}: {factor:.2f}x enhancement")
    except Exception as e:
        print(f"Warning: Could not load pulsed beam enhancements: {e}")
    
    # Final assessment
    print("\n" + "=" * 50)
    print("VIABILITY ASSESSMENT")
    print("=" * 50)
    
    total_pathways = len(economics_results)
    print(f"\nTotal pathways analyzed: {total_pathways}")
    print(f"Economically viable pathways: {viable_count}")
    print(f"Viability rate: {viable_count/total_pathways*100:.1f}%")
    
    if viable_count > 0:
        print(f"\n🎯 SUCCESS! Found {viable_count} economically viable pathway(s)")
        print("   Meeting criteria:")
        print("   • ≥0.1 mg Au/g conversion")
        print("   • Economic FOM ≥ 0.1")
        print("   • Positive profit margin")
        
        best_pathway = economics_results[0]
        print(f"\n🏆 Best pathway: {best_pathway[1].pathway_name}")
        print(f"   FOM: {best_pathway[2]['economic_fom']:.2f}")
        print(f"   Conversion: {best_pathway[2]['conversion_mg_per_g']:.3f} mg/g")
        
        print(f"\n✅ RECOMMENDATION: PROCEED TO OUTSOURCE MICRO-RUNS")
        print("   Draft RFQ for the top viable pathway(s)")
        
    else:
        print(f"\n⚠️  No pathways currently meet economic viability thresholds")
        print("   Consider:")
        print("   • Pulsed beam enhancements")
        print("   • Alternative target isotopes")
        print("   • Higher beam powers")
        print("   • Process optimization")
        
        print(f"\n🔄 RECOMMENDATION: EXPAND SEARCH OR ENHANCE CONDITIONS")
    
    print("\n" + "=" * 70)
    print("Enhanced pathway demonstration completed! ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
