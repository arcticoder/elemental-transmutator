#!/usr/bin/env python3
"""
Enhanced Pathway Test Suite
===========================

Test suite to validate the new transmutation pathways and ensure
they are correctly loaded and analyzed.
"""

import sys
import unittest
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from atomic_binder import EnhancedAtomicDataBinder
from comprehensive_analyzer import EnhancedComprehensiveAnalyzer

class TestEnhancedPathways(unittest.TestCase):
    """Test cases for enhanced transmutation pathways."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.binder = EnhancedAtomicDataBinder()
        self.analyzer = EnhancedComprehensiveAnalyzer()
        logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    def test_enhanced_isotope_loading(self):
        """Test that new isotopes are properly loaded."""
        
        # Check that new alternative feedstocks are present
        required_isotopes = [
            'Bi-209', 'Pt-195', 'Ir-191', 'Ta-181', 'U-238', 'Th-232'
        ]
        
        for isotope in required_isotopes:
            self.assertIn(isotope, self.binder.target_isotopes, 
                         f"Missing isotope: {isotope}")
        
        # Check isotope properties
        bi209 = self.binder.target_isotopes['Bi-209']
        self.assertEqual(bi209['Z'], 83)
        self.assertEqual(bi209['A'], 209)
        self.assertEqual(bi209['abundance'], 1.0)
        
        pt195 = self.binder.target_isotopes['Pt-195']
        self.assertEqual(pt195['Z'], 78)
        self.assertEqual(pt195['A'], 195)
    
    def test_pathway_loading(self):
        """Test that enhanced pathways are properly loaded."""
        
        pathways = self.binder.load_enhanced_pathways()
        
        # Should have multiple pathways
        self.assertGreater(len(pathways), 5, "Should have at least 6 pathways")
        
        # Check specific pathways exist
        expected_pathways = [
            'bi209_gamma_n_cascade',
            'pt195_neutron_loss', 
            'ir191_proton_alpha',
            'ta_hg_two_stage',
            'th_pb_converter'
        ]
        
        for pathway_name in expected_pathways:
            self.assertIn(pathway_name, pathways, 
                         f"Missing pathway: {pathway_name}")
        
        # Validate pathway structure
        test_pathway = pathways['bi209_gamma_n_cascade']
        self.assertEqual(test_pathway.initial_isotope, 'Bi-209')
        self.assertEqual(test_pathway.final_isotope, 'Au-197')
        self.assertGreater(len(test_pathway.steps), 0)
        self.assertIsInstance(test_pathway.total_probability, float)
    
    def test_economic_calculations(self):
        """Test economic viability calculations."""
        
        pathways = self.binder.load_enhanced_pathways()
        
        # Test economic calculation for a pathway
        test_pathway = pathways['pt195_neutron_loss']
        economics = self.binder.calculate_pathway_economics(test_pathway)
        
        # Check that all required economic metrics are present
        required_metrics = [
            'feedstock_cost_per_g', 'energy_cost_per_g', 'total_cost_per_g',
            'product_value_per_g', 'profit_per_g', 'profit_margin',
            'conversion_mg_per_g', 'economic_fom', 'viable'
        ]
          for metric in required_metrics:
            self.assertIn(metric, economics, f"Missing metric: {metric}")
            # Convert numpy types to native Python types for testing
            value = economics[metric]
            if hasattr(value, 'item'):  # numpy scalar
                value = value.item()
            self.assertIsInstance(value, (int, float, bool))
    
    def test_pathway_ranking(self):
        """Test pathway ranking by economics."""
        
        ranked_pathways = self.binder.rank_pathways_by_economics()
        
        # Should return list of tuples
        self.assertIsInstance(ranked_pathways, list)
        self.assertGreater(len(ranked_pathways), 0)
        
        # Check ranking order (should be sorted by economic FOM)
        foms = [data[1]['economics']['economic_fom'] for data in ranked_pathways]
        self.assertEqual(foms, sorted(foms, reverse=True), 
                        "Pathways should be sorted by FOM (descending)")
    
    def test_pulsed_beam_enhancements(self):
        """Test pulsed beam enhancement loading."""
        
        enhancements = self.binder.load_pulsed_beam_enhancements()
        
        # Check that enhancement factors are loaded
        self.assertIn('Bi-209', enhancements)
        self.assertIn('Pt-195', enhancements)
        self.assertIn('Ta-181', enhancements)
        
        # Check enhancement structure
        bi209_enhancements = enhancements['Bi-209']
        self.assertIn('gamma_n', bi209_enhancements)
        self.assertGreater(bi209_enhancements['gamma_n'], 1.0)
    
    def test_comprehensive_analysis(self):
        """Test that comprehensive analysis runs without errors."""
        
        # Run a minimal analysis
        results = self.analyzer.run_enhanced_analysis(
            sensitivity_samples=16,  # Very small for fast testing
            optimization_detailed=False,
            include_pulsed_beams=True
        )
        
        # Check that all expected sections are present
        expected_sections = [
            'total_pathways_analyzed', 'pathway_rankings', 'viable_pathways',
            'viable_count', 'recommendations'
        ]
        
        for section in expected_sections:
            self.assertIn(section, results, f"Missing section: {section}")
        
        # Check that we analyzed some pathways
        self.assertGreater(results['total_pathways_analyzed'], 0)
        
        # Check recommendation structure
        recommendations = results['recommendations']
        self.assertIn('overall_recommendation', recommendations)
        self.assertIn('confidence_level', recommendations)
        self.assertIn('reasoning', recommendations)
        self.assertIn('next_steps', recommendations)
    
    def test_viability_thresholds(self):
        """Test that viability thresholds are correctly applied."""
        
        pathways = self.binder.load_enhanced_pathways()
        
        # Create a test pathway that should be viable
        test_pathway = list(pathways.values())[0]
        
        # Calculate economics
        economics = self.binder.calculate_pathway_economics(test_pathway)
        
        # Check threshold evaluation
        thresholds = self.analyzer.economic_thresholds
        
        meets_conversion = economics['conversion_mg_per_g'] >= thresholds['min_conversion_mg_per_g']
        meets_cost = economics['total_cost_per_g'] <= thresholds['max_cost_per_g_cad']
        meets_margin = economics['profit_margin'] >= thresholds['min_profit_margin']
        meets_fom = economics['economic_fom'] >= thresholds['min_economic_fom']
        
        expected_viable = meets_conversion and meets_cost and meets_margin and meets_fom
        self.assertEqual(economics['viable'], expected_viable)
    
    def test_new_isotope_costs(self):
        """Test that new isotope costs are reasonable."""
        
        # Check that costs are positive and within reasonable ranges
        isotope_costs = {
            'Bi-209': (0.1, 1.0),     # Should be reasonable
            'Pt-195': (20.0, 50.0),   # Expensive but not crazy
            'Ir-191': (40.0, 100.0),  # Very expensive
            'Ta-181': (1.0, 3.0),     # Moderate
            'U-238': (0.01, 0.1),     # Very cheap
            'Th-232': (0.5, 2.0)      # Reasonable
        }
        
        for isotope, (min_cost, max_cost) in isotope_costs.items():
            actual_cost = self.binder.target_isotopes[isotope]['cost_per_g']
            self.assertGreaterEqual(actual_cost, min_cost, 
                                   f"{isotope} cost too low: {actual_cost}")
            self.assertLessEqual(actual_cost, max_cost, 
                                f"{isotope} cost too high: {actual_cost}")
    
    def test_pathway_probabilities(self):
        """Test that pathway probabilities are reasonable."""
        
        pathways = self.binder.load_enhanced_pathways()
        
        for name, pathway in pathways.items():
            # Total probability should be between 0 and 1
            self.assertGreaterEqual(pathway.total_probability, 0.0, 
                                   f"{name} has negative probability")
            self.assertLessEqual(pathway.total_probability, 1.0, 
                                f"{name} has probability > 1")
            
            # Individual steps should have reasonable branching ratios
            for step in pathway.steps:
                if 'branching_ratio' in step:
                    br = step['branching_ratio']
                    self.assertGreaterEqual(br, 0.0, f"{name} step has negative branching ratio")
                    self.assertLessEqual(br, 1.0, f"{name} step has branching ratio > 1")


def run_pathway_validation():
    """Run comprehensive pathway validation."""
    
    print("=" * 60)
    print("ENHANCED PATHWAY VALIDATION SUITE")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Additional validation
    print("\n" + "=" * 40)
    print("ADDITIONAL VALIDATION CHECKS")
    print("=" * 40)
    
    binder = EnhancedAtomicDataBinder()
    
    # Load and display pathway summary
    pathways = binder.load_enhanced_pathways()
    print(f"\nLoaded {len(pathways)} enhanced pathways:")
    
    for name, pathway in pathways.items():
        print(f"  • {pathway.pathway_name}")
        print(f"    {pathway.initial_isotope} → {pathway.final_isotope}")
        print(f"    Probability: {pathway.total_probability:.4f}")
        print(f"    Steps: {len(pathway.steps)}")
    
    # Test economic ranking
    print(f"\n" + "=" * 40)
    print("ECONOMIC RANKING TEST")
    print("=" * 40)
    
    ranked = binder.rank_pathways_by_economics()
    print(f"\nTop 5 pathways by economic FOM:")
    
    for i, (name, data) in enumerate(ranked[:5], 1):
        economics = data['economics']
        viable_status = "✅" if economics['viable'] else "❌"
        print(f"{i}. {name} {viable_status}")
        print(f"   FOM: {economics['economic_fom']:.3f}")
        print(f"   Conversion: {economics['conversion_mg_per_g']:.3f} mg/g")
        print(f"   Cost: ${economics['total_cost_per_g']:.4f}/g")
    
    viable_count = sum(1 for _, data in ranked if data['economics']['viable'])
    print(f"\nPathways meeting viability thresholds: {viable_count}/{len(ranked)}")
    
    if viable_count > 0:
        print("🎯 SUCCESS: Found economically viable pathways!")
    else:
        print("⚠️  No pathways currently meet all viability thresholds")
        print("   Consider pulsed beam enhancements or alternative targets")
    
    print("\nValidation completed successfully! ✅")


if __name__ == "__main__":
    run_pathway_validation()
