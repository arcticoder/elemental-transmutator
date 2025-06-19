#!/usr/bin/env python3
"""
Cheap Feedstock Selector
========================

Identifies and optimizes selection of low-cost feedstock materials for 
rhodium replicator operation. Targets materials ≤ $1/kg for economic viability.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

@dataclass
class FeedstockCandidate:
    """Represents a potential low-cost feedstock material."""
    isotope: str
    atomic_mass: float          # amu
    abundance: float            # natural abundance (0-1)
    market_price: float         # $/kg
    availability: str           # "abundant", "common", "moderate"
    cross_section_data: Dict    # spallation cross-sections by energy
    neutron_number: int
    proton_number: int

class FeedstockSelector:
    """Optimizes selection of cheap feedstock for rhodium production."""
    
    def __init__(self):
        self.candidates = self._initialize_feedstock_database()
        self.lv_enhancement_factors = {}
        
    def _initialize_feedstock_database(self) -> Dict[str, FeedstockCandidate]:
        """Initialize database of cheap feedstock candidates."""
        return {
            "Fe-56": FeedstockCandidate(
                isotope="Fe-56",
                atomic_mass=55.845,
                abundance=0.9175,
                market_price=0.12,  # $/kg (steel scrap)
                availability="abundant",
                cross_section_data={
                    "proton_100MeV": {"fragments": 850, "total": 950},
                    "proton_150MeV": {"fragments": 1200, "total": 1400},
                    "deuteron_120MeV": {"fragments": 950, "total": 1100}
                },
                neutron_number=30,
                proton_number=26
            ),
            "Al-27": FeedstockCandidate(
                isotope="Al-27",
                atomic_mass=26.982,
                abundance=1.0,
                market_price=0.85,  # $/kg (aluminum scrap)
                availability="abundant", 
                cross_section_data={
                    "proton_100MeV": {"fragments": 420, "total": 520},
                    "proton_150MeV": {"fragments": 680, "total": 780},
                    "deuteron_120MeV": {"fragments": 520, "total": 620}
                },
                neutron_number=14,
                proton_number=13
            ),
            "Si-28": FeedstockCandidate(
                isotope="Si-28",
                atomic_mass=27.977,
                abundance=0.9223,
                market_price=0.45,  # $/kg (silica sand)
                availability="abundant",
                cross_section_data={
                    "proton_100MeV": {"fragments": 380, "total": 480},
                    "proton_150MeV": {"fragments": 620, "total": 720},
                    "deuteron_120MeV": {"fragments": 480, "total": 580}
                },
                neutron_number=14,
                proton_number=14
            ),
            "Ca-40": FeedstockCandidate(
                isotope="Ca-40",
                atomic_mass=39.963,
                abundance=0.9694,
                market_price=0.25,  # $/kg (limestone)
                availability="abundant",
                cross_section_data={
                    "proton_100MeV": {"fragments": 620, "total": 720},
                    "proton_150MeV": {"fragments": 920, "total": 1020},
                    "deuteron_120MeV": {"fragments": 720, "total": 820}
                },
                neutron_number=20,
                proton_number=20
            ),
            "Mg-24": FeedstockCandidate(
                isotope="Mg-24",
                atomic_mass=23.985,
                abundance=0.7899,
                market_price=0.95,  # $/kg (magnesium metal)
                availability="common",
                cross_section_data={
                    "proton_100MeV": {"fragments": 320, "total": 420},
                    "proton_150MeV": {"fragments": 520, "total": 620},
                    "deuteron_120MeV": {"fragments": 420, "total": 520}
                },
                neutron_number=12,
                proton_number=12
            ),
            "Ti-48": FeedstockCandidate(
                isotope="Ti-48",
                atomic_mass=47.948,
                abundance=0.7372,
                market_price=0.75,  # $/kg (titanium dioxide)
                availability="common",
                cross_section_data={
                    "proton_100MeV": {"fragments": 720, "total": 820},
                    "proton_150MeV": {"fragments": 1050, "total": 1150},
                    "deuteron_120MeV": {"fragments": 820, "total": 920}
                },
                neutron_number=26,
                proton_number=22
            )
        }
    
    def calculate_economic_score(self, candidate: FeedstockCandidate, 
                               beam_energy: float, beam_type: str) -> float:
        """Calculate economic viability score for feedstock."""
        # Base cross-section for fragmentation
        reaction_key = f"{beam_type}_{int(beam_energy/1e6)}MeV"
        cross_section = candidate.cross_section_data.get(reaction_key, {}).get("fragments", 100)
        
        # Economic factors
        price_factor = 1.0 / (candidate.market_price + 0.01)  # Lower price = higher score
        abundance_factor = candidate.abundance
        cross_section_factor = cross_section / 1000.0  # Normalize
        
        # Availability bonus
        availability_bonus = {"abundant": 1.5, "common": 1.2, "moderate": 1.0}[candidate.availability]
        
        score = (price_factor * abundance_factor * cross_section_factor * availability_bonus)
        return score
    
    def rank_feedstocks(self, beam_energy: float = 120e6, 
                       beam_type: str = "proton") -> List[Tuple[str, float]]:
        """Rank feedstock candidates by economic viability."""
        scores = []
        
        for isotope, candidate in self.candidates.items():
            score = self.calculate_economic_score(candidate, beam_energy, beam_type)
            scores.append((isotope, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def get_optimal_feedstock(self, beam_config: Dict) -> str:
        """Get the most economical feedstock for given beam configuration."""
        rankings = self.rank_feedstocks(
            beam_energy=beam_config.get("energy", 120e6),
            beam_type=beam_config.get("type", "proton")
        )
        return rankings[0][0]  # Return top-ranked isotope
    
    def estimate_yield_potential(self, feedstock: str, mass_kg: float,
                               beam_energy: float, lv_enhancement: float = 1000.0) -> Dict:
        """Estimate rhodium yield potential from feedstock."""
        if feedstock not in self.candidates:
            return {"error": "Unknown feedstock"}
        
        candidate = self.candidates[feedstock]
        
        # Calculate number of target nuclei
        mass_per_nucleus = candidate.atomic_mass * 1.66054e-27  # kg
        target_nuclei = mass_kg / mass_per_nucleus
        
        # Estimate multi-stage yield (simplified model)
        beam_type = "proton" if beam_energy > 100e6 else "deuteron"
        reaction_key = f"{beam_type}_{int(beam_energy/1e6)}MeV"
        
        base_cross_section = candidate.cross_section_data.get(reaction_key, {}).get("fragments", 100)
        enhanced_cross_section = base_cross_section * 1e-27 * lv_enhancement  # cm²
        
        # Multi-stage efficiency (Stage A → B → C)
        stage_a_efficiency = 0.15  # Fe → mid-mass fragments
        stage_b_efficiency = 0.08  # fragments → Ag/Cd
        stage_c_efficiency = 0.25  # Ag/Cd → Rh
        
        total_efficiency = stage_a_efficiency * stage_b_efficiency * stage_c_efficiency
        
        # Estimate rhodium nuclei produced
        potential_reactions = target_nuclei * enhanced_cross_section * 1e15  # flux assumption
        rh_nuclei = potential_reactions * total_efficiency
        
        # Convert to mass
        rh_mass_per_nucleus = 103.0 * 1.66054e-27  # kg (Rh-103)
        rh_mass = rh_nuclei * rh_mass_per_nucleus
        
        return {
            "feedstock": feedstock,
            "input_mass_kg": mass_kg,
            "input_cost": mass_kg * candidate.market_price,
            "estimated_rh_nuclei": rh_nuclei,
            "estimated_rh_mass_kg": rh_mass,
            "estimated_rh_value": rh_mass * 25000,  # $25k/kg rhodium
            "net_profit": rh_mass * 25000 - mass_kg * candidate.market_price,
            "profit_ratio": (rh_mass * 25000) / (mass_kg * candidate.market_price + 1e-10)
        }
    
    def generate_feedstock_report(self) -> Dict:
        """Generate comprehensive feedstock analysis report."""
        beam_configs = [
            {"type": "proton", "energy": 100e6},
            {"type": "proton", "energy": 150e6},
            {"type": "deuteron", "energy": 120e6}
        ]
        
        report = {
            "timestamp": "2025-06-18",
            "analysis": "cheap_feedstock_optimization",
            "beam_configurations": [],
            "feedstock_rankings": {},
            "economic_projections": {}
        }
        
        for config in beam_configs:
            config_name = f"{config['type']}_{config['energy']/1e6:.0f}MeV"
            rankings = self.rank_feedstocks(config["energy"], config["type"])
            
            report["beam_configurations"].append(config_name)
            report["feedstock_rankings"][config_name] = rankings
            
            # Economic projections for top 3 feedstocks
            projections = []
            for isotope, score in rankings[:3]:
                projection = self.estimate_yield_potential(
                    feedstock=isotope,
                    mass_kg=1e-3,  # 1 kg test case
                    beam_energy=config["energy"],
                    lv_enhancement=5000.0
                )
                projections.append(projection)
            
            report["economic_projections"][config_name] = projections
        
        return report

def main():
    """Demonstrate feedstock selection and analysis."""
    print("🏭 CHEAP FEEDSTOCK SELECTOR FOR RHODIUM REPLICATOR")
    print("=" * 55)
    print("📍 Optimizing low-cost materials (≤ $1/kg) for Rh production")
    print("")
    
    selector = FeedstockSelector()
    
    # Test different beam configurations
    beam_configs = [
        {"type": "proton", "energy": 120e6, "name": "120 MeV protons"},
        {"type": "deuteron", "energy": 100e6, "name": "100 MeV deuterons"},
        {"type": "proton", "energy": 150e6, "name": "150 MeV protons"}
    ]
    
    for config in beam_configs:
        print(f"🔬 Configuration: {config['name']}")
        print("-" * 30)
        
        rankings = selector.rank_feedstocks(config["energy"], config["type"])
        
        print("Top 5 Feedstock Materials:")
        for i, (isotope, score) in enumerate(rankings[:5]):
            candidate = selector.candidates[isotope]
            print(f"  {i+1}. {isotope}: Score {score:.2f}")
            print(f"     Price: ${candidate.market_price:.2f}/kg, Abundance: {candidate.abundance:.1%}")
        
        # Detailed analysis for top candidate
        top_feedstock = rankings[0][0]
        analysis = selector.estimate_yield_potential(
            feedstock=top_feedstock,
            mass_kg=1.0,  # 1 kg feedstock
            beam_energy=config["energy"],
            lv_enhancement=7500.0
        )
        
        print(f"\n💰 Economic Analysis for 1 kg {top_feedstock}:")
        print(f"  Input cost: ${analysis['input_cost']:.2f}")
        print(f"  Estimated Rh yield: {analysis['estimated_rh_mass_kg']*1e6:.1f} mg")
        print(f"  Estimated Rh value: ${analysis['estimated_rh_value']:.2f}")
        print(f"  Net profit: ${analysis['net_profit']:.2f}")
        print(f"  Profit ratio: {analysis['profit_ratio']:.0f}×")
        print("")
    
    # Generate comprehensive report
    report = selector.generate_feedstock_report()
    
    print("📊 FEEDSTOCK OPTIMIZATION SUMMARY")
    print("=" * 40)
    print("Best overall feedstock candidates:")
    
    # Find consistently top-performing feedstocks
    all_rankings = report["feedstock_rankings"]
    feedstock_scores = {}
    
    for config_name, rankings in all_rankings.items():
        for i, (isotope, score) in enumerate(rankings):
            if isotope not in feedstock_scores:
                feedstock_scores[isotope] = []
            feedstock_scores[isotope].append((len(rankings) - i, score))  # Rank points
    
    # Calculate average performance
    avg_performance = {}
    for isotope, scores in feedstock_scores.items():
        avg_rank = np.mean([s[0] for s in scores])
        avg_score = np.mean([s[1] for s in scores])
        avg_performance[isotope] = (avg_rank + avg_score)
    
    # Sort by average performance
    sorted_feedstocks = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
    
    for i, (isotope, performance) in enumerate(sorted_feedstocks[:5]):
        candidate = selector.candidates[isotope]
        print(f"  {i+1}. {isotope}: ${candidate.market_price:.2f}/kg, {candidate.availability}")
    
    print("\n✅ FEEDSTOCK SELECTION COMPLETE")
    print("🚀 Ready for multi-stage transmutation network implementation")
    
    return report

if __name__ == "__main__":
    main()
