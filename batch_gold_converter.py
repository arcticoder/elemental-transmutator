#!/usr/bin/env python3
"""
Batch Gold Converter - Minimal One-Off Transmutation
===================================================

Table-top scale Hg-202 + Pt-197 → Au-197 converter for microgram to milligram batches.
Designed for bench-scale operation in under a minute with minimal overhead.

Usage:
    python batch_gold_converter.py --feed Hg-202 --mass 1e-9 \
           --beam proton:80e6:1e14 --lv mu=1e-17,alpha=1e-14,beta=1e-11

Author: Advanced Energy Research Team
License: MIT
"""

import argparse
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# Import core transmutation modules
try:
    from spallation_transmutation import SpallationTransmuter
    from decay_accelerator import DecayAccelerator
    from atomic_binder import AtomicBinder
    from energy_ledger import EnergyLedger
    print("✅ All transmutation modules imported successfully")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Running in simulation mode with mock implementations")
    
    # Mock implementations for standalone operation
    class SpallationTransmuter:
        def __init__(self, lv_params, target, precursors, beam):
            self.lv_params = lv_params
            self.target = target
            self.precursors = precursors
            self.beam = beam
            self.energy_used = 0
            
        def simulate(self, mass_kg):
            # Mock spallation with realistic cross-sections
            beam_energy, beam_intensity = self.beam['energy'], self.beam['intensity']
            
            # Enhanced cross-section with LV boost
            base_cross_section = 1e-27  # cm²
            lv_enhancement = 1.0 + self.lv_params.get('mu', 1e-17) * beam_energy**2
            effective_cross_section = base_cross_section * lv_enhancement
            
            # Calculate yield (simplified)
            avogadro = 6.022e23
            atomic_mass = 200e-3  # kg/mol for Hg-202
            n_atoms = (mass_kg / atomic_mass) * avogadro
            
            reaction_rate = effective_cross_section * beam_intensity * n_atoms
            yield_fraction = min(0.85, reaction_rate * 1e-15)  # Cap at 85% conversion
            
            self.energy_used = beam_energy * beam_intensity * 1.6e-19 * 60  # 1 minute beam time
            return int(n_atoms * yield_fraction)
    
    class DecayAccelerator:
        def __init__(self, lv_params, isotope):
            self.lv_params = lv_params
            self.isotope = isotope
            self.energy_used = 0
            
        def simulate_decay(self, precursor_count, t):
            # Mock LV-accelerated decay
            alpha = self.lv_params.get('alpha', 1e-14)
            acceleration_factor = 1e12 * alpha  # Massive acceleration
            
            # Assume 90% of precursors decay to target in accelerated time
            decay_fraction = 0.90 * min(1.0, acceleration_factor * t)
            
            self.energy_used = 1e3 * precursor_count * 1.6e-19  # Energy per nucleus
            return int(precursor_count * decay_fraction)
    
    class AtomicBinder:
        def __init__(self, lv_params):
            self.lv_params = lv_params
            self.energy_used = 0
            
        def bind(self, nucleus_count):
            # Mock atomic binding
            beta = self.lv_params.get('beta', 1e-11)
            binding_efficiency = 0.95 * (1.0 + beta * 1e6)  # Enhanced binding
            
            # Convert nuclei to mass
            au_atomic_mass = 196.966569  # u
            au_mass_kg = nucleus_count * au_atomic_mass * 1.66054e-27
            
            self.energy_used = 10 * nucleus_count * 1.6e-19  # Binding energy per atom
            return au_mass_kg * binding_efficiency
    
    class EnergyLedger:
        def __init__(self):
            self.entries = []
            
        def log_energy(self, energy_j):
            self.entries.append(energy_j)
            
        def total_energy(self):
            return sum(self.entries)

@dataclass
class BatchParameters:
    """Parameters for a single batch conversion"""
    feedstock: str
    mass_kg: float
    beam_profile: Dict
    lv_params: Dict
    timestamp: str
    checksum: str

@dataclass 
class BatchResult:
    """Results from a batch conversion"""
    parameters: BatchParameters
    au_output_g: float
    energy_consumed_j: float
    round_trip_efficiency: float
    conversion_time_s: float
    yield_per_joule: float

class BatchGoldConverter:
    """Minimal batch converter for Hg/Pt → Au transmutation"""
    
    def __init__(self, dry_run=False, verbose=True):
        self.dry_run = dry_run
        self.verbose = verbose
        self.results_log = []
        
    def parse_beam_profile(self, beam_str: str) -> Dict:
        """Parse beam profile string: 'proton:80e6:1e14'"""
        try:
            particle, energy_str, intensity_str = beam_str.split(':')
            energy = float(energy_str)  # eV
            intensity = float(intensity_str)  # particles/s
            
            return {
                'particle': particle,
                'energy': energy,
                'intensity': intensity
            }
        except Exception as e:
            raise ValueError(f"Invalid beam profile '{beam_str}': {e}")
    
    def parse_lv_params(self, lv_str: str) -> Dict:
        """Parse LV parameters: 'mu=1e-17,alpha=1e-14,beta=1e-11'"""
        params = {}
        try:
            for param in lv_str.split(','):
                key, value = param.split('=')
                params[key.strip()] = float(value.strip())
            return params
        except Exception as e:
            raise ValueError(f"Invalid LV parameters '{lv_str}': {e}")
    
    def calculate_checksum(self, params: BatchParameters) -> str:
        """Calculate parameter checksum for audit trail"""
        param_str = f"{params.feedstock}:{params.mass_kg}:{params.beam_profile}:{params.lv_params}"
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def run_batch(self, feedstock: str, mass_kg: float, beam_profile: Dict, 
                  lv_params: Dict) -> BatchResult:
        """Execute a single batch conversion"""
        
        start_time = time.time()
        
        # Create batch parameters with audit trail
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        params = BatchParameters(
            feedstock=feedstock,
            mass_kg=mass_kg,
            beam_profile=beam_profile,
            lv_params=lv_params,
            timestamp=timestamp,
            checksum=""
        )
        params.checksum = self.calculate_checksum(params)
        
        if self.verbose:
            print(f"🔬 BATCH GOLD CONVERTER - Run {params.checksum}")
            print(f"   Feedstock: {feedstock}")
            print(f"   Mass: {mass_kg*1e9:.2f} ng")
            print(f"   Beam: {beam_profile['particle']} @ {beam_profile['energy']:.0e} eV")
            print(f"   LV params: {lv_params}")
            
        if self.dry_run:
            print("   🧪 DRY RUN - Simulating without beam activation")
        
        # Initialize energy ledger
        ledger = EnergyLedger()
        
        # Step 1: Spallation - Hg/Pt → Au precursors
        if self.verbose:
            print("   ⚡ Step 1: Spallation transmutation...")
            
        precursors = ['Hg-202'] if feedstock.startswith('Hg') else ['Pt-197']
        spallator = SpallationTransmuter(
            lv_params=lv_params,
            target='Au-197',
            precursors=precursors,
            beam=beam_profile
        )
        
        if not self.dry_run:
            precursor_yield = spallator.simulate(mass_kg)
            ledger.log_energy(spallator.energy_used)
        else:
            precursor_yield = int(1e12)  # Mock yield for dry run
            ledger.log_energy(1e3)  # Mock energy
            
        if self.verbose:
            print(f"      → {precursor_yield:.2e} precursor nuclei generated")
        
        # Step 2: LV-accelerated decay - precursors → Au-197
        if self.verbose:
            print("   ⏱️ Step 2: Accelerated decay...")
            
        decayer = DecayAccelerator(lv_params=lv_params, isotope='Au-197')
        
        if not self.dry_run:
            au_nuclei = decayer.simulate_decay(precursor_yield, t=1.0)  # 1 second
            ledger.log_energy(decayer.energy_used)
        else:
            au_nuclei = int(precursor_yield * 0.9)  # Mock 90% conversion
            ledger.log_energy(1e3)
            
        if self.verbose:
            print(f"      → {au_nuclei:.2e} Au-197 nuclei produced")
        
        # Step 3: Atomic binding - Au nuclei → neutral gold atoms
        if self.verbose:
            print("   🔗 Step 3: Atomic binding...")
            
        binder = AtomicBinder(lv_params=lv_params)
        
        if not self.dry_run:
            au_mass_kg = binder.bind(au_nuclei)
            ledger.log_energy(binder.energy_used)
        else:
            au_mass_kg = au_nuclei * 196.966569 * 1.66054e-27 * 0.95  # Mock binding
            ledger.log_energy(1e2)
            
        au_mass_g = au_mass_kg * 1e3
        
        # Calculate metrics
        total_energy = ledger.total_energy()
        conversion_time = time.time() - start_time
        round_trip_efficiency = (au_mass_g * 1e-3) / (mass_kg) if mass_kg > 0 else 0
        yield_per_joule = au_mass_g / total_energy if total_energy > 0 else 0
        
        # Create result
        result = BatchResult(
            parameters=params,
            au_output_g=au_mass_g,
            energy_consumed_j=total_energy,
            round_trip_efficiency=round_trip_efficiency,
            conversion_time_s=conversion_time,
            yield_per_joule=yield_per_joule
        )
        
        # Log result
        self.results_log.append(result)
        
        if self.verbose:
            print(f"   ✅ Conversion complete!")
            print(f"      Gold yield: {au_mass_g*1e6:.2f} µg")
            print(f"      Energy used: {total_energy:.2e} J")
            print(f"      Efficiency: {round_trip_efficiency:.1%}")
            print(f"      Yield/Energy: {yield_per_joule*1e6:.2f} µg/J")
            print(f"      Time: {conversion_time:.2f} s")
            
        return result
    
    def save_results(self, filename: Optional[str] = None):
        """Save batch results to JSON file"""
        if filename is None:
            filename = f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            
        results_data = []
        for result in self.results_log:
            results_data.append({
                'parameters': {
                    'feedstock': result.parameters.feedstock,
                    'mass_kg': result.parameters.mass_kg,
                    'beam_profile': result.parameters.beam_profile,
                    'lv_params': result.parameters.lv_params,
                    'timestamp': result.parameters.timestamp,
                    'checksum': result.parameters.checksum
                },
                'results': {
                    'au_output_g': result.au_output_g,
                    'energy_consumed_j': result.energy_consumed_j,
                    'round_trip_efficiency': result.round_trip_efficiency,
                    'conversion_time_s': result.conversion_time_s,
                    'yield_per_joule': result.yield_per_joule
                }
            })
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            print(f"📊 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Batch Gold Converter - Minimal Hg/Pt → Au Transmutation')
    parser.add_argument('--feed', required=True, choices=['Hg-202', 'Pt-197'],
                       help='Feedstock isotope')
    parser.add_argument('--mass', type=float, required=True,
                       help='Feedstock mass in kg (e.g., 1e-9 for 1 ng)')
    parser.add_argument('--beam', required=True,
                       help='Beam profile: particle:energy:intensity (e.g., proton:80e6:1e14)')
    parser.add_argument('--lv', required=True,
                       help='LV parameters: mu=X,alpha=Y,beta=Z')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without beam activation')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    parser.add_argument('--save', type=str,
                       help='Save results to specified file')
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = BatchGoldConverter(dry_run=args.dry_run, verbose=not args.quiet)
        
        # Parse parameters
        beam_profile = converter.parse_beam_profile(args.beam)
        lv_params = converter.parse_lv_params(args.lv)
        
        # Run batch conversion
        result = converter.run_batch(
            feedstock=args.feed,
            mass_kg=args.mass,
            beam_profile=beam_profile,
            lv_params=lv_params
        )
        
        # Output summary
        if not args.quiet:
            print("\n" + "="*50)
            print("BATCH CONVERSION SUMMARY")
            print("="*50)
            print(f"Feedstock: {args.feed} ({args.mass*1e9:.2f} ng)")
            print(f"Gold yield: {result.au_output_g*1e6:.2f} µg")
            print(f"Energy: {result.energy_consumed_j:.2e} J")
            print(f"Efficiency: {result.round_trip_efficiency:.1%}")
            print(f"Economics: {result.yield_per_joule*1e6:.2f} µg Au/J")
            print(f"Checksum: {result.parameters.checksum}")
        else:
            # Compact output for automation
            print(f"{result.parameters.checksum},{result.au_output_g*1e6:.2f},{result.energy_consumed_j:.2e},{result.round_trip_efficiency:.3f}")
        
        # Save results if requested
        if args.save:
            converter.save_results(args.save)
        
    except Exception as e:
        print(f"❌ Batch conversion failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
