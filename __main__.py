#!/usr/bin/env python3
"""
Elemental Transmutator Main Entry Point
=======================================

Element-agnostic transmutation engine driven by configuration.
Supports any target isotope (Au, Pt, Pd, etc.) from configurable feedstock.
"""

import json
import logging
from spallation_transmutation import SpallationTransmuter, SpallationConfig
from decay_accelerator import DecayAccelerator, DecayConfig
from atomic_binder import AtomicBinder
from energy_ledger import EnergyLedger

def main():
    """Main transmutation pipeline."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found! Please create configuration file.")
        return
    except Exception as e:
        logger.error(f"Error loading config.json: {e}")
        return
    
    # Extract configuration parameters
    target_isotope = cfg.get("target_isotope", "Au-197")
    feedstock_isotope = cfg.get("feedstock_isotope", "Fe-56")
    beam_profile = cfg.get("beam_profile", {})
    lv_params = cfg.get("lv_params", {})
    duration_s = cfg.get("duration_s", 60)
    decay_time_s = cfg.get("decay_time_s", 1)
    
    logger.info(f"Starting transmutation: {feedstock_isotope} → {target_isotope}")
    
    # Initialize energy ledger
    ledger = EnergyLedger()
    
    # Step 1: Spallation Transmutation
    spallation_config = SpallationConfig(
        target_isotope=target_isotope,
        feedstock_isotope=feedstock_isotope,
        mu_lv=lv_params.get("mu", 1e-17),
        alpha_lv=lv_params.get("alpha", 1e-14),
        beta_lv=lv_params.get("beta", 1e-11),
        beam_type=beam_profile.get("type", "deuteron"),
        beam_energy=beam_profile.get("energy_MeV", 80) * 1e6,  # Convert to eV
        beam_flux=beam_profile.get("flux", 1e14),
        beam_duration=duration_s
    )
    
    transmuter = SpallationTransmuter(spallation_config, ledger)
    yields = transmuter.simulate(duration=duration_s)
    
    # Step 2: Decay Acceleration (if needed)
    if decay_time_s > 0:
        decay_config = DecayConfig(
            target_isotope=target_isotope,
            mu_lv=lv_params.get("mu", 1e-17),
            alpha_lv=lv_params.get("alpha", 1e-14),
            beta_lv=lv_params.get("beta", 1e-11),
            acceleration_time=decay_time_s
        )
        
        decayer = DecayAccelerator(decay_config, ledger)
        final_nuclei = decayer.simulate_decay(yields, t=decay_time_s)
    else:
        final_nuclei = yields
    
    # Step 3: Atomic Binding
    binder = AtomicBinder(lv_params, target_isotope)
    atoms = binder.bind(final_nuclei)
    
    # Final results
    print(f"\n{'='*50}")
    print(f"TRANSMUTATION COMPLETE")
    print(f"{'='*50}")
    print(f"Target element: {target_isotope}")
    print(f"Total mass produced: {atoms.mass*1e6:.3f} mg")
    print(f"Atoms bound: {atoms.atoms_bound:.2e}")
    print(f"Binding efficiency: {atoms.binding_efficiency:.2%}")
    
    # Economic analysis
    economic_params = cfg.get("economic_params", {})
    if economic_params:
        target_price = economic_params.get("target_market_price_per_kg", 62000000)  # Au price
        revenue = atoms.mass * target_price
        
        feedstock_cost = economic_params.get("feedstock_cost_per_kg", 0.12)
        feedstock_mass = spallation_config.target_mass
        material_cost = feedstock_mass * feedstock_cost
        
        energy_cost_kwh = economic_params.get("energy_cost_per_kwh", 0.10)
        energy_used_kwh = ledger.get_total_energy_input() / 3.6e6  # Convert J to kWh
        energy_cost = energy_used_kwh * energy_cost_kwh
        
        overhead = economic_params.get("facility_overhead_per_hour", 1000)
        time_hours = duration_s / 3600
        overhead_cost = overhead * time_hours
        
        total_cost = material_cost + energy_cost + overhead_cost
        profit = revenue - total_cost
        roi = (profit / total_cost * 100) if total_cost > 0 else 0
        
        print(f"\nECONOMIC ANALYSIS:")
        print(f"Revenue: ${revenue:,.2f}")
        print(f"Costs: ${total_cost:,.2f}")
        print(f"  - Materials: ${material_cost:,.2f}")
        print(f"  - Energy: ${energy_cost:,.2f}")
        print(f"  - Overhead: ${overhead_cost:,.2f}")
        print(f"Profit: ${profit:,.2f}")
        print(f"ROI: {roi:.1f}%")
    
    # Save results
    results = {
        "target_isotope": target_isotope,
        "feedstock_isotope": feedstock_isotope,
        "mass_produced_kg": atoms.mass,
        "atoms_bound": atoms.atoms_bound,
        "binding_efficiency": atoms.binding_efficiency,
        "energy_input_j": ledger.get_total_energy_input(),
        "config_used": cfg
    }
    
    with open("transmutation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to transmutation_results.json")

if __name__ == "__main__":
    main()
