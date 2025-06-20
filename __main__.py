#!/usr/bin/env python3
"""
Elemental Transmutator Main Entry Point
=======================================

Element-agnostic transmutation engine driven by configuration.
Supports any target isotope (Au, Pt, Pd, etc.) from configurable feedstock.
"""

import json
import logging
import numpy as np

# Simple direct imports
import spallation_transmutation
import photonuclear_transmutation
import decay_accelerator
import atomic_binder
import energy_ledger

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
    
    logger.info(f"Starting transmutation: {feedstock_isotope} -> {target_isotope}")
    
    # Initialize energy ledger
    ledger = energy_ledger.EnergyLedger()
    
    # Step 1: Choose transmutation method based on config
    use_photonuclear = cfg.get("use_photonuclear", False)
    sample_mass_g = 1.0  # Standard sample size
    
    if use_photonuclear:
        logger.info("Using photonuclear (GDR) transmutation pathway")
        
        # Create photonuclear config
        photon_config = photonuclear_transmutation.PhotonuclearConfig(
            target_isotope=target_isotope,
            feedstock_isotope=feedstock_isotope,
            mu_lv=lv_params.get("mu", 2.5e-12),
            alpha_lv=lv_params.get("alpha", 0.85),
            beta_lv=lv_params.get("beta", 0.65),
            gamma_energy_MeV=cfg.get("photon_beam", {}).get("energy_MeV", 15.0),
            photon_flux=cfg.get("photon_beam", {}).get("flux", 1e13),
            beam_duration_s=duration_s
        )
        
        transmuter = photonuclear_transmutation.PhotonuclearTransmuter(photon_config)
        transmutation_result = transmuter.transmute_sample(sample_mass_g, duration_s)
        
    else:
        logger.info("Using spallation transmutation pathway")
        
        # Create spallation config
        spallation_config = spallation_transmutation.TransmutationConfig(
            target_isotope=target_isotope,
            feedstock_isotope=feedstock_isotope,
            mu_lv=lv_params.get("mu", 1e-17),
            alpha_lv=lv_params.get("alpha", 1e-14),
            beta_lv=lv_params.get("beta", 1e-11),
            beam_type=beam_profile.get("type", "deuteron"),
            beam_energy=beam_profile.get("energy_MeV", 80) * 1e6,  # Convert to eV
            beam_flux=beam_profile.get("flux", 1e14),
            irradiation_time_s=duration_s
        )
        
        transmuter = spallation_transmutation.SpallationTransmuter(spallation_config)
        transmutation_result = transmuter.transmute_sample(sample_mass_g, duration_s)
    
    # Extract the total yield for subsequent processing
    total_yield_mass_g = transmutation_result['summary']['total_yield_mass_g']
    
    # Step 2: Decay Acceleration (if needed)
    if decay_time_s > 0:
        decay_config = decay_accelerator.DecayConfig(
            target_isotope=target_isotope,
            mu_lv=lv_params.get("mu", 1e-17),
            alpha_lv=lv_params.get("alpha", 1e-14),
            beta_lv=lv_params.get("beta", 1e-11),
            acceleration_time=decay_time_s
        )
        
        decayer = decay_accelerator.DecayAccelerator(decay_config, ledger)
        # For decay acceleration, use the yield mass as input nuclei
        final_nuclei = decayer.simulate_decay({'nuclei': total_yield_mass_g * 6.022e23 / 197}, t=decay_time_s)
    else:
        final_nuclei = {'nuclei': total_yield_mass_g * 6.022e23 / 197}  # Convert mass to nuclei count
    
    # Step 3: Atomic Binding
    binder = atomic_binder.AtomicBinder(lv_params, target_isotope)
    atoms = binder.bind(final_nuclei)    # Final results
    print(f"\n{'='*50}")
    print(f"TRANSMUTATION COMPLETE")
    print(f"{'='*50}")
    print(f"Target element: {target_isotope}")
    print(f"Input mass: {sample_mass_g:.3f} g")
    print(f"Output mass: {total_yield_mass_g*1000:.6f} mg")
    print(f"Conversion efficiency: {transmutation_result['summary']['conversion_efficiency']:.6f}%")
    print(f"LV enhancement: {transmutation_result['summary']['lv_total_enhancement']:.2f}×")
    
    # Economic analysis
    economic_params = cfg.get("economic_params", {})
    if economic_params:
        target_price = economic_params.get("target_market_price_per_kg", 62000000)  # Au price
        revenue = total_yield_mass_g * target_price / 1000  # Convert g to kg
        
        feedstock_cost = economic_params.get("feedstock_cost_per_kg", 0.12)
        material_cost = sample_mass_g * feedstock_cost / 1000  # Convert g to kg
        
        energy_cost_kwh = economic_params.get("energy_cost_per_kwh", 0.10)
        
        # Check if LV self-powered mode is enabled
        use_lv_self_powered = cfg.get("use_lv_self_powered", False)
        
        if use_lv_self_powered:
            # Zero energy cost for LV self-powered operation
            energy_used_kwh = 0.0
            energy_cost = 0.0
            logger.info("Using LV self-powered mode: Zero energy cost")
        else:
            # Estimate energy cost based on transmutation method
            if use_photonuclear:
                # Photon beam power estimation
                photon_energy_j = cfg.get("photon_beam", {}).get("energy_MeV", 15.0) * 1.6e-13  # Convert MeV to J
                photon_flux = cfg.get("photon_beam", {}).get("flux", 1e13)
                beam_area_m2 = np.pi * 0.01**2  # Assume 1 cm radius beam
                beam_power_w = photon_energy_j * photon_flux * beam_area_m2
                energy_used_kwh = beam_power_w * (duration_s / 3600) / 1000  # Convert to kWh
            else:
                # Spallation beam power estimation  
                beam_power_mw = 0.1  # Assume 100 kW for simplicity
                energy_used_kwh = beam_power_mw * 1000 * (duration_s / 3600)  # Convert MW to kW and s to h
            
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
        "sample_mass_g": sample_mass_g,
        "output_mass_g": total_yield_mass_g,        "conversion_efficiency": transmutation_result['summary']['conversion_efficiency'],
        "lv_enhancement": transmutation_result['summary']['lv_total_enhancement'],
        "energy_used_kwh": energy_used_kwh if 'energy_used_kwh' in locals() else 0,
        "config_used": cfg
    }
    
    with open("transmutation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to transmutation_results.json")

if __name__ == "__main__":
    main()
