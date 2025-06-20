#!/usr/bin/env python3

import numpy as np

def test_photon_calculations():
    """Test photon flux calculations used in simulation."""
    
    # Typical simulation parameters
    beam_current_ua = 10.0
    beam_energy_mev = 20.0
    irradiation_time_s = 3600 * 5  # 5 hours 
    target_mass_g = 4.0
    target_density_g_cm3 = 11.3

    print(f"Input parameters:")
    print(f"  Beam current: {beam_current_ua} μA")
    print(f"  Beam energy: {beam_energy_mev} MeV")
    print(f"  Irradiation time: {irradiation_time_s} s ({irradiation_time_s/3600:.1f} hours)")
    print(f"  Target mass: {target_mass_g} g")
    print(f"  Target density: {target_density_g_cm3} g/cm³")

    # Calculate photon fluence (same as simulation code)
    beam_power_mev_per_s = beam_current_ua * 1e-6 * 6.24e12 * beam_energy_mev
    total_photons = beam_power_mev_per_s * irradiation_time_s / beam_energy_mev

    # Target area calculation
    target_volume_cm3 = target_mass_g / target_density_g_cm3
    target_radius_cm = (target_volume_cm3 / (2 * np.pi))**(1/3)
    target_area_cm2 = np.pi * target_radius_cm**2

    fluence_per_cm2 = total_photons / target_area_cm2

    print(f"\nCalculated values:")
    print(f"  Beam power: {beam_power_mev_per_s:.2e} MeV/s")
    print(f"  Total photons: {total_photons:.2e}")
    print(f"  Target volume: {target_volume_cm3:.2f} cm³")
    print(f"  Target radius: {target_radius_cm:.2f} cm")
    print(f"  Target area: {target_area_cm2:.2f} cm²")
    print(f"  Fluence: {fluence_per_cm2:.2e} photons/cm²")
    
    # Test with atomic efficiency
    try:
        from atomic_binder import AtomicDataBinder
        
        binder = AtomicDataBinder()
        target_mix = {'Pb-208': 0.5, 'Bi-209': 0.3, 'Tl-203': 0.2}
        
        efficiency = binder.calculate_gold_production_efficiency(target_mix, beam_energy_mev, fluence_per_cm2)
        gold_atoms = int(efficiency * total_photons)
        gold_mass_mg = gold_atoms * 197 / 6.022e23 * 1000
        
        print(f"\nGold production estimate:")
        print(f"  Production efficiency: {efficiency:.2e}")
        print(f"  Gold atoms produced: {gold_atoms:,}")
        print(f"  Gold mass: {gold_mass_mg:.9f} mg")
        
    except ImportError as e:
        print(f"\nCould not import atomic_binder: {e}")

if __name__ == "__main__":
    test_photon_calculations()
