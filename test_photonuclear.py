#!/usr/bin/env python3
import photonuclear_transmutation

transmuter = photonuclear_transmutation.PhotonuclearTransmuter()
result = transmuter.transmute_sample(1.0)

print('🚀 PHOTONUCLEAR GDR TRANSMUTATION RESULTS:')
print(f'Input: 1.0 g {result["summary"]["feedstock_isotope"]}')
print(f'Output: {result["summary"]["output_mass_g"]:.6f} g {result["summary"]["target_isotope"]}')
print(f'Conversion efficiency: {result["summary"]["conversion_efficiency"]:.6f}%')
print(f'Cross-section: {result["cross_section_mb"]:.1f} mb')
print(f'LV enhancement: {result["summary"]["lv_total_enhancement"]:.2f}×')
print(f'Expected improvement vs spallation: {result["summary"]["yield_improvement_vs_spallation"]}')

# Test energy optimization
print(f'\n🔬 OPTIMIZING GAMMA ENERGY:')
optimization = transmuter.optimize_gamma_energy()
print(f'Optimal energy: {optimization["optimal_energy_MeV"]:.1f} MeV')
print(f'Optimal yield: {optimization["optimal_yield_g"]:.6f} g')
print(f'Improvement factor: {optimization["improvement_factor"]:.2f}×')
