#!/usr/bin/env python3
import spallation_transmutation

transmuter = spallation_transmutation.SpallationTransmuter()
result = transmuter.transmute_sample(1.0)

print('Direct spallation result:')
print(f'Total yield: {result["summary"]["total_yield_mass_g"]:.8f} g')
print(f'Conversion efficiency: {result["summary"]["conversion_efficiency"]:.8f}%')

for channel, data in result.items():
    if channel != 'summary':
        print(f'{channel}: {data["transmuted_mass_g"]:.8f} g, σ={data["cross_section_mb"]:.2f} mb')
