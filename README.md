# Elemental Transmutator

An element-agnostic nuclear transmutation engine leveraging Lorentz violation (LV) physics for efficient production of any target isotope.

## Features

- **Element-Agnostic**: Configure any target isotope (Au, Pt, Pd, etc.) from any feedstock
- **LV-Enhanced**: Uses Lorentz violation physics for enhanced cross-sections and accelerated decay
- **Configurable**: JSON-based configuration for easy parameter adjustment
- **Economic Analysis**: Built-in cost/revenue analysis for production feasibility
- **Modular Design**: Separate modules for spallation, decay acceleration, and atomic binding

## Quick Start

1. **Configure your target**: Edit `config.json` to specify your desired element
```json
{
  "target_isotope": "Au-197",
  "feedstock_isotope": "Fe-56",
  "beam_profile": {
    "type": "deuteron",
    "energy_MeV": 80,
    "flux": 1e14
  },
  "lv_params": {
    "mu": 1e-17,
    "alpha": 1e-14,
    "beta": 1e-11
  }
}
```

2. **Run transmutation**:
```bash
python __main__.py
```

## Supported Elements

The system supports any element via atomic number mapping:
- **Gold (Au)**: Au-197 - Premium precious metal
- **Platinum (Pt)**: Pt-195 - Industrial catalyst applications  
- **Palladium (Pd)**: Pd-105 - Automotive catalysts
- **Rhodium (Rh)**: Rh-103 - High-value catalyst
- **Iron (Fe)**: Fe-56 - Cheap feedstock material
- And many more...

## Configuration Examples

### Gold Production
```json
{
  "target_isotope": "Au-197",
  "feedstock_isotope": "Fe-56",
  "economic_params": {
    "target_market_price_per_kg": 62000000
  }
}
```

### Platinum Production
```json
{
  "target_isotope": "Pt-195", 
  "feedstock_isotope": "Fe-56",
  "economic_params": {
    "target_market_price_per_kg": 30000000
  }
}
```

## Physics Overview

### Spallation Transmutation
- **Cross-sections**: Enhanced from mb to barns via LV effects
- **Direct production**: Single-step spallation vs multi-step decay chains
- **Energy range**: 20-200 MeV proton/deuteron beams

### LV Enhancement Formula
```
σ = σ₀ × (A_feedstock)^α × (E_beam)^β × f_LV
```

Where:
- `σ₀`: Base cross-section (50 mb)
- `α`: Mass dependence (0.7)
- `β`: Energy dependence (0.3)  
- `f_LV`: Lorentz violation enhancement factor

### Decay Acceleration
- **Rate enhancement**: 10³-10⁶× faster decay via LV field engineering
- **Matrix elements**: Modified by μ coefficient
- **Phase space**: Enhanced by β coefficient

## Economic Analysis

The system provides automatic economic analysis including:
- **Revenue**: Mass produced × market price
- **Costs**: Materials + energy + facility overhead
- **ROI**: Return on investment calculation
- **Break-even**: Analysis for commercial viability

## Output

Results are saved to `transmutation_results.json`:
```json
{
  "target_isotope": "Au-197",
  "feedstock_isotope": "Fe-56", 
  "mass_produced_kg": 1.23e-9,
  "atoms_bound": 3.76e+15,
  "binding_efficiency": 0.99,
  "energy_input_j": 767000000
}
```

## Module Structure

- **`spallation_transmutation.py`**: High-energy spallation for direct isotope production
- **`decay_accelerator.py`**: LV-enhanced nuclear decay acceleration
- **`atomic_binder.py`**: Electron capture and atomic assembly
- **`energy_ledger.py`**: Comprehensive energy accounting
- **`__main__.py`**: Main execution pipeline

## Mathematics

The core transmutation equation for each stage `i`:

$$Y_i = N_{\rm feedstock} \cdot \sigma_i(E) \cdot \Phi_i \cdot t_i$$

Where:
- $N_{\rm feedstock}$: Number of target nuclei
- $\sigma_i(E)$: LV-enhanced cross-section (barns)
- $\Phi_i$: Beam flux (particles/cm²/s)  
- $t_i$: Irradiation time (s)

## Requirements

- Python 3.7+
- NumPy
- JSON (built-in)
- Logging (built-in)

## License

MIT License - Free for research and commercial use.

---

**Note**: This is a theoretical framework for nuclear transmutation research. Actual implementation would require sophisticated accelerator facilities and safety protocols.
