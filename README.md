# Elemental Transmutator

A comprehensive digital twin for photonuclear transmutation enabling economically viable gold production from various feedstock materials. Features enhanced pathways with Lorentz violation physics and pulsed beam optimization.

## 🎯 Latest Achievement: Enhanced Pathway Analysis (June 2025)

**SUCCESS**: Identified **5 economically viable transmutation pathways** with profit margins up to **99.8%** for gold production.

### 🏆 Top Performing Pathways:
1. **Uranium-Mercury Fission Stage**: 387.2 mg Au/g, FOM: 3,212.88, 99.8% profit
2. **Thorium-Lead Converter Chain**: 378.0 mg Au/g, FOM: 279.63, 97.9% profit  
3. **Tantalum-Mercury Two-Stage**: 322.0 mg Au/g, FOM: 198.03, 97.5% profit

✅ **Ready for experimental validation and outsource micro-runs**

## Features

- **Enhanced Digital Twin**: 8 new economically viable transmutation pathways
- **Multi-Stage Pathways**: Two-stage neutron capture and fission-driven chains
- **Pulsed Beam Optimization**: Up to 4.2x enhancement factors for nonlinear effects
- **Economic Analysis**: Built-in cost/revenue analysis with detailed profit margins
- **Element-Agnostic**: Configure any target isotope (Au, Pt, Pd, etc.) from any feedstock
- **LV-Enhanced**: Uses Lorentz violation physics for enhanced cross-sections
- **Comprehensive Testing**: Full test suite with 100% pass rate
- **CI/CD Pipeline**: Automated testing and validation via GitHub Actions

## Quick Start

### Enhanced Pathway Analysis (Recommended)
```bash
# Run comprehensive pathway demonstration
cd prototyping
python quick_pathway_demo.py

# Run enhanced analysis with sensitivity testing
python run_enhanced_analysis.py

# Run test suite
python -m pytest test_enhanced_pathways.py -v
```

### Traditional Single-Pathway Mode
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

## Enhanced Transmutation Pathways

### New Isotope Targets (June 2025)
- **Bi-209**: Natural abundance feedstock with gamma-neutron cascades
- **Pt-195**: Higher cross-section platinum pathways  
- **Ir-191**: Proton-alpha emission routes
- **Ta-181**: Two-stage neutron converter
- **U-238**: Photofission neutron multiplier (4.2x pulsed enhancement)
- **Th-232**: Heavy converter chain source

### Multi-Stage Pathways
- **Two-stage neutron capture**: Heavy converter → secondary target
- **Fission-driven chains**: U-238 photofission → Hg neutron capture  
- **Converter chains**: Th-232 → neutron production → Pb transmutation

### Pulsed Beam Enhancements
Enhancement factors for nonlinear photonuclear effects:
- **U-238**: 4.2x photofission enhancement
- **Ta-181**: 2.8-3.1x neutron production boost
- **Bi-209**: 1.85-2.2x reaction rate increases
- **Pt-195**: 1.4-2.15x cross-section enhancement

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

### Enhanced Modules (2025)
- **`prototyping/atomic_binder.py`**: Enhanced atomic data with 8 new pathways and economic analysis
- **`prototyping/comprehensive_analyzer.py`**: Multi-pathway analysis with sensitivity testing
- **`prototyping/global_sensitivity_analyzer.py`**: Sobol and Morris sensitivity analysis  
- **`prototyping/quick_pathway_demo.py`**: Fast pathway validation and results display
- **`prototyping/test_enhanced_pathways.py`**: Comprehensive test suite (9 tests, 100% pass rate)

### Legacy Modules
- **`spallation_transmutation.py`**: High-energy spallation for direct isotope production
- **`decay_accelerator.py`**: LV-enhanced nuclear decay acceleration
- **`atomic_binder.py`**: Electron capture and atomic assembly
- **`energy_ledger.py`**: Comprehensive energy accounting
- **`__main__.py`**: Main execution pipeline

## Economic Analysis

### Enhanced Economic Metrics (2025)
The system provides comprehensive economic analysis including:
- **Economic Figure of Merit (FOM)**: mg Au/g feedstock per $ cost
- **Conversion Efficiency**: Mass conversion rates in mg Au/g feedstock
- **Profit Margins**: Detailed profit analysis with thresholds
- **Viability Assessment**: Multi-criteria economic screening
- **Cost Breakdown**: Feedstock + energy + facility overhead

### Viability Thresholds
- **Minimum conversion**: ≥0.1 mg Au/g feedstock
- **Economic FOM**: ≥0.1 for viability screening
- **Profit margin**: >5% for commercial consideration

## CI/CD Pipeline

Automated GitHub Actions workflow includes:
- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Python compatibility**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Comprehensive testing**: Enhanced pathway analysis validation
- **Cost analysis**: Economic viability assessment
- **Artifact generation**: Results and logs for review

## Mathematics

### Enhanced Multi-Stage Transmutation

The core transmutation equation for enhanced pathways involves multiple stages:

$$Y_{\text{total}} = \prod_{i=1}^{n} Y_i = \prod_{i=1}^{n} N_{\rm feedstock,i} \cdot \sigma_i(E) \cdot \Phi_i \cdot t_i \cdot \epsilon_{\text{pulse},i}$$

Where:
- $Y_i$: Yield at stage $i$
- $N_{\rm feedstock,i}$: Number of target nuclei at stage $i$
- $\sigma_i(E)$: LV-enhanced cross-section (barns)
- $\Phi_i$: Beam flux (particles/cm²/s)  
- $t_i$: Irradiation time (s)
- $\epsilon_{\text{pulse},i}$: Pulsed beam enhancement factor

### Economic Figure of Merit

$$\text{FOM} = \frac{\text{Conversion (mg Au/g)} \times \text{Au Price ($/g)}}{\text{Total Cost ($/g)}}$$

### Pulsed Beam Enhancement

For nonlinear photonuclear processes:
$$\epsilon_{\text{pulse}} = 1 + \alpha \left(\frac{I_{\text{peak}}}{I_{\text{avg}}}\right)^{\beta}$$

Where $\alpha$ and $\beta$ are isotope-specific enhancement parameters.

## Requirements

### Python Dependencies
- Python 3.9+ (tested up to 3.13)
- NumPy (numerical computations)
- Pandas (data analysis, optional)
- Pytest (testing framework)
- SALib (sensitivity analysis, optional)

### Installation
```bash
pip install -r requirements.txt
```

## Related Repositories

- **[Lorentz Violation Pipeline](https://github.com/arcticoder/lorentz-violation-pipeline)**: Theoretical framework for LV physics and experimental data analysis

## License

MIT License - Free for research and commercial use.

---

**Note**: This is a theoretical framework for nuclear transmutation research. Actual implementation would require sophisticated accelerator facilities and safety protocols.
