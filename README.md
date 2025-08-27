# Elemental Transmutator

## Related Repositories

- [energy](https://github.com/arcticoder/energy): Central meta-repo for all energy and quantum research. The elemental transmutator is integrated for simulation, digital twin, and advanced energy applications.
- [unified-gut-polymerization](https://github.com/arcticoder/unified-gut-polymerization): Shares theoretical models and simulation infrastructure for matter transmutation and GUT-scale processes.
- [polymerized-lqg-matter-transporter](https://github.com/arcticoder/polymerized-lqg-matter-transporter): Related for matter transport and transformation at the quantum level.

All repositories are part of the [arcticoder](https://github.com/arcticoder) ecosystem and link back to the energy framework for unified documentation and integration.

This repository contains research-stage code and simulation artifacts exploring photonuclear transmutation pathways. The included materials are intended for hypothesis exploration and reproducibility of simulation results rather than as validated experimental protocols or production-ready processes.

## Summary — Research-Stage Results (June 2025)

Preliminary simulation runs identified candidate transmutation pathways under the models' stated assumptions. Reported metrics (yields, FOM, and profit estimates) are model outputs and subject to substantial uncertainty; they should be treated as provisional and used only to guide further study and experimental design.

### Example Candidate Pathways (model outputs)
- Reported candidate pathways include uranium-based, thorium-based, and tantalum-based chains. Detailed numeric results are simulation artifacts and require uncertainty quantification and experimental validation.

**Note:** These results are not experimentally validated and are not an endorsement of operational feasibility.

## Features (research-stage)

- Digital twin & analysis prototypes for pathway exploration
- Multi-stage pathway modeling for simulation studies
- Pulsed-beam modeling (simulation outputs; not validated experimentally)
- Economic scenario tools for illustrative analysis
- Configurable isotope examples for simulation
- Model-parameterized LV terms included for sensitivity studies
- Unit tests and CI for software correctness (does not validate experimental claims)

## Quick Start

### Enhanced Pathway Analysis (Recommended)
```bash
# Run pathway demonstration
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
Enhancement factors reported by the simulation models (examples):
- U-238: model-reported values up to ~4.2× (simulation)
- Ta-181: model-reported values ~2.8–3.1× (simulation)
- Bi-209: model-reported values ~1.85–2.2× (simulation)
- Pt-195: model-reported values ~1.4–2.15× (simulation)

These enhancement factors are conditional on modeling assumptions and require experimental validation before use in operational planning.

## Supported Elements

The system supports many elements via atomic number mapping (examples listed below):

- **`prototyping/test_enhanced_pathways.py`**: Comprehensive test suite (9 tests; currently passing in CI)

### Gold Production
- **Rate enhancement**: Model-estimated rate enhancements reported up to 10^3–10^6× under certain parameterizations in simulation; these values are model outputs and require experimental validation and uncertainty quantification.
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

Simulation outputs are saved to `transmutation_results.json`. Example records included in the repository are model outputs and should be treated as simulation artifacts rather than experimental measurements. Maintain provenance information (seed, environment, parameters) in `docs/` to support reproducibility.

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
The system provides economic analysis including:
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

The Unlicense - Free for research and educational use. Users are responsible for complying with applicable laws and safety protocols.

---

## Scope / Validation & Limitations

- **Research-stage artifacts**: Content in this repository consists primarily of simulation code, example runs, and exploratory analyses. It is intended for research, reproducibility, and hypothesis generation.
- **Uncertainty**: Reported numeric outputs should be accompanied by uncertainty quantification and sensitivity analysis before being used for decision-making. If maintainers publish numeric claims, attach UQ artifacts (confidence intervals, sensitivity results) under `docs/`.
- **Experimental caution**: Physical experimentation with radiation-producing equipment requires institutional oversight, safety approvals, and regulatory compliance. Do not attempt any experimental work without appropriate facilities and authorizations.
- **Provenance**: Re-run simulations using the provided scripts and document runtime environment, random seeds, and parameter files in `docs/` to enable independent verification.

If you are a maintainer preparing public-facing summaries, prefer conservative phrasing and link to `docs/` artifacts that demonstrate reproducibility and UQ.
