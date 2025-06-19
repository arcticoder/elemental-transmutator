#!/usr/bin/env python3
"""
Experimental Data Integration Module
===================================

Real-world test data ingestion, parameter fitting, and simulation updating framework.
Bridges simulation predictions with experimental reality for robust pilot plant operation.

Author: Advanced Energy Research Team
License: MIT
"""

import numpy as np
import pandas as pd
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2, t
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentalDataPoint:
    """Single experimental measurement with metadata"""
    timestamp: str
    measurement_type: str  # 'cross_section', 'lv_parameter', 'beam_efficiency', 'yield'
    target_isotope: str
    beam_energy: float  # MeV
    measured_value: float
    uncertainty: float
    systematic_error: float
    experimental_conditions: Dict[str, Any]
    detector_calibration: Dict[str, float]
    environmental_factors: Dict[str, float]
    operator_id: str
    run_id: str

@dataclass
class ParameterFitResult:
    """Results from parameter fitting to experimental data"""
    parameter_name: str
    fitted_value: float
    uncertainty: float
    confidence_interval_95: Tuple[float, float]
    chi_squared: float
    degrees_of_freedom: int
    p_value: float
    fit_quality: str  # 'excellent', 'good', 'acceptable', 'poor'
    correlation_matrix: Dict[str, Dict[str, float]]

class ExperimentalDataIntegrator:
    """
    Integrates real-world experimental data with simulation models.
    Performs parameter fitting, uncertainty quantification, and model updating.
    """
    
    def __init__(self, data_dir: str = "experimental_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage for experimental data
        self.data_points: List[ExperimentalDataPoint] = []
        self.fit_results: Dict[str, ParameterFitResult] = {}
        
        # Simulation parameter bounds (physical constraints)
        self.parameter_bounds = {
            'lv_xi': (1e-20, 1e-15),  # Lorentz violation parameter range
            'lv_eta': (1e-22, 1e-17),  # SME parameter range
            'beam_focusing_efficiency': (0.1, 0.95),
            'target_density_factor': (0.8, 1.2),
            'detector_efficiency': (0.05, 0.99),
            'cross_section_scaling': (0.1, 10.0),
        }
        
        # Reference simulation values (to be updated)
        self.reference_parameters = {
            'lv_xi': 3.2e-18,
            'lv_eta': 1.1e-19,
            'beam_focusing_efficiency': 0.85,
            'target_density_factor': 1.0,
            'detector_efficiency': 0.15,
            'cross_section_scaling': 1.0,
        }
        
        # Experimental data schemas for validation
        self.required_fields = [
            'timestamp', 'measurement_type', 'target_isotope', 
            'beam_energy', 'measured_value', 'uncertainty'
        ]
        
    def ingest_data_csv(self, csv_path: str) -> int:
        """
        Ingest experimental data from CSV file.
        
        Returns:
            Number of data points successfully ingested
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading experimental data from {csv_path}")
            
            # Validate required columns
            missing_cols = [col for col in self.required_fields if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            ingested_count = 0
            for _, row in df.iterrows():
                try:
                    # Parse experimental conditions from JSON string if present
                    exp_conditions = {}
                    if 'experimental_conditions' in row and pd.notna(row['experimental_conditions']):
                        exp_conditions = json.loads(row['experimental_conditions'])
                    
                    # Parse detector calibration
                    detector_cal = {}
                    if 'detector_calibration' in row and pd.notna(row['detector_calibration']):
                        detector_cal = json.loads(row['detector_calibration'])
                    
                    # Parse environmental factors
                    env_factors = {}
                    if 'environmental_factors' in row and pd.notna(row['environmental_factors']):
                        env_factors = json.loads(row['environmental_factors'])
                    
                    data_point = ExperimentalDataPoint(
                        timestamp=row['timestamp'],
                        measurement_type=row['measurement_type'],
                        target_isotope=row['target_isotope'],
                        beam_energy=float(row['beam_energy']),
                        measured_value=float(row['measured_value']),
                        uncertainty=float(row['uncertainty']),
                        systematic_error=float(row.get('systematic_error', 0.0)),
                        experimental_conditions=exp_conditions,
                        detector_calibration=detector_cal,
                        environmental_factors=env_factors,
                        operator_id=row.get('operator_id', 'unknown'),
                        run_id=row.get('run_id', f"run_{ingested_count}")
                    )
                    
                    self.data_points.append(data_point)
                    ingested_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to parse row {row.name}: {e}")
                    
            logger.info(f"Successfully ingested {ingested_count} data points")
            return ingested_count
            
        except Exception as e:
            logger.error(f"Failed to ingest data from {csv_path}: {e}")
            return 0
    
    def ingest_data_realtime(self, data_point: ExperimentalDataPoint) -> bool:
        """
        Ingest single real-time experimental data point.
        
        Returns:
            Success status
        """
        try:
            # Validate data point
            if data_point.uncertainty <= 0:
                raise ValueError("Uncertainty must be positive")
            
            if data_point.measured_value <= 0 and data_point.measurement_type in ['cross_section', 'yield']:
                raise ValueError("Cross-section and yield measurements must be positive")
            
            self.data_points.append(data_point)
            
            # Auto-save for real-time data
            self.save_data_backup()
            
            logger.info(f"Ingested real-time data: {data_point.measurement_type} for {data_point.target_isotope}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest real-time data: {e}")
            return False
    
    def filter_data(self, 
                   measurement_type: Optional[str] = None,
                   target_isotope: Optional[str] = None,
                   energy_range: Optional[Tuple[float, float]] = None,
                   time_range: Optional[Tuple[str, str]] = None) -> List[ExperimentalDataPoint]:
        """Filter experimental data by criteria"""
        
        filtered_data = self.data_points.copy()
        
        if measurement_type:
            filtered_data = [d for d in filtered_data if d.measurement_type == measurement_type]
        
        if target_isotope:
            filtered_data = [d for d in filtered_data if d.target_isotope == target_isotope]
        
        if energy_range:
            e_min, e_max = energy_range
            filtered_data = [d for d in filtered_data if e_min <= d.beam_energy <= e_max]
        
        if time_range:
            t_start, t_end = time_range
            filtered_data = [d for d in filtered_data 
                           if t_start <= d.timestamp <= t_end]
        
        return filtered_data
    
    def fit_lv_parameters(self, measurement_type: str = 'cross_section') -> Dict[str, ParameterFitResult]:
        """
        Fit Lorentz violation parameters to experimental cross-section data.
        
        Returns:
            Dictionary of fitted parameters with uncertainties
        """
        # Filter relevant data
        data = self.filter_data(measurement_type=measurement_type)
        
        if len(data) < 3:
            raise ValueError(f"Insufficient data points for fitting: {len(data)}")
        
        logger.info(f"Fitting LV parameters to {len(data)} {measurement_type} measurements")
        
        # Extract data arrays
        energies = np.array([d.beam_energy for d in data])
        measurements = np.array([d.measured_value for d in data])
        uncertainties = np.array([d.uncertainty for d in data])
        
        # Define theoretical model (simplified LV-enhanced cross-section)
        def lv_cross_section_model(energy, xi, eta, sigma_0):
            """LV-enhanced cross-section model"""
            # Standard cross-section with LV enhancement
            lv_enhancement = 1.0 + xi * (energy / 100.0)**2 + eta * np.log(energy / 10.0)
            return sigma_0 * lv_enhancement * (energy / 100.0)**(-0.5)
        
        # Objective function for fitting
        def objective(params):
            xi, eta, sigma_0 = params
            model_pred = lv_cross_section_model(energies, xi, eta, sigma_0)
            chi_sq = np.sum(((measurements - model_pred) / uncertainties)**2)
            return chi_sq
        
        # Parameter bounds for fitting
        bounds = [
            self.parameter_bounds['lv_xi'],
            self.parameter_bounds['lv_eta'],
            (1e-30, 1e-24)  # Cross-section scaling bound
        ]
        
        # Initial guess
        initial_guess = [
            self.reference_parameters['lv_xi'],
            self.reference_parameters['lv_eta'],
            1e-27  # Typical cross-section scale
        ]
        
        # Perform fitting using differential evolution (global optimizer)
        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
        
        if not result.success:
            logger.warning("Parameter fitting did not converge")
        
        xi_fit, eta_fit, sigma_0_fit = result.x
        chi_squared = result.fun
        dof = len(data) - 3  # degrees of freedom
        
        # Calculate parameter uncertainties using Hessian approximation
        def hessian_objective(params):
            return objective(params)
        
        # Numerical Hessian calculation
        eps = 1e-8
        hessian = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                params_pp = result.x.copy()
                params_pm = result.x.copy()
                params_mp = result.x.copy()
                params_mm = result.x.copy()
                
                params_pp[i] += eps
                params_pp[j] += eps
                params_pm[i] += eps
                params_pm[j] -= eps
                params_mp[i] -= eps
                params_mp[j] += eps
                params_mm[i] -= eps
                params_mm[j] -= eps
                
                hessian[i, j] = (objective(params_pp) - objective(params_pm) - 
                               objective(params_mp) + objective(params_mm)) / (4 * eps**2)
        
        # Covariance matrix and uncertainties
        try:
            cov_matrix = np.linalg.inv(hessian / 2.0)  # Factor of 2 for chi-squared
            uncertainties_fit = np.sqrt(np.diag(cov_matrix))
        except:
            logger.warning("Could not calculate parameter uncertainties")
            uncertainties_fit = [0.1 * abs(val) for val in result.x]
            cov_matrix = np.eye(3)
        
        # Statistical tests
        p_value = 1.0 - chi2.cdf(chi_squared, dof) if dof > 0 else 0.0
        
        # Determine fit quality
        reduced_chi_sq = chi_squared / dof if dof > 0 else float('inf')
        if reduced_chi_sq < 1.2:
            fit_quality = 'excellent'
        elif reduced_chi_sq < 2.0:
            fit_quality = 'good'
        elif reduced_chi_sq < 5.0:
            fit_quality = 'acceptable'
        else:
            fit_quality = 'poor'
        
        # Create fit results
        param_names = ['lv_xi', 'lv_eta', 'cross_section_scaling']
        fit_results = {}
        
        for i, (name, value, uncertainty) in enumerate(zip(param_names, result.x, uncertainties_fit)):
            # 95% confidence interval
            t_val = t.ppf(0.975, dof) if dof > 0 else 1.96
            ci_95 = (value - t_val * uncertainty, value + t_val * uncertainty)
            
            # Correlation matrix
            correlation = {}
            for j, other_name in enumerate(param_names):
                if i != j and cov_matrix[i, i] > 0 and cov_matrix[j, j] > 0:
                    correlation[other_name] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
                else:
                    correlation[other_name] = 0.0
            
            fit_results[name] = ParameterFitResult(
                parameter_name=name,
                fitted_value=value,
                uncertainty=uncertainty,
                confidence_interval_95=ci_95,
                chi_squared=chi_squared,
                degrees_of_freedom=dof,
                p_value=p_value,
                fit_quality=fit_quality,
                correlation_matrix={name: correlation}
            )
        
        # Update reference parameters
        self.reference_parameters['lv_xi'] = xi_fit
        self.reference_parameters['lv_eta'] = eta_fit
        self.reference_parameters['cross_section_scaling'] = sigma_0_fit
        
        # Store results
        self.fit_results.update(fit_results)
        
        logger.info(f"Parameter fitting complete. χ²/dof = {reduced_chi_sq:.2f}, p-value = {p_value:.3f}")
        
        return fit_results
    
    def update_simulation_parameters(self) -> Dict[str, float]:
        """
        Update simulation parameters based on experimental fits.
        
        Returns:
            Updated parameter dictionary
        """
        updated_params = self.reference_parameters.copy()
        
        # Apply fitted parameters if available
        for param_name, fit_result in self.fit_results.items():
            if fit_result.fit_quality in ['excellent', 'good', 'acceptable']:
                updated_params[param_name] = fit_result.fitted_value
                logger.info(f"Updated {param_name} = {fit_result.fitted_value:.2e} ± {fit_result.uncertainty:.2e}")
            else:
                logger.warning(f"Poor fit quality for {param_name}, keeping reference value")
        
        return updated_params
    
    def validate_simulation_predictions(self, 
                                      measurement_type: str = 'yield',
                                      tolerance: float = 0.2) -> Dict[str, Any]:
        """
        Validate simulation predictions against experimental data.
        
        Args:
            measurement_type: Type of measurement to validate
            tolerance: Acceptable relative error
            
        Returns:
            Validation results dictionary
        """
        data = self.filter_data(measurement_type=measurement_type)
        
        if len(data) == 0:
            return {'status': 'no_data', 'message': 'No experimental data available for validation'}
        
        # Simple validation: compare mean experimental value to simulation prediction
        exp_values = [d.measured_value for d in data]
        exp_uncertainties = [d.uncertainty for d in data]
        
        # Weighted mean of experimental data
        weights = [1.0 / u**2 for u in exp_uncertainties]
        weighted_mean = np.average(exp_values, weights=weights)
        weighted_std = np.sqrt(1.0 / sum(weights))
        
        # Get simulation prediction (placeholder - would call actual simulation)
        sim_prediction = self.get_simulation_prediction(measurement_type, data[0].target_isotope, data[0].beam_energy)
        
        # Calculate relative error
        relative_error = abs(weighted_mean - sim_prediction) / weighted_mean
        
        # Determine validation status
        if relative_error <= tolerance:
            status = 'validated'
        elif relative_error <= 2 * tolerance:
            status = 'marginal'
        else:
            status = 'failed'
        
        validation_results = {
            'status': status,
            'experimental_mean': weighted_mean,
            'experimental_uncertainty': weighted_std,
            'simulation_prediction': sim_prediction,
            'relative_error': relative_error,
            'tolerance': tolerance,
            'data_points': len(data),
            'recommendation': self.get_validation_recommendation(status, relative_error)
        }
        
        logger.info(f"Validation complete: {status} (error: {relative_error:.1%})")
        
        return validation_results
    
    def get_simulation_prediction(self, measurement_type: str, isotope: str, energy: float) -> float:
        """
        Get simulation prediction for comparison with experimental data.
        This would interface with the actual simulation modules.
        """
        # Placeholder implementation - would call actual simulation
        if measurement_type == 'cross_section':
            # LV-enhanced cross-section
            xi = self.reference_parameters['lv_xi']
            eta = self.reference_parameters['lv_eta']
            sigma_0 = self.reference_parameters['cross_section_scaling']
            
            lv_enhancement = 1.0 + xi * (energy / 100.0)**2 + eta * np.log(energy / 10.0)
            return sigma_0 * lv_enhancement * (energy / 100.0)**(-0.5)
        
        elif measurement_type == 'yield':
            # Simple yield model
            base_yield = 1e-6  # atoms/second
            efficiency = self.reference_parameters['beam_focusing_efficiency']
            return base_yield * efficiency * (energy / 100.0)
        
        else:
            return 1.0  # Default placeholder
    
    def get_validation_recommendation(self, status: str, error: float) -> str:
        """Generate recommendations based on validation results"""
        if status == 'validated':
            return "Simulation parameters validated. Proceed with pilot operations."
        elif status == 'marginal':
            return f"Marginal validation (error: {error:.1%}). Consider parameter refinement."
        else:
            return f"Validation failed (error: {error:.1%}). Simulation parameters need revision."
    
    def generate_calibration_report(self) -> Dict[str, Any]:
        """Generate comprehensive calibration and validation report"""
        
        timestamp = datetime.datetime.now().isoformat()
        
        # Summary statistics
        total_data_points = len(self.data_points)
        measurement_types = list(set(d.measurement_type for d in self.data_points))
        isotopes = list(set(d.target_isotope for d in self.data_points))
        
        # Fit quality summary
        fit_summary = {}
        for param_name, fit_result in self.fit_results.items():
            fit_summary[param_name] = {
                'value': fit_result.fitted_value,
                'uncertainty': fit_result.uncertainty,
                'quality': fit_result.fit_quality,
                'p_value': fit_result.p_value
            }
        
        # Updated parameters
        updated_params = self.update_simulation_parameters()
        
        report = {
            'report_metadata': {
                'timestamp': timestamp,
                'total_data_points': total_data_points,
                'measurement_types': measurement_types,
                'target_isotopes': isotopes
            },
            'fitted_parameters': fit_summary,
            'updated_simulation_parameters': updated_params,
            'data_quality_metrics': self.calculate_data_quality_metrics(),
            'recommendations': self.generate_recommendations(),
            'next_experiments': self.suggest_next_experiments()
        }
        
        return report
    
    def calculate_data_quality_metrics(self) -> Dict[str, float]:
        """Calculate data quality metrics"""
        if not self.data_points:
            return {}
        
        # Statistical measures
        uncertainties = [d.uncertainty / d.measured_value for d in self.data_points if d.measured_value > 0]
        
        metrics = {
            'mean_relative_uncertainty': np.mean(uncertainties) if uncertainties else 0.0,
            'data_coverage_score': len(set(d.target_isotope for d in self.data_points)) / 10.0,  # Normalize by expected isotopes
            'energy_range_coverage': self.calculate_energy_coverage(),
            'temporal_consistency': self.calculate_temporal_consistency()
        }
        
        return metrics
    
    def calculate_energy_coverage(self) -> float:
        """Calculate energy range coverage score"""
        if not self.data_points:
            return 0.0
        
        energies = [d.beam_energy for d in self.data_points]
        energy_range = max(energies) - min(energies)
        target_range = 1000.0  # MeV
        
        return min(energy_range / target_range, 1.0)
    
    def calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of measurements"""
        # Placeholder - would analyze measurement reproducibility over time
        return 0.85  # Assume good consistency
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Check fit quality
        poor_fits = [name for name, result in self.fit_results.items() if result.fit_quality == 'poor']
        if poor_fits:
            recommendations.append(f"Improve data quality for parameters: {', '.join(poor_fits)}")
        
        # Check data coverage
        data_types = set(d.measurement_type for d in self.data_points)
        if 'cross_section' not in data_types:
            recommendations.append("Collect cross-section measurement data")
        if 'yield' not in data_types:
            recommendations.append("Collect yield measurement data")
        
        # Energy range
        if self.calculate_energy_coverage() < 0.5:
            recommendations.append("Expand beam energy range for measurements")
        
        if not recommendations:
            recommendations.append("Data quality is sufficient for pilot operations")
        
        return recommendations
    
    def suggest_next_experiments(self) -> List[Dict[str, Any]]:
        """Suggest next experiments to improve parameter estimates"""
        suggestions = []
        
        # Identify parameters with large uncertainties
        for param_name, fit_result in self.fit_results.items():
            relative_uncertainty = fit_result.uncertainty / abs(fit_result.fitted_value)
            if relative_uncertainty > 0.2:  # More than 20% uncertainty
                suggestions.append({
                    'parameter': param_name,
                    'current_uncertainty': relative_uncertainty,
                    'suggested_measurement': 'high-precision cross-section at multiple energies',
                    'priority': 'high' if relative_uncertainty > 0.5 else 'medium'
                })
        
        return suggestions
    
    def save_data_backup(self) -> str:
        """Save experimental data backup"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.data_dir / f"experimental_data_backup_{timestamp}.json"
        
        backup_data = {
            'timestamp': timestamp,
            'data_points': [asdict(dp) for dp in self.data_points],
            'fit_results': {name: asdict(result) for name, result in self.fit_results.items()},
            'reference_parameters': self.reference_parameters
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"Data backup saved to {backup_path}")
        return str(backup_path)
    
    def load_data_backup(self, backup_path: str) -> bool:
        """Load experimental data from backup"""
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Restore data points
            self.data_points = []
            for dp_dict in backup_data['data_points']:
                self.data_points.append(ExperimentalDataPoint(**dp_dict))
            
            # Restore fit results
            self.fit_results = {}
            for name, result_dict in backup_data['fit_results'].items():
                self.fit_results[name] = ParameterFitResult(**result_dict)
            
            # Restore reference parameters
            self.reference_parameters.update(backup_data['reference_parameters'])
            
            logger.info(f"Loaded {len(self.data_points)} data points from backup")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load backup: {e}")
            return False

def demo_experimental_data_integration():
    """Demonstrate experimental data integration capabilities"""
    
    print("🧪 EXPERIMENTAL DATA INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize integrator
    integrator = ExperimentalDataIntegrator()
    
    # Generate synthetic experimental data for demonstration
    print("\n📊 Generating synthetic experimental data...")
    
    np.random.seed(42)
    synthetic_data = []
    
    isotopes = ['Fe-56', 'Ni-58', 'Cu-63', 'Zn-64']
    measurement_types = ['cross_section', 'yield', 'beam_efficiency']
    
    for i in range(50):
        isotope = np.random.choice(isotopes)
        meas_type = np.random.choice(measurement_types)
        energy = np.random.uniform(50, 500)  # MeV
        
        # Generate realistic measurement with LV enhancement
        if meas_type == 'cross_section':
            base_value = 1e-27 * (energy / 100.0)**(-0.5)
            lv_enhancement = 1.0 + 3.2e-18 * (energy / 100.0)**2
            true_value = base_value * lv_enhancement
            measured_value = np.random.normal(true_value, 0.1 * true_value)
            uncertainty = 0.1 * true_value
        
        elif meas_type == 'yield':
            base_yield = 1e-6 * energy / 100.0
            measured_value = np.random.normal(base_yield, 0.15 * base_yield)
            uncertainty = 0.15 * base_yield
        
        else:  # beam_efficiency
            measured_value = np.random.normal(0.85, 0.05)
            uncertainty = 0.05
        
        timestamp = f"2024-01-{(i%30)+1:02d}T{(i%24):02d}:00:00"
        
        data_point = ExperimentalDataPoint(
            timestamp=timestamp,
            measurement_type=meas_type,
            target_isotope=isotope,
            beam_energy=energy,
            measured_value=max(measured_value, 0),  # Ensure positive
            uncertainty=uncertainty,
            systematic_error=0.02 * measured_value,
            experimental_conditions={'temperature': 300, 'pressure': 1.0},
            detector_calibration={'efficiency': 0.15, 'resolution': 0.01},
            environmental_factors={'humidity': 0.45, 'magnetic_field': 0.1},
            operator_id=f"operator_{(i%3)+1}",
            run_id=f"run_{i:03d}"
        )
        
        synthetic_data.append(data_point)
        integrator.ingest_data_realtime(data_point)
    
    print(f"✅ Ingested {len(synthetic_data)} synthetic data points")
    
    # Demonstrate parameter fitting
    print("\n🔧 Fitting Lorentz violation parameters...")
    
    try:
        fit_results = integrator.fit_lv_parameters('cross_section')
        
        print("\n📈 Parameter Fitting Results:")
        for param_name, result in fit_results.items():
            print(f"  {param_name}:")
            print(f"    Value: {result.fitted_value:.2e} ± {result.uncertainty:.2e}")
            print(f"    95% CI: [{result.confidence_interval_95[0]:.2e}, {result.confidence_interval_95[1]:.2e}]")
            print(f"    Fit Quality: {result.fit_quality}")
            print(f"    p-value: {result.p_value:.3f}")
    
    except Exception as e:
        print(f"❌ Parameter fitting failed: {e}")
    
    # Demonstrate validation
    print("\n✅ Validating simulation predictions...")
    
    validation_results = integrator.validate_simulation_predictions('yield')
    print(f"  Status: {validation_results['status']}")
    print(f"  Relative Error: {validation_results['relative_error']:.1%}")
    print(f"  Recommendation: {validation_results['recommendation']}")
    
    # Generate comprehensive report
    print("\n📊 Generating calibration report...")
    
    report = integrator.generate_calibration_report()
    
    print(f"\n📋 Calibration Report Summary:")
    print(f"  Data Points: {report['report_metadata']['total_data_points']}")
    print(f"  Measurement Types: {report['report_metadata']['measurement_types']}")
    print(f"  Target Isotopes: {report['report_metadata']['target_isotopes']}")
    
    print(f"\n🎯 Data Quality Metrics:")
    for metric, value in report['data_quality_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experimental_integration_demo_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    backup_path = integrator.save_data_backup()
    
    print(f"\n💾 Results saved:")
    print(f"  Report: {results_file}")
    print(f"  Data Backup: {backup_path}")
    
    print(f"\n🎉 Experimental data integration demonstration complete!")
    
    return integrator, report

if __name__ == "__main__":
    # Run demonstration
    integrator, report = demo_experimental_data_integration()
