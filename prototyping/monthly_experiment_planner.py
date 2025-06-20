#!/usr/bin/env python3
"""
Monthly Experiment Planner
==========================

Plans and schedules outsourced micro-experiments within budget constraints.
Implements the 3-month roadmap for Monte Carlo validation.
"""

import argparse
import json
import logging
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path

class MonthlyExperimentPlanner:
    """Planner for monthly micro-experiment campaigns."""
    
    def __init__(self):
        """Initialize the experiment planner."""
        self.logger = logging.getLogger(__name__)
        
        # Default budget constraints
        self.monthly_budget_cad = 100.0
        self.safety_buffer = 0.85  # Use 85% of budget to account for overruns
        
        self.logger.info("Monthly experiment planner initialized")
    
    def load_optimization_results(self) -> Dict:
        """Load results from Monte Carlo optimization."""
        try:
            with open("experiment_specs/optimal_recipe.json", 'r') as f:
                recipe = json.load(f)
                return recipe
        except FileNotFoundError:
            # Default values for testing
            return {
                'feedstock_g': 1.0,
                'beam_energy_mev': 15.0,
                'total_dose_kgy': 10.0,
                'predicted_au_mg': 0.008,
                'predicted_cost_cad': 55.0,
                'yield_per_cad': 0.000145,
                'confidence_interval_mg': [0.002, 0.015]
            }
    
    def plan_month_1(self) -> Dict:
        """Plan Month 1: Complete MC simulation + optimizer; pick recipe."""
        
        return {
            "month": 1,
            "phase": "Digital Optimization",
            "budget_allocated": 0.0,
            "activities": [
                {
                    "week": 1,
                    "activity": "Complete Monte Carlo digital twin",
                    "deliverables": [
                        "Geant4/MCNP validation",
                        "Cross-section database calibration",
                        "Uncertainty quantification"
                    ],
                    "cost": 0.0
                },
                {
                    "week": 2,
                    "activity": "Implement Bayesian optimizer",
                    "deliverables": [
                        "Multi-parameter optimization",
                        "Yield/cost ratio maximization",
                        "Constraint handling for practical limits"
                    ],
                    "cost": 0.0
                },
                {
                    "week": 3,
                    "activity": "Recipe optimization campaign",
                    "deliverables": [
                        "1000+ simulation runs",
                        "Pareto frontier analysis",
                        "Sensitivity analysis"
                    ],
                    "cost": 0.0
                },
                {
                    "week": 4,
                    "activity": "Lock in one-shot recipe",
                    "deliverables": [
                        "Optimal recipe specification",
                        "Vendor RFQ generation",
                        "Risk assessment"
                    ],
                    "cost": 0.0
                }
            ],
            "success_criteria": {
                "recipe_yield": ">= 0.01 mg Au predicted",
                "cost_efficiency": ">= 0.0001 mg Au per CAD",
                "detection_probability": "> 80%"
            },
            "risks": [
                "Cross-section uncertainties too large",
                "No economically viable recipe found",
                "Vendor availability issues"
            ]
        }
    
    def plan_month_2(self, recipe: Dict) -> Dict:
        """Plan Month 2: First micro-experiment run."""
        
        experiment_cost = min(recipe.get('predicted_cost_cad', 55.0), 
                            self.monthly_budget_cad * self.safety_buffer)
        
        return {
            "month": 2,
            "phase": "First Micro-Experiment",
            "budget_allocated": experiment_cost,
            "activities": [
                {
                    "week": 1,
                    "activity": "Sample preparation and vendor coordination",
                    "deliverables": [
                        f"{recipe.get('feedstock_g', 1.0):.1f}g Pb-208 pellet preparation",
                        "Irradiation vendor contract",
                        "Analysis vendor contract",
                        "Shipping arrangements"
                    ],
                    "cost": 10.0
                },
                {
                    "week": 2,
                    "activity": "Gamma irradiation at vendor facility",
                    "deliverables": [
                        f"{recipe.get('total_dose_kgy', 10.0):.1f} kGy Co-60 irradiation",
                        "Dosimetry certification",
                        "Activity measurement",
                        "Chain of custody documentation"
                    ],
                    "cost": experiment_cost * 0.4  # ~40% of total cost
                },
                {
                    "week": 3,
                    "activity": "ICP-MS analysis",
                    "deliverables": [
                        "Gold content assay",
                        "Matrix element analysis",
                        "Quality control verification",
                        "Uncertainty assessment"
                    ],
                    "cost": experiment_cost * 0.5  # ~50% of total cost
                },
                {
                    "week": 4,
                    "activity": "Data analysis and model calibration",
                    "deliverables": [
                        "Yield measurement vs. prediction",
                        "Cross-section correction factors",
                        "Updated Monte Carlo model",
                        "Experiment report"
                    ],
                    "cost": 0.0
                }
            ],
            "success_criteria": {
                "sample_integrity": "No contamination or loss",
                "dose_accuracy": "Within 10% of target",
                "detection_limit": f"< {recipe.get('predicted_au_mg', 0.01)/10:.4f} mg Au",
                "measurement_uncertainty": "< 25%"
            },
            "risks": [
                "Sample contamination during handling",
                "Vendor quality issues",
                "Below-detection-limit results",
                "Cost overruns"
            ]
        }
    
    def plan_month_3(self, recipe: Dict, month2_results: Dict = None) -> Dict:
        """Plan Month 3: Model calibration and second run if needed."""
        
        # Assume we learned something from Month 2
        updated_cost = recipe.get('predicted_cost_cad', 55.0) * 0.9  # 10% cost reduction
        remaining_budget = self.monthly_budget_cad * self.safety_buffer
        
        return {
            "month": 3,
            "phase": "Calibration & Validation",
            "budget_allocated": remaining_budget,
            "activities": [
                {
                    "week": 1,
                    "activity": "Model calibration with real data",
                    "deliverables": [
                        "Updated cross-section parameters",
                        "Systematic uncertainty quantification",
                        "Bias correction implementation",
                        "Prediction interval validation"
                    ],
                    "cost": 0.0
                },
                {
                    "week": 2,
                    "activity": "Optimized recipe generation",
                    "deliverables": [
                        "Second-generation recipe",
                        "Improved yield prediction",
                        "Cost optimization",
                        "Vendor negotiations"
                    ],
                    "cost": 0.0
                },
                {
                    "week": 3,
                    "activity": "Second micro-experiment (if budget allows)",
                    "deliverables": [
                        "Confirmation experiment",
                        "Reproducibility assessment",
                        "Process validation",
                        "Scale-up feasibility"
                    ],
                    "cost": min(updated_cost, remaining_budget) if remaining_budget >= updated_cost else 0.0
                },
                {
                    "week": 4,
                    "activity": "Business case development",
                    "deliverables": [
                        "Economic viability assessment",
                        "Scale-up requirements",
                        "Technology roadmap",
                        "Investment proposal"
                    ],
                    "cost": 0.0
                }
            ],
            "success_criteria": {
                "model_accuracy": "Prediction within 50% of measurement",
                "process_reproducibility": "CV < 30% between runs",
                "economic_viability": "Positive ROI at 10g scale",
                "technology_readiness": "TRL 3-4 achieved"
            },
            "decision_points": {
                "continue_development": "If yield > 0.005 mg Au per run",
                "scale_up_planning": "If yield > 0.02 mg Au per run",
                "pivot_to_alternatives": "If yield < 0.001 mg Au per run"
            }
        }
    
    def generate_quarterly_plan(self) -> Dict:
        """Generate complete 3-month development plan."""
        
        # Load optimization results
        recipe = self.load_optimization_results()
        
        # Plan each month
        month1_plan = self.plan_month_1()
        month2_plan = self.plan_month_2(recipe)
        month3_plan = self.plan_month_3(recipe)
        
        # Calculate totals
        total_budget = (month1_plan['budget_allocated'] + 
                       month2_plan['budget_allocated'] + 
                       month3_plan['budget_allocated'])
        
        quarterly_plan = {
            "plan_metadata": {
                "generated_date": datetime.now().isoformat(),
                "planning_horizon": "3_months",
                "budget_constraint": f"{self.monthly_budget_cad * 3:.2f} CAD",
                "strategy": "Outsourced micro-experiments"
            },
            "objective": {
                "primary": "Validate photonuclear gold production",
                "secondary": "Develop scalable process",
                "success_metric": "Real yield data within 3 months"
            },
            "monthly_plans": {
                "month_1": month1_plan,
                "month_2": month2_plan,
                "month_3": month3_plan
            },
            "budget_summary": {
                "total_budget_cad": self.monthly_budget_cad * 3,
                "allocated_budget_cad": total_budget,
                "safety_reserve_cad": (self.monthly_budget_cad * 3) - total_budget,
                "cost_breakdown": {
                    "digital_optimization": 0.0,
                    "first_experiment": month2_plan['budget_allocated'],
                    "second_experiment": month3_plan['budget_allocated'],
                    "overhead": 0.0
                }
            },
            "risk_mitigation": {
                "budget_overrun": "85% budget allocation with 15% reserve",
                "vendor_failure": "Multiple vendor options pre-qualified",
                "detection_limit": "Conservative yield predictions",
                "contamination": "Strict chain of custody protocols"
            },
            "key_milestones": [
                {
                    "month": 1,
                    "milestone": "Optimal recipe locked in",
                    "success_criteria": "Predicted yield > 0.01 mg Au"
                },
                {
                    "month": 2,
                    "milestone": "First real yield measurement",
                    "success_criteria": "Detectable gold production"
                },
                {
                    "month": 3,
                    "milestone": "Process validation",
                    "success_criteria": "Reproducible results"
                }
            ]
        }
        
        return quarterly_plan
    
    def save_plan(self, plan: Dict):
        """Save quarterly plan to file."""
        
        Path("experiment_specs").mkdir(exist_ok=True)
        
        with open("experiment_specs/quarterly_plan.json", 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2)
        
        # Generate human-readable summary
        self._generate_plan_summary(plan)
        
        self.logger.info("Quarterly plan saved to experiment_specs/")
    
    def _generate_plan_summary(self, plan: Dict):
        """Generate human-readable plan summary."""
        
        summary = f"""
PHOTONUCLEAR GOLD PRODUCTION - 3-MONTH ROADMAP
==============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Strategy: Outsourced micro-experiments
Total Budget: ${plan['budget_summary']['total_budget_cad']:.2f} CAD

MONTH 1: DIGITAL OPTIMIZATION
-----------------------------
Phase: {plan['monthly_plans']['month_1']['phase']}
Budget: ${plan['monthly_plans']['month_1']['budget_allocated']:.2f}
Goal: Complete Monte Carlo simulation and lock in optimal recipe

Key Activities:
"""
        
        for activity in plan['monthly_plans']['month_1']['activities']:
            summary += f"  Week {activity['week']}: {activity['activity']}\n"
        
        summary += f"""
Success Criteria: Recipe with ≥ 0.01 mg Au predicted yield

MONTH 2: FIRST MICRO-EXPERIMENT
-------------------------------
Phase: {plan['monthly_plans']['month_2']['phase']}
Budget: ${plan['monthly_plans']['month_2']['budget_allocated']:.2f}
Goal: Execute first outsourced irradiation + analysis cycle

Key Activities:
"""
        
        for activity in plan['monthly_plans']['month_2']['activities']:
            summary += f"  Week {activity['week']}: {activity['activity']}\n"
        
        summary += f"""
Success Criteria: Detectable gold measurement with <25% uncertainty

MONTH 3: CALIBRATION & VALIDATION
---------------------------------
Phase: {plan['monthly_plans']['month_3']['phase']}
Budget: ${plan['monthly_plans']['month_3']['budget_allocated']:.2f}
Goal: Calibrate model and validate process reproducibility

Key Activities:
"""
        
        for activity in plan['monthly_plans']['month_3']['activities']:
            summary += f"  Week {activity['week']}: {activity['activity']}\n"
        
        summary += f"""
BUDGET BREAKDOWN:
- Month 1 (Digital): ${plan['budget_summary']['cost_breakdown']['digital_optimization']:.2f}
- Month 2 (First Exp): ${plan['budget_summary']['cost_breakdown']['first_experiment']:.2f}
- Month 3 (Validation): ${plan['budget_summary']['cost_breakdown']['second_experiment']:.2f}
- Safety Reserve: ${plan['budget_summary']['safety_reserve_cad']:.2f}

EXPECTED OUTCOMES:
By Month 3 end, you will have:
✓ Real experimental data on gold yield
✓ Calibrated Monte Carlo model
✓ Economic viability assessment
✓ Decision point for scale-up or pivot

NEXT STEPS:
- If yield ≥ 0.02 mg Au: Plan scale-up to 10g batches
- If yield 0.005-0.02 mg Au: Optimize process parameters
- If yield < 0.005 mg Au: Consider alternative approaches

==============================================
"""
        
        with open("experiment_specs/quarterly_plan_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary)

def main():
    """Main planning routine."""
    parser = argparse.ArgumentParser(description='Generate monthly experiment plan')
    parser.add_argument('--budget', type=float, default=100.0,
                       help='Monthly budget in CAD')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize planner
    planner = MonthlyExperimentPlanner()
    planner.monthly_budget_cad = args.budget
    
    # Generate plan
    plan = planner.generate_quarterly_plan()
    
    # Save to files
    planner.save_plan(plan)
    
    # Print to stdout for CI
    with open("experiment_specs/quarterly_plan_summary.txt", 'r', encoding='utf-8') as f:
        print(f.read())
    
    return 0

if __name__ == '__main__':
    exit(main())
