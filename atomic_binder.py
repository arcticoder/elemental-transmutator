#!/usr/bin/env python3
"""
Atomic Binder Module
====================

Element-agnostic atomic binding module for converting free nuclei into neutral atoms.
Handles electron capture and atomic assembly for any target element.
"""

import numpy as np
import json
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class AtomicBinderResult:
    """Result from atomic binding process."""
    element: str
    mass: float  # kg
    atoms_bound: int
    binding_efficiency: float

class AtomicBinder:
    """
    Element-agnostic atomic binder for converting nuclei to atoms.
    """
    
    def __init__(self, lv_params: Dict[str, float], target_isotope: str):
        self.lv_params = lv_params
        self.target_isotope = target_isotope
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_config()
        
    def _load_config(self):
        """Load configuration from config.json if available."""
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
                self.target_isotope = cfg.get("target_isotope", self.target_isotope)
        except FileNotFoundError:
            self.logger.info("No config.json found, using default configuration")
        except Exception as e:
            self.logger.warning(f"Error loading config.json: {e}")
    
    def _parse_isotope(self, isotope: str) -> Tuple[int, int]:
        """Parse isotope string to get atomic number (Z) and mass number (A)."""
        element_map = {
            "Fe": 26, "Ag": 47, "Au": 79, "Pt": 78, "Pd": 46, "Rh": 45,
            "Cd": 48, "Cu": 29, "Ni": 28, "Co": 27, "Zn": 30
        }
        
        parts = isotope.split("-")
        element = parts[0]
        mass_number = int(parts[1])
        atomic_number = element_map.get(element, 79)  # Default to Au
        
        return atomic_number, mass_number
    
    def bind(self, nuclei_yields: Dict[str, float]) -> AtomicBinderResult:
        """
        Bind nuclei into neutral atoms.
        
        Args:
            nuclei_yields: Dictionary of nucleus counts by isotope
            
        Returns:
            AtomicBinderResult with binding statistics
        """
        total_nuclei = 0
        total_mass = 0.0
        
        for isotope, nuclei_count in nuclei_yields.items():
            if isotope == "total":
                continue
                
            z, a = self._parse_isotope(isotope)
            
            # Calculate binding efficiency (LV-enhanced)
            binding_efficiency = self._calculate_binding_efficiency(z)
            
            # Apply binding efficiency
            bound_nuclei = nuclei_count * binding_efficiency
            
            # Calculate mass
            nucleus_mass = a * 1.66054e-27  # kg
            isotope_mass = bound_nuclei * nucleus_mass
            
            total_nuclei += bound_nuclei
            total_mass += isotope_mass
            
            print(f"\n{isotope} binding:")
            print(f"  Input nuclei: {nuclei_count:.2e}")
            print(f"  Binding efficiency: {binding_efficiency:.2%}")
            print(f"  Bound atoms: {bound_nuclei:.2e}")
            print(f"  Mass: {isotope_mass*1e6:.3f} mg")
        
        return AtomicBinderResult(
            element=self.target_isotope,
            mass=total_mass,
            atoms_bound=int(total_nuclei),
            binding_efficiency=binding_efficiency
        )
    
    def _calculate_binding_efficiency(self, atomic_number: int) -> float:
        """Calculate LV-enhanced binding efficiency."""
        # Base efficiency depends on atomic number
        base_efficiency = min(0.95, 0.7 + 0.001 * atomic_number)
        
        # LV enhancement
        lv_enhancement = 1.0 + abs(self.lv_params.get("mu", 1e-17)) / 1e-18 * 0.1
        
        return min(0.99, base_efficiency * lv_enhancement)
