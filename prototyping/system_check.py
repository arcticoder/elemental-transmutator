#!/usr/bin/env python3
"""
Ultra-quick system check for elemental transmutator.
Just validates core functionality without long-running demos.
"""

import sys
import os
import json
from pathlib import Path

# Set up encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("ELEMENTAL TRANSMUTATOR - SYSTEM CHECK")
    print("=" * 40)
    
    # Test 1: Core imports
    print("\n1. Testing core imports...")
    try:
        import photonuclear_transmutation
        import energy_ledger
        import atomic_binder
        print("   ✓ All core modules imported successfully")
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        return 1
    
    # Test 2: Prototyping modules
    print("\n2. Testing prototyping modules...")
    try:
        from prototyping import gamma_beam_controller
        from prototyping import target_cell_monitor
        from prototyping import data_logger
        print("   ✓ All prototyping modules imported successfully")
    except Exception as e:
        print(f"   ✗ Prototyping import failed: {e}")
        return 1
    
    # Test 3: Configuration files
    print("\n3. Testing configuration files...")
    config_path = Path(__file__).parent / "hardware_config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("   ✓ Hardware configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Config load failed: {e}")
        return 1
    
    # Test 4: Create a simple physics calculation
    print("\n4. Testing physics engine...")
    try:
        config = photonuclear_transmutation.PhotonuclearConfig(
            feedstock_isotope="Pb-208",
            gamma_energy_MeV=16.5
        )
        print("   ✓ Physics configuration created successfully")
    except Exception as e:
        print(f"   ✗ Physics test failed: {e}")
        return 1
    
    print("\n" + "=" * 40)
    print("SYSTEM CHECK PASSED! ✓")
    print("All core components are functional.")
    print("System ready for demonstration.")
    print("=" * 40)
    return 0

if __name__ == "__main__":
    sys.exit(main())
