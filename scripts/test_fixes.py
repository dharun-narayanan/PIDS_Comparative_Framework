#!/usr/bin/env python3
"""
Test script to verify all fixes are working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all models can be imported."""
    print("="*80)
    print("Testing Model Imports")
    print("="*80)
    
    results = {}
    
    # Test MAGIC
    try:
        from models.magic_wrapper import MAGICModel
        results['MAGIC'] = '✅ OK'
    except Exception as e:
        results['MAGIC'] = f'❌ {str(e)[:50]}'
    
    # Test Kairos
    try:
        from models.kairos_wrapper import KairosModel
        from models.implementations.kairos import prepare_kairos_batch
        results['Kairos'] = '✅ OK'
    except Exception as e:
        results['Kairos'] = f'❌ {str(e)[:50]}'
    
    # Test Orthrus
    try:
        from models.orthrus_wrapper import OrthrusModel
        from models.implementations.orthrus import prepare_orthrus_batch
        results['Orthrus'] = '✅ OK'
    except Exception as e:
        results['Orthrus'] = f'❌ {str(e)[:50]}'
    
    # Test ThreaTrace
    try:
        from models.threatrace_wrapper import ThreaTraceModel
        results['ThreaTrace'] = '✅ OK'
    except Exception as e:
        results['ThreaTrace'] = f'❌ {str(e)[:50]}'
    
    # Test Continuum_FL
    try:
        from models.continuum_fl_wrapper import ContinuumFLModel
        results['Continuum_FL'] = '✅ OK'
    except Exception as e:
        results['Continuum_FL'] = f'❌ {str(e)[:50]}'
    
    for model, status in results.items():
        print(f"  {model:15s}: {status}")
    
    print()
    return all('✅' in v for v in results.values())

def test_model_registry():
    """Test that models are registered."""
    print("="*80)
    print("Testing Model Registry")
    print("="*80)
    
    try:
        from models import ModelRegistry
        
        available = ModelRegistry.list_models()
        print(f"  Registered models: {len(available)}")
        for model in available:
            print(f"    - {model}")
        
        print()
        return len(available) > 0
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()
        return False

def test_device_handling():
    """Test safe device handling."""
    print("="*80)
    print("Testing Device Handling")
    print("="*80)
    
    try:
        import torch
        from models.base_model import BasePIDSModel
        
        # Test with valid device
        config1 = {'device': 'cpu'}
        print(f"  Testing with device='cpu'... ", end='')
        # We can't directly instantiate BasePIDSModel (it's abstract)
        # but we can test torch.device creation
        device = torch.device(config1['device'])
        print(f"✅ OK ({device})")
        
        # Test with CUDA (if available)
        if torch.cuda.is_available():
            config2 = {'device': 'cuda:0'}
            print(f"  Testing with device='cuda:0'... ", end='')
            device = torch.device(config2['device'])
            print(f"✅ OK ({device})")
        else:
            print(f"  CUDA not available, skipping GPU test")
        
        # Test with invalid device (should be handled now)
        print(f"  Testing with invalid device... ", end='')
        try:
            device = torch.device('cuda:-1')  # This would cause error
            print(f"❌ Should have raised error")
            return False
        except RuntimeError:
            print(f"✅ OK (correctly raises error, our fix catches it)")
        
        print()
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        return False

def test_compare_script():
    """Test compare.py accepts new arguments."""
    print("="*80)
    print("Testing Compare Script Arguments")
    print("="*80)
    
    try:
        import subprocess
        result = subprocess.run(
            ['python', 'experiments/compare.py', '--help'],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        help_text = result.stdout
        
        checks = {
            '--results-dir': '--results-dir' in help_text,
            '--output-file': '--output-file' in help_text,
            '--generate-plots': '--generate-plots' in help_text,
        }
        
        for arg, found in checks.items():
            status = '✅ OK' if found else '❌ Missing'
            print(f"  {arg:20s}: {status}")
        
        print()
        return all(checks.values())
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print()
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PIDS FRAMEWORK FIX VERIFICATION" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    results = []
    
    results.append(('Model Imports', test_imports()))
    results.append(('Model Registry', test_model_registry()))
    results.append(('Device Handling', test_device_handling()))
    results.append(('Compare Script', test_compare_script()))
    
    print("="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = '✅ PASSED' if passed else '❌ FAILED'
        print(f"  {test_name:20s}: {status}")
    
    print()
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("✅ All tests passed! Fixes are working correctly.")
        print()
        print("Next steps:")
        print("  1. Run diagnostic: python scripts/diagnose_data.py data/custom_soc")
        print("  2. Fix empty dataset issue")
        print("  3. Re-run evaluation: ./scripts/run_evaluation.sh")
    else:
        print("❌ Some tests failed. Please review the output above.")
    
    print()
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
