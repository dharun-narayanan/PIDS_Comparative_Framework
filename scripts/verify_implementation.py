#!/usr/bin/env python3
"""
PIDS Comparative Framework - Comprehensive Verification Script

This script performs comprehensive verification of:
1. Model implementations and imports
2. Framework components
3. Dependencies and environment
4. Checkpoint availability
5. Configuration files

Updated: October 14, 2025 to match revised framework
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_failure(text):
    """Print failure message."""
    print(f"❌ {text}")


def print_warning(text):
    """Print warning message."""
    print(f"⚠️  {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ️  {text}")


def test_model_imports():
    """Test all model imports"""
    print_header("Model Implementation Verification")
    
    results = {}
    
    # Test MAGIC
    print("\nTesting MAGIC imports...")
    try:
        from models.implementations.magic import (
            build_model, GMAEModel, GAT,
            sce_loss, evaluate_entity_level_using_knn
        )
        results['magic'] = True
        print_success("MAGIC imports successful")
    except Exception as e:
        results['magic'] = False
        print_failure(f"MAGIC imports failed: {e}")
    
    # Test Kairos
    print("\nTesting Kairos imports...")
    try:
        from models.implementations.kairos import (
            GraphAttentionEmbedding, LinkPredictor,
            TimeEncoder, setup_kairos_model
        )
        results['kairos'] = True
        print_success("Kairos imports successful")
    except Exception as e:
        results['kairos'] = False
        print_failure(f"Kairos imports failed: {e}")
    
    # Test Orthrus
    print("\nTesting Orthrus imports...")
    try:
        from models.implementations.orthrus import (
            Orthrus, OrthrusEncoder,
            get_encoder, get_decoders,
            setup_orthrus_model
        )
        results['orthrus'] = True
        print_success("Orthrus imports successful")
    except Exception as e:
        results['orthrus'] = False
        print_failure(f"Orthrus imports failed: {e}")
    
    # Test Continuum_FL
    print("\nTesting Continuum_FL imports...")
    try:
        from models.implementations.continuum_fl import (
            STGNN_AutoEncoder, STGNN, GAT, RNN_Cells,
            sce_loss, setup_continuum_fl_model
        )
        results['continuum_fl'] = True
        print_success("Continuum_FL imports successful")
    except Exception as e:
        results['continuum_fl'] = False
        print_failure(f"Continuum_FL imports failed: {e}")
    
    # Test ThreaTrace
    print("\nTesting ThreaTrace imports...")
    try:
        from models.implementations.threatrace import (
            ThreaTraceModel, SketchGenerator,
            setup_threatrace_model
        )
        results['threatrace'] = True
        print_success("ThreaTrace imports successful")
    except Exception as e:
        results['threatrace'] = False
        print_failure(f"ThreaTrace imports failed: {e}")
    
    return results


def test_model_registry():
    """Test ModelRegistry functionality."""
    print_header("Model Registry Verification")
    
    try:
        from models import ModelRegistry
        
        # List all registered models
        models = ModelRegistry.list_models()
        
        print(f"\nRegistered models: {len(models)}")
        for model in models:
            print(f"  ✓ {model}")
        
        # Expected models
        expected_models = [
            'magic', 'magic_streamspot', 'magic_darpa',
            'kairos', 'orthrus', 'threatrace',
            'continuum_fl', 'continuum_fl_streamspot', 'continuum_fl_darpa'
        ]
        
        missing_models = set(expected_models) - set(models)
        if missing_models:
            print_warning(f"Missing models: {', '.join(missing_models)}")
            return False
        else:
            print_success("All expected models are registered")
            return True
            
    except Exception as e:
        print_failure(f"ModelRegistry test failed: {e}")
        return False


def test_checkpoints():
    """Test checkpoint availability."""
    print_header("Checkpoint Verification")
    
    checkpoints_dir = project_root / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print_failure(f"Checkpoints directory not found: {checkpoints_dir}")
        return False
    
    expected_checkpoints = {
        'magic': ['*.pt'],
        'kairos': ['*.pt', '*.pkl'],
        'orthrus': ['*.pkl'],
        'threatrace': ['darpatc/', 'streamspot/', 'unicornsc/'],
        'continuum_fl': ['*.pt']
    }
    
    results = {}
    
    for model, patterns in expected_checkpoints.items():
        model_dir = checkpoints_dir / model
        
        if not model_dir.exists():
            print_warning(f"{model:15s} - Directory not found: {model_dir}")
            results[model] = False
            continue
        
        found_files = []
        for pattern in patterns:
            if '/' in pattern:
                # Directory pattern
                subdir = model_dir / pattern.rstrip('/')
                if subdir.exists() and subdir.is_dir():
                    found_files.append(pattern)
            else:
                # File pattern
                matches = list(model_dir.glob(pattern))
                found_files.extend([f.name for f in matches])
        
        if found_files:
            print_success(f"{model:15s} - Found {len(found_files)} checkpoint(s)")
            results[model] = True
        else:
            print_warning(f"{model:15s} - No checkpoints found")
            results[model] = False
    
    return all(results.values())


def test_dependencies():
    """Test framework dependencies."""
    print_header("Dependency Verification")
    
    dependencies = {
        'Core': {
            'torch': '1.12.0',
            'numpy': '1.20.0',
            'pandas': '1.3.0',
            'scipy': '1.7.0',
        },
        'Graph Libraries': {
            'dgl': '1.0.0',
            'networkx': '2.0',
        },
        'Utilities': {
            'tqdm': None,
            'matplotlib': '3.0.0',
        }
    }
    
    all_good = True
    
    for category, deps in dependencies.items():
        print(f"\n{category}:")
        for package, min_version in deps.items():
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print_success(f"{package:20s} - version {version}")
            except ImportError:
                print_failure(f"{package:20s} - NOT INSTALLED")
                all_good = False
    
    # Check optional dependencies
    print(f"\nOptional Dependencies:")
    optional_deps = ['sklearn', 'yaml', 'torch_geometric']
    for package in optional_deps:
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'yaml':
                import yaml
                version = getattr(yaml, '__version__', 'unknown')
            elif package == 'torch_geometric':
                import torch_geometric
                version = torch_geometric.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            print_success(f"{package:20s} - version {version}")
        except ImportError:
            print_warning(f"{package:20s} - NOT INSTALLED (optional)")
    
    return all_good


def test_directory_structure():
    """Test directory structure."""
    print_header("Directory Structure Verification")
    
    required_dirs = [
        'data',
        'models',
        'models/implementations',
        'utils',
        'experiments',
        'scripts',
        'configs',
        'checkpoints',
        'results',
        'logs',
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_success(f"{dir_path:40s} - EXISTS")
        else:
            print_failure(f"{dir_path:40s} - MISSING")
            all_good = False
    
    return all_good


def test_configuration_files():
    """Test configuration files."""
    print_header("Configuration Files Verification")
    
    required_configs = [
        'configs/models/magic.yaml',
        'configs/models/kairos.yaml',
        'configs/models/orthrus.yaml',
        'configs/models/threatrace.yaml',
        'configs/models/continuum_fl.yaml',
        'configs/datasets/custom_soc.yaml',
        'configs/datasets/cadets_e3.yaml',
        'configs/datasets/streamspot.yaml',
        'configs/experiments/compare_all.yaml',
        'configs/experiments/train_single.yaml',
    ]
    
    all_good = True
    
    for config_path in required_configs:
        full_path = project_root / config_path
        if full_path.exists():
            print_success(f"{config_path}")
        else:
            print_warning(f"{config_path} - MISSING (may be optional)")
    
    return all_good


def test_scripts():
    """Test required scripts."""
    print_header("Scripts Verification")
    
    required_scripts = [
        ('scripts/setup.sh', True),
        ('scripts/setup_models.py', False),
        ('scripts/preprocess_data.py', False),
        ('scripts/run_evaluation.sh', True),
        ('scripts/verify_installation.py', False),
        ('experiments/train.py', False),
        ('experiments/evaluate.py', False),
        ('experiments/compare.py', False),
    ]
    
    all_good = True
    
    for script_path, needs_executable in required_scripts:
        full_path = project_root / script_path
        if full_path.exists():
            if needs_executable:
                is_executable = os.access(full_path, os.X_OK)
                if is_executable:
                    print_success(f"{script_path:40s} - EXISTS (executable)")
                else:
                    print_warning(f"{script_path:40s} - EXISTS (not executable)")
                    print_info(f"   Fix with: chmod +x {script_path}")
            else:
                print_success(f"{script_path:40s} - EXISTS")
        else:
            print_failure(f"{script_path:40s} - MISSING")
            all_good = False
    
    return all_good


def test_environment():
    """Test conda environment."""
    print_header("Environment Verification")
    
    # Check if in conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print_success(f"Conda environment: {conda_env}")
        if conda_env != 'pids_framework':
            print_warning(f"Expected 'pids_framework' but got '{conda_env}'")
    else:
        print_warning("Not in a conda environment")
        print_info("Activate with: conda activate pids_framework")
    
    # Check Python version
    version = sys.version_info
    print(f"\nPython version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 10:
        print_success("Python version is compatible (3.8-3.10)")
        return True
    else:
        print_failure(f"Python {version.major}.{version.minor} may not be compatible")
        return False


def print_summary(test_results):
    """Print summary of all tests."""
    print_header("Verification Summary")
    
    # Calculate totals
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print()
    
    # Show individual results
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
    
    print()
    
    if passed_tests == total_tests:
        print("=" * 80)
        print_success("ALL TESTS PASSED! Framework is ready to use.")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Setup models: python scripts/setup_models.py --all")
        print("2. Preprocess data: python scripts/preprocess_data.py --input-dir ../custom_dataset/")
        print("3. Run evaluation: ./scripts/run_evaluation.sh")
        print("\n📖 See Setup_New.md for detailed instructions")
        return True
    else:
        print("=" * 80)
        print_warning("SOME TESTS FAILED. Please review the errors above.")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("1. Run setup: ./scripts/setup.sh")
        print("2. Activate environment: conda activate pids_framework")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Setup models: python scripts/setup_models.py --all")
        print("\n📖 See Setup_New.md troubleshooting section for more help")
        return False


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("  PIDS Comparative Framework - Comprehensive Verification")
    print("=" * 80)
    print(f"Framework root: {project_root}")
    
    # Run all tests
    test_results = {
        'Environment': test_environment(),
        'Dependencies': test_dependencies(),
        'Directory Structure': test_directory_structure(),
        'Model Implementations': all(test_model_imports().values()),
        'Model Registry': test_model_registry(),
        'Configuration Files': test_configuration_files(),
        'Scripts': test_scripts(),
        'Checkpoints': test_checkpoints(),
    }
    
    # Print summary
    success = print_summary(test_results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
