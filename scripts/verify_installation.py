#!/usr/bin/env python3
"""
PIDS Comparative Framework - Verification Script

This script verifies that all components of the framework are properly
installed and configured.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header(title):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_failure(message):
    """Print failure message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message."""
    print(f"⚠️  {message}")

def check_python_version():
    """Check Python version."""
    print_header("Python Environment")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python version is compatible (3.8+)")
        return True
    else:
        print_failure(f"Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False

def check_core_dependencies():
    """Check core dependencies."""
    print_header("Core Dependencies")
    
    dependencies = {
        'torch': '1.12.0',
        'numpy': '1.20.0',
        'scipy': '1.7.0',
        'pandas': '1.3.0',
        'sklearn': '1.0.0',
        'yaml': '5.4.0',
        'matplotlib': '3.4.0',
    }
    
    all_good = True
    
    for package, min_version in dependencies.items():
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'yaml':
                import yaml
                version = yaml.__version__ if hasattr(yaml, '__version__') else 'unknown'
            else:
                module = __import__(package)
                version = module.__version__
            
            print_success(f"{package:15s} - version {version}")
        except ImportError:
            print_failure(f"{package:15s} - NOT INSTALLED")
            all_good = False
    
    return all_good

def check_deep_learning_frameworks():
    """Check deep learning frameworks."""
    print_header("Deep Learning Frameworks")
    
    all_good = True
    
    # Check PyTorch
    try:
        import torch
        print_success(f"PyTorch        - version {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print_success(f"CUDA           - version {torch.version.cuda}")
            print_success(f"GPU Device     - {torch.cuda.get_device_name(0)}")
            print_success(f"GPU Count      - {torch.cuda.device_count()}")
        else:
            print_warning("CUDA           - NOT AVAILABLE (CPU only)")
    except ImportError:
        print_failure("PyTorch        - NOT INSTALLED")
        all_good = False
    
    # Check DGL
    try:
        import dgl
        print_success(f"DGL            - version {dgl.__version__}")
    except ImportError:
        print_failure("DGL            - NOT INSTALLED")
        all_good = False
    
    # Check PyTorch Geometric
    try:
        import torch_geometric
        print_success(f"PyTorch Geom.  - version {torch_geometric.__version__}")
        
        # Check PyG components
        try:
            import torch_scatter
            print_success(f"  torch-scatter - version {torch_scatter.__version__}")
        except ImportError:
            print_warning("  torch-scatter - NOT INSTALLED")
        
        try:
            import torch_sparse
            print_success(f"  torch-sparse  - version {torch_sparse.__version__}")
        except ImportError:
            print_warning("  torch-sparse  - NOT INSTALLED")
            
        try:
            import torch_cluster
            print_success(f"  torch-cluster - version {torch_cluster.__version__}")
        except ImportError:
            print_warning("  torch-cluster - NOT INSTALLED")
            
    except ImportError:
        print_failure("PyTorch Geom.  - NOT INSTALLED")
        all_good = False
    
    return all_good

def check_model_integrations():
    """Check model integrations using ModelBuilder."""
    print_header("Model Integrations (ModelBuilder)")
    
    all_good = True
    
    try:
        from models.model_builder import ModelBuilder
        from pathlib import Path
        
        # Initialize ModelBuilder
        model_builder = ModelBuilder(config_dir="configs/models")
        
        # List available model configs
        config_dir = Path("configs/models")
        model_configs = list(config_dir.glob("*.yaml"))
        model_names = [f.stem for f in model_configs if f.stem != "template"]
        
        print(f"Found {len(model_names)} model configurations:")
        for model_name in sorted(model_names):
            print_success(f"  {model_name}.yaml")
        
        # Expected models
        expected_models = [
            'magic', 'kairos', 'orthrus', 'threatrace', 'continuum_fl'
        ]
        
        missing_models = set(expected_models) - set(model_names)
        if missing_models:
            print_warning(f"Missing model configs: {', '.join(missing_models)}")
            all_good = False
        else:
            print_success("All expected models are registered")
            
    except Exception as e:
        print_failure(f"Error checking models: {e}")
        all_good = False
    
    return all_good

def check_directory_structure():
    """Check directory structure."""
    print_header("Directory Structure")
    
    required_dirs = [
        'data',
        'models',
        'utils',
        'experiments',
        'scripts',
        'configs',
        'configs/models',
        'configs/datasets',
        'configs/experiments',
        'checkpoints',
        'results',
        'logs',
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_success(f"{dir_path:25s} - EXISTS")
        else:
            print_failure(f"{dir_path:25s} - MISSING")
            all_good = False
    
    return all_good

def check_configuration_files():
    """Check configuration files."""
    print_header("Configuration Files")
    
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
            print_failure(f"{config_path} - MISSING")
            all_good = False
    
    return all_good

def check_scripts():
    """Check scripts."""
    print_header("Scripts")
    
    required_scripts = [
        'scripts/setup.sh',
        'scripts/preprocess_data.py',
        'experiments/train.py',
        'experiments/evaluate_pipeline.py',
    ]
    
    all_good = True
    
    for script_path in required_scripts:
        full_path = project_root / script_path
        if full_path.exists():
            is_executable = full_path.stat().st_mode & 0o111
            if is_executable or script_path.endswith('.py'):
                print_success(f"{script_path:35s} - EXISTS (executable)")
            else:
                print_warning(f"{script_path:35s} - EXISTS (not executable)")
        else:
            print_failure(f"{script_path:35s} - MISSING")
            all_good = False
    
    return all_good

def check_documentation():
    """Check documentation."""
    print_header("Documentation")
    
    required_docs = [
        'README.md',
        'QUICKSTART.md',
        'IMPLEMENTATION_COMPLETE.md',
        'FRAMEWORK_GUIDE.md',
    ]
    
    all_good = True
    
    for doc_path in required_docs:
        full_path = project_root / doc_path
        if full_path.exists():
            size = full_path.stat().st_size
            print_success(f"{doc_path:30s} - {size:,} bytes")
        else:
            print_failure(f"{doc_path:30s} - MISSING")
            all_good = False
    
    return all_good

def check_external_models():
    """Check external model directories."""
    print_header("External Model Directories")
    
    external_models = {
        'MAGIC': '../MAGIC',
        'Kairos': '../kairos',
        'Orthrus': '../orthrus',
        'ThreaTrace': '../threaTrace',
        'Continuum_FL': '../Continuum_FL',
    }
    
    all_good = True
    
    for model_name, model_path in external_models.items():
        full_path = project_root / model_path
        if full_path.exists():
            print_success(f"{model_name:15s} - {full_path}")
        else:
            print_warning(f"{model_name:15s} - NOT FOUND (optional)")
    
    return all_good

def print_summary(results):
    """Print summary of checks."""
    print_header("Verification Summary")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"\nTotal checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    
    if passed_checks == total_checks:
        print("\n" + "="*80)
        print("🎉 ALL CHECKS PASSED! Framework is ready to use.")
        print("="*80)
        print("\nNext steps:")
        print("1. Install model dependencies: ./scripts/install_model_deps.sh --all")
        print("2. Copy pretrained weights: python scripts/download_weights.py --copy-existing")
        print("3. Preprocess data: python scripts/preprocess_data.py --input-dir ../custom_dataset/")
        print("4. Start training: python experiments/train.py --model magic --dataset custom_soc")
        print("\n📖 See QUICKSTART.md for detailed instructions")
        return True
    else:
        print("\n" + "="*80)
        print("⚠️  SOME CHECKS FAILED. Please review the errors above.")
        print("="*80)
        print("\nTroubleshooting:")
        print("1. Run setup script: ./scripts/setup.sh")
        print("2. Activate environment: conda activate pids_framework")
        print("3. Install dependencies: ./scripts/install_model_deps.sh --all")
        print("\n📖 See docs/installation.md for detailed troubleshooting")
        return False

def main():
    """Run all verification checks."""
    print("="*80)
    print("  PIDS Comparative Framework - Verification")
    print("="*80)
    print(f"Framework root: {project_root}")
    
    results = {
        'Python Version': check_python_version(),
        'Core Dependencies': check_core_dependencies(),
        'Deep Learning Frameworks': check_deep_learning_frameworks(),
        'Model Integrations': check_model_integrations(),
        'Directory Structure': check_directory_structure(),
        'Configuration Files': check_configuration_files(),
        'Scripts': check_scripts(),
        'Documentation': check_documentation(),
        'External Models': check_external_models(),
    }
    
    success = print_summary(results)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
