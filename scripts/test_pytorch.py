#!/usr/bin/env python3
"""
Test script to verify PyTorch installation and diagnose issues.
Usage: python scripts/test_pytorch.py
"""

import sys
import os

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_environment():
    """Check basic environment settings"""
    print_section("Environment Information")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"MKL_THREADING_LAYER: {os.environ.get('MKL_THREADING_LAYER', 'Not set')}")
    
    # Check if in conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
    print(f"Conda environment: {conda_env}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")

def test_pytorch():
    """Test PyTorch import and basic functionality"""
    print_section("PyTorch Import Test")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        
        # Test basic operations
        try:
            x = torch.tensor([1.0, 2.0, 3.0])
            y = x * 2
            print(f"✓ Basic tensor operations working: {y.tolist()}")
        except Exception as e:
            print(f"✗ Tensor operations failed: {e}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"✗ PyTorch import failed!")
        print(f"  Error: {e}")
        print("\nSuggested fixes:")
        print("  1. Set environment variable: export MKL_THREADING_LAYER=GNU")
        print("  2. Run fix script: ./scripts/fix_pytorch_mkl.sh")
        print("  3. See docs/PYTORCH_MKL_FIX.md for detailed solutions")
        return False

def test_cuda():
    """Test CUDA availability"""
    print_section("CUDA Information")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            
            # Test CUDA tensor operations
            try:
                x = torch.tensor([1.0, 2.0, 3.0]).cuda()
                y = x * 2
                print(f"\n✓ CUDA tensor operations working: {y.cpu().tolist()}")
            except Exception as e:
                print(f"\n✗ CUDA tensor operations failed: {e}")
        else:
            print("\nCUDA not available. This is normal if:")
            print("  - You don't have an NVIDIA GPU")
            print("  - NVIDIA drivers are not installed")
            print("  - PyTorch was installed with CPU-only support")
            print("\nTo check NVIDIA drivers, run: nvidia-smi")
        
        return True
        
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        return False

def test_dependencies():
    """Test other key dependencies"""
    print_section("Dependency Check")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('pandas', 'Pandas'),
        ('networkx', 'NetworkX'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
        ('yaml', 'PyYAML'),
    ]
    
    all_ok = True
    for module_name, display_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:20s} {version}")
        except ImportError:
            print(f"✗ {display_name:20s} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def test_graph_libraries():
    """Test graph processing libraries"""
    print_section("Graph Library Check")
    
    # Test PyTorch Geometric
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
        
        # Test basic PyG functionality
        try:
            from torch_geometric.data import Data
            import torch
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            print(f"  ✓ Basic PyG operations working (nodes: {data.num_nodes}, edges: {data.num_edges})")
        except Exception as e:
            print(f"  ✗ PyG operations failed: {e}")
            
    except ImportError as e:
        print(f"✗ PyTorch Geometric not installed: {e}")
    
    # Test DGL
    try:
        import dgl
        print(f"✓ DGL {dgl.__version__}")
        
        # Test basic DGL functionality
        try:
            import torch
            g = dgl.graph(([0, 1, 2], [1, 2, 0]))
            g.ndata['feat'] = torch.randn(3, 5)
            print(f"  ✓ Basic DGL operations working (nodes: {g.num_nodes()}, edges: {g.num_edges()})")
        except Exception as e:
            print(f"  ✗ DGL operations failed: {e}")
            
    except ImportError as e:
        print(f"✗ DGL not installed: {e}")

def run_all_tests():
    """Run all tests and return overall status"""
    print("\n" + "=" * 60)
    print("  PIDS Framework - PyTorch Installation Test")
    print("=" * 60)
    
    results = {
        'environment': check_environment(),
        'pytorch': test_pytorch(),
        'cuda': test_cuda(),
        'dependencies': test_dependencies(),
        'graph_libs': test_graph_libraries(),
    }
    
    print_section("Test Summary")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        if passed is not None:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name.replace('_', ' ').title():20s} {status}")
    
    if all_passed:
        print("\n" + "=" * 60)
        print("  All tests passed! PyTorch is ready to use.")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("  Some tests failed. See error messages above.")
        print("  Run ./scripts/fix_pytorch_mkl.sh to fix common issues.")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
