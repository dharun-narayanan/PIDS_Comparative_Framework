#!/usr/bin/env python3
"""
Quick test to verify the dataset loader fix works with edges data.
"""

import sys
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_dataset_loading():
    """Test loading custom_soc dataset."""
    print("="*80)
    print("Testing Custom SOC Dataset Loading")
    print("="*80)
    
    # Load preprocessed file to check its structure
    data_path = Path('data/custom_soc')
    pkl_file = data_path / 'custom_soc_graph.pkl'
    
    if not pkl_file.exists():
        print(f"✗ Preprocessed file not found: {pkl_file}")
        return False
    
    print(f"✓ Found preprocessed file: {pkl_file}")
    
    try:
        with open(pkl_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        print(f"  Type: {type(graph_data)}")
        print(f"  Keys: {list(graph_data.keys()) if isinstance(graph_data, dict) else 'N/A'}")
        
        # Check for events or edges
        if isinstance(graph_data, dict):
            has_events = 'events' in graph_data
            has_edges = 'edges' in graph_data
            
            if has_events:
                num_events = len(graph_data['events'])
                print(f"  ✓ Has 'events' key: {num_events} events")
            elif has_edges:
                num_edges = len(graph_data['edges'])
                print(f"  ✓ Has 'edges' key: {num_edges} edges")
                print(f"    (Will be used as events)")
            else:
                print(f"  ✗ No 'events' or 'edges' key found!")
                return False
        
        print()
        
    except Exception as e:
        print(f"✗ Error loading preprocessed file: {e}")
        return False
    
    # Now test the dataset loader
    print("Testing dataset loader with fixed code...")
    
    try:
        from data.dataset import CustomSOCDataset
        from utils.common import load_config
        
        # Load config
        config_path = Path('configs/datasets/custom_soc.yaml')
        if config_path.exists():
            config = load_config(config_path)
        else:
            config = {'time_window': 3600}
        
        # Create dataset
        dataset = CustomSOCDataset(data_path, config, split='test')
        
        print(f"✅ Dataset loaded successfully!")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Number of events: {len(dataset.events)}")
        
        if len(dataset) > 0:
            print(f"  Sample 0 type: {type(dataset.data[0])}")
            if isinstance(dataset.data[0], dict):
                print(f"  Sample 0 keys: {list(dataset.data[0].keys())[:5]}")
        
        print()
        
        # Check if we have actual data
        if len(dataset.events) > 0:
            print("✅ SUCCESS: Dataset has events and can be used for evaluation!")
            return True
        else:
            print("⚠ WARNING: Dataset loaded but has 0 events")
            return False
            
    except Exception as e:
        print(f"✗ Error testing dataset loader: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "DATASET LOADER FIX TEST" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    success = test_dataset_loading()
    
    print("="*80)
    if success:
        print("✅ TEST PASSED - Dataset loader fix is working!")
        print()
        print("Next step: Re-run evaluation")
        print("  ./scripts/run_evaluation.sh --data-path data/custom_soc --skip-preprocess")
    else:
        print("❌ TEST FAILED - Check the error messages above")
        print()
        print("Possible issues:")
        print("  1. Preprocessed file has unexpected structure")
        print("  2. Missing dependencies")
        print("  3. Configuration file issues")
    print("="*80)
    print()
    
    sys.exit(0 if success else 1)
