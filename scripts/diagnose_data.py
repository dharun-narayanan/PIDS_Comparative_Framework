#!/usr/bin/env python3
"""
Diagnostic script to investigate empty dataset issue.
Run this to understand why preprocessed data has 0 events.
"""

import sys
import json
import pickle
from pathlib import Path

def check_source_files(data_path):
    """Check source JSON files."""
    print("="*80)
    print("CHECKING SOURCE JSON FILES")
    print("="*80)
    
    data_path = Path(data_path)
    json_files = ['endpoint_process.json', 'endpoint_file.json', 'endpoint_network.json']
    
    for filename in json_files:
        filepath = data_path / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"✓ {filename}: {len(data)} entries")
                        if len(data) > 0:
                            print(f"  Sample keys: {list(data[0].keys())[:5]}")
                    elif isinstance(data, dict):
                        print(f"✓ {filename}: dict with {len(data)} keys")
                        print(f"  Keys: {list(data.keys())[:5]}")
                    else:
                        print(f"⚠ {filename}: unexpected type {type(data)}")
            except json.JSONDecodeError as e:
                print(f"✗ {filename}: JSON decode error - {e}")
            except Exception as e:
                print(f"✗ {filename}: {e}")
        else:
            print(f"✗ {filename}: NOT FOUND")
    print()

def check_preprocessed_file(data_path):
    """Check preprocessed pickle file."""
    print("="*80)
    print("CHECKING PREPROCESSED FILE")
    print("="*80)
    
    data_path = Path(data_path)
    pkl_file = data_path / 'custom_soc_graph.pkl'
    
    if not pkl_file.exists():
        print(f"✗ Preprocessed file not found: {pkl_file}")
        return
    
    print(f"✓ Found: {pkl_file}")
    print(f"  Size: {pkl_file.stat().st_size / 1024:.2f} KB")
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())}")
            
            # Count actual events/edges
            num_events = 0
            if 'events' in data:
                num_events = len(data['events'])
                print(f"    ✓ Found 'events' key with {num_events} entries")
            elif 'edges' in data:
                num_events = len(data['edges'])
                print(f"    ✓ Found 'edges' key with {num_events} entries (will be used as events)")
            else:
                print(f"    ⚠ No 'events' or 'edges' key found!")
            
            # Show all keys with their sizes
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    print(f"    {key}: {len(value)} items")
                else:
                    print(f"    {key}: {type(value).__name__} = {value}")
                    
            # Show statistics if available
            if 'stats' in data and isinstance(data['stats'], dict):
                print(f"\n  Graph Statistics:")
                for stat_key, stat_val in data['stats'].items():
                    print(f"    {stat_key}: {stat_val}")
                    
        elif hasattr(data, '__dict__'):
            print(f"  Attributes: {list(data.__dict__.keys())[:10]}")
            if hasattr(data, 'num_nodes'):
                print(f"    num_nodes: {data.num_nodes}")
            if hasattr(data, 'num_edges'):
                print(f"    num_edges: {data.num_edges}")
        else:
            print(f"  Content: {str(data)[:200]}")
            
    except Exception as e:
        print(f"✗ Error reading pickle file: {e}")
    print()

def check_data_loader(data_path):
    """Try loading data using the framework's dataset loader."""
    print("="*80)
    print("CHECKING DATASET LOADER")
    print("="*80)
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from data.dataset import get_dataset
        
        dataset = get_dataset('custom_soc', data_path, split='test')
        print(f"✓ Dataset loaded successfully")
        print(f"  Length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample type: {type(sample)}")
            if hasattr(sample, 'x'):
                print(f"  Sample x shape: {sample.x.shape}")
            if hasattr(sample, 'edge_index'):
                print(f"  Sample edge_index shape: {sample.edge_index.shape}")
                
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
    print()

def suggest_fixes():
    """Suggest potential fixes."""
    print("="*80)
    print("SUGGESTED FIXES")
    print("="*80)
    print("""
1. If source JSON files are empty or missing:
   - Check that you copied the correct files to data/custom_soc/
   - Verify JSON format matches expected schema
   - Example format: [{"timestamp": ..., "event_type": ..., ...}, ...]

2. If source files have data but preprocessed is empty:
   - Re-run preprocessing with verbose logging:
     python scripts/preprocess_data.py \\
         --dataset custom_soc \\
         --data-path data/custom_soc \\
         --output-dir data/custom_soc \\
         --verbose

3. If preprocessing filters are too strict:
   - Check configs/datasets/custom_soc.yaml
   - Adjust filtering parameters
   - Set min_events, time_window, etc.

4. If data format is incompatible:
   - Review data/dataset.py to see expected format
   - Create custom data loader if needed
   - See EXTEND.md for adding custom datasets

5. Quick test with sample data:
   - Try with DARPA dataset to verify framework works:
     ./scripts/run_evaluation.sh --dataset cadets --data-path data/darpa
""")

def main():
    """Main diagnostic routine."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PIDS DATASET DIAGNOSTIC TOOL" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # Check default path
    data_path = Path('data/custom_soc')
    
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    
    print(f"Checking data directory: {data_path}")
    print()
    
    if not data_path.exists():
        print(f"✗ Directory not found: {data_path}")
        print(f"  Please provide correct path: python {sys.argv[0]} <data_path>")
        return
    
    # Run diagnostics
    check_source_files(data_path)
    check_preprocessed_file(data_path)
    check_data_loader(data_path)
    suggest_fixes()
    
    print("="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
