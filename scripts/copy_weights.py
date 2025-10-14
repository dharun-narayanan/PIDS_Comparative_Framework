#!/usr/bin/env python3
"""
Simple script to copy existing pretrained weights from MAGIC, Continuum_FL, and Orthrus directories.
This script doesn't require any external dependencies like gdown or requests.
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def copy_weights(source_dir: Path, dest_dir: Path, model_name: str):
    """Copy weights from source to destination."""
    if not source_dir.exists():
        print(f"⚠️  Source directory not found: {source_dir}")
        return 0
    
    dest_model_dir = dest_dir / model_name
    dest_model_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    # Check for .pt and .pkl files
    checkpoints = list(source_dir.glob("*.pt")) + list(source_dir.glob("*.pkl"))
    
    for checkpoint in checkpoints:
        dest_file = dest_model_dir / checkpoint.name
        
        if dest_file.exists():
            print(f"  ⏭️  Skipping (already exists): {checkpoint.name}")
            continue
        
        try:
            shutil.copy2(checkpoint, dest_file)
            print(f"  ✓ Copied: {checkpoint.name} -> {model_name}/{checkpoint.name}")
            count += 1
        except Exception as e:
            print(f"  ✗ Failed to copy {checkpoint.name}: {e}")
    
    return count


def main():
    print("="*80)
    print("  PIDS Framework - Copy Existing Pretrained Weights")
    print("="*80)
    print()
    
    # Destination directory
    dest_dir = project_root / 'checkpoints'
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    # Copy MAGIC weights
    print("📦 Checking MAGIC weights...")
    magic_dir = project_root.parent / "MAGIC" / "checkpoints"
    count = copy_weights(magic_dir, dest_dir, "magic")
    total_copied += count
    if count > 0:
        print(f"  ✓ Copied {count} MAGIC checkpoint(s)")
    print()
    
    # Copy Continuum_FL weights
    print("📦 Checking Continuum_FL weights...")
    continuum_dir = project_root.parent / "Continuum_FL" / "checkpoints"
    count = copy_weights(continuum_dir, dest_dir, "continuum_fl")
    total_copied += count
    if count > 0:
        print(f"  ✓ Copied {count} Continuum_FL checkpoint(s)")
    print()
    
    # Copy Orthrus weights
    print("📦 Checking Orthrus weights...")
    orthrus_dir = project_root.parent / "orthrus" / "weights"
    count = copy_weights(orthrus_dir, dest_dir, "orthrus")
    total_copied += count
    if count > 0:
        print(f"  ✓ Copied {count} Orthrus checkpoint(s)")
    print()
    
    # Copy Kairos weights (if any)
    print("📦 Checking Kairos weights...")
    kairos_dir = project_root.parent / "kairos" / "DARPA"
    # Check subdirectories for weights
    kairos_count = 0
    for subdir in ["CADETS_E3", "THEIA_E3", "CLEARSCOPE_E3"]:
        subdir_path = kairos_dir / subdir
        if subdir_path.exists():
            count = copy_weights(subdir_path, dest_dir, "kairos")
            kairos_count += count
    if kairos_count > 0:
        print(f"  ✓ Copied {kairos_count} Kairos checkpoint(s)")
    total_copied += kairos_count
    print()
    
    # Copy ThreaTrace weights (if any)
    print("📦 Checking ThreaTrace weights...")
    threatrace_dir = project_root.parent / "threaTrace" / "models"
    count = copy_weights(threatrace_dir, dest_dir, "threatrace")
    total_copied += count
    if count > 0:
        print(f"  ✓ Copied {count} ThreaTrace checkpoint(s)")
    print()
    
    # Summary
    print("="*80)
    if total_copied > 0:
        print(f"✅ Successfully copied {total_copied} checkpoint(s)")
        print(f"📁 Checkpoints saved to: {dest_dir}")
        print()
        print("Next steps:")
        print("  1. Verify checkpoints: ls -lR checkpoints/")
        print("  2. Run evaluation: ./scripts/run_evaluation.sh")
    else:
        print("⚠️  No new checkpoints copied")
        print()
        print("Possible reasons:")
        print("  • Checkpoints already exist in destination")
        print("  • Source directories don't contain .pt or .pkl files")
        print("  • Source directories not found in parent directory")
        print()
        print("Source directories checked:")
        print(f"  • {project_root.parent / 'MAGIC' / 'checkpoints'}")
        print(f"  • {project_root.parent / 'Continuum_FL' / 'checkpoints'}")
        print(f"  • {project_root.parent / 'orthrus' / 'weights'}")
        print(f"  • {project_root.parent / 'kairos' / 'DARPA'}")
        print(f"  • {project_root.parent / 'threaTrace' / 'models'}")
    print("="*80)


if __name__ == '__main__':
    main()
