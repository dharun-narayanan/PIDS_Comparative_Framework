#!/usr/bin/env python3
"""
Download pretrained model weights.

This script downloads pretrained weights for PIDS models from various sources
including Google Drive, GitHub releases, and direct URLs.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# Pretrained model weights URLs
PRETRAINED_WEIGHTS = {
    'magic': {
        'streamspot': {
            'url': 'https://drive.google.com/file/d/MAGIC_STREAMSPOT_ID/view',
            'filename': 'checkpoint-streamspot.pt',
            'description': 'MAGIC trained on StreamSpot dataset'
        },
        'cadets_e3': {
            'url': 'https://drive.google.com/file/d/MAGIC_CADETS_ID/view',
            'filename': 'checkpoint-cadets-e3.pt',
            'description': 'MAGIC trained on DARPA CADETS E3'
        },
        'theia_e3': {
            'url': 'https://drive.google.com/file/d/MAGIC_THEIA_ID/view',
            'filename': 'checkpoint-theia-e3.pt',
            'description': 'MAGIC trained on DARPA THEIA E3'
        },
        'trace_e3': {
            'url': 'https://drive.google.com/file/d/MAGIC_TRACE_ID/view',
            'filename': 'checkpoint-trace-e3.pt',
            'description': 'MAGIC trained on DARPA TRACE E3'
        }
    },
    'kairos': {
        'cadets_e3': {
            'url': 'https://github.com/kairos/releases/download/v1.0/kairos-cadets-e3.pt',
            'filename': 'checkpoint-cadets-e3.pt',
            'description': 'Kairos trained on DARPA CADETS E3'
        }
    },
    'orthrus': {
        'darpa': {
            'url': 'https://github.com/crimson-unicorn/orthrus/releases/download/v1.0/orthrus-darpa.pt',
            'filename': 'checkpoint-darpa.pt',
            'description': 'Orthrus trained on DARPA datasets'
        }
    },
    'threatrace': {
        'streamspot': {
            'url': 'https://github.com/threatrace/releases/download/v1.0/threatrace-streamspot.pt',
            'filename': 'checkpoint-streamspot.pt',
            'description': 'ThreaTrace trained on StreamSpot'
        }
    }
}


def download_from_google_drive(url: str, output_path: Path):
    """Download file from Google Drive."""
    try:
        # Import gdown only when needed
        try:
            import gdown
        except ImportError:
            logger.error("gdown not installed. Install with: pip install gdown")
            logger.info("For copying existing weights, use --copy-existing flag instead")
            return False
        
        # Extract file ID from URL
        if 'drive.google.com' in url:
            gdown.download(url, str(output_path), quiet=False, fuzzy=True)
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        return False


def download_from_url(url: str, output_path: Path):
    """Download file from direct URL with progress bar."""
    try:
        # Import requests and tqdm only when needed
        try:
            import requests
            from tqdm import tqdm
        except ImportError:
            logger.error("requests or tqdm not installed. Install with: pip install requests tqdm")
            return False
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        logger.error(f"Failed to download from URL: {e}")
        return False


def download_weight(model: str, variant: str, output_dir: Path):
    """Download a specific model weight."""
    if model not in PRETRAINED_WEIGHTS:
        logger.error(f"Unknown model: {model}")
        return False
    
    if variant not in PRETRAINED_WEIGHTS[model]:
        logger.error(f"Unknown variant for {model}: {variant}")
        return False
    
    weight_info = PRETRAINED_WEIGHTS[model][variant]
    url = weight_info['url']
    filename = weight_info['filename']
    
    # Create model-specific directory
    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = model_dir / filename
    
    # Check if already downloaded
    if output_path.exists():
        logger.info(f"Weight already exists: {output_path}")
        return True
    
    logger.info(f"Downloading {model}/{variant}: {weight_info['description']}")
    logger.info(f"URL: {url}")
    
    # Try Google Drive first
    if 'drive.google.com' in url:
        success = download_from_google_drive(url, output_path)
    else:
        success = download_from_url(url, output_path)
    
    if success:
        logger.info(f"Successfully downloaded to: {output_path}")
        return True
    else:
        logger.error(f"Failed to download {model}/{variant}")
        return False


def list_available_weights():
    """List all available pretrained weights."""
    print("\n" + "="*80)
    print("Available Pretrained Weights")
    print("="*80 + "\n")
    
    for model, variants in PRETRAINED_WEIGHTS.items():
        print(f"Model: {model}")
        print("-" * 40)
        for variant, info in variants.items():
            print(f"  • {variant:15s} - {info['description']}")
        print()


def copy_existing_weights(source_dir: Path, output_dir: Path):
    """Copy existing weights from Continuum_FL or MAGIC directories."""
    logger.info("Checking for existing weights in project...")
    
    # Check Continuum_FL checkpoints
    continuum_checkpoints = project_root.parent / "Continuum_FL" / "checkpoints"
    if continuum_checkpoints.exists():
        logger.info(f"Found Continuum_FL checkpoints: {continuum_checkpoints}")
        
        for checkpoint in continuum_checkpoints.glob("checkpoint-*.pt"):
            # Determine model and variant from filename
            filename = checkpoint.name
            
            # Copy to appropriate model directory
            if 'cadets' in filename.lower():
                dest = output_dir / 'magic' / filename
            elif 'theia' in filename.lower():
                dest = output_dir / 'magic' / filename
            elif 'trace' in filename.lower():
                dest = output_dir / 'magic' / filename
            elif 'streamspot' in filename.lower():
                dest = output_dir / 'magic' / filename
            else:
                dest = output_dir / 'magic' / filename
            
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            if not dest.exists():
                import shutil
                shutil.copy2(checkpoint, dest)
                logger.info(f"Copied: {checkpoint.name} -> {dest}")
    
    # Check MAGIC checkpoints
    magic_checkpoints = project_root.parent / "MAGIC" / "checkpoints"
    if magic_checkpoints.exists():
        logger.info(f"Found MAGIC checkpoints: {magic_checkpoints}")
        
        for checkpoint in magic_checkpoints.glob("*.pt"):
            dest = output_dir / 'magic' / checkpoint.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            if not dest.exists():
                import shutil
                shutil.copy2(checkpoint, dest)
                logger.info(f"Copied: {checkpoint.name} -> {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained PIDS model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=list(PRETRAINED_WEIGHTS.keys()) + ['all'],
        help='Model to download weights for'
    )
    
    parser.add_argument(
        '--variant',
        type=str,
        help='Specific variant to download (e.g., streamspot, cadets_e3)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / 'checkpoints',
        help='Output directory for weights (default: checkpoints/)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available pretrained weights'
    )
    
    parser.add_argument(
        '--copy-existing',
        action='store_true',
        help='Copy existing weights from Continuum_FL/MAGIC directories'
    )
    
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Copy weights for all models (same as --model all --copy-existing)'
    )
    
    args = parser.parse_args()
    
    # List available weights
    if args.list:
        list_available_weights()
        return
    
    # Copy existing weights for all models
    if args.all_models:
        logger.info("Copying existing weights for all models...")
        copy_existing_weights(project_root.parent, args.output_dir)
        return
    
    # Copy existing weights
    if args.copy_existing:
        copy_existing_weights(project_root.parent, args.output_dir)
        return
    
    # Download weights
    if not args.model:
        parser.print_help()
        print("\nUse --list to see available weights")
        return
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model == 'all':
        # Download all weights
        for model in PRETRAINED_WEIGHTS:
            for variant in PRETRAINED_WEIGHTS[model]:
                download_weight(model, variant, output_dir)
    else:
        if args.variant:
            # Download specific variant
            download_weight(args.model, args.variant, output_dir)
        else:
            # Download all variants for model
            for variant in PRETRAINED_WEIGHTS[args.model]:
                download_weight(args.model, variant, output_dir)
    
    logger.info("\nDownload complete!")
    logger.info(f"Weights saved to: {output_dir}")


if __name__ == '__main__':
    main()
