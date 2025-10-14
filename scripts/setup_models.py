#!/usr/bin/env python3
"""
PIDS Framework - Unified Model Setup Script

This script handles:
1. Installing model-specific dependencies
2. Checking for and copying local pretrained weights
3. Downloading missing pretrained weights from official sources

Replaces: copy_weights.py, download_weights.py, install_model_deps.sh

Usage:
    python scripts/setup_models.py --all                    # Setup all models
    python scripts/setup_models.py --models magic kairos    # Setup specific models
    python scripts/setup_models.py --download-only          # Only download weights
    python scripts/setup_models.py --list                   # List available models
"""

import os
import sys
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
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

# Model configurations with pretrained weight sources from GitHub repos
MODEL_CONFIGS = {
    'magic': {
        'name': 'MAGIC',
        'description': 'Masked Graph Autoencoder for APT Detection',
        'requirements': 'requirements/magic.txt',
        'github_repo': 'https://github.com/FDUDSDE/MAGIC',
        'local_fallback': [
            project_root.parent / 'MAGIC' / 'checkpoints',
        ],
        'weights': {
            'streamspot': {
                'filename': 'checkpoint-streamspot.pt',
                'url': 'https://raw.githubusercontent.com/FDUDSDE/MAGIC/main/checkpoints/checkpoint-streamspot.pt',
                'description': 'MAGIC trained on StreamSpot dataset'
            },
            'cadets': {
                'filename': 'checkpoint-cadets.pt',
                'url': 'https://raw.githubusercontent.com/FDUDSDE/MAGIC/main/checkpoints/checkpoint-cadets.pt',
                'description': 'MAGIC trained on DARPA CADETS'
            },
            'theia': {
                'filename': 'checkpoint-theia.pt',
                'url': 'https://raw.githubusercontent.com/FDUDSDE/MAGIC/main/checkpoints/checkpoint-theia.pt',
                'description': 'MAGIC trained on DARPA THEIA'
            },
            'trace': {
                'filename': 'checkpoint-trace.pt',
                'url': 'https://raw.githubusercontent.com/FDUDSDE/MAGIC/main/checkpoints/checkpoint-trace.pt',
                'description': 'MAGIC trained on DARPA TRACE'
            },
            'wget': {
                'filename': 'checkpoint-wget.pt',
                'url': 'https://raw.githubusercontent.com/FDUDSDE/MAGIC/main/checkpoints/checkpoint-wget.pt',
                'description': 'MAGIC trained on Wget dataset'
            }
        }
    },
    'kairos': {
        'name': 'Kairos',
        'description': 'Practical Intrusion Detection with Whole-system Provenance',
        'requirements': 'requirements/kairos.txt',
        'github_repo': 'https://github.com/ubc-provenance/kairos',
        'local_fallback': [
            project_root.parent / 'kairos' / 'DARPA',
        ],
        'weights': {
            'google_drive_folder': {
                'filename': 'kairos_models',  # This will be a folder
                'url': 'https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C',
                'description': 'Kairos pretrained models from Google Drive',
                'gdrive': True,
                'folder_id': '1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C'
            }
        }
    },
    'orthrus': {
        'name': 'Orthrus',
        'description': 'High Quality Attribution in Provenance-based IDS',
        'requirements': 'requirements/orthrus.txt',
        'github_repo': 'https://github.com/ubc-provenance/orthrus',
        'local_fallback': [
            project_root.parent / 'orthrus' / 'weights',
        ],
        'weights': {
            'cadets_e3': {
                'filename': 'CADETS_E3.pkl',
                'url': 'https://raw.githubusercontent.com/ubc-provenance/orthrus/main/weights/CADETS_E3.pkl',
                'description': 'Orthrus CADETS E3 model'
            },
            'clearscope_e3': {
                'filename': 'CLEARSCOPE_E3.pkl',
                'url': 'https://raw.githubusercontent.com/ubc-provenance/orthrus/main/weights/CLEARSCOPE_E3.pkl',
                'description': 'Orthrus CLEARSCOPE E3 model'
            },
            'clearscope_e5': {
                'filename': 'CLEARSCOPE_E5.pkl',
                'url': 'https://raw.githubusercontent.com/ubc-provenance/orthrus/main/weights/CLEARSCOPE_E5.pkl',
                'description': 'Orthrus CLEARSCOPE E5 model'
            },
            'theia_e3': {
                'filename': 'THEIA_E3.pkl',
                'url': 'https://raw.githubusercontent.com/ubc-provenance/orthrus/main/weights/THEIA_E3.pkl',
                'description': 'Orthrus THEIA E3 model'
            },
            'theia_e5': {
                'filename': 'THEIA_E5.pkl',
                'url': 'https://raw.githubusercontent.com/ubc-provenance/orthrus/main/weights/THEIA_E5.pkl',
                'description': 'Orthrus THEIA E5 model'
            }
        }
    },
    'threatrace': {
        'name': 'ThreaTrace',
        'description': 'Scalable Graph-based Threat Detection',
        'requirements': 'requirements/threatrace.txt',
        'github_repo': 'https://github.com/Provenance-IDS/threaTrace',
        'local_fallback': [
            project_root.parent / 'threaTrace' / 'example_models',
        ],
        'weights': {
            'git_sparse_checkout': {
                'repo_url': 'https://github.com/Provenance-IDS/threaTrace.git',
                'sparse_paths': ['example_models'],
                'description': 'ThreaTrace example models (140+ files in 3 subdirectories: darpatc, streamspot, unicornsc)'
            }
        }
    },
    'continuum_fl': {
        'name': 'Continuum_FL',
        'description': 'Federated Learning for PIDS',
        'requirements': 'requirements/continuum_fl.txt',
        'github_repo': 'https://github.com/kamelferrahi/Continuum_FL',
        'local_fallback': [
            project_root.parent / 'Continuum_FL' / 'checkpoints',
            project_root.parent / 'Continuum_FL' / 'result',
        ],
        'weights': {
            'streamspot': {
                'filename': 'checkpoint-streamspot.pt',
                'url': 'https://raw.githubusercontent.com/kamelferrahi/Continuum_FL/master/checkpoints/checkpoint-streamspot.pt',
                'description': 'Continuum_FL trained on StreamSpot'
            },
            'cadets_e3': {
                'filename': 'checkpoint-cadets-e3.pt',
                'url': 'https://raw.githubusercontent.com/kamelferrahi/Continuum_FL/master/checkpoints/checkpoint-cadets-e3.pt',
                'description': 'Continuum_FL trained on CADETS E3'
            },
            'theia_e3': {
                'filename': 'checkpoint-theia-e3.pt',
                'url': 'https://raw.githubusercontent.com/kamelferrahi/Continuum_FL/master/checkpoints/checkpoint-theia-e3.pt',
                'description': 'Continuum_FL trained on THEIA E3'
            },
            'trace_e3': {
                'filename': 'checkpoint-trace-e3.pt',
                'url': 'https://raw.githubusercontent.com/kamelferrahi/Continuum_FL/master/checkpoints/checkpoint-trace-e3.pt',
                'description': 'Continuum_FL trained on TRACE E3'
            },
            'clearscope_e3': {
                'filename': 'checkpoint-clearscope-e3.pt',
                'url': 'https://raw.githubusercontent.com/kamelferrahi/Continuum_FL/master/checkpoints/checkpoint-clearscope-e3.pt',
                'description': 'Continuum_FL trained on CLEARSCOPE E3'
            }
        }
    }
}


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def install_dependencies(model: str) -> bool:
    """Install model-specific dependencies."""
    config = MODEL_CONFIGS.get(model)
    if not config:
        logger.error(f"Unknown model: {model}")
        return False
    
    req_file = project_root / config['requirements']
    if not req_file.exists():
        logger.warning(f"Requirements file not found: {req_file}")
        return False
    
    logger.info(f"Installing dependencies for {config['name']}...")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(req_file)],
            check=True,
            capture_output=True
        )
        logger.info(f"✓ {config['name']} dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install {config['name']} dependencies")
        logger.error(e.stderr.decode())
        return False


def copy_local_weights(model: str) -> int:
    """Copy weights from local directories as fallback if available."""
    config = MODEL_CONFIGS.get(model)
    if not config:
        return 0
    
    dest_dir = project_root / 'checkpoints' / model
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    # Special handling for ThreaTrace - copy entire directory structure
    if model == 'threatrace':
        for local_path in config.get('local_fallback', []):
            if not local_path.exists():
                continue
            
            logger.info(f"   Checking local fallback: {local_path.parent.name}/{local_path.name}")
            
            # Copy the entire example_models directory structure to checkpoints/threatrace/
            if dest_dir.exists() and any(dest_dir.iterdir()):
                logger.info(f"      ⏭️  Skipping (already exists): example_models/")
                logger.info(f"         Target: {dest_dir}")
                return 1  # Count as success since it exists
            
            try:
                shutil.copytree(local_path, dest_dir, dirs_exist_ok=True)
                logger.info(f"      ✓ Copied directory: example_models/ → checkpoints/{model}/")
                logger.info(f"         Target: {dest_dir}")
                return 1
            except Exception as e:
                logger.warning(f"      ✗ Failed to copy example_models/: {e}")
                return 0
        
        return 0
    
    # Standard checkpoint file copying for other models
    # Check each local fallback path
    for local_path in config.get('local_fallback', []):
        if not local_path.exists():
            continue
        
        logger.info(f"   Checking local fallback: {local_path.name}")
        
        # Find all checkpoint files
        checkpoint_patterns = ['*.pt', '*.pkl', '*.pth', '*.ckpt']
        checkpoints = []
        for pattern in checkpoint_patterns:
            checkpoints.extend(local_path.glob(pattern))
            # Also check subdirectories
            checkpoints.extend(local_path.rglob(pattern))
        
        # Remove duplicates
        checkpoints = list(set(checkpoints))
        
        if not checkpoints:
            logger.info(f"      No checkpoints found in {local_path}")
            continue
        
        # Copy each checkpoint
        for checkpoint in checkpoints:
            dest_file = dest_dir / checkpoint.name
            
            if dest_file.exists():
                logger.info(f"      ⏭️  Skipping (already exists): {checkpoint.name}")
                continue
            
            try:
                shutil.copy2(checkpoint, dest_file)
                logger.info(f"      ✓ Copied: {checkpoint.name}")
                copied_count += 1
            except Exception as e:
                logger.warning(f"      ✗ Failed to copy {checkpoint.name}: {e}")
    
    return copied_count


def git_sparse_checkout(repo_url: str, sparse_paths: list, output_dir: Path) -> bool:
    """Download specific directories from a git repository using sparse-checkout."""
    temp_dir = None
    try:
        # Check if git is available
        if not shutil.which('git'):
            logger.error("❌ git not installed")
            return False
        
        # Create temp directory for sparse checkout
        temp_dir = output_dir.parent / f".git_temp_{output_dir.name}"
        
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"   Initializing sparse checkout from: {repo_url}")
        logger.info(f"   Downloading directories: {', '.join(sparse_paths)}")
        
        # Initialize git repo with sparse-checkout
        subprocess.run(['git', 'init'], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=temp_dir, check=True, capture_output=True)
        subprocess.run(['git', 'config', 'core.sparseCheckout', 'true'], cwd=temp_dir, check=True, capture_output=True)
        
        # Configure sparse-checkout paths
        sparse_checkout_file = temp_dir / '.git' / 'info' / 'sparse-checkout'
        sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_checkout_file.write_text('\n'.join(sparse_paths) + '\n')
        
        # Pull only the specified directories
        logger.info("   Downloading files (this may take a few minutes)...")
        result = subprocess.run(
            ['git', 'pull', 'origin', 'master'],
            cwd=temp_dir,
            capture_output=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode != 0:
            logger.error(f"❌ git pull failed: {result.stderr.decode()}")
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return False
        
        # Move the downloaded directories to output location
        success = False
        for sparse_path in sparse_paths:
            source_path = temp_dir / sparse_path
            if source_path.exists():
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                shutil.move(str(source_path), str(output_dir))
                logger.info(f"   ✓ Downloaded directory: {output_dir.name}/")
                success = True
        
        # Clean up temp directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return success
        
    except subprocess.TimeoutExpired:
        logger.error("❌ git sparse-checkout timed out (repository too large)")
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False
    except Exception as e:
        logger.error(f"❌ git sparse-checkout failed: {e}")
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def download_weights(model: str, force_download: bool = False) -> int:
    """Download missing weights for a model from GitHub or other sources."""
    config = MODEL_CONFIGS.get(model)
    if not config:
        return 0
    
    dest_dir = project_root / 'checkpoints' / model
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    failed_downloads = []
    
    logger.info(f"Downloading weights for {config['name']}...")
    logger.info(f"   Repository: {config['github_repo']}")
    
    for variant, weight_info in config.get('weights', {}).items():
        # Handle git sparse-checkout (for ThreaTrace)
        if variant == 'git_sparse_checkout':
            repo_url = weight_info.get('repo_url')
            sparse_paths = weight_info.get('sparse_paths', [])
            
            if not repo_url or not sparse_paths:
                logger.info(f"   ℹ️  git sparse-checkout: Missing configuration")
                continue
            
            # Target directory is checkpoints/<model>/ (standard location)
            target_dir = dest_dir
            
            # Skip if already exists and not forcing download
            if target_dir.exists() and any(target_dir.iterdir()) and not force_download:
                logger.info(f"   ⏭️  {variant}: Already exists (checkpoints/{model}/)")
                continue
            
            logger.info(f"   📥 {variant}: {weight_info['description']}")
            
            if git_sparse_checkout(repo_url, sparse_paths, target_dir):
                downloaded_count += 1
            else:
                failed_downloads.append(variant)
                logger.warning(f"      ⚠️  Git sparse-checkout failed")
                logger.info(f"      💡 Alternative: git clone {repo_url} && cp -r example_models checkpoints/{model}/")
            continue
        
        filename = weight_info.get('filename')
        url = weight_info.get('url')
        
        # Skip if no filename specified
        if not filename:
            logger.info(f"   ℹ️  {variant}: No filename specified (using local fallback only)")
            continue
            
        dest_file = dest_dir / filename
        
        # Skip if already exists and not forcing download
        if dest_file.exists() and not force_download:
            logger.info(f"   ⏭️  {variant}: Already exists ({filename})")
            continue
        
        # Skip if no URL
        if not url:
            logger.info(f"   ℹ️  {variant}: No download URL available (using local fallback)")
            continue
        
        # Check if manual download is required
        if weight_info.get('manual', False):
            logger.info(f"   ⚠️  {variant}: {weight_info['description']}")
            logger.info(f"      Manual download required from: {url}")
            failed_downloads.append(variant)
            continue
        
        logger.info(f"   📥 {variant}: {weight_info['description']}")
        
        # Check for Google Drive parameters
        is_gdrive = weight_info.get('gdrive', False)
        folder_id = weight_info.get('folder_id')
        svn_export = weight_info.get('svn_export', False)
        is_directory = weight_info.get('is_directory', False)
        
        if download_weight_from_url(url, dest_file, is_gdrive=is_gdrive, folder_id=folder_id, 
                                   svn_export=svn_export, is_directory=is_directory):
            downloaded_count += 1
        else:
            failed_downloads.append(variant)
            logger.warning(f"      ⚠️  Download failed for {variant}")
    
    if failed_downloads:
        logger.warning(f"   Failed to download {len(failed_downloads)} variant(s): {', '.join(failed_downloads)}")
        logger.info(f"   Checking local fallback directories...")
    
    return downloaded_count


def download_weight_from_url(url: str, output_path: Path, is_gdrive: bool = False, 
                            folder_id: Optional[str] = None, svn_export: bool = False, 
                            is_directory: bool = False) -> bool:
    """Download a weight file from a URL using curl, wget, gdown, or svn export."""
    try:
        # Handle SVN export for GitHub directories (ThreaTrace)
        if svn_export and 'github.com' in url and '/trunk/' in url:
            logger.info("📦 GitHub directory download detected (using svn export)")
            
            # Check if svn is available
            svn_available = shutil.which('svn') is not None
            
            if not svn_available:
                logger.error("❌ svn not installed. Install with: sudo apt-get install subversion (Ubuntu) or brew install subversion (macOS)")
                logger.info(f"   Or manually clone repository and copy: {url.replace('/trunk/', '/tree/master/')}")
                return False
            
            try:
                # Create output directory
                output_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"   Downloading directory with svn export...")
                logger.info(f"   Target: {output_path}")
                
                # Use svn export to download the directory
                result = subprocess.run(
                    ['svn', 'export', url, str(output_path), '--force'],
                    check=True,
                    capture_output=True,
                    timeout=600  # 10 minutes for large directories
                )
                
                # Check if directory has files
                if output_path.exists() and any(output_path.iterdir()):
                    logger.info(f"   ✓ Downloaded directory: {output_path.name}/")
                    return True
                else:
                    logger.error("❌ Downloaded directory is empty")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error("❌ svn export timed out (directory too large)")
                logger.info("   💡 Try downloading manually or use local fallback")
                return False
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ svn export failed: {e.stderr.decode()}")
                logger.info(f"   💡 Manual download: git clone {url.replace('/trunk/', '').split('/tree/')[0]}.git")
                return False
        
        # Handle Google Drive downloads
        if is_gdrive or 'drive.google.com' in url or folder_id:
            logger.info("📦 Google Drive download detected")
            
            try:
                import gdown
                
                if folder_id:
                    # Download entire folder
                    logger.info(f"   Downloading Google Drive folder: {folder_id}")
                    logger.info("   ⚠️  This may take a while for large folders...")
                    
                    # Create output directory
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        # Download folder contents
                        gdown.download_folder(
                            id=folder_id,
                            output=str(output_path.parent),
                            quiet=False,
                            use_cookies=False
                        )
                        logger.info(f"   ✓ Downloaded Google Drive folder to: {output_path.parent}")
                        return True
                    except Exception as e:
                        logger.error(f"❌ gdown folder download failed: {e}")
                        logger.info("   💡 Alternative: Download manually from:")
                        logger.info(f"      https://drive.google.com/drive/folders/{folder_id}")
                        return False
                        
                elif 'drive.google.com' in url:
                    # Extract file ID from URL
                    logger.info(f"   Downloading with gdown from: {url}")
                    
                    try:
                        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
                        
                        if output_path.exists() and output_path.stat().st_size > 0:
                            logger.info(f"   ✓ Downloaded: {output_path.name}")
                            return True
                        else:
                            raise Exception("Downloaded file is empty or doesn't exist")
                            
                    except Exception as e:
                        logger.error(f"❌ gdown download failed: {e}")
                        logger.info("   💡 Try downloading manually from the Google Drive link")
                        return False
                        
            except ImportError:
                logger.error("❌ gdown not installed. Install with: pip install gdown")
                logger.info(f"   Or manually download from: {url}")
                if folder_id:
                    logger.info(f"   Folder URL: https://drive.google.com/drive/folders/{folder_id}")
                logger.info(f"   Save to: {output_path.parent}")
                return False
        
        # Check if curl is available
        curl_available = shutil.which('curl') is not None
        wget_available = shutil.which('wget') is not None
        
        if not curl_available and not wget_available:
            logger.warning("Neither curl nor wget found. Trying Python requests...")
            try:
                import requests
                from tqdm import tqdm
                
                logger.info(f"   Downloading from: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(
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
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                logger.info(f"   ✓ Downloaded: {output_path.name}")
                return True
                
            except ImportError:
                logger.error("❌ requests not installed. Install with: pip install requests tqdm")
                logger.info(f"   Or manually download from: {url}")
                logger.info(f"   Save to: {output_path}")
                return False
            except Exception as e:
                logger.error(f"❌ Download failed: {e}")
                return False
        
        # Try curl first (more common on macOS)
        if curl_available:
            logger.info(f"   Downloading with curl: {output_path.name}")
            try:
                result = subprocess.run(
                    ['curl', '-L', '-f', '-o', str(output_path), url],
                    check=True,
                    capture_output=True,
                    timeout=300
                )
                logger.info(f"   ✓ Downloaded: {output_path.name}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ curl failed: {e.stderr.decode()}")
                # Try wget as fallback
                if wget_available:
                    logger.info("   Trying wget as fallback...")
                else:
                    return False
        
        # Try wget
        if wget_available:
            logger.info(f"   Downloading with wget: {output_path.name}")
            try:
                result = subprocess.run(
                    ['wget', '-O', str(output_path), url],
                    check=True,
                    capture_output=True,
                    timeout=300
                )
                logger.info(f"   ✓ Downloaded: {output_path.name}")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ wget failed: {e.stderr.decode()}")
                return False
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


def setup_model(model: str, install_deps: bool = True, download: bool = True, copy_local: bool = True) -> Dict[str, int]:
    """Setup a single model (dependencies, weights).
    
    Order of operations:
    1. Install dependencies
    2. Download weights from GitHub/official sources (primary method)
    3. Fallback to copying from local directories if download fails
    """
    config = MODEL_CONFIGS.get(model)
    if not config:
        logger.error(f"Unknown model: {model}")
        return {'installed': 0, 'copied': 0, 'downloaded': 0}
    
    print_header(f"Setting up {config['name']}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"GitHub: {config['github_repo']}")
    
    results = {'installed': 0, 'copied': 0, 'downloaded': 0}
    
    # Install dependencies
    if install_deps:
        if install_dependencies(model):
            results['installed'] = 1
    
    # Download weights from GitHub/official sources (PRIMARY METHOD)
    if download:
        results['downloaded'] = download_weights(model)
        if results['downloaded'] > 0:
            logger.info(f"✓ Downloaded {results['downloaded']} checkpoint(s) from GitHub/official sources")
    
    # Fallback to local copies if download failed or incomplete
    if copy_local:
        # Check if we need to copy (some weights may not have been downloaded)
        dest_dir = project_root / 'checkpoints' / model
        existing_weights = list(dest_dir.glob('*.pt')) if dest_dir.exists() else []
        
        if len(existing_weights) == 0:
            logger.info(f"No weights downloaded, checking local fallback directories...")
            results['copied'] = copy_local_weights(model)
            if results['copied'] > 0:
                logger.info(f"✓ Copied {results['copied']} checkpoint(s) from local fallback")
        else:
            logger.info(f"✓ Found {len(existing_weights)} weight file(s) already present")
    
    return results


def list_models():
    """List all available models and their configurations."""
    print_header("Available Models")
    
    for model_id, config in MODEL_CONFIGS.items():
        print(f"\n{config['name']} ({model_id})")
        print("-" * 40)
        print(f"  Description: {config['description']}")
        print(f"  GitHub Repo: {config['github_repo']}")
        print(f"  Requirements: {config['requirements']}")
        print(f"  Pretrained Weights:")
        for variant, weight_info in config.get('weights', {}).items():
            manual = weight_info.get('manual', False)
            if manual:
                status = "⚠️  Manual"
            elif weight_info.get('url'):
                status = "📥 Download"
            else:
                status = "📂 Local Only"
            print(f"    • {variant:20s} - {weight_info['description'][:60]}")
            print(f"      {status}")
            if weight_info.get('url') and not manual:
                print(f"      {weight_info['url']}")


def main():
    parser = argparse.ArgumentParser(
        description="PIDS Framework - Unified Model Setup (Downloads from GitHub)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup all models (downloads weights from GitHub)
  python scripts/setup_models.py --all
  
  # Setup specific models
  python scripts/setup_models.py --models magic kairos orthrus
  
  # Only install dependencies (skip weights)
  python scripts/setup_models.py --all --no-download --no-copy
  
  # Force re-download all weights
  python scripts/setup_models.py --all --force-download
  
  # List available models and their GitHub repos
  python scripts/setup_models.py --list

Note: This script primarily downloads weights from official GitHub repositories.
      Local directories are used as fallback only if downloads fail.
"""
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Setup all models'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODEL_CONFIGS.keys()),
        help='Specific models to setup'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models and exit'
    )
    
    parser.add_argument(
        '--no-install',
        action='store_true',
        help='Skip dependency installation'
    )
    
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Skip copying local weights'
    )
    
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip downloading weights'
    )
    
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download weights (skip dependencies and local copy)'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download weights even if they exist'
    )
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list:
        list_models()
        return
    
    # Determine which models to setup
    models_to_setup = []
    if args.all:
        models_to_setup = list(MODEL_CONFIGS.keys())
    elif args.models:
        models_to_setup = args.models
    else:
        parser.print_help()
        print("\nError: Please specify --all or --models")
        return
    
    # Determine what to do
    install_deps = not (args.no_install or args.download_only)
    copy_local = not (args.no_copy or args.download_only)
    download = not args.no_download
    force_download = args.force_download
    
    print_header("PIDS Framework - Model Setup (GitHub Download)")
    logger.info(f"Setting up models: {', '.join(models_to_setup)}")
    logger.info(f"Strategy: Download from GitHub → Fallback to local if needed")
    logger.info(f"Actions: Install deps={install_deps}, Download={download}, Copy local fallback={copy_local}")
    if force_download:
        logger.info(f"Force download: Enabled (will re-download existing weights)")
    
    # Setup each model
    total_results = {'installed': 0, 'copied': 0, 'downloaded': 0}
    
    for model in models_to_setup:
        results = setup_model(model, install_deps, download, copy_local)
        for key in total_results:
            total_results[key] += results[key]
    
    # Print summary
    print_header("Setup Summary")
    logger.info(f"✓ Dependencies installed for {total_results['installed']} model(s)")
    logger.info(f"✓ Downloaded {total_results['downloaded']} checkpoint(s) from GitHub/official sources")
    logger.info(f"✓ Copied {total_results['copied']} checkpoint(s) from local fallback")
    
    logger.info("\nCheckpoints saved to: checkpoints/")
    logger.info("\nNext steps:")
    logger.info("  1. Verify weights: ls -lh checkpoints/*/")
    logger.info("  2. Preprocess data: python scripts/preprocess_data.py")
    logger.info("  3. Run evaluation: ./scripts/run_evaluation.sh")


if __name__ == '__main__':
    main()
