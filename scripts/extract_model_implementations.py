#!/usr/bin/env python3
"""
Script to create standalone implementations for all PIDS models.

This script extracts the core components from external model repositories
and creates self-contained implementations within the framework.
"""

import os
import sys
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Framework root directory
FRAMEWORK_ROOT = Path(__file__).parent.parent
IMPLEMENTATIONS_DIR = FRAMEWORK_ROOT / 'models' / 'implementations'
EXTERNAL_MODELS_DIR = FRAMEWORK_ROOT.parent

# Model configurations
MODELS_TO_EXTRACT = {
    'kairos': {
        'source_dir': EXTERNAL_MODELS_DIR / 'kairos' / 'DARPA' / 'CADETS_E3',
        'files_to_extract': ['model.py', 'kairos_utils.py', 'config.py'],
        'dependencies': ['torch', 'torch_geometric']
    },
    'orthrus': {
        'source_dir': EXTERNAL_MODELS_DIR / 'orthrus' / 'src',
        'files_to_extract': ['model.py', 'encoders.py', 'decoders.py', 'temporal.py'],
        'dependencies': ['torch', 'torch_geometric']
    },
    'continuum_fl': {
        'source_dir': EXTERNAL_MODELS_DIR / 'Continuum_FL' / 'model',
        'files_to_extract': ['model.py', 'gat.py', 'rnn.py', 'eval.py'],
        'dependencies': ['torch', 'dgl']
    },
    'threatrace': {
        'source_dir': EXTERNAL_MODELS_DIR / 'threaTrace' / 'scripts',
        'files_to_extract': ['train_darpatc.py'],
        'dependencies': ['torch', 'scikit-learn']
    }
}


def create_init_file(model_name: str, model_dir: Path):
    """Create __init__.py file for model implementation."""
    init_content = f'''"""
{model_name.upper()} Model Implementation
Standalone implementation extracted and adapted for PIDS Comparative Framework.

This is a self-contained version that doesn't depend on external repositories.
"""

# Import main components here
# from .model import {model_name.capitalize()}Model

__all__ = []
'''
    
    init_file = model_dir / '__init__.py'
    with open(init_file, 'w') as f:
        f.write(init_content)
    
    logger.info(f"Created __init__.py for {model_name}")


def extract_model_implementation(model_name: str, config: dict):
    """Extract model implementation from external repository."""
    source_dir = Path(config['source_dir'])
    target_dir = IMPLEMENTATIONS_DIR / model_name
    
    if not source_dir.exists():
        logger.warning(f"Source directory not found for {model_name}: {source_dir}")
        logger.info(f"Skipping {model_name} extraction")
        return False
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created implementation directory: {target_dir}")
    
    # Create __init__.py
    create_init_file(model_name, target_dir)
    
    # Copy files (we'll adapt them manually later)
    for filename in config['files_to_extract']:
        source_file = source_dir / filename
        if source_file.exists():
            target_file = target_dir / filename
            logger.info(f"Found {filename}, will need manual adaptation")
            # Note: We don't copy directly because files need adaptation
        else:
            logger.warning(f"File not found: {source_file}")
    
    # Create a README for manual adaptation
    readme_content = f"""# {model_name.upper()} Implementation

## Source
Extracted from: `{source_dir}`

## Files to Adapt
{chr(10).join(f'- {f}' for f in config['files_to_extract'])}

## Dependencies
{chr(10).join(f'- {d}' for d in config['dependencies'])}

## Adaptation Checklist
- [ ] Remove external path dependencies
- [ ] Extract core model classes
- [ ] Adapt imports to use relative imports
- [ ] Remove unused code
- [ ] Add proper documentation
- [ ] Test standalone functionality

## Notes
This implementation should be self-contained and not depend on the original repository.
"""
    
    readme_file = target_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created README for {model_name}")
    return True


def main():
    """Main extraction process."""
    logger.info("=" * 60)
    logger.info("PIDS Model Implementation Extractor")
    logger.info("=" * 60)
    
    # Ensure implementations directory exists
    IMPLEMENTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract each model
    extracted_count = 0
    for model_name, config in MODELS_TO_EXTRACT.items():
        logger.info(f"\nProcessing {model_name}...")
        if extract_model_implementation(model_name, config):
            extracted_count += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"Models processed: {len(MODELS_TO_EXTRACT)}")
    logger.info(f"Implementations created: {extracted_count}")
    
    if extracted_count < len(MODELS_TO_EXTRACT):
        logger.warning(f"\nSome models could not be extracted. Manual implementation required.")
    
    logger.info("\nNext steps:")
    logger.info("1. Review README files in each implementation directory")
    logger.info("2. Manually adapt the core model files")
    logger.info("3. Test each implementation independently")
    logger.info("4. Update wrapper files to use standalone implementations")
    logger.info("\nSee docs/ARCHITECTURE_AND_EXTENSIBILITY.md for detailed guidance.")


if __name__ == '__main__':
    main()
