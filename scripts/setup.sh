#!/bin/bash

# PIDS Comparative Framework - Unified Setup Script
# This script sets up the complete framework environment including:
#   - Conda environment creation
#   - All dependencies installation
#   - PyTorch MKL fix (integrated)
#   - Directory structure creation
#   - Installation verification
#
# Prerequisites:
#   - Conda (Anaconda or Miniconda) must be installed
#   - CUDA 11.6+ (optional, for GPU support)
#
# Usage:
#   ./scripts/setup.sh

set -e  # Exit on error

echo "============================================"
echo "PIDS Comparative Framework - Complete Setup"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Step 1: Check if conda is installed
print_step "Step 1/7: Checking for Conda installation"
if ! command -v conda &> /dev/null; then
    print_error "Conda not found! Please install Anaconda or Miniconda first."
    echo ""
    echo "Installation instructions:"
    echo "  macOS/Linux: https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "Quick install (Linux/macOS):"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname -s)-$(uname -m).sh"
    echo "  bash Miniconda3-latest-$(uname -s)-$(uname -m).sh"
    echo ""
    exit 1
fi

print_info "Conda found: $(conda --version)"
echo ""

# Step 2: Create conda environment
ENV_NAME="pids_framework"
print_step "Step 2/7: Creating conda environment: ${ENV_NAME}"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
        print_info "Creating new environment..."
        conda env create -f environment.yml -n ${ENV_NAME}
    else
        print_info "Keeping existing environment."
        skip_env_creation=true
    fi
else
    print_info "Creating new environment from environment.yml..."
    conda env create -f environment.yml -n ${ENV_NAME}
fi

echo ""

# Step 3: Initialize conda for shell
print_step "Step 3/7: Initializing Conda for shell"
conda init bash 2>/dev/null || true
conda init zsh 2>/dev/null || true

# Source conda
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" 2>/dev/null || true
fi
if [ -f "$HOME/.zshrc" ]; then
    source "$HOME/.zshrc" 2>/dev/null || true
fi

# Initialize conda in current shell
eval "$(conda shell.bash hook)"

echo ""

# Step 4: Activate environment
print_step "Step 4/7: Activating environment"
conda activate ${ENV_NAME}

# Verify activation
ACTIVE_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$ACTIVE_ENV" != "$ENV_NAME" ]; then
    print_warning "Automatic activation may have failed. Please manually activate:"
    echo "  conda activate ${ENV_NAME}"
else
    print_info "Environment activated: ${ENV_NAME}"
fi

echo ""

# Optional preinstall: Intel OpenMP and prebuilt PyG wheels
# This helps avoid pip trying to build torch-scatter/torch-sparse/torch-cluster
# from source (which can fail if torch's native libs can't be imported during build).
print_step "Step 4.1/7: Pre-install optional runtime helpers"
print_info "Installing intel-openmp and prebuilt PyG wheels (best-effort; non-fatal)"

# Install Intel OpenMP runtime into the active environment (provides ITT/OpenMP symbols)
if ! conda install -c conda-forge intel-openmp -y; then
    print_warning "intel-openmp install failed. Continuing; this may require manual intervention."
else
    print_info "intel-openmp installed"
fi

# Attempt to install prebuilt PyG wheels for PyTorch 1.12.1 + CUDA 11.6 so pip won't build from source.
# If your environment uses a different PyTorch/CUDA combo, update the wheel index URL accordingly.
PYG_WHL_INDEX="https://data.pyg.org/whl/torch-1.12.1+cu116.html"
if ! python -m pip install -U -f "$PYG_WHL_INDEX" \
    torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 torch-geometric==2.1.0 dgl==1.0.0; then
    print_warning "Prebuilt PyG/dgl install failed or fell back to source builds. The script will continue but you may need to retry these installs manually."
else
    print_info "Prebuilt PyG and DGL (where available) installed successfully"
fi


# Step 5: Apply PyTorch MKL Fix (integrated from fix_pytorch_mkl.sh)
print_step "Step 5/7: Applying PyTorch MKL threading fix"
print_info "Setting MKL_THREADING_LAYER=GNU to prevent symbol conflicts..."

# Create activation/deactivation scripts for the conda environment
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

# Create activation script
cat > "${CONDA_PREFIX}/etc/conda/activate.d/mkl_fix.sh" << 'EOF'
#!/bin/bash
export MKL_THREADING_LAYER=GNU
EOF

# Create deactivation script
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/mkl_fix.sh" << 'EOF'
#!/bin/bash
unset MKL_THREADING_LAYER
EOF

chmod +x "${CONDA_PREFIX}/etc/conda/activate.d/mkl_fix.sh"
chmod +x "${CONDA_PREFIX}/etc/conda/deactivate.d/mkl_fix.sh"

# Apply the fix for current session
export MKL_THREADING_LAYER=GNU

print_info "✓ MKL threading fix applied (will auto-activate with environment)"

# Test PyTorch import
print_info "Testing PyTorch import..."
if python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully!')" 2>/dev/null; then
    print_info "✓ PyTorch is working!"
else
    print_warning "PyTorch import test failed. Trying alternative fix..."
    
    # Method 2: Reinstall with compatible MKL
    print_info "Reinstalling MKL to compatible version..."
    conda install "mkl<2024" -c conda-forge --force-reinstall -y
    
    # Test again
    if python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully!')" 2>/dev/null; then
        print_info "✓ PyTorch is working after MKL reinstall!"
    else
        print_error "PyTorch still not working. Manual intervention may be required."
        print_error "Please see documentation for advanced troubleshooting."
    fi
fi

echo ""

# Step 6: Create directory structure
print_step "Step 6/7: Creating directory structure"
mkdir -p data/{darpa_tc,streamspot,custom,processed,cache}
mkdir -p checkpoints/{magic,kairos,orthrus,threatrace,continuum_fl}
mkdir -p results/{experiments,comparisons,evaluation,plots}
mkdir -p logs
mkdir -p configs/{models,datasets,experiments}

print_info "Directory structure created:"
echo "  ├── data/           - Dataset storage"
echo "  ├── checkpoints/    - Model checkpoints"
echo "  ├── results/        - Experiment results"
echo "  ├── logs/           - Training logs"
echo "  └── configs/        - Configuration files"
echo ""

# Step 7: Verify installation
print_step "Step 7/7: Verifying installation"

print_info "Checking Python version..."
python --version

print_info "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    print_error "PyTorch verification failed"
    exit 1
}

print_info "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

print_info "Checking DGL installation..."
python -c "import dgl; print(f'DGL version: {dgl.__version__}')" || print_warning "DGL not installed or not working"

print_info "Checking PyTorch Geometric..."
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')" || print_warning "PyG not installed or not working"

echo ""

# Final verification
print_info "Running comprehensive dependency check..."
python -c "
import sys
try:
    import torch
    import dgl
    import numpy as np
    import sklearn
    import yaml
    print('✓ All core dependencies installed successfully')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo -e "${GREEN}✓ Setup completed successfully!${NC}"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the environment (if not already active):"
    echo "   conda activate ${ENV_NAME}"
    echo ""
    echo "2. Setup models and download pretrained weights:"
    echo "   python scripts/setup_models.py --all"
    echo ""
    echo "3. Preprocess your custom SOC data:"
    echo "   python scripts/preprocess_data.py --input-dir ../custom_dataset/"
    echo ""
    echo "4. Run evaluation on all models:"
    echo "   ./scripts/run_evaluation.sh"
    echo ""
    echo "📖 For detailed instructions, see:"
    echo "   - setup.md        - Step-by-step setup guide"
    echo "   - README.md       - Complete documentation"
    echo ""
else
    print_error "Core dependencies verification failed!"
    exit 1
fi
