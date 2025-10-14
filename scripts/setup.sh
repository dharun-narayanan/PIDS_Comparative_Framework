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
print_step "Step 6/7: Creating directory structure and installing PyTorch Geometric"

# First create directories
mkdir -p data/{darpa_tc,streamspot,custom,processed,cache}
mkdir -p checkpoints/{magic,kairos,orthrus,threatrace,continuum_fl}
mkdir -p results/{experiments,comparisons,evaluation,plots}
mkdir -p logs
mkdir -p configs/{models,datasets,experiments}

print_info "Directory structure created"

# Now install PyTorch Geometric and extensions with MKL fix applied
print_info "Installing PyTorch Geometric and extensions (with MKL fix applied)..."

# Detect CUDA version for appropriate wheels
PYTHON_CHECK='import torch
try:
    cuda = torch.version.cuda
    if cuda and "11.6" in str(cuda):
        print("cu116")
    elif cuda and "11.3" in str(cuda):
        print("cu113")
    else:
        print("cpu")
except:
    print("cpu")
'

PYG_CUDA_TAG=$(python -c "$PYTHON_CHECK")

if [ "${PYG_CUDA_TAG}" = "cu116" ]; then
    TORCH_VERSION="1.12.1"
    CUDA_VERSION="cu116"
    print_info "Detected CUDA 11.6, installing CUDA-enabled PyG extensions"
elif [ "${PYG_CUDA_TAG}" = "cu113" ]; then
    TORCH_VERSION="1.12.1"
    CUDA_VERSION="cu113"
    print_info "Detected CUDA 11.3, installing CUDA-enabled PyG extensions"
else
    TORCH_VERSION="1.12.1"
    CUDA_VERSION="cpu"
    print_info "Installing CPU-only PyG extensions"
fi

# Install PyTorch Geometric extensions with appropriate CUDA support
print_info "Installing torch-scatter, torch-sparse, torch-cluster..."
pip install --no-cache-dir torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-cache-dir torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install --no-cache-dir torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html

# Install PyTorch Geometric itself
print_info "Installing torch-geometric..."
pip install --no-cache-dir torch-geometric==2.1.0

print_info "✓ PyTorch Geometric and extensions installed successfully"

echo ""
echo "Directory structure:"
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
