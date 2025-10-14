#!/bin/bash

# PIDS Comparative Framework - Automated Setup Script
# This script sets up the framework environment and dependencies
#
# Prerequisites:
#   - Conda (Anaconda or Miniconda) must be installed
#   - CUDA 11.6+ (optional, for GPU support)
#
# Usage:
#   ./scripts/setup.sh

set -e  # Exit on error

echo "============================================"
echo "PIDS Comparative Framework Setup"
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
print_step "Step 1/8: Checking for Conda installation"
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
print_step "Step 2/8: Creating conda environment: ${ENV_NAME}"

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
print_step "Step 3/8: Initializing Conda for shell"
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
print_step "Step 4/8: Activating environment"
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

# Step 5: Install PyTorch Geometric and additional dependencies
print_step "Step 5/8: Installing PyTorch Geometric and additional dependencies"
print_info "This may take a few minutes..."

# Install PyG extension wheels that match the installed PyTorch/CUDA build.
# We try to import torch to detect the CUDA version. If that fails, fall back
# to checking for cudatoolkit in the conda environment. Default to CPU wheels.
PYG_TORCH_VERSION="1.12.1"
WHEEL_INDEX="https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+cpu.html"

PYTHON_CHECK='import importlib,sys
try:
    import torch
    v = getattr(torch, "version", None)
    cuda = getattr(torch, "version", None) and torch.version.cuda
    if cuda is not None and "11.6" in str(cuda):
        print("cu116")
    else:
        print("cpu")
except Exception:
    # fallback to checking conda-installed cudatoolkit
    try:
        import subprocess, json
        out = subprocess.check_output(["conda", "list", "cudatoolkit", "--json"]).decode()
        entries = json.loads(out)
        if entries and any("11.6" in e.get("version","") for e in entries):
            print("cu116")
        else:
            print("cpu")
    except Exception:
        print("cpu")
'

PYG_CUDA_TAG=$(python - <<PY
${PYTHON_CHECK}
PY
)

if [ "${PYG_CUDA_TAG}" = "cu116" ]; then
    WHEEL_INDEX="https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+cu116.html"
    print_info "Detected CUDA 11.6-compatible PyTorch. Installing CUDA wheels (+cu116)."
else
    print_info "Installing CPU wheels for PyG extensions. If you have CUDA available, re-run with CUDA-enabled PyTorch."
fi

# Attempt installation from multiple wheel indices to improve compatibility.
# Preferred order: detected index, cu116, cu113, cpu.
declare -a TRY_INDICES=()
TRY_INDICES+=("${WHEEL_INDEX}")
TRY_INDICES+=("https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+cu116.html")
TRY_INDICES+=("https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+cu113.html")
TRY_INDICES+=("https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+cpu.html")

INSTALL_SUCCESS=0
for IDX in "${TRY_INDICES[@]}"; do
    print_info "Trying PyG wheel index: ${IDX}"
    # Try to install extensions from this index
    if pip install --no-cache-dir --quiet torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 -f ${IDX}; then
        if pip install --no-cache-dir --quiet torch-geometric==2.1.0; then
            print_info "Installed PyG extensions successfully using index: ${IDX}"
            INSTALL_SUCCESS=1
            break
        else
            print_warning "torch-geometric install failed for index ${IDX}; trying next index"
        fi
    else
        print_warning "PyG extension wheel install failed for index ${IDX}; trying next index"
    fi
done

if [ ${INSTALL_SUCCESS} -ne 1 ]; then
    print_error "Failed to install PyG extension wheels from known indices."
    print_error "You can try manual installation (see QUICKSTART.md) or install a matching PyTorch build first."
    exit 1
fi

# Install additional utilities
pip install --quiet tqdm scikit-learn pandas matplotlib seaborn pyyaml

print_info "PyTorch Geometric installed successfully"
echo ""

# Step 6: Create directory structure
print_step "Step 6/8: Creating directory structure"
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
print_step "Step 7/8: Verifying installation"

print_info "Checking Python version..."
python --version

print_info "Checking PyTorch installation..."
# Set MKL threading layer to avoid symbol conflicts in the current session
export MKL_THREADING_LAYER=GNU
if ! python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
    print_warning "PyTorch import failed. Attempting automated fix using scripts/fix_pytorch_mkl.sh..."

    # Try to run packaged fix script which handles activation scripts and MKL reinstall
    if [ -x "$(pwd)/scripts/fix_pytorch_mkl.sh" ]; then
        print_info "Running scripts/fix_pytorch_mkl.sh (may prompt for conda privileges)..."
        ./scripts/fix_pytorch_mkl.sh || true
    else
        print_warning "scripts/fix_pytorch_mkl.sh not found or not executable. Falling back to manual steps."
    fi

    # After running the fix script (or fallback), try again
    if MKL_THREADING_LAYER=GNU python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null; then
        print_info "PyTorch working with MKL_THREADING_LAYER=GNU"
        # Ensure activation script exists (if CONDA_PREFIX is set)
        if [ -n "${CONDA_PREFIX}" ]; then
            mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
            echo 'export MKL_THREADING_LAYER=GNU' > "${CONDA_PREFIX}/etc/conda/activate.d/mkl_fix.sh"
        fi
        print_info "✓ MKL threading fix applied (activation script created if possible)"
    else
        print_error "PyTorch installation verification failed after automated fix."
        print_error "You can try manual fixes:" 
        echo "  1) Export for current session: export MKL_THREADING_LAYER=GNU"
        echo "  2) Run the fix script: chmod +x scripts/fix_pytorch_mkl.sh && ./scripts/fix_pytorch_mkl.sh"
        echo "  3) Reinstall a compatible MKL: conda install 'mkl<2024' -c conda-forge --force-reinstall -y"
        echo "  4) Reinstall PyTorch matching your CUDA: conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.6 -c pytorch -y"
        exit 1
    fi
else
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
fi

print_info "Checking CUDA availability..."
MKL_THREADING_LAYER=GNU python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

print_info "Checking DGL installation..."
MKL_THREADING_LAYER=GNU python -c "import dgl; print(f'DGL version: {dgl.__version__}')" || print_warning "DGL not installed or not working"

print_info "Checking PyTorch Geometric..."
MKL_THREADING_LAYER=GNU python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')" || print_warning "PyG not installed or not working"

echo ""

# Step 8: Final instructions
print_step "Step 8/8: Setup complete!"
echo ""
echo "============================================"
echo "  Setup Summary"
echo "============================================"
echo "✅ Conda environment created: ${ENV_NAME}"
echo "✅ Dependencies installed"
echo "✅ Directory structure created"
echo "✅ Installation verified"
echo ""
echo "============================================"
echo "  Next Steps"
echo "============================================"
echo ""
echo "1. Activate the environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "2. Install model-specific dependencies:"
echo "   ./scripts/install_model_deps.sh --all"
echo ""
echo "3. Copy existing pretrained weights:"
echo "   python scripts/download_weights.py --copy-existing"
echo ""
echo "4. Preprocess your custom SOC data:"
echo "   python scripts/preprocess_data.py --input-dir ../custom_dataset/"
echo ""
echo "5. Start training:"
echo "   python experiments/train.py --model magic --dataset custom_soc"
echo ""
echo "📖 For detailed instructions, see:"
echo "   - QUICKSTART.md      - Quick start guide"
echo "   - README.md          - Complete documentation"
echo ""
echo "============================================"
echo ""
MKL_THREADING_LAYER=GNU python -c "
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
    print_info "Core dependencies verification passed!"
else
    print_error "Core dependencies verification failed!"
    exit 1
fi

echo ""
echo "============================================"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate ${ENV_NAME}"
echo "2. Download model weights: python scripts/download_weights.py"
echo "3. Prepare datasets: See docs/datasets.md"
echo "4. Run example: python experiments/train.py --help"
echo ""
echo "For model-specific dependencies, run:"
echo "  ./scripts/install_model_deps.sh --models magic kairos orthrus"
echo ""
