#!/bin/bash

# Quick fix for PyTorch MKL threading issue
# This script resolves the "undefined symbol: iJIT_NotifyEvent" error

set -e

echo "============================================"
echo "PyTorch MKL Threading Issue Fix"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_error "No conda environment activated!"
    echo "Please activate your environment first:"
    echo "  conda activate pids_framework"
    exit 1
fi

print_info "Active environment: $CONDA_DEFAULT_ENV"
echo ""

# Method 1: Set environment variable permanently
print_info "Method 1: Setting MKL_THREADING_LAYER=GNU permanently..."
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

print_info "✓ Environment variable will be set automatically on activation"
echo ""

# Apply the fix for current session
export MKL_THREADING_LAYER=GNU

# Test PyTorch import
print_info "Testing PyTorch import..."
if python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully!')" 2>/dev/null; then
    print_info "✓ PyTorch is working!"
    echo ""
    
    # Show CUDA info
    python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('Number of GPUs:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
else
    print_warning "PyTorch still not working. Trying Method 2..."
    echo ""
    
    # Method 2: Reinstall with compatible MKL
    print_info "Method 2: Reinstalling MKL to compatible version..."
    conda install "mkl<2024" -c conda-forge --force-reinstall -y
    
    # Test again
    if python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully!')" 2>/dev/null; then
        print_info "✓ PyTorch is working after MKL reinstall!"
    else
        print_error "PyTorch still not working. Trying Method 3..."
        echo ""
        
        # Method 3: Reinstall PyTorch
        print_info "Method 3: Reinstalling PyTorch..."
        conda uninstall pytorch torchvision torchaudio -y
        conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -y
        
        if python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully!')" 2>/dev/null; then
            print_info "✓ PyTorch is working after reinstall!"
        else
            print_error "All methods failed. Manual intervention required."
            echo ""
            echo "Please try:"
            echo "  1. Recreate the conda environment from scratch"
            echo "  2. Use pip-installed PyTorch instead:"
            echo "     pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116"
            exit 1
        fi
    fi
fi

echo ""
echo "============================================"
echo "Fix Applied Successfully!"
echo "============================================"
echo ""
echo "The environment variable MKL_THREADING_LAYER=GNU"
echo "will be set automatically whenever you activate"
echo "the '${CONDA_DEFAULT_ENV}' environment."
echo ""
echo "To test manually, run:"
echo "  python -c 'import torch; print(torch.__version__)'"
echo ""
