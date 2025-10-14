#!/bin/bash
# One-liner to fix PyTorch MKL issue
# Usage: curl -sSL https://raw.githubusercontent.com/.../quick_fix.sh | bash
#    OR: ./scripts/quick_fix.sh

echo "🔧 Applying PyTorch MKL fix..."

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ No conda environment active. Run: conda activate pids_framework"
    exit 1
fi

# Create activation script directory
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"

# Add MKL fix
echo 'export MKL_THREADING_LAYER=GNU' > "${CONDA_PREFIX}/etc/conda/activate.d/mkl_fix.sh"
chmod +x "${CONDA_PREFIX}/etc/conda/activate.d/mkl_fix.sh"

# Apply for current session
export MKL_THREADING_LAYER=GNU

# Test
if python -c "import torch; print(f'✅ PyTorch {torch.__version__} working!')" 2>/dev/null; then
    echo "✅ Fix applied successfully!"
    echo "MKL_THREADING_LAYER=GNU will be set automatically on environment activation."
else
    echo "⚠️  Basic fix didn't work. Run full fix script:"
    echo "   ./scripts/fix_pytorch_mkl.sh"
    exit 1
fi
