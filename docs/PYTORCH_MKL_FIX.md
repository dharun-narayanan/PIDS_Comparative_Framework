# PyTorch MKL Threading Issue - Troubleshooting Guide

## Problem

When trying to import PyTorch, you encounter the error:

```
ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

## Root Cause

This error occurs due to a compatibility issue between PyTorch and Intel MKL (Math Kernel Library) threading layers. The `iJIT_NotifyEvent` symbol is part of Intel VTune profiling libraries, and version mismatches between MKL and PyTorch can cause this import failure.

## Solutions

### Quick Fix (Recommended for Immediate Use)

Run the automated fix script:

```bash
cd ~/PIDS/PIDS_Comparative_Framework
conda activate pids_framework
chmod +x scripts/fix_pytorch_mkl.sh
./scripts/fix_pytorch_mkl.sh
```

This script will:
1. Set `MKL_THREADING_LAYER=GNU` environment variable permanently
2. Test PyTorch import
3. If needed, reinstall compatible MKL version
4. If still failing, reinstall PyTorch

### Manual Fix Methods

#### Method 1: Set Environment Variable

The simplest solution is to tell MKL to use GNU OpenMP threading:

```bash
export MKL_THREADING_LAYER=GNU
```

To make this permanent for your conda environment:

```bash
conda activate pids_framework
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export MKL_THREADING_LAYER=GNU' > $CONDA_PREFIX/etc/conda/activate.d/mkl_fix.sh
```

Now the variable will be set automatically whenever you activate the environment.

#### Method 2: Downgrade MKL

Install a compatible MKL version:

```bash
conda activate pids_framework
conda install "mkl<2024" -c conda-forge --force-reinstall
```

#### Method 3: Reinstall PyTorch

Completely reinstall PyTorch:

```bash
conda activate pids_framework
conda uninstall pytorch torchvision torchaudio
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch
```

#### Method 4: Use pip-installed PyTorch

Switch to pip installation (may have better compatibility):

```bash
conda activate pids_framework
conda uninstall pytorch torchvision torchaudio
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

For CPU-only:

```bash
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

## Verification

After applying any fix, verify PyTorch works:

```bash
# Test basic import
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Full system info
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
print('cuDNN version:', torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A')
print('Number of GPUs:', torch.cuda.device_count())
"
```

## Prevention for Future Installations

When creating new conda environments, specify MKL version:

```yaml
dependencies:
  - python=3.10
  - pytorch=1.12.1
  - mkl<2024  # Pin MKL to compatible version
```

Or set the environment variable in your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
# For all conda environments
export MKL_THREADING_LAYER=GNU
```

## Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Intel MKL Threading Layer Documentation](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-linux-developer-guide/top/linking-your-application-with-the-intel-oneapi-math-kernel-library/linking-in-detail/linking-with-threading-libraries.html)
- [GitHub Issue: PyTorch MKL Symbol Issues](https://github.com/pytorch/pytorch/issues/123097)

## Common Issues

### Issue: Environment variable not persisting

**Solution:** Make sure you're setting it in the conda environment activation script, not just in your shell session.

### Issue: CUDA not available after fix

**Solution:** The MKL fix only affects CPU operations. If CUDA isn't available, check:
- NVIDIA drivers are installed: `nvidia-smi`
- CUDA toolkit is installed: `nvcc --version`
- PyTorch was installed with CUDA support: `python -c "import torch; print(torch.version.cuda)"`

### Issue: Different error after applying fix

**Solution:** Try a clean reinstall:

```bash
conda deactivate
conda env remove -n pids_framework
conda env create -f environment.yml -n pids_framework
conda activate pids_framework
./scripts/fix_pytorch_mkl.sh
```

## System-Specific Notes

### Ubuntu/Debian Linux
If using system Python/BLAS, you may need to install Intel MKL separately:
```bash
sudo apt-get install intel-mkl
```

### RHEL/CentOS/Rocky Linux
```bash
sudo yum install intel-mkl
```

### macOS
MKL issues are less common on macOS, but if encountered:
```bash
brew install intel-mkl
```

## Contact

If none of these solutions work, please open an issue with:
- Output of `conda list`
- Output of `python -c "import sys; print(sys.version)"`
- Full error traceback
- OS and architecture (`uname -a`)
