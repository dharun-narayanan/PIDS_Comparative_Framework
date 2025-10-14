# PyTorch MKL Issue - Immediate Fix Instructions

## Problem Summary

Your setup script failed at Step 7 with this error:
```
ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

This is a **common and fixable** issue caused by incompatibility between PyTorch and Intel MKL threading libraries.

---

## ⚡ IMMEDIATE FIX (Choose One)

### Option 1: Run Automated Fix Script (RECOMMENDED)

```bash
cd ~/PIDS/PIDS_Comparative_Framework
conda activate pids_framework
chmod +x scripts/fix_pytorch_mkl.sh
./scripts/fix_pytorch_mkl.sh
```

This will:
- ✅ Automatically detect and apply the fix
- ✅ Set environment variables permanently
- ✅ Reinstall MKL if needed
- ✅ Verify PyTorch is working

---

### Option 2: Quick Manual Fix (30 seconds)

```bash
# Activate your environment
conda activate pids_framework

# Set the fix permanently
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export MKL_THREADING_LAYER=GNU' > $CONDA_PREFIX/etc/conda/activate.d/mkl_fix.sh

# Apply for current session
export MKL_THREADING_LAYER=GNU

# Test it works
python -c "import torch; print(f'✓ PyTorch {torch.__version__} working!')"
```

---

### Option 3: Reinstall Compatible MKL

```bash
conda activate pids_framework
conda install "mkl<2024" -c conda-forge --force-reinstall -y
python -c "import torch; print(f'✓ PyTorch {torch.__version__} working!')"
```

---

## ✅ Verify the Fix

After applying any fix method above, run the test script:

```bash
python scripts/test_pytorch.py
```

Expected output:
```
============================================================
  PIDS Framework - PyTorch Installation Test
============================================================

============================================================
  Environment Information
============================================================
Python version: 3.10.13
...
MKL_THREADING_LAYER: GNU

============================================================
  PyTorch Import Test
============================================================
✓ PyTorch imported successfully
  Version: 1.12.1
✓ Basic tensor operations working: [2.0, 4.0, 6.0]

...

============================================================
  Test Summary
============================================================
Environment              ✓ PASSED
Pytorch                  ✓ PASSED
Cuda                     ✓ PASSED
Dependencies             ✓ PASSED
Graph Libs               ✓ PASSED

============================================================
  All tests passed! PyTorch is ready to use.
============================================================
```

---

## 🔄 Continue Setup

Once PyTorch is working, continue with the framework setup:

```bash
# Re-run the setup script (it will skip completed steps)
./scripts/setup.sh

# Or manually continue with the next steps:

# Step 1: Install model-specific dependencies
./scripts/install_model_deps.sh --all

# Step 2: Copy existing pretrained weights
python scripts/download_weights.py --copy-existing

# Step 3: Preprocess your custom SOC data
python scripts/preprocess_data.py --input-dir ../custom_dataset/

# Step 4: Run evaluation on all models
./scripts/run_evaluation.sh
```

---

## 📋 What Changed

I've updated your framework with these fixes:

### 1. **Updated setup.sh** (`scripts/setup.sh`)
   - Now automatically detects and fixes MKL issues during installation
   - Applies environment variable fix permanently
   - Includes fallback methods if primary fix fails

### 2. **New automated fix script** (`scripts/fix_pytorch_mkl.sh`)
   - Standalone script to fix PyTorch import issues
   - Try 3 different methods automatically
   - Provides detailed feedback

### 3. **New test script** (`scripts/test_pytorch.py`)
   - Comprehensive PyTorch installation testing
   - Checks CUDA, dependencies, graph libraries
   - Easy to diagnose issues

### 4. **Documentation** (`docs/PYTORCH_MKL_FIX.md`)
   - Complete troubleshooting guide
   - Multiple fix methods explained
   - Prevention tips for future installations

### 5. **Updated README** (`README.md`)
   - Added PyTorch MKL issue to troubleshooting section
   - Links to detailed fix guide

---

## 🎯 Next Steps After Fix

1. **Verify**: `python scripts/test_pytorch.py`
2. **Continue Setup**: `./scripts/setup.sh` or manual steps above
3. **Start Using**: `./scripts/run_evaluation.sh`

---

## 💡 Why This Happens

The `iJIT_NotifyEvent` symbol comes from Intel VTune profiling libraries. When:
- PyTorch expects one version of Intel MKL
- But finds a different version installed
- The symbol definitions don't match
- Import fails

The fix tells MKL to use GNU OpenMP instead of Intel OpenMP, avoiding the version mismatch.

---

## 🆘 If Still Not Working

1. **Check the detailed guide**: `docs/PYTORCH_MKL_FIX.md`
2. **Try pip-installed PyTorch**:
   ```bash
   conda activate pids_framework
   conda uninstall pytorch torchvision torchaudio -y
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

3. **Recreate environment from scratch**:
   ```bash
   conda deactivate
   conda env remove -n pids_framework
   conda env create -f environment.yml -n pids_framework
   conda activate pids_framework
   ./scripts/fix_pytorch_mkl.sh
   ```

---

## ✅ This is a Common Issue

- Affects many PyTorch users on Linux
- Well-documented with known fixes
- Does NOT indicate a serious problem
- Quick to resolve (< 5 minutes)

You're one command away from a working installation! 🚀
