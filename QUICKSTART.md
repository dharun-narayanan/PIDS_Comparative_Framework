# PIDS Framework - Quick Start Guide

## 🎯 Overview

This framework evaluates **5 pretrained PIDS models** on your custom SOC data:
- **MAGIC** - Masked Graph Autoencoder
- **Kairos** - Temporal GNN
- **Orthrus** - Multi-Decoder Contrastive Learning
- **ThreaTrace** - Scalable Graph Processing
- **Continuum_FL** - Federated Learning PIDS

**Default workflow**: Download pretrained weights → Preprocess data → Evaluate models → Compare performance

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ **Conda** installed (Anaconda or Miniconda)
- ✅ **Python 3.8+** (will be installed via conda)
 - ✅ **Graphviz (system binary)** - required for some visualization tools. The Python package is installed via pip in the environment, but the system `graphviz` binary (provides `dot`) should be installed separately. On macOS:
```bash
# Install Graphviz via Homebrew
brew install graphviz
# Verify
dot -V
```
On Debian/Ubuntu:
```bash
sudo apt-get update && sudo apt-get install -y graphviz
```
- ✅ **Git** (for cloning repositories)
- ✅ **50GB+** free disk space
- ✅ **16GB+** RAM (8GB minimum)
- ⚙️ **GPU** (optional - framework runs on CPU by default)

**Check Conda installation:**
```bash
conda --version
# If not installed, download from: https://docs.conda.io/en/latest/miniconda.html
```

---

## 🚀 Complete Setup Guide (Step-by-Step)

### Step 1: Navigate to Framework Directory

```bash
# Navigate to the framework
cd /Users/lakshmid/Downloads/Research/Final\ PIDS/PIDS_Files/PIDS_Comparative_Framework

# Verify you're in the correct directory
ls
# Expected output: README.md, QUICKSTART.md, scripts/, models/, etc.
```

---

### Step 2: Run Automated Setup

This will create a conda environment and install all dependencies.

```bash
# Make setup script executable (if not already)
chmod +x scripts/setup.sh

# Run the setup script
./scripts/setup.sh
```

**What this does:**
1. ✅ Checks for Conda installation
2. ✅ Creates `pids_framework` conda environment (Python 3.10)
3. ✅ Installs PyTorch 1.12.1 with CUDA 11.6
4. ✅ Installs DGL 1.0.0 (Deep Graph Library)
5. ✅ Installs PyTorch Geometric 2.1.0
6. ✅ Installs all framework dependencies
7. ✅ Creates necessary directories

**Expected output:**
```
Step 1/8: Checking for Conda installation...
✓ Conda is installed: conda 23.x.x

Step 2/8: Creating pids_framework environment...
✓ Environment created successfully

...

Step 8/8: Verifying installation...
✓ Installation complete!
```

**If setup fails**, see [Troubleshooting](#troubleshooting) section below.

---

### Step 3: Activate Conda Environment

```bash
# Activate the environment
conda activate pids_framework

# Verify activation
echo $CONDA_DEFAULT_ENV
# Should output: pids_framework
```

**Important:** You must activate this environment every time you use the framework.

---

### Step 4: Install Model-Specific Dependencies

Each model has specific dependencies that need to be installed.

```bash
# Install dependencies for ALL models (recommended)
./scripts/install_model_deps.sh --all

# OR install for specific models only
./scripts/install_model_deps.sh --models magic kairos orthrus
```

**This installs:**
- MAGIC: Additional graph processing libraries
- Kairos: Temporal modeling dependencies
- Orthrus: Contrastive learning packages
- ThreaTrace: GraphChi libraries
- Continuum_FL: Federated learning tools

**Expected output:**
```
Installing dependencies for magic...
✓ MAGIC dependencies installed

Installing dependencies for kairos...
✓ Kairos dependencies installed

...
All model dependencies installed successfully!
```

---

### Step 5: Download/Copy Pretrained Model Weights

**Option A: Copy from Existing Checkpoints (Recommended)**

If you already have pretrained weights in the parent directories (MAGIC/, Continuum_FL/, etc.):

```bash
# Copy existing weights from MAGIC, Continuum_FL directories
python scripts/download_weights.py --copy-existing --all-models

# Verify weights were copied
ls checkpoints/
# Expected: magic/, kairos/, orthrus/, threatrace/, continuum_fl/
```

**What this does:**
- Scans `../MAGIC/checkpoints/` for MAGIC weights
- Scans `../Continuum_FL/checkpoints/` for Continuum_FL weights
- Copies all found checkpoints to framework's `checkpoints/` directory
- Organizes by model name

**Expected output:**
```
Checking for existing weights in project...
Found MAGIC checkpoints: ../MAGIC/checkpoints
Copied: checkpoint-cadets-e3.pt -> checkpoints/magic/checkpoint-cadets-e3.pt
Copied: checkpoint-streamspot.pt -> checkpoints/magic/checkpoint-streamspot.pt
...
✓ All existing weights copied
```

**Option B: Download from URLs (If weights not available locally)**

```bash
# List available pretrained weights
python scripts/download_weights.py --list

# Download specific model weights
python scripts/download_weights.py --model magic --variant cadets_e3

# Download all available weights
python scripts/download_weights.py --model all
```

**Note:** Some weights may require Google Drive authentication or may not be publicly available. Use Option A if you have local weights.

---

### Step 6: Prepare Your Custom Dataset

Place your SOC data in the `custom_dataset` directory (sibling to framework).

#### 6.1: Create Custom Dataset Directory

```bash
# Navigate to parent directory
cd ..

# Create custom_dataset directory if it doesn't exist
mkdir -p custom_dataset

# Copy your SOC logs
cp /path/to/your/logs/*.json custom_dataset/

# Verify data is present
ls custom_dataset/
# Expected: endpoint_file.json, endpoint_network.json, endpoint_process.json (or similar)

# Return to framework directory
cd PIDS_Comparative_Framework
```

#### 6.2: Data Format Requirements

Your JSON files should contain provenance events with:
- **Process events**: process creation, execution, termination
- **File events**: read, write, create, delete
- **Network events**: connect, bind, send, receive

**Example JSON format:**
```json
{
  "events": [
    {
      "timestamp": "2025-10-13T10:30:00Z",
      "event_type": "process_create",
      "pid": 1234,
      "ppid": 1000,
      "cmdline": "/bin/bash script.sh",
      "user": "admin"
    },
    {
      "timestamp": "2025-10-13T10:30:01Z",
      "event_type": "file_read",
      "pid": 1234,
      "file_path": "/etc/passwd",
      "operation": "read"
    }
  ]
}
```

#### 6.3: Download Benchmark Datasets (Optional)

If you want to test on standard benchmarks:

**DARPA TC Dataset:**
```bash
# Create data directory
mkdir -p data/darpa

# Download DARPA CADETS E3 (example - adjust URL)
# Note: DARPA datasets are large (10GB+) and require access approval
# Visit: https://github.com/darpa-i2o/Transparent-Computing
```

**StreamSpot Dataset:**
```bash
# Create data directory
mkdir -p data/streamspot

# Clone StreamSpot repository
git clone https://github.com/sbustreamspot/sbustreamspot-data.git data/streamspot
```

---

### Step 7: Verify Installation

Run the verification script to ensure everything is set up correctly.

```bash
# Run verification
python scripts/verify_installation.py

# Expected output:
# ✓ Python version: 3.10.x
# ✓ PyTorch installed: 1.12.1
# ✓ DGL installed: 1.0.0
# ✓ PyTorch Geometric installed: 2.1.0
# ✓ All 9 models registered
# ✓ Checkpoints directory exists
# ✓ Configuration files present
# Installation verification complete!
```

**If any checks fail**, see the [Troubleshooting](#troubleshooting) section.

---

### Step 8: Run Evaluation (Primary Workflow)

Now you're ready to evaluate pretrained models on your data!

```bash
# Evaluate ALL models on your custom data (CPU by default)
./scripts/run_evaluation.sh

# Expected output:
# ════════════════════════════════════════════════════════
#     PIDS Comparative Framework - Evaluation Workflow
# ════════════════════════════════════════════════════════
# 
# Configuration:
#   Model(s):      all
#   Dataset:       custom_soc
#   Data Path:     ../custom_dataset
#   Output Dir:    results/evaluation_20251013_103000
#
# Step 1/4: Downloading Pretrained Weights
# ✓ Weights downloaded successfully
#
# Step 2/4: Preprocessing Custom Dataset
# ✓ Data preprocessing completed
#
# Step 3/4: Running Model Evaluation
# Evaluating magic...
# ✓ magic evaluation completed
# Evaluating kairos...
# ✓ kairos evaluation completed
# ...
#
# Step 4/4: Generating Comparison Report
# ✓ Comparison report generated
#
# ✓ EVALUATION COMPLETED SUCCESSFULLY!
#
# Results saved to: results/evaluation_20251013_103000
```

**Results location:** `results/evaluation_YYYYMMDD_HHMMSS/`

---

### Step 9: View Results

```bash
# Navigate to results directory
cd results/evaluation_YYYYMMDD_HHMMSS/

# View comparison report
cat comparison_report.json

# Example output:
# {
#   "dataset": "custom_soc",
#   "timestamp": "2025-10-13T10:30:00",
#   "models": {
#     "magic": {
#       "auc_roc": 0.9245,
#       "auc_pr": 0.8532,
#       "f1": 0.8710,
#       "precision": 0.8456,
#       "recall": 0.8973
#     },
#     "kairos": {
#       "auc_roc": 0.9156,
#       ...
#     }
#   }
# }

# Check individual model logs
tail -n 50 magic_evaluation.log
tail -n 50 kairos_evaluation.log

# View predictions (if saved)
head predictions.json
```

---

## 📊 Understanding Your Results

### Key Metrics Explained

| Metric | Description | Good Value | What It Means |
|--------|-------------|------------|---------------|
| **AUROC** | Area Under ROC Curve | > 0.90 | How well model separates normal vs attack |
| **AUPRC** | Area Under Precision-Recall | > 0.80 | Performance on imbalanced data |
| **F1-Score** | Harmonic mean of precision & recall | > 0.85 | Overall detection accuracy |
| **Precision** | TP / (TP + FP) | > 0.80 | How many detections are correct |
| **Recall** | TP / (TP + FN) | > 0.85 | How many attacks are caught |
| **FPR** | False Positive Rate | < 0.10 | False alarm rate (lower is better) |

### Interpreting Results

**Excellent Performance (AUROC > 0.90)**
- ✅ Model works well on your data
- ✅ Deploy to production
- ✅ Monitor performance over time

**Good Performance (AUROC 0.80-0.90)**
- ⚠️ Acceptable but room for improvement
- Consider threshold tuning
- May benefit from fine-tuning

**Poor Performance (AUROC < 0.80)**
- ❌ Model not suitable for your data
- Try different model
- Consider retraining on labeled data

---

## 🔧 Advanced Options

### Evaluate Specific Model

```bash
# MAGIC only (fastest evaluation)
./scripts/run_evaluation.sh --model magic

# Kairos only
./scripts/run_evaluation.sh --model kairos

# Continuum_FL only
./scripts/run_evaluation.sh --model continuum_fl
```

### Custom Data Path

```bash
# Evaluate on data from different location
./scripts/run_evaluation.sh --data-path /path/to/soc/logs
```

### Use GPU (if available)

```bash
# Check if GPU is available
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Force GPU usage for evaluation
CUDA_VISIBLE_DEVICES=0 ./scripts/run_evaluation.sh

# Specify GPU device (if multiple GPUs)
CUDA_VISIBLE_DEVICES=1 ./scripts/run_evaluation.sh --model magic
```

### Skip Steps (Already Completed)

```bash
# Skip downloading weights (already downloaded)
./scripts/run_evaluation.sh --skip-download

# Skip preprocessing (data already preprocessed)
./scripts/run_evaluation.sh --skip-preprocess

# Skip both
./scripts/run_evaluation.sh --skip-download --skip-preprocess
```

### Custom Output Directory

```bash
# Specify where to save results
./scripts/run_evaluation.sh --output-dir results/my_evaluation

# Results will be in: results/my_evaluation/
```

---

## 🔄 Advanced Feature: Retrain Models (Optional)

> ⚠️ **Only needed if pretrained models achieve AUROC < 0.80 on your data**

### Prerequisites for Retraining

- Labeled attack data (ground truth)
- Sufficient data (thousands of events)
- Time (training can take hours)

### Step-by-Step Retraining

#### 1. Prepare Labeled Data

```bash
# Create ground truth file
cat > data/custom_soc/ground_truth.json << EOF
{
  "attacks": [
    {
      "start_time": "2025-10-13T08:00:00Z",
      "end_time": "2025-10-13T09:00:00Z",
      "attack_type": "apt",
      "entities": [1234, 1235, 1236]
    }
  ]
}
EOF
```

#### 2. Preprocess Data with Labels

```bash
# Preprocess with ground truth
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset/ \
    --labels-file data/custom_soc/ground_truth.json \
    --output-dir data/custom_soc \
    --dataset-name custom_soc
```

#### 3. Train Model (CPU by default)

```bash
# Train MAGIC on custom data (CPU)
python experiments/train.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 100 \
    --batch-size 32 \
    --device -1 \
    --save-dir checkpoints

# Train on GPU (if available)
python experiments/train.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 100 \
    --batch-size 32 \
    --device 0
```

#### 4. Evaluate Retrained Model

```bash
# Evaluate the retrained model
python experiments/evaluate.py \
    --model magic \
    --checkpoint checkpoints/magic_custom_soc_best.pt \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --device -1
```

#### 5. Compare Pretrained vs Retrained

```bash
# Compare performance
python experiments/compare.py \
    --checkpoints checkpoints/magic/checkpoint-cadets-e3.pt checkpoints/magic_custom_soc_best.pt \
    --labels "Pretrained" "Retrained" \
    --dataset custom_soc \
    --output-dir results/comparison
```

### Fine-tune Pretrained Model

```bash
# Fine-tune (transfer learning) instead of training from scratch
python experiments/train.py \
    --model magic \
    --pretrained checkpoints/magic/checkpoint-cadets-e3.pt \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 50 \
    --learning-rate 0.0001 \
    --device -1
```

---

## 🐛 Troubleshooting

### Issue 1: Conda Not Found

**Error:** `conda: command not found`

**Solution:**
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or on macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Restart terminal and retry
```

### Issue 2: Environment Not Activated

**Error:** `conda environment 'pids_framework' not activated`

**Solution:**
```bash
# Activate environment
conda activate pids_framework

# Verify
echo $CONDA_DEFAULT_ENV  # Should show: pids_framework

# Add to ~/.bashrc or ~/.zshrc for auto-activation (optional)
echo "conda activate pids_framework" >> ~/.zshrc
```

### Issue 3: Data Path Does Not Exist

**Error:** `Data path does not exist: ../custom_dataset`

**Solution:**
```bash
# Create directory and add data
mkdir -p ../custom_dataset

# Copy your JSON logs
cp /path/to/your/logs/*.json ../custom_dataset/

# Verify
ls -la ../custom_dataset/
```

### Issue 4: Pretrained Weights Not Found

**Error:** `Pretrained weights not found for model: magic`

**Solution:**
```bash
# Option 1: Copy from existing checkpoints
python scripts/download_weights.py --copy-existing --all-models

# Option 2: Download from URLs
python scripts/download_weights.py --model magic

# Option 3: Manual copy
mkdir -p checkpoints/magic
cp ../MAGIC/checkpoints/*.pt checkpoints/magic/
cp ../Continuum_FL/checkpoints/*.pt checkpoints/continuum_fl/

# Verify
ls -la checkpoints/
```

### Issue 5: Out of Memory

**Error:** `RuntimeError: CUDA out of memory` or `MemoryError`

**Solution:**
```bash
# Solution 1: Use CPU (default, safer)
python experiments/train.py --device -1

# Solution 2: Reduce batch size
# Edit configs/models/MODEL.yaml:
training:
  batch_size: 8  # or even smaller: 4, 2, 1

# Solution 3: Process data in chunks
python scripts/preprocess_data.py --chunk-size 1000
```

### Issue 6: Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Ensure environment is activated
conda activate pids_framework

# Reinstall PyTorch
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# Verify
python -c "import torch; print(torch.__version__)"
```

### PyTorch MKL / "undefined symbol: iJIT_NotifyEvent" error

If you see an import error mentioning `iJIT_NotifyEvent` (or other MKL-related symbol errors), this is usually caused by a mismatch between PyTorch and Intel MKL threading libraries. The repository includes an automated fixer to resolve this quickly.

Preferred automated fix:

```bash
# Activate environment first
conda activate pids_framework

# Make the fix script executable and run it
chmod +x scripts/fix_pytorch_mkl.sh
./scripts/fix_pytorch_mkl.sh

# Re-run verification
python scripts/test_pytorch.py
```

Quick manual fix (applies to current session and can be made permanent):

```bash
# For current session
export MKL_THREADING_LAYER=GNU

# To persist for the conda env
conda activate pids_framework
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export MKL_THREADING_LAYER=GNU' > $CONDA_PREFIX/etc/conda/activate.d/mkl_fix.sh
```

If the automated script and manual fix do not work, try reinstalling a compatible MKL version:

```bash
conda activate pids_framework
conda install "mkl<2024" -c conda-forge --force-reinstall -y
```

Verification:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print('CUDA available:', torch.cuda.is_available())"
```

See `docs/PYTORCH_MKL_FIX.md` and `scripts/fix_pytorch_mkl.sh` for more details.

### Issue 7: Model Not Registered

**Error:** `KeyError: 'magic'` or `Model not found`

**Solution:**
```bash
# Reinstall model dependencies
./scripts/install_model_deps.sh --models magic

# Verify registration
python -c "from models import list_available_models; print(list_available_models())"

# Expected output: ['magic', 'kairos', 'orthrus', 'threatrace', 'continuum_fl', ...]
```

### Issue 8: Permission Denied

**Error:** `Permission denied: ./scripts/run_evaluation.sh`

**Solution:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Retry
./scripts/run_evaluation.sh
```

---

## 📚 Additional Resources

- **README.md** - Complete framework documentation
- **configs/** - Configuration files for models and datasets
- **scripts/** - All utility scripts with `--help` options
- **experiments/** - Training, evaluation, and comparison scripts

### Get Help

```bash
# Script help
./scripts/run_evaluation.sh --help
python scripts/download_weights.py --help
python scripts/preprocess_data.py --help
python experiments/train.py --help
python experiments/evaluate.py --help
python experiments/compare.py --help

# Model information
python -c "from models import list_available_models; print(list_available_models())"

# Check installation
python scripts/verify_installation.py
```

---

## 🎯 Quick Reference

### Essential Commands

```bash
# 1. Setup (once)
./scripts/setup.sh
conda activate pids_framework

# 2. Install model dependencies (once)
./scripts/install_model_deps.sh --all

# 3. Download/copy weights (once)
python scripts/download_weights.py --copy-existing --all-models

# 4. Verify setup (once)
python scripts/verify_installation.py

# 5. Prepare data (once per dataset)
mkdir -p ../custom_dataset
cp /path/to/logs/*.json ../custom_dataset/

# 6. Run evaluation (main workflow)
./scripts/run_evaluation.sh

# 7. View results
cat results/evaluation_*/comparison_report.json
```

### Quick Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│                  PIDS Framework Workflow                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Setup:                                                  │
│  1. ./scripts/setup.sh                                   │
│  2. conda activate pids_framework                        │
│  3. ./scripts/install_model_deps.sh --all                │
│  4. python scripts/download_weights.py --copy-existing   │
│                                                           │
│  Prepare Data:                                           │
│  5. mkdir -p ../custom_dataset                           │
│  6. cp logs/*.json ../custom_dataset/                    │
│                                                           │
│  Evaluate:                                               │
│  7. ./scripts/run_evaluation.sh                          │
│                                                           │
│  Results:                                                │
│  8. cat results/evaluation_*/comparison_report.json      │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🎉 Success Checklist

Before running evaluation, verify:

- ✅ Conda environment `pids_framework` is activated
- ✅ All model dependencies installed (`./scripts/install_model_deps.sh --all`)
- ✅ Pretrained weights downloaded/copied to `checkpoints/`
- ✅ Custom dataset present in `../custom_dataset/`
- ✅ Verification script passes (`python scripts/verify_installation.py`)

If all checks pass, you're ready to run:
```bash
./scripts/run_evaluation.sh
```

---

**Need more help?** See [README.md](README.md) for complete documentation.
