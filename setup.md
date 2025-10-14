# PIDS Framework - Complete Setup Guide

## 🎯 Overview

This framework evaluates **5 pretrained PIDS models** on your custom SOC data:
- **MAGIC** - Masked Graph Autoencoder
- **Kairos** - Temporal GNN
- **Orthrus** - Multi-Decoder Contrastive Learning
- **ThreaTrace** - Scalable Graph Processing
- **Continuum_FL** - Federated Learning PIDS

**Default workflow**: Setup environment → Setup models & weights → Preprocess data → Evaluate models → Compare performance

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ **Conda** installed (Anaconda or Miniconda)
- ✅ **Python 3.8+** (will be installed via conda)
- ✅ **Graphviz (system binary)** - required for visualization (optional but recommended). 
On macOS:
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

## 🚀 Streamlined Setup (3 Simple Steps)

The framework has been optimized to minimize setup steps. You now need only **3 commands** to get started!

### Step 1: Navigate to Framework Directory

```bash
# Navigate to the framework
cd ../PIDS_Comparative_Framework

# Verify you're in the correct directory
ls
# Expected output: README.md, setup.md, scripts/, models/, etc.
```

---

### Step 2: Run Complete Environment Setup

**This single script does everything:**
- ✅ Creates conda environment
- ✅ Installs all core dependencies
- ✅ Applies PyTorch MKL fix automatically
- ✅ Creates directory structure
- ✅ Verifies installation

```bash
# Make setup script executable (if not already)
chmod +x scripts/setup.sh

# Run the complete setup script
./scripts/setup.sh
```

**What this does:**
1. ✅ Checks for Conda installation
2. ✅ Creates `pids_framework` conda environment (Python 3.10)
3. ✅ Installs PyTorch 1.12.1 with CUDA 11.6
4. ✅ Installs DGL 1.0.0 (Deep Graph Library)
5. ✅ Installs PyTorch Geometric 2.1.0
6. ✅ Applies MKL threading fix automatically (no separate step needed!)
7. ✅ Creates necessary directories
8. ✅ Verifies installation

**Expected output:**
```
Step 1/7: Checking for Conda installation...
✓ Conda is installed: conda 23.x.x

Step 2/7: Creating pids_framework environment...
✓ Environment created successfully

Step 3/7: Initializing Conda for shell...
✓ Conda initialized

Step 4/7: Activating environment...
✓ Environment activated: pids_framework

Step 5/7: Applying PyTorch MKL threading fix...
✓ MKL threading fix applied (will auto-activate with environment)
✓ PyTorch is working!

Step 6/7: Creating directory structure...
✓ Directory structure created

Step 7/7: Verifying installation...
✓ All core dependencies installed successfully

============================================
✓ Setup completed successfully!
============================================
```

**If setup fails**, the script includes automatic fixes for common issues. See [Troubleshooting](#troubleshooting) section if problems persist.

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

### Step 4: Setup Models and Pretrained Weights

**This unified script handles everything:**
- ✅ Installs model-specific dependencies
- ✅ **Downloads pretrained weights from GitHub repositories** (primary method)
- ✅ Falls back to local directories if GitHub download fails

```bash
# Setup ALL models (recommended - downloads from GitHub!)
python scripts/setup_models.py --all

# OR setup specific models only
python scripts/setup_models.py --models magic kairos orthrus

# List available models and their GitHub repos
python scripts/setup_models.py --list
```

**How it works:**
1. **Primary Method**: Downloads weights directly from official GitHub repositories:
   - **MAGIC**: https://github.com/FDUDSDE/MAGIC
   - **Continuum_FL**: https://github.com/kamelferrahi/Continuum_FL
   - **ThreaTrace**: https://github.com/threaTrace-detector/threaTrace
   - **Orthrus**: https://github.com/ubc-provenance/orthrus
   - **Kairos**: https://github.com/ubc-provenance/kairos

2. **Fallback Method**: If GitHub download fails, searches local directories:
   - `../MAGIC/checkpoints/`
   - `../Continuum_FL/checkpoints/`
   - `../orthrus/weights/`
   - `../kairos/DARPA/`
   - `../threaTrace/example_models/`

**Expected output:**
```
================================================================================
  PIDS Framework - Model Setup (GitHub Download)
================================================================================
Setting up models: magic, kairos, orthrus, threatrace, continuum_fl
Strategy: Download from GitHub → Fallback to local if needed

================================================================================
  Setting up MAGIC
================================================================================
Description: Masked Graph Autoencoder for APT Detection
GitHub: https://github.com/FDUDSDE/MAGIC

Installing dependencies for MAGIC...
✓ MAGIC dependencies installed

Downloading weights for MAGIC...
   Repository: https://github.com/FDUDSDE/MAGIC
   📥 streamspot: MAGIC trained on StreamSpot dataset
   Downloading with curl: checkpoint-streamspot.pt
   ✓ Downloaded: checkpoint-streamspot.pt
   📥 cadets: MAGIC trained on DARPA CADETS
   ✓ Downloaded: checkpoint-cadets.pt
   ...
✓ Downloaded 5 checkpoint(s) from GitHub/official sources

...

================================================================================
  Setup Summary
================================================================================
✓ Dependencies installed for 5 model(s)
✓ Downloaded 18 checkpoint(s) from GitHub/official sources
✓ Copied 2 checkpoint(s) from local fallback

Checkpoints saved to: checkpoints/

Next steps:
  1. Verify weights: ls -lh checkpoints/*/
  2. Preprocess data: python scripts/preprocess_data.py
  3. Run evaluation: ./scripts/run_evaluation.sh
```

**Note on Special Cases:**
- **Kairos**: Pre-trained weights available from [Google Drive](https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C) (manual download required)
- **Orthrus**: Pre-trained weights available from [Zenodo](https://zenodo.org/records/14641605) (automatic download via curl/wget)

The script will automatically use `curl` or `wget` (whichever is available) to download weights.

---

### Step 5: Prepare Your Custom Dataset

Place your SOC provenance data in the `custom_dataset` directory.

```bash
# Ensure you're in the framework directory
cd ~/PIDS/PIDS_Comparative_Framework  # Adjust path as needed

# Check if custom_dataset directory exists (it should be a sibling directory)
ls -la ../custom_dataset/

# Expected files:
# endpoint_file.json       - File system events (read, write, create, delete)
# endpoint_network.json    - Network events (connect, send, receive)
# endpoint_process.json    - Process events (create, execute, terminate)
```

**If custom_dataset doesn't exist or is empty:**

```bash
# Navigate to parent directory
cd ..

# Create custom_dataset directory
mkdir -p custom_dataset

# Copy your SOC logs (adjust source path)
cp /path/to/your/soc/logs/*.json custom_dataset/

# Verify files are present
ls -lh custom_dataset/
# Expected: At least one .json file with provenance events

# Return to framework directory
cd PIDS_Comparative_Framework
```

**Required Data Format:**
- JSON files containing provenance events
- Events should include: timestamp, event_type, entity information (process, file, or network)
- See existing `custom_dataset/` files for format examples

---

### Step 6: Preprocess the Dataset

Before evaluation, the data must be preprocessed to convert JSON logs into graph structures.

```bash
# Activate environment (if not already active)
conda activate pids_framework

# Preprocess custom dataset
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc

# Expected output:
# ════════════════════════════════════════════════════════
#   Preprocessing Dataset: custom_soc
# ════════════════════════════════════════════════════════
# Input directory: ../custom_dataset
# Output directory: data/custom_soc
# 
# Step 1/4: Loading JSON files...
# ✓ Loaded endpoint_file.json (1234 events)
# ✓ Loaded endpoint_network.json (567 events)
# ✓ Loaded endpoint_process.json (890 events)
# Total events: 2691
# 
# Step 2/4: Building provenance graph...
# ✓ Created 450 nodes (processes, files, sockets)
# ✓ Created 2241 edges (dependencies)
# 
# Step 3/4: Extracting features...
# ✓ Node features extracted (128-dim)
# ✓ Edge features extracted (64-dim)
# 
# Step 4/4: Saving preprocessed data...
# ✓ Saved graph: data/custom_soc/graph.bin
# ✓ Saved features: data/custom_soc/features.pt
# ✓ Saved metadata: data/custom_soc/metadata.json
# 
# ✓ Preprocessing completed successfully!
```

**Verify preprocessed data:**

```bash
# Check output directory
ls -lh data/custom_soc/

# Expected files:
# graph.bin         - DGL graph structure
# features.pt       - Node/edge features (PyTorch tensor)
# metadata.json     - Dataset statistics and info
```

**Common preprocessing options:**

```bash
# Preprocess with custom window size (temporal graph segmentation)
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --window-size 3600  # 1-hour time windows

# Preprocess with specific event types only
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --event-types process file  # Exclude network events

# Get help on all options
python scripts/preprocess_data.py --help
```

---

### Step 7: Run Evaluation

Now you're ready to evaluate all pretrained models on your preprocessed data!

```bash
# Run evaluation on ALL models (recommended)
./scripts/run_evaluation.sh --data-path data/custom_soc

# Expected output:
# ════════════════════════════════════════════════════════
#     PIDS Framework - Model Evaluation
# ════════════════════════════════════════════════════════
# 
# Configuration:
#   Models:        magic, kairos, orthrus, threatrace, continuum_fl
#   Dataset:       custom_soc
#   Data Path:     data/custom_soc
#   Device:        CPU
#   Output:        results/evaluation_20251014_143000
#
# Evaluating MAGIC...
# ✓ Loaded checkpoint: checkpoints/magic/checkpoint-cadets.pt
# ✓ Evaluation completed (AUROC: 0.9245, F1: 0.8710)
#
# Evaluating Kairos...
# ✓ Loaded checkpoint: checkpoints/kairos/model_cadets.pt
# ✓ Evaluation completed (AUROC: 0.9156, F1: 0.8523)
#
# Evaluating Orthrus...
# ✓ Loaded checkpoint: checkpoints/orthrus/provdetector_main.pkl
# ✓ Evaluation completed (AUROC: 0.9087, F1: 0.8402)
#
# Evaluating ThreaTrace...
# ✓ Loaded checkpoint: checkpoints/threatrace/darpatc/cadets/
# ✓ Evaluation completed (AUROC: 0.8956, F1: 0.8234)
#
# Evaluating Continuum_FL...
# ✓ Loaded checkpoint: checkpoints/continuum_fl/checkpoint-cadets-e3.pt
# ✓ Evaluation completed (AUROC: 0.9123, F1: 0.8601)
#
# ════════════════════════════════════════════════════════
#   Evaluation Summary
# ════════════════════════════════════════════════════════
# ✓ All 5 models evaluated successfully
# ✓ Results saved to: results/evaluation_20251014_143000
# ✓ Comparison report: results/evaluation_20251014_143000/comparison.json
```

**Evaluate specific model only:**

```bash
# Evaluate MAGIC only (fastest)
./scripts/run_evaluation.sh --model magic --data-path data/custom_soc

# Evaluate Kairos only
./scripts/run_evaluation.sh --model kairos --data-path data/custom_soc

# Evaluate multiple specific models
./scripts/run_evaluation.sh --models magic kairos orthrus --data-path data/custom_soc
```

**GPU acceleration (if available):**

```bash
# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Run on GPU
./scripts/run_evaluation.sh --data-path data/custom_soc --device 0

# Specify GPU device (if multiple GPUs)
CUDA_VISIBLE_DEVICES=1 ./scripts/run_evaluation.sh --data-path data/custom_soc
```

---

### Step 8: View Results

Check the evaluation results and comparison report.

```bash
# Navigate to results directory (use actual timestamp)
cd results/evaluation_20251014_143000

# View comparison report (JSON format)
cat comparison.json

# Example output:
# {
#   "dataset": "custom_soc",
#   "timestamp": "2025-10-14T14:30:00Z",
#   "models": {
#     "magic": {
#       "auroc": 0.9245,
#       "auprc": 0.8532,
#       "f1": 0.8710,
#       "precision": 0.8456,
#       "recall": 0.8973,
#       "fpr": 0.0823,
#       "checkpoint": "checkpoints/magic/checkpoint-cadets.pt"
#     },
#     "kairos": {
#       "auroc": 0.9156,
#       "auprc": 0.8421,
#       ...
#     }
#   },
#   "best_model": "magic",
#   "best_auroc": 0.9245
# }

# View individual model results
cat magic_results.json
cat kairos_results.json

# Check detailed logs
tail -n 100 magic_evaluation.log
tail -n 100 kairos_evaluation.log

# View predictions (if saved)
head -n 20 predictions.csv
```

**Understanding the results:**

| Metric | Description | Good Value |
|--------|-------------|------------|
| **AUROC** | Area Under ROC Curve | > 0.90 |
| **AUPRC** | Area Under Precision-Recall | > 0.80 |
| **F1** | Harmonic mean of precision/recall | > 0.85 |
| **Precision** | Correct detections / Total detections | > 0.80 |
| **Recall** | Detected attacks / Total attacks | > 0.85 |
| **FPR** | False Positive Rate | < 0.10 |

**Next steps based on results:**

- **AUROC > 0.90**: ✅ Excellent! Model ready for deployment
- **AUROC 0.80-0.90**: ⚠️ Good, consider fine-tuning for better performance
- **AUROC < 0.80**: ❌ Poor fit, consider retraining on labeled data (see Advanced section)

---

## 📊 Complete Workflow Summary

Here's the complete workflow from start to finish:

```bash
# 1. Setup environment (one-time)
./scripts/setup.sh
conda activate pids_framework

# 2. Setup models and weights (one-time)
python scripts/setup_models.py --all

# 3. Prepare your data (one-time per dataset)
# Copy your SOC logs to ../custom_dataset/
ls -la ../custom_dataset/*.json

# 4. Preprocess data (one-time per dataset)
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc

# 5. Run evaluation (main workflow)
./scripts/run_evaluation.sh --data-path data/custom_soc

# 6. View results
cat results/evaluation_*/comparison.json
```

---

## 🔧 Advanced Options

### Preprocessing Options

```bash
# Custom time window (default: 3600 seconds)
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --window-size 7200  # 2-hour windows

# Filter specific event types
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --event-types process file  # Exclude network events

# See all options
python scripts/preprocess_data.py --help
```

### Evaluation Options

```bash
# Evaluate specific models only
./scripts/run_evaluation.sh --models magic kairos --data-path data/custom_soc

# Use GPU (if available)
./scripts/run_evaluation.sh --data-path data/custom_soc --device 0

# Custom output directory
./scripts/run_evaluation.sh --data-path data/custom_soc --output-dir results/my_eval

# See all options
./scripts/run_evaluation.sh --help
```

---

## 🔄 Advanced: Retraining Models (Optional)

> ⚠️ **Only needed if pretrained models achieve AUROC < 0.80 on your data**

### Prerequisites
- Labeled attack data (ground truth)
- Sufficient training data (thousands of events)
- Time (training can take hours)

### Quick Retraining Guide

```bash
# 1. Prepare labeled data
cat > data/custom_soc/ground_truth.json << EOF
{
  "attacks": [
    {
      "start_time": "2025-10-14T08:00:00Z",
      "end_time": "2025-10-14T09:00:00Z",
      "attack_type": "apt",
      "entities": [1234, 1235, 1236]
    }
  ]
}
EOF

# 2. Preprocess with labels
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --labels-file data/custom_soc/ground_truth.json

# 3. Train model (CPU)
python experiments/train.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 100 \
    --batch-size 32 \
    --device -1

# 4. Evaluate retrained model
python experiments/evaluate.py \
    --model magic \
    --checkpoint checkpoints/magic_custom_soc_best.pt \
    --dataset custom_soc \
    --data-path data/custom_soc
```

**Fine-tuning (recommended over training from scratch):**
```bash
python experiments/train.py \
    --model magic \
    --pretrained checkpoints/magic/checkpoint-cadets.pt \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 50 \
    --learning-rate 0.0001 \
    --device -1
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Conda Not Found
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Restart terminal and retry
```

#### 2. Environment Not Activated
```bash
conda activate pids_framework
# Verify: echo $CONDA_DEFAULT_ENV
```

#### 3. Custom Dataset Not Found
```bash
# Create and populate directory
mkdir -p ../custom_dataset
cp /path/to/your/logs/*.json ../custom_dataset/
ls -la ../custom_dataset/
```

#### 4. Pretrained Weights Missing
```bash
# Re-run model setup
python scripts/setup_models.py --all

# Or copy from local directories
python scripts/setup_models.py --models magic --copy-local
```

#### 5. Out of Memory
```bash
# Use CPU instead of GPU
./scripts/run_evaluation.sh --device -1

# Or reduce batch size in preprocessing
python scripts/preprocess_data.py --batch-size 16
```

#### 6. PyTorch MKL Error
**Error:** `undefined symbol: iJIT_NotifyEvent`

**Solution:** The setup script already applies the fix automatically. If error persists:
```bash
conda activate pids_framework
export MKL_THREADING_LAYER=GNU
# Test: python -c "import torch; print(torch.__version__)"
```

#### 7. Import Errors
```bash
# Ensure environment is activated
conda activate pids_framework

# Reinstall dependencies
./scripts/setup.sh
```

#### 8. Permission Denied
```bash
chmod +x scripts/*.sh
./scripts/run_evaluation.sh
```

### Get Help

```bash
# Script help
python scripts/preprocess_data.py --help
./scripts/run_evaluation.sh --help
python scripts/setup_models.py --help

# Check installation
python -c "import torch; import dgl; import torch_geometric; print('All imports OK')"
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

### Essential Workflow

```bash
# ONE-TIME SETUP (Steps 1-4)
# 1. Setup environment
./scripts/setup.sh
conda activate pids_framework

# 2. Setup models
python scripts/setup_models.py --all

# 3. Prepare data (copy your SOC logs)
ls -la ../custom_dataset/*.json

# 4. Preprocess data
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc

# MAIN WORKFLOW (Steps 5-6)
# 5. Run evaluation
./scripts/run_evaluation.sh --data-path data/custom_soc

# 6. View results
cat results/evaluation_*/comparison.json
```

### Quick Commands

```bash
# Activate environment
conda activate pids_framework

# Evaluate specific model
./scripts/run_evaluation.sh --models magic --data-path data/custom_soc

# Use GPU
./scripts/run_evaluation.sh --data-path data/custom_soc --device 0

# Get help
python scripts/preprocess_data.py --help
./scripts/run_evaluation.sh --help
python scripts/setup_models.py --help
```

### Workflow Diagram

```
┌─────────────────────────────────────────────────┐
│         PIDS Framework - Simple Workflow        │
├─────────────────────────────────────────────────┤
│                                                  │
│  ONE-TIME SETUP:                                │
│  1. ./scripts/setup.sh                          │
│  2. python scripts/setup_models.py --all        │
│  3. Copy data → ../custom_dataset/*.json        │
│  4. python scripts/preprocess_data.py           │
│                                                  │
│  MAIN WORKFLOW:                                 │
│  5. ./scripts/run_evaluation.sh                 │
│  6. cat results/evaluation_*/comparison.json    │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 🎉 Pre-Evaluation Checklist

Before running evaluation, verify:

- ✅ Environment activated: `conda activate pids_framework`
- ✅ Models setup: `ls -la checkpoints/*/`
- ✅ Data present: `ls -la ../custom_dataset/*.json`
- ✅ Data preprocessed: `ls -la data/custom_soc/graph.bin`

**If all checks pass:**
```bash
./scripts/run_evaluation.sh --data-path data/custom_soc
```

---

**Need more help?** See [README.md](README.md) for complete documentation.
