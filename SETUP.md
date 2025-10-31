# PIDS Comparative Framework - Complete Setup Guide
---

## 🎯 Overview

This guide provides **complete setup instructions** for the PIDS Comparative Framework. For architectural details and framework capabilities, see **[README.md](README.md)**.

**What's in this guide:**
- Step-by-step installation (automated + manual)
- Model setup and pretrained weights download
- Custom SOC data preprocessing
- Running evaluations and analyzing results
- Comprehensive troubleshooting
- Advanced configuration options

### Supported Models

- **MAGIC** - Masked Graph Autoencoder (DGL-based, 5 checkpoints)
- **Kairos** - Temporal GNN with sketching (8 checkpoints)
- **Orthrus** - Multi-Decoder Contrastive Learning (5 checkpoints)
- **ThreaTrace** - Locality-Sensitive Hashing (140+ models)
- **Continuum_FL** - Federated Learning PIDS (5 checkpoints)

### Typical Workflow

```text
Setup Environment → Download Weights → Preprocess Your Data → Evaluate Models → Analyze Results
     (10 mins)          (5-15 mins)        (5-60 mins)          (10-30 mins)      (ongoing)
```

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Model Setup & Checkpoints](#model-setup--checkpoints)
5. [Data Preprocessing](#data-preprocessing)
6. [Running Evaluation](#running-evaluation)
7. [Analyzing Results](#analyzing-results)
   - [Visualizing Attack Graphs](#visualizing-attack-graphs)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Configuration](#configuration)
11. [Command Reference](#command-reference)

---

## 📋 Prerequisites

### Required

- ✅ **Conda** (Anaconda or Miniconda)
  - Download: https://docs.conda.io/en/latest/miniconda.html
  - Verify: `conda --version`

- ✅ **Python 3.8-3.10** (installed via conda environment)

- ✅ **16GB+ RAM** (8GB minimum, 32GB recommended for large datasets)

- ✅ **50GB+ free disk space**
  - Framework: ~5GB
  - Pretrained weights: ~10GB
  - Datasets: 10-30GB (depending on your data)

### Optional (Enhanced Functionality)

- ⚙️ **GPU with CUDA 11.6+** (optional - framework runs on CPU by default)
  - Speeds up evaluation 5-10x
  - Verify: `nvidia-smi`

- 🔧 **Download Tools** (usually pre-installed):
  - `curl` or `wget` - For downloading weights
  - `git` - For repository operations
  - `svn` (Subversion) - For ThreaTrace weights download
    - macOS: `brew install subversion`
    - Ubuntu: `sudo apt-get install subversion`

- 📦 **Google Drive Downloads** (for Kairos only):
  - `gdown` package (auto-installed, but may need manual intervention)

### Check Your System

```bash
# Check Conda
conda --version  # Should show: conda 23.x.x or higher

# Check available disk space
df -h .  # Should show 50GB+ available

# Check RAM
free -h  # Linux
vm_stat | grep free  # macOS

# Optional: Check GPU
nvidia-smi  # Should show GPU info if available
```

---

## Quick Start

For most users, these commands will get you up and running:

```bash
# Step 1: Navigate to framework directory
cd PIDS_Comparative_Framework

# Step 2: Run complete setup (creates environment, installs dependencies)
./scripts/setup.sh

# Step 3: Activate environment
conda activate pids_framework

# Step 4: Setup models and download pretrained weights
python scripts/download_checkpoints.py --all

# Step 5: Run evaluation on your data
./scripts/run_evaluation.sh --data-path ../custom_dataset

# Step 6: Visualize attack graphs
# Local visualization (opens HTML locally)
./scripts/visualize_attacks.sh

# OR — start an HTTP server for remote access (useful with VS Code Remote / SSH)
./scripts/visualize_attacks.sh --serve
```

**Total time:** 15-30 minutes (depending on download speeds)

---

## Detailed Installation

### Step 1: Clone or Navigate to Framework

```bash
# If you already have the framework:
cd /path/to/PIDS_Comparative_Framework

# If cloning from repository:
git clone https://github.com/yourusername/PIDS_Comparative_Framework.git
cd PIDS_Comparative_Framework

# Verify you're in the correct directory
ls -la
# Expected: README.md, Setup.md, scripts/, models/, experiments/, etc.
```

### Step 2: Run Automated Setup Script

The `setup.sh` script performs **complete environment setup** in 7 automated steps:

```bash
# Make script executable (if needed)
chmod +x scripts/setup.sh

# Run setup
./scripts/setup.sh
```

#### What `setup.sh` Does:

```
Step 1/7: Checking for Conda installation
  ✓ Verifies conda is installed
  ✓ Shows conda version

Step 2/7: Creating conda environment
  ✓ Creates 'pids_framework' environment from environment.yml
  ✓ Installs Python 3.10
  ✓ Installs PyTorch 1.12.1 with CUDA 11.6 support
  ✓ Installs DGL 1.0.0 (Deep Graph Library)
  ✓ Installs core dependencies (numpy, pandas, sklearn, etc.)

Step 3/7: Initializing Conda for shell
  ✓ Configures conda for bash/zsh shells
  ✓ Enables conda activate command

Step 4/7: Activating environment
  ✓ Activates pids_framework environment
  ✓ Verifies activation

Step 5/7: Applying PyTorch MKL threading fix (AUTOMATIC)
  ✓ Sets MKL_THREADING_LAYER=GNU
  ✓ Creates activation/deactivation scripts
  ✓ Tests PyTorch import
  ✓ Falls back to MKL reinstall if needed

Step 6/7: Creating directory structure & Installing PyTorch Geometric
  ✓ Creates data/, checkpoints/, results/, logs/, configs/ directories
  ✓ Installs torch-scatter 2.1.0
  ✓ Installs torch-sparse 0.6.16
  ✓ Installs torch-cluster 1.6.0
  ✓ Installs torch-geometric 2.1.0
  ✓ Auto-detects CUDA version for appropriate wheels

Step 7/7: Verifying installation
  ✓ Checks Python version
  ✓ Verifies PyTorch import and CUDA availability
  ✓ Checks DGL installation
  ✓ Checks PyTorch Geometric components
  ✓ Runs comprehensive dependency check
```

**Expected output:**
```
============================================
✓ Setup completed successfully!
============================================

Next steps:

1. Activate the environment (if not already active):
   conda activate pids_framework

2. Setup models and download pretrained weights:
   python scripts/download_checkpoints.py --all

3. Preprocess your custom SOC data:
   python scripts/preprocess_data.py --input-dir ../custom_dataset/

4. Run evaluation on all models:
   ./scripts/run_evaluation.sh
```

#### Important Notes:

- ⚠️ **MKL Fix is Automatic** - You don't need to manually set environment variables
- ⚠️ **PyTorch Geometric Included** - No need for separate installation
- ⚠️ **Environment Activation** - Scripts auto-activate in subshells, but you should manually activate for interactive use

### Step 3: Activate Conda Environment

```bash
# Activate the environment
conda activate pids_framework

# Verify activation
echo $CONDA_DEFAULT_ENV
# Should output: pids_framework

# Check Python version
python --version
# Should output: Python 3.10.x

# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"
# Should output: PyTorch 1.12.1+cu116 (or +cpu)
```

**⚠️ Important:** You must activate this environment every time you use the framework!

### Step 4: Setup Models and Download Pretrained Weights

The `download_checkpoints.py` script handles:
1. Installing model-specific dependencies
2. Downloading pretrained weights from official GitHub repositories
3. Falling back to local directories if downloads fail

```bash
# Setup ALL models (recommended)
python scripts/download_checkpoints.py --all

# OR setup specific models only
python scripts/download_checkpoints.py --models magic kairos orthrus
```

#### Download Strategy:

**Primary Method: GitHub Download**
- Downloads weights directly from official repositories
- Uses curl/wget (automatically selected)
- Handles special cases (Google Drive, git sparse-checkout)

**Fallback Method: Local Copy**
- Searches for weights in local directories:
  - `../MAGIC/checkpoints/`
  - `../Continuum_FL/checkpoints/`
  - `../orthrus/weights/`
  - `../kairos/DARPA/`
  - `../threaTrace/example_models/`

#### Expected Output:

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
   📥 theia: MAGIC trained on DARPA THEIA
      ✓ Downloaded: checkpoint-theia.pt
   📥 trace: MAGIC trained on DARPA TRACE
      ✓ Downloaded: checkpoint-trace.pt
   📥 wget: MAGIC trained on Wget dataset
      ✓ Downloaded: checkpoint-wget.pt
✓ Downloaded 5 checkpoint(s) from GitHub/official sources

================================================================================
  Setting up Kairos
================================================================================
Description: Practical Intrusion Detection with Whole-system Provenance
GitHub: https://github.com/ubc-provenance/kairos

Installing dependencies for Kairos...
✓ Kairos dependencies installed

Downloading weights for Kairos...
   Repository: https://github.com/ubc-provenance/kairos
   ⚠️  google_drive_folder: Kairos pretrained models from Google Drive
      Manual download required from: https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C

   Checking local fallback: kairos/DARPA
      ⏭️  Skipping (no checkpoints found)

[... similar output for Orthrus, ThreaTrace, Continuum_FL ...]

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

#### Verify Downloaded Weights:

```bash
# Check checkpoint directory structure
ls -lh checkpoints/*/

# Expected structure:
checkpoints/
├── magic/
│   ├── checkpoint-streamspot.pt
│   ├── checkpoint-cadets.pt
│   ├── checkpoint-theia.pt
│   ├── checkpoint-trace.pt
│   └── checkpoint-wget.pt
├── kairos/
│   └── [requires manual download]
├── orthrus/
│   ├── CADETS_E3.pkl
│   ├── CLEARSCOPE_E3.pkl
│   ├── CLEARSCOPE_E5.pkl
│   ├── THEIA_E3.pkl
│   └── THEIA_E5.pkl
├── threatrace/
│   ├── darpatc/
│   ├── streamspot/
│   └── unicornsc/
└── continuum_fl/
    ├── checkpoint-streamspot.pt
    ├── checkpoint-cadets-e3.pt
    ├── checkpoint-theia-e3.pt
    ├── checkpoint-trace-e3.pt
    └── checkpoint-clearscope-e3.pt
```

### Step 5: Verify Installation

```bash
# Run comprehensive verification
python scripts/verify_installation.py
```

**Expected output:**
```
================================================================================
  PIDS Comparative Framework - Verification
================================================================================

================================================================================
  Python Environment
================================================================================
Python version: 3.10.x
✅ Python version is compatible (3.8+)

================================================================================
  Core Dependencies
================================================================================
✅ torch          - version 1.12.1
✅ numpy          - version 1.23.5
✅ scipy          - version 1.10.1
✅ pandas         - version 1.5.3
✅ sklearn        - version 1.2.2
✅ yaml           - version 6.0
✅ matplotlib     - version 3.7.1

================================================================================
  Deep Learning Frameworks
================================================================================
✅ PyTorch        - version 1.12.1
✅ CUDA           - version 11.6 (or ⚠️ NOT AVAILABLE - CPU only)
✅ DGL            - version 1.0.0
✅ PyTorch Geom.  - version 2.1.0
  ✅ torch-scatter - version 2.1.0
  ✅ torch-sparse  - version 0.6.16
  ✅ torch-cluster - version 1.6.0

================================================================================
  Model Integrations
================================================================================
Found 9 registered models:
  ✅ magic
  ✅ magic_streamspot
  ✅ magic_darpa
  ✅ kairos
  ✅ orthrus
  ✅ threatrace
  ✅ continuum_fl
  ✅ continuum_fl_streamspot
  ✅ continuum_fl_darpa
✅ All expected models are registered

================================================================================
  Directory Structure
================================================================================
✅ data                     - EXISTS
✅ models                   - EXISTS
✅ utils                    - EXISTS
✅ experiments              - EXISTS
✅ scripts                  - EXISTS
✅ configs                  - EXISTS
✅ checkpoints              - EXISTS
✅ results                  - EXISTS
✅ logs                     - EXISTS

================================================================================
  Configuration Files
================================================================================
✅ configs/models/magic.yaml
✅ configs/models/kairos.yaml
✅ configs/models/orthrus.yaml
✅ configs/models/threatrace.yaml
✅ configs/models/continuum_fl.yaml
✅ configs/datasets/custom_soc.yaml
✅ configs/datasets/cadets_e3.yaml
✅ configs/datasets/streamspot.yaml
✅ configs/experiments/compare_all.yaml
✅ configs/experiments/train_single.yaml

================================================================================
  Verification Summary
================================================================================

Total checks: 9
Passed: 9
Failed: 0

================================================================================
🎉 ALL CHECKS PASSED! Framework is ready to use.
================================================================================
```

---

## 🔧 Model-Specific Setup

Each model has unique requirements and weight sources:

### MAGIC

**Description:** Masked Graph Autoencoder for APT Detection  
**GitHub:** https://github.com/FDUDSDE/MAGIC  
**Dependencies:** DGL 1.0.0, torch-geometric  
**Weights:** ✅ Auto-downloaded from GitHub  
**Special Requirements:** None

**Available Pretrained Weights:**
- `checkpoint-streamspot.pt` - StreamSpot dataset
- `checkpoint-cadets.pt` - DARPA TC CADETS
- `checkpoint-theia.pt` - DARPA TC THEIA
- `checkpoint-trace.pt` - DARPA TC TRACE
- `checkpoint-wget.pt` - Wget attack dataset

### Kairos

**Description:** Practical Intrusion Detection with Whole-system Provenance  
**GitHub:** https://github.com/ubc-provenance/kairos  
**Dependencies:** psycopg2, sqlalchemy (for database access)  
**Weights:** ⚠️ **Manual download required**  
**Special Requirements:** Google Drive access

**Manual Download Instructions:**
```bash
# Option 1: Using gdown (if installed)
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C -O checkpoints/kairos/

# Option 2: Manual browser download
# 1. Visit: https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C
# 2. Download all files
# 3. Save to: checkpoints/kairos/
```

### Orthrus

**Description:** High Quality Attribution in Provenance-based IDS  
**GitHub:** https://github.com/ubc-provenance/orthrus  
**Dependencies:** torch-geometric, contrastive learning libraries  
**Weights:** ✅ Auto-downloaded from GitHub (or Zenodo)  
**Special Requirements:** None

**Available Pretrained Weights:**
- `CADETS_E3.pkl` - DARPA TC CADETS Engagement 3
- `CLEARSCOPE_E3.pkl` - DARPA TC CLEARSCOPE E3
- `CLEARSCOPE_E5.pkl` - DARPA TC CLEARSCOPE E5
- `THEIA_E3.pkl` - DARPA TC THEIA E3
- `THEIA_E5.pkl` - DARPA TC THEIA E5

**Alternative Source:** https://zenodo.org/records/14641605

### ThreaTrace

**Description:** Scalable Sketch-based Threat Detection  
**GitHub:** https://github.com/Provenance-IDS/threaTrace  
**Dependencies:** scikit-learn, graph processing libraries  
**Weights:** ✅ Auto-downloaded via git sparse-checkout (large)  
**Special Requirements:** `git` or `svn` (Subversion)

**Download Details:**
- **Size:** ~500MB (140+ files in 3 subdirectories)
- **Method:** Git sparse-checkout or SVN export
- **Structure:** `example_models/` with subdirectories:
  - `darpatc/` - DARPA TC models
  - `streamspot/` - StreamSpot models
  - `unicornsc/` - Unicorn SC models

**Manual Download (if automated fails):**
```bash
# Method 1: Full clone and copy
git clone https://github.com/Provenance-IDS/threaTrace.git /tmp/threatrace
cp -r /tmp/threatrace/example_models checkpoints/threatrace/

# Method 2: SVN export (faster)
svn export https://github.com/Provenance-IDS/threaTrace/trunk/example_models checkpoints/threatrace
```

### Continuum_FL

**Description:** Federated Learning for Provenance-based IDS  
**GitHub:** https://github.com/kamelferrahi/Continuum_FL  
**Dependencies:** mpi4py, federated learning libraries  
**Weights:** ✅ Auto-downloaded from GitHub  
**Special Requirements:** MPI (for federated learning features)

**Available Pretrained Weights:**
- `checkpoint-streamspot.pt` - StreamSpot dataset
- `checkpoint-cadets-e3.pt` - DARPA TC CADETS E3
- `checkpoint-theia-e3.pt` - DARPA TC THEIA E3
- `checkpoint-trace-e3.pt` - DARPA TC TRACE E3
- `checkpoint-clearscope-e3.pt` - DARPA TC CLEARSCOPE E3

---

## Preparing Custom Data

### Data Requirements

Your SOC data must be in **JSON format** with provenance events. The framework supports:

1. **Elastic/ELK Stack format**
2. **Custom JSON format** (with field mapping)
3. **NDJSON** (newline-delimited JSON)

### Required Event Types

At least one of the following:
- **Process events** - Process creation, execution, termination
- **File events** - File read, write, create, delete
- **Network events** - Network connections, send, receive

### Example Data Format

**Elastic/ELK Format:**
```json
{
  "@timestamp": "2024-10-14T10:30:00.000Z",
  "event": {
    "kind": "event",
    "category": ["process"],
    "type": ["start"]
  },
  "process": {
    "pid": 1234,
    "name": "bash",
    "executable": "/bin/bash",
    "command_line": "bash -c 'ls -la'",
    "parent": {
      "pid": 1000,
      "name": "systemd"
    }
  },
  "user": {
    "name": "root",
    "id": "0"
  },
  "host": {
    "name": "web-server-01"
  }
}
```

### Place Your Data

```bash
# Create custom_dataset directory (if not exists)
cd /path/to/PIDS_Files
mkdir -p custom_dataset

# Copy your JSON files
cp /path/to/your/logs/*.json custom_dataset/

# Verify files
ls -lh custom_dataset/
# Expected: At least one .json file with provenance events
```

### Preprocess Your Data

```bash
# Navigate to framework directory
cd PIDS_Comparative_Framework

# Activate environment (if not already active)
conda activate pids_framework

# Run preprocessing
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc
```

**Expected Output:**
```
════════════════════════════════════════════════════════
  Preprocessing Dataset: custom_soc
════════════════════════════════════════════════════════
Input directory: ../custom_dataset
Output directory: data/custom_soc

Step 1/4: Loading JSON files...
✓ Loaded endpoint_file.json (1234 events)
✓ Loaded endpoint_network.json (567 events)
✓ Loaded endpoint_process.json (890 events)
Total events: 2691

Step 2/4: Building provenance graph...
✓ Created 450 nodes (processes, files, sockets)
✓ Created 2241 edges (dependencies)

Step 3/4: Extracting features...
✓ Node features extracted (128-dim)
✓ Edge features extracted (64-dim)

Step 4/4: Saving preprocessed data...
✓ Saved graph: data/custom_soc/custom_soc_graph.pkl
✓ Saved features: data/custom_soc/custom_soc_features.pt
✓ Saved metadata: data/custom_soc/custom_soc_metadata.json

✓ Preprocessing completed successfully!
```

**Verify Preprocessed Data:**
```bash
# Check output directory
ls -lh data/custom_soc/

# Expected files:
# custom_soc_graph.pkl       - DGL/PyG graph structure (~50-500MB)
# custom_soc_features.pt     - Node/edge features (PyTorch tensor)
# custom_soc_metadata.json   - Dataset statistics and info
```

### Advanced Preprocessing Options

```bash
# Use Elastic/ELK schema explicitly
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --schema elastic

# Custom time window (for temporal graphs)
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --time-window 3600  # 1-hour windows

# Filter specific event types
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --event-types process file  # Exclude network events

# Large dataset optimization
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --chunk-size 50000 \  # Process 50K events at a time
    --verbose              # Show detailed progress
```

---

## Running Evaluation

### Quick Evaluation (All Models)

```bash
# Activate environment
conda activate pids_framework

# Run evaluation on all models
./scripts/run_evaluation.sh
```

This automatically:
1. Checks conda environment activation
2. Sets up model weights (if needed)
3. Detects if data is already preprocessed
4. Runs preprocessing if needed
5. Evaluates all 5 models
6. Generates comparison report

### Evaluation with Preprocessed Data

If you've already preprocessed your data:

```bash
./scripts/run_evaluation.sh \
    --data-path data/custom_soc \
    --skip-preprocess
```

### Evaluate Specific Model

```bash
# Evaluate only MAGIC
./scripts/run_evaluation.sh --model magic

# Evaluate Kairos
./scripts/run_evaluation.sh --model kairos

# Evaluate multiple specific models
./scripts/run_evaluation.sh --model magic
./scripts/run_evaluation.sh --model orthrus
```

### Custom Data Path

```bash
# If your data is in a different location
./scripts/run_evaluation.sh \
    --data-path /path/to/your/preprocessed/data \
    --dataset my_custom_dataset
```

### Full Workflow Example

```bash
# Complete evaluation workflow
cd PIDS_Comparative_Framework
conda activate pids_framework

# Step 1: Preprocess your data (if not done)
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc

# Step 2: Run evaluation
./scripts/run_evaluation.sh \
    --data-path data/custom_soc \
    --skip-preprocess

# Step 3: View results
ls -lh results/evaluation_*/
cat results/evaluation_*/comparison_report.json
```

### Expected Output

```
════════════════════════════════════════════════════════════════
    PIDS Comparative Framework - Evaluation Workflow
════════════════════════════════════════════════════════════════

Configuration:
  Model(s):      all
  Dataset:       custom_soc
  Data Path:     data/custom_soc
  Output Dir:    results/evaluation_20251014_143000

────────────────────────────────────────────────────────────────
Step 1/4: Setting up Models and Pretrained Weights
────────────────────────────────────────────────────────────────
✓ Model weights setup successfully

────────────────────────────────────────────────────────────────
Step 2/4: Checking Preprocessed Data
────────────────────────────────────────────────────────────────
✓ Preprocessed data found
  Using: data/custom_soc

────────────────────────────────────────────────────────────────
Step 3/5: Running Model Evaluation
────────────────────────────────────────────────────────────────
Evaluating magic...
✓ magic evaluation completed (Separation Ratio: 1.95, Critical Anomalies: 921)

Evaluating kairos...
✓ kairos evaluation completed (Separation Ratio: 1.82, Critical Anomalies: 856)

Evaluating orthrus...
✓ orthrus evaluation completed (Separation Ratio: 1.74, Critical Anomalies: 812)

Evaluating threatrace...
✓ threatrace evaluation completed (Separation Ratio: 1.68, Critical Anomalies: 789)

Evaluating continuum_fl...
✓ continuum_fl evaluation completed (Separation Ratio: 1.79, Critical Anomalies: 843)

────────────────────────────────────────────────────────────────
Step 4/5: Analyzing Anomalies
────────────────────────────────────────────────────────────────
Analyzing magic anomalies...
✓ magic anomaly analysis completed (1000 top anomalies)

Analyzing kairos anomalies...
✓ kairos anomaly analysis completed (1000 top anomalies)

Analyzing orthrus anomalies...
✓ orthrus anomaly analysis completed (1000 top anomalies)

Analyzing threatrace anomalies...
✓ threatrace anomaly analysis completed (1000 top anomalies)

Analyzing continuum_fl anomalies...
✓ continuum_fl anomaly analysis completed (1000 top anomalies)

Generating ensemble consensus...
✓ Ensemble consensus report generated (127 high-confidence anomalies)

────────────────────────────────────────────────────────────────
Step 5/5: Generating Comparison Report
────────────────────────────────────────────────────────────────
✓ Comparison report generated

════════════════════════════════════════════════════════════════
✓ EVALUATION COMPLETED SUCCESSFULLY!
════════════════════════════════════════════════════════════════

Results saved to: results/evaluation_20251014_143000

Next steps:
  1. Review results:        ls results/evaluation_20251014_143000
  2. View comparison:       cat results/evaluation_20251014_143000/comparison_report.json
  3. View anomalies:        cat results/evaluation_20251014_143000/magic_anomalies.json
  4. View consensus:        cat results/evaluation_20251014_143000/ensemble_consensus.json
  5. Check model logs:      tail results/evaluation_20251014_143000/*.log
```

**Note:** The framework uses **unsupervised metrics** (Score Separation Ratio, anomaly counts) as primary metrics. Supervised metrics (AUROC, F1) are shown only if ground truth labels exist.

---

## Analyzing Results

### Visualizing Attack Graphs

After running model evaluations, visualize and compare attack graphs across multiple models using the `visualize_attack_graphs.py` utility.

#### Features

- **Interactive Multi-Model Comparison** - Side-by-side comparison of attack graphs
- **Attack Path Reconstruction** - Backward/forward provenance traversal from anomalies  
- **Entity Clustering** - Group related entities and events by type, temporal proximity, or attack path
- **Multiple Export Formats** - HTML (interactive), JSON (summary), GraphML (for Gephi/Cytoscape)
- **Auto-Open Browser** - Automatically opens visualization in your default browser
- **Attack Path Ranking** - Ranks attack paths by severity and anomaly scores

#### Quick Usage

**Option 1: Using the Convenience Script (Recommended)**

```bash
# Activate environment
conda activate pids_framework

# Visualize with default settings (99th percentile, top 100 anomalies)
./scripts/visualize_attacks.sh

# Customize thresholds
./scripts/visualize_attacks.sh \
  --threshold 99.9 \
  --top-k 50 \
  --top-paths 20

# Custom output directory
./scripts/visualize_attacks.sh \
  --output-dir results/my_attack_viz \
  --cluster-by temporal

# Show help
./scripts/visualize_attacks.sh --help
```

**Remote Server / VS Code Remote Usage**

When running the framework on a remote server (e.g., via VS Code Remote SSH), use the `--serve` flag to start an HTTP server with port forwarding:

```bash
# Start visualization with HTTP server for remote access
./scripts/visualize_attacks.sh --serve

# The script will:
# 1. Generate the visualization
# 2. Start HTTP server on port 8000
# 3. Keep running until you press Ctrl+C

# To access the visualization:
# 1. In VS Code, open the PORTS panel (bottom, next to TERMINAL)
# 2. VS Code should auto-detect port 8000
# 3. Click the globe icon 🌐 next to port 8000
# 4. The visualization opens directly in your local browser!

# If port not auto-detected:
#   - Click "Forward a Port" in PORTS panel
#   - Enter: 8000
#   - Then click the globe icon 🌐

# Stop the server when done viewing:
#   - Press Ctrl+C in the terminal
```

**Option 2: Using Python Script Directly (Advanced)**

```bash
# Activate environment
conda activate pids_framework

# Visualize all models
python utils/visualize_attack_graphs.py \
  --artifacts-dir artifacts \
  --models magic kairos orthrus threatrace continuum_fl \
  --output-dir results/attack_viz

# Customize thresholds and paths  
python utils/visualize_attack_graphs.py \
  --threshold-percentile 99.0 \
  --top-k 50 \
  --top-paths 20 \
  --cluster-by entity

# Visualize specific models only
python utils/visualize_attack_graphs.py \
  --models magic kairos \
  --output-dir results/magic_vs_kairos
```

#### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--artifacts-dir` | `artifacts` | Directory containing model artifacts with inference results |
| `--models` | All 5 models | Space-separated list of models to visualize |
| `--output-dir` | `results/attack_graph_visualization` | Output directory for visualizations |
| `--threshold-percentile` | `95.0` | Percentile threshold for anomalies (e.g., 90, 95, 99, 99.9) |
| `--top-k` | `100` | Number of top anomalies to include in graph |
| `--top-paths` | `10` | Number of attack paths to extract and visualize |
| `--cluster-by` | `entity` | Clustering strategy: `entity`, `temporal`, or `path` |
| `--serve` | Off | Start HTTP server for remote access (recommended for VS Code Remote) |
| `--no-browser` | Off | Skip auto-opening browser (useful for headless servers) |

#### Generated Output

```
results/attack_graph_visualization/
├── attack_graph_viewer.html              # Interactive visualization (auto-opens)
├── attack_summary.json                   # Attack statistics and paths
├── magic_attack_graph.graphml            # GraphML for external tools
├── kairos_attack_graph.graphml
├── orthrus_attack_graph.graphml
├── threatrace_attack_graph.graphml
└── continuum_fl_attack_graph.graphml
```

#### Interactive Viewer Features

The HTML viewer provides:
- **Model Tabs** - Switch between different model visualizations
- **Attack Path Highlighting** - View top attack paths with severity scores
- **Entity Tooltips** - Hover over nodes to see detailed entity information (type, attributes, timestamps)
- **Provenance Edges** - Colored edges showing different dependency relationships
- **Attack Statistics** - Summary metrics for each model (anomaly counts, separation ratios)
- **Zoom/Pan Controls** - Interactive graph navigation with mouse/keyboard
- **Export Options** - Download subgraphs or specific attack paths

#### Example Workflow

```bash
# Step 1: Run evaluation on your data
./scripts/run_evaluation.sh --data-path data/custom_soc

# Step 2: Visualize attack graphs with high threshold
python utils/visualize_attack_graphs.py \
  --artifacts-dir artifacts \
  --output-dir results/attack_viz \
  --threshold-percentile 99.0 \
  --top-k 100 \
  --top-paths 20

# Output:
# ✓ Loaded graph and scores for magic
# ✓ Loaded graph and scores for kairos  
# ✓ Loaded graph and scores for orthrus
# ✓ Reconstructed attack graph for magic (95 nodes, 234 edges)
# ✓ Extracted 20 attack paths
# ✓ Interactive visualization: results/attack_viz/attack_graph_viewer.html
# ✓ Opened visualization in default browser

# Step 3: View interactive HTML (auto-opened in browser)
# Navigate to: results/attack_viz/attack_graph_viewer.html

# Step 4: Import GraphML into Gephi for advanced analysis
# Open Gephi → Import → results/attack_viz/magic_attack_graph.graphml
```

#### Integration with External Tools

Export GraphML files to:
- **Gephi** - Advanced graph visualization, community detection, layout algorithms
- **Cytoscape** - Network analysis, biological network tools adapted for provenance
- **Neo4j** - Import into graph database for complex queries and pattern matching
- **Python NetworkX** - Custom analysis scripts and graph algorithms

#### Attack Summary JSON Structure

```json
{
  "magic": {
    "attack_graph": {
      "num_nodes": 95,
      "num_edges": 234,
      "node_types": {"process": 45, "file": 35, "socket": 15},
      "edge_types": {"read": 80, "write": 60, "execute": 40, "connect": 54}
    },
    "attack_paths": [
      {
        "path_id": 1,
        "severity": 0.98,
        "length": 5,
        "nodes": ["proc_123", "file_456", "proc_789", "socket_012", "file_345"],
        "description": "Process execution chain leading to network exfiltration"
      }
    ],
    "statistics": {
      "anomaly_count": 95,
      "score_separation_ratio": 2.45,
      "temporal_span": "2024-10-14 09:30:00 to 2024-10-14 11:45:00"
    }
  }
}
```

#### Programmatic Usage

```python
from utils.visualize_attack_graphs import AttackGraphReconstructor, MultiModelVisualizer
import pickle
from pathlib import Path

# Load preprocessed graph
with open('data/custom_soc/graph.pkl', 'rb') as f:
    graph_data = pickle.load(f)

# Load model inference results
with open('artifacts/magic/model_inference/output.pkl', 'rb') as f:
    inference_result = pickle.load(f)

# Reconstruct attack graph
reconstructor = AttackGraphReconstructor(
    graph_data=graph_data,
    scores=inference_result['scores'],
    threshold_percentile=99.0
)

attack_graph = reconstructor.reconstruct_attack_graph(
    top_k=100,
    cluster_by='entity'
)

# Extract attack paths
attack_paths = reconstructor.extract_attack_paths(
    attack_graph=attack_graph,
    top_k=10
)

# Create visualization
visualizer = MultiModelVisualizer(output_dir='results/my_viz')
html_path = visualizer.create_interactive_comparison(
    model_graphs={'magic': attack_graph},
    model_paths={'magic': attack_paths},
    model_scores={'magic': inference_result['scores']}
)

print(f"Visualization saved to: {html_path}")
```

#### Clustering Strategies

**Entity Clustering** (default):
- Groups nodes by entity type (process, file, socket)
- Best for understanding attack patterns by resource type
- Useful for identifying lateral movement, data exfiltration

**Temporal Clustering**:
- Groups events by time proximity
- Best for understanding attack timeline and progression
- Useful for identifying attack phases (reconnaissance, exploitation, persistence)

**Path Clustering**:
- Groups nodes by attack path membership
- Best for understanding distinct attack chains
- Useful for identifying parallel attack vectors

#### Troubleshooting

**Issue: No visualization generated**
```bash
# Check if plotly is installed
pip install plotly kaleido

# Verify artifacts exist
ls -lh artifacts/*/model_inference/
```

**Issue: Empty attack graph**
```bash
# Lower the threshold to include more events
python utils/visualize_attack_graphs.py --threshold-percentile 90.0

# Increase top-k to include more anomalies
python utils/visualize_attack_graphs.py --top-k 200
```

**Issue: Browser doesn't auto-open**
```bash
# Manually open the HTML file
firefox results/attack_graph_visualization/attack_graph_viewer.html
# or
google-chrome results/attack_graph_visualization/attack_graph_viewer.html
# or
xdg-open results/attack_graph_visualization/attack_graph_viewer.html
```

---

## Advanced Features

### Understanding Unsupervised Evaluation

The framework uses **unsupervised anomaly detection metrics** by default, suitable for unlabeled SOC data:

#### Primary Metrics

1. **Score Separation Ratio** (std/mean)
   - Measures how well the model separates anomalous from normal behavior
   - Higher ratio = better separation between anomaly scores
   - Used for model ranking
   - Threshold-independent metric

2. **Anomaly Score Distribution**
   - Mean, median, standard deviation
   - Percentiles (75th, 90th, 95th, 99th)
   - Critical anomalies: events > 99th percentile
   - High-risk anomalies: events 95-99th percentile

#### Automatic Anomaly Analysis

For each model, the framework automatically:
- Extracts top 1000 highest-scoring anomalies
- Analyzes temporal patterns (hourly, daily distributions)
- Identifies most suspicious entities (processes, files, hosts)
- Characterizes attack patterns (edge types, node features)
- Generates ensemble consensus (anomalies flagged by multiple models)

#### Using Results

**View model rankings:**
```bash
cat results/evaluation_*/comparison_report.json | grep -A 5 "model_rankings"
```

**Investigate specific anomalies:**
```bash
# View top anomalies for MAGIC
cat results/evaluation_*/magic_anomalies.json | jq '.top_anomalies[:10]'

# View ensemble consensus (high-confidence anomalies)
cat results/evaluation_*/ensemble_consensus.json
```

**Supervised Metrics (Optional):**
If your dataset has ground truth labels, the framework will also calculate:
- AUROC, AUPRC, F1-Score, Precision, Recall
- These are shown in results but not used for model ranking

---

### Training Models on Custom Data (Reference Only)

**⚠️ Important Note:** The framework is primarily designed for **evaluation with pretrained models**. Training functionality (`experiments/train.py`) is provided for reference but is not actively maintained and may require additional setup.

**For most users:** Use the pretrained models provided by the original model authors.

If you need to retrain models:

```bash
# Train MAGIC on custom data (reference implementation)
python experiments/train.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output-dir results/training/magic

# Train with GPU
python experiments/train.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --device 0  # GPU device ID

# Resume training from checkpoint
python experiments/train.py \
    --model magic \
    --checkpoint checkpoints/magic/checkpoint-streamspot.pt \
    --resume
```

**Note:** Training script uses the old ModelRegistry system and may need updates to work with the new ModelBuilder architecture.

### Custom Configuration

Models can be configured via YAML files in `configs/models/`:

```bash
# Copy default configuration
cp configs/models/magic.yaml configs/models/magic_custom.yaml

# Edit configuration (adjust architecture, training params, etc.)
nano configs/models/magic_custom.yaml

# Use custom configuration for evaluation
python experiments/evaluate_pipeline.py \
    --models magic \
    --dataset custom_soc \
    --data-path data/custom_soc
```

### Batch Processing Multiple Datasets

```bash
# Create a batch evaluation script
cat > batch_evaluate.sh << 'EOF'
#!/bin/bash
for dataset in dataset1 dataset2 dataset3; do
    echo "Evaluating $dataset..."
    ./scripts/run_evaluation.sh \
        --data-path data/$dataset \
        --dataset $dataset \
        --output-dir results/${dataset}_evaluation
done
EOF

chmod +x batch_evaluate.sh
./batch_evaluate.sh
```

### GPU Memory Optimization

If you run out of GPU memory:

```bash
# Reduce batch size
python experiments/evaluate_pipeline.py \
    --models magic \
    --batch-size 4 \  # Reduce from default 32
    --device 0

# Use CPU instead
python experiments/evaluate_pipeline.py \
    --models magic \
    --device -1  # -1 = CPU
```

---

## Command Reference

### setup.sh

**Purpose:** Complete environment setup (automated)

```bash
./scripts/setup.sh
```

**No arguments** - fully automated script

**What it does:**
- Creates conda environment from `environment.yml`
- Installs PyTorch 1.12.1 with CUDA 11.6
- Installs DGL 1.0.0
- Installs PyTorch Geometric (torch-scatter, torch-sparse, torch-cluster)
- Applies MKL threading fix automatically
- Creates directory structure
- Verifies installation

**Time:** 10-15 minutes

---

### download_checkpoints.py

**Purpose:** Download pretrained weights and install model-specific dependencies

```bash
# Setup all models (recommended)
python scripts/download_checkpoints.py --all

# Setup specific models
python scripts/download_checkpoints.py --models magic kairos orthrus

# List available models and sources
python scripts/download_checkpoints.py --list

# Force re-download existing weights
python scripts/download_checkpoints.py --all --force-download

# Only install dependencies (skip weight download)
python scripts/download_checkpoints.py --all --no-download

# Only download weights (skip dependencies)
python scripts/download_checkpoints.py --download-only --all

# Skip local fallback (GitHub only)
python scripts/download_checkpoints.py --all --no-copy
```

**Arguments:**
- `--all` - Setup all 5 models
- `--models MODEL [MODEL ...]` - Setup specific models (magic, kairos, orthrus, threatrace, continuum_fl)
- `--list` - List all models with GitHub sources and exit
- `--no-install` - Skip dependency installation
- `--no-copy` - Skip copying from local fallback directories
- `--no-download` - Skip downloading weights
- `--download-only` - Only download weights (skip deps and local copy)
- `--force-download` - Force re-download existing weights

**Time:** 5-20 minutes (depending on download speeds)

---

### preprocess_data.py

**Purpose:** Convert JSON logs to graph format for model evaluation

```bash
# Basic preprocessing
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc

# Advanced options
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --schema elastic \              # Schema type: 'elastic' or 'custom'
    --time-window 3600 \            # Time window in seconds
    --event-types process file \    # Filter event types
    --chunk-size 50000 \            # Events per chunk
    --verbose                       # Detailed progress
```

**Required Arguments:**
- `--input-dir PATH` - Directory with JSON log files
- `--output-dir PATH` - Directory to save preprocessed data
- `--dataset-name NAME` - Name for the dataset

**Optional Arguments:**
- `--schema {elastic,custom}` - Data schema format (default: elastic)
- `--time-window SECONDS` - Time window for temporal graphs (default: 3600)
- `--event-types TYPE [TYPE ...]` - Event types to include (process, file, network)
- `--chunk-size NUM` - Events per chunk for large files (default: 10000)
- `--config PATH` - Custom configuration file
- `--verbose` - Show detailed progress information

**Output Files:**
- `<dataset>_graph.pkl` - Graph structure (DGL/PyG format)
- `<dataset>_features.pt` - Node and edge features (PyTorch tensor)
- `<dataset>_metadata.json` - Dataset statistics and metadata

**Time:** 1-30 minutes (depending on dataset size)

---

### run_evaluation.sh

**Purpose:** Complete evaluation workflow (setup → preprocess → evaluate → compare)

```bash
# Evaluate all models on custom data
./scripts/run_evaluation.sh

# Evaluate specific model
./scripts/run_evaluation.sh --model magic

# Use preprocessed data (skip preprocessing)
./scripts/run_evaluation.sh \
    --data-path data/custom_soc \
    --skip-preprocess

# Custom output directory
./scripts/run_evaluation.sh \
    --output-dir results/my_evaluation

# Skip weight download (use existing)
./scripts/run_evaluation.sh --skip-download
```

**Arguments:**
- `--model MODEL` - Evaluate specific model (magic, kairos, orthrus, threatrace, continuum_fl, all)
- `--dataset NAME` - Dataset name (default: custom_soc)
- `--data-path PATH` - Path to preprocessed data or JSON source files
- `--skip-download` - Skip downloading pretrained weights
- `--skip-preprocess` - Skip data preprocessing (use if already preprocessed)
- `--output-dir DIR` - Output directory for results
- `--help, -h` - Show help message

**Time:** 10-60 minutes (depending on dataset size and number of models)

---

### verify_installation.py

**Purpose:** Verify framework installation and dependencies

```bash
python scripts/verify_installation.py
```

**No arguments**

**Checks performed:**
1. Python version (3.8+ required)
2. Core dependencies (torch, numpy, pandas, etc.)
3. Deep learning frameworks (PyTorch, DGL, PyG)
4. Model integrations (all 5 model wrappers)
5. Directory structure (12 required directories)
6. Configuration files (10+ YAML configs)
7. Scripts (7 required scripts)
8. Documentation (4 markdown files)
9. External models (5 optional external directories)

**Exit code:** 0 if all checks pass, 1 if any check fails

---

### evaluate_pipeline.py (Task-Based Evaluation)

**Purpose:** Modern task-based evaluation using the pipeline architecture

```bash
# Basic evaluation - single model
python experiments/evaluate_pipeline.py \
    --models magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --checkpoints-dir checkpoints

# Evaluate multiple models
python experiments/evaluate_pipeline.py \
    --models magic kairos orthrus \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --checkpoints-dir checkpoints \
    --batch-size 16 \               # Batch size
    --device 0 \                    # GPU device ID (-1 for CPU)
    --cache-dir .cache \            # Cache intermediate results
    --output-dir results/eval

# Evaluate all models
python experiments/evaluate_pipeline.py \
    --models all \
    --dataset streamspot \
    --data-path data/preprocessed/streamspot
```

**Arguments:**
- `--models MODEL [MODEL ...]` - One or more models to evaluate (or "all")
- `--dataset NAME` - Dataset name
- `--data-path PATH` - Path to preprocessed data
- `--checkpoints-dir PATH` - Directory containing checkpoints (default: checkpoints/)
- `--batch-size NUM` - Batch size (default: 32)
- `--device NUM` - Device: -1 for CPU, 0+ for GPU (default: -1)
- `--cache-dir PATH` - Cache directory for pipeline (default: .cache/)
- `--output-dir PATH` - Output directory (default: results/)

**Features:**
- Task-based pipeline with 9 stages (load_data → calculate_metrics)
- Automatic caching of intermediate results
- Multi-model evaluation in single run
- Dynamic model construction via ModelBuilder
- Per-model YAML configurations

---

### visualize_attack_graphs.py

**Purpose:** Visualize and compare attack graphs from multiple models

```bash
# Visualize all models with default settings
python utils/visualize_attack_graphs.py

# Visualize specific models
python utils/visualize_attack_graphs.py \
    --models magic kairos orthrus

# Customize thresholds and output
python utils/visualize_attack_graphs.py \
    --artifacts-dir artifacts \
    --output-dir results/attack_viz \
    --threshold-percentile 99.0 \
    --top-k 100 \
    --top-paths 20 \
    --cluster-by entity

# Focus on high-severity anomalies
python utils/visualize_attack_graphs.py \
    --threshold-percentile 99.9 \
    --top-k 50 \
    --top-paths 10
```

**Arguments:**
- `--artifacts-dir PATH` - Artifacts directory with model outputs (default: artifacts/)
- `--models MODEL [MODEL ...]` - Models to visualize (default: all 5 models)
- `--output-dir PATH` - Output directory for visualizations (default: results/attack_graph_visualization/)
- `--threshold-percentile FLOAT` - Percentile threshold for anomalies, 90-99.9 (default: 95.0)
- `--top-k NUM` - Number of top anomalies to include (default: 100)
- `--top-paths NUM` - Number of attack paths to extract (default: 10)
- `--cluster-by STRATEGY` - Clustering strategy: entity, temporal, or path (default: entity)

**Features:**
- Attack path reconstruction with provenance traversal
- Interactive HTML visualization (auto-opens in browser)
- Multi-model comparison side-by-side
- Entity clustering and attack path ranking
- Multiple export formats: HTML, JSON, GraphML
- Compatible with Gephi, Cytoscape, Neo4j

**Output Files:**
- `attack_graph_viewer.html` - Interactive visualization
- `attack_summary.json` - Attack statistics and paths
- `{model}_attack_graph.graphml` - GraphML exports for each model

**Time:** 1-5 minutes depending on number of models and graph size

---

### visualize_attacks.sh

**Purpose:** Convenience wrapper for attack graph visualization with simplified arguments

```bash
# Visualize with default settings
./scripts/visualize_attacks.sh

# Customize thresholds
./scripts/visualize_attacks.sh \
    --threshold 99.9 \
    --top-k 50 \
    --top-paths 20

# Custom output and clustering
./scripts/visualize_attacks.sh \
    --output-dir results/my_attack_viz \
    --cluster-by temporal \
    --threshold 99

# Show help
./scripts/visualize_attacks.sh --help
```

**Arguments:**
- `--threshold FLOAT` - Percentile threshold for anomalies, 90-99.9 (default: 99)
- `--top-k NUM` - Number of top anomalies to include (default: 100)
- `--top-paths NUM` - Number of attack paths to extract (default: 15)
- `--cluster-by STRATEGY` - Clustering strategy: entity, temporal, or path (default: entity)
- `--output-dir PATH` - Output directory for visualizations (default: results/attack_graph_visualization/)
- `--help` - Show help message

**Features:**
- Simplified command-line interface
- Automatic browser opening
- Progress feedback
- All models visualized by default
- Same output as `visualize_attack_graphs.py`

**Output Files:**
- `attack_graph_viewer.html` - Interactive visualization (auto-opens in browser)
- `attack_summary.json` - Attack statistics and paths
- `{model}_attack_graph.graphml` - GraphML exports for each model

**Time:** 1-5 minutes depending on number of models and graph size

**Note:** This script internally calls `utils/visualize_attack_graphs.py` with appropriate parameters.

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Conda environment not activated"

**Symptom:**
```
Error: Conda environment 'pids_framework' is not activated!
Please run: conda activate pids_framework
```

**Solution:**
```bash
conda activate pids_framework

# Verify activation
echo $CONDA_DEFAULT_ENV
# Should output: pids_framework
```

---

#### Issue 2: PyTorch import fails with symbol error

**Symptom:**
```
OSError: /lib/x86_64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block
OR
undefined symbol: iJIT_NotifyEvent
```

**Solution:** This is automatically fixed by `setup.sh`, but if it persists:

```bash
conda activate pids_framework

# The MKL fix is already applied, but verify:
echo $MKL_THREADING_LAYER
# Should output: GNU

# If not set, manually export:
export MKL_THREADING_LAYER=GNU

# Test PyTorch:
python -c "import torch; print(torch.__version__)"

# If still fails, reinstall MKL:
conda install "mkl<2024" -c conda-forge --force-reinstall -y
```

---

#### Issue 3: Git sparse-checkout timeout (ThreaTrace)

**Symptom:**
```
❌ git sparse-checkout timed out (repository too large)
```

**Solution:** Manual download:

```bash
# Method 1: Full clone and copy
git clone https://github.com/Provenance-IDS/threaTrace.git /tmp/threatrace
cp -r /tmp/threatrace/example_models checkpoints/threatrace/

# Method 2: SVN export (faster, requires svn)
# Install svn first:
# macOS: brew install subversion
# Ubuntu: sudo apt-get install subversion

svn export https://github.com/Provenance-IDS/threaTrace/trunk/example_models checkpoints/threatrace/

# Verify:
ls -R checkpoints/threatrace/
```

---

#### Issue 4: Google Drive download fails (Kairos)

**Symptom:**
```
❌ gdown not installed or download failed
```

**Solution:** Manual download from Google Drive:

```bash
# Method 1: Using gdown (if not already installed)
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C -O checkpoints/kairos/

# Method 2: Browser download
# 1. Open: https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C
# 2. Click "Download" button
# 3. Extract downloaded zip
# 4. Copy all files to: checkpoints/kairos/

# Verify:
ls -lh checkpoints/kairos/
```

---

#### Issue 5: Preprocessed data not found

**Symptom:**
```
Error: No preprocessed data found
```

**Solution:** Check filename and location:

```bash
# run_evaluation.sh looks for specific files:
ls data/custom_soc/custom_soc_graph.pkl  # NEW format
ls data/custom_soc/graph.pkl             # OLD format

# If file has different name, create symlink:
cd data/custom_soc
ln -s your_graph_file.pkl custom_soc_graph.pkl

# OR re-run preprocessing with correct dataset name:
python scripts/preprocess_data.py \
    --input-dir ../../custom_dataset \
    --output-dir . \
    --dataset-name custom_soc
```

---

#### Issue 6: Out of memory during evaluation

**Symptom:**
```
Killed
OR
RuntimeError: CUDA out of memory
```

**Solution:** Reduce memory usage:

```bash
# For CPU OOM:
python experiments/evaluate_pipeline.py \
    --models magic \
    --batch-size 4     # Reduce from default 32

# For GPU OOM:
python experiments/evaluate_pipeline.py \
    --models magic \
    --device -1  # Use CPU instead of GPU

# OR reduce batch size on GPU:
python experiments/evaluate_pipeline.py \
    --models magic \
    --batch-size 4 \
    --device 0
```

---

#### Issue 7: Model checkpoint not found

**Symptom:**
```
Error: Checkpoint not found: checkpoints/magic/checkpoint-streamspot.pt
```

**Solution:** Verify and re-download checkpoints:

```bash
# Check what checkpoints exist:
ls -R checkpoints/

# Re-download missing checkpoints:
python scripts/download_checkpoints.py --all --force-download

# OR download specific model:
python scripts/download_checkpoints.py --models magic --force-download

# Verify checkpoint structure:
ls -lh checkpoints/magic/
# Should contain .pt files
```

---

#### Issue 8: Conda activate not working

**Symptom:**
```
conda: command not found
OR
CommandNotFoundError: Your shell has not been properly configured
```

**Solution:** Initialize conda:

```bash
# Find conda installation
which conda
# OR
find ~ -name conda 2>/dev/null

# Initialize conda (replace with your conda path)
~/anaconda3/bin/conda init bash  # For bash
~/anaconda3/bin/conda init zsh   # For zsh

# Restart shell
exec $SHELL

# Try activating again
conda activate pids_framework
```

---

#### Issue 9: Insufficient disk space

**Symptom:**
```
No space left on device
```

**Solution:** Free up space or use different directory:

```bash
# Check disk space
df -h .

# Clean conda cache
conda clean --all -y

# Clean pip cache
pip cache purge

# Remove unnecessary checkpoints (if you only need specific models)
rm -rf checkpoints/threatrace  # Large (500MB)

# Use external drive for data
mkdir -p /path/to/external/drive/pids_data
ln -s /path/to/external/drive/pids_data data/custom_soc
```

---

#### Issue 10: Permission denied errors

**Symptom:**
```
Permission denied: ./scripts/setup.sh
```

**Solution:** Fix file permissions:

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Verify permissions
ls -la scripts/

# Run setup script
./scripts/setup.sh
```

---

### Getting More Help

If you encounter issues not covered here:

1. **Check logs:**
   ```bash
   # Evaluation logs
   tail -n 100 results/evaluation_*/magic_evaluation.log
   
   # System logs
   tail -n 100 logs/framework.log
   ```

2. **Enable verbose output:**
   ```bash
   python scripts/preprocess_data.py --verbose
   python experiments/evaluate_pipeline.py --models magic --dataset streamspot
   ```

3. **Test individual components:**
   ```bash
   # Test PyTorch
   python -c "import torch; print(torch.__version__)"
   
   # Test DGL
   python -c "import dgl; print(dgl.__version__)"
   
   # Test model imports
   python scripts/verify_implementation.py
   ```

4. **Check GitHub Issues:**
   - Framework issues: https://github.com/yourusername/PIDS_Comparative_Framework/issues
   - Model-specific issues: Check individual model repositories

---

## Configuration

### Model Configuration Files

Each model has a YAML configuration file in `configs/models/`:

```bash
configs/models/
├── magic.yaml          # MAGIC configuration
├── kairos.yaml         # Kairos configuration
├── orthrus.yaml        # Orthrus configuration
├── threatrace.yaml     # ThreaTrace configuration
└── continuum_fl.yaml   # Continuum_FL configuration
```

**Example: `configs/models/magic.yaml`**
```yaml
model:
  name: magic
  type: autoencoder
  architecture:
    encoder:
      hidden_dim: 128
      num_layers: 3
      dropout: 0.1
    decoder:
      hidden_dim: 128
      num_layers: 3
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: adam
  scheduler: cosine

evaluation:
  batch_size: 64
  k_neighbors: 5
  detection_level: entity
```

### Dataset Configuration Files

Dataset configurations are in `configs/datasets/`:

```bash
configs/datasets/
├── cadets_e3.yaml      # DARPA TC CADETS E3
├── streamspot.yaml     # StreamSpot dataset
└── custom_soc.yaml     # Template for custom data
```

**Example: `configs/datasets/custom_soc.yaml`**
```yaml
dataset:
  name: custom_soc
  type: provenance_graph
  
  paths:
    data_dir: data/custom_soc
    graph_file: custom_soc_graph.pkl
    features_file: custom_soc_features.pt
    metadata_file: custom_soc_metadata.json
  
  format:
    schema: elastic  # or 'custom'
    timestamp_field: '@timestamp'
    event_field: 'event.type'
  
  preprocessing:
    time_window: 3600  # seconds
    event_types:
      - process
      - file
      - network
    feature_dim: 128
```

### Experiment Configuration Files

Experiment configurations are in `configs/experiments/`:

```bash
configs/experiments/
├── compare_all.yaml    # Compare all models
└── train_single.yaml   # Train single model
```

### Modifying Configurations

To customize model behavior:

```bash
# 1. Copy default configuration
cp configs/models/magic.yaml configs/models/magic_custom.yaml

# 2. Edit configuration
nano configs/models/magic_custom.yaml

# 3. Use custom configuration in training/evaluation
python experiments/train.py \
    --model magic \
    --config configs/models/magic_custom.yaml
```

---

## Additional Resources

### Model Documentation

Each model has its own documentation:
- **MAGIC:** https://github.com/FDUDSDE/MAGIC
- **Kairos:** https://github.com/ubc-provenance/kairos
- **Orthrus:** https://github.com/ubc-provenance/orthrus
- **ThreaTrace:** https://github.com/threaTrace-detector/threaTrace
- **Continuum_FL:** https://github.com/kamelferrahi/Continuum_FL

### Papers

- **[MAGIC — "MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning"](https://arxiv.org/abs/2310.09831)**
- **[Kairos — "Kairos: Practical Intrusion Detection and Investigation using Whole-system Provenance"](https://ieeexplore.ieee.org/document/10646673)**
- **[Orthrus — "Achieving High Quality of Attribution in Provenance-based Intrusion Detection Systems"](https://www.usenix.org/conference/usenixsecurity25/presentation/jiang-baoxiang)**
- **[ThreaTrace — "Enabling Refinable Cross-Host Attack Investigation with Efficient Data Flow Tagging and Tracking"](https://ieeexplore.ieee.org/document/9899459)**
- **[Continuum_FL — "Federated Learning for Intrusion Detection Systems"](https://www.researchgate.net/publication/387767270_CONTINUUM_Detecting_APT_Attacks_through_Spatial-Temporal_Graph_Neural_Networks)**

---

### Installation Checklist

- [ ] Conda installed and verified
- [ ] Framework directory accessible
- [ ] `./scripts/setup.sh` executed successfully
- [ ] Environment activated: `conda activate pids_framework`
- [ ] `python scripts/download_checkpoints.py --all` completed
- [ ] Pretrained weights downloaded (check `checkpoints/`)
- [ ] `python scripts/verify_installation.py` passes all checks

### Evaluation Checklist

- [ ] Custom data in JSON format
- [ ] Data placed in `custom_dataset/` directory
- [ ] `python scripts/preprocess_data.py` completed
- [ ] Preprocessed files exist in `data/custom_soc/`
- [ ] `./scripts/run_evaluation.sh` executed
- [ ] Results saved in `results/evaluation_*/`
- [ ] Comparison report generated
- [ ] Attack graphs visualized: `python utils/visualize_attack_graphs.py`
- [ ] Interactive visualization opened in browser

### Next Steps

1. **Review results:** Check `results/` directory for evaluation metrics
2. **Compare models:** Analyze comparison report to find best model
3. **Deploy:** Integrate best-performing model into your SOC pipeline
4. **Retrain (optional):** Train models on your specific data for better performance

---

## Conclusion

You've successfully set up the PIDS Comparative Framework! The framework is now ready to evaluate state-of-the-art intrusion detection models on your custom SOC data.

For questions or issues:
- Check the [Troubleshooting](#troubleshooting) section
- Review model-specific documentation
- Open an issue on GitHub
