# PIDS Comparative Framework

<div align="center">

**A Unified Framework for Evaluating State-of-the-Art Provenance-based Intrusion Detection Systems (PIDS)**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Quick Start](#quick-start) | [Installation](#installation) | [Models](#supported-models) | [Configuration](#configuration) | [Troubleshooting](#troubleshooting)

</div>

> **📚 Documentation**: This README provides complete framework documentation. For quick setup, see [QUICKSTART.md](QUICKSTART.md).

---

## 🎯 **Primary Use Case: Evaluate Pretrained Models**

This framework is designed to **evaluate pretrained PIDS models on your custom SOC data**. The models are already trained on benchmark datasets (DARPA, StreamSpot) and ready to detect intrusions in your environment.

✅ **Default Workflow**: Evaluate pretrained weights → Compare performance → Deploy best model  
✅ **CPU-First**: Runs on CPU by default (no GPU required)  
✅ **GPU Support**: Automatically uses GPU when available  
🔄 **Advanced Feature**: Retrain models on custom data (optional)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Supported Models](#supported-models)
- [Supported Datasets](#supported-datasets)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [Results](#results)
- [Extending the Framework](#extending-the-framework)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## 🔍 Overview

The **PIDS Comparative Framework** is a production-ready platform designed for Security Operations Centers (SOC) to:

✅ **Evaluate** pretrained PIDS models on your custom SOC data (2GB+ logs supported)  
✅ **Compare** multiple models with consistent benchmarking and metrics  
✅ **Deploy** the best-performing model for your specific environment  
✅ **Train** models on custom data (advanced feature, optional)  
✅ **Fine-tune** existing models using transfer learning (advanced feature)  
✅ **Extend** easily with new models through a clean interface  

### What is Provenance-based Intrusion Detection?

Provenance graphs capture system-level information flows (process→file, process→network) to model normal behavior and detect anomalous activities indicative of cyber attacks. This framework integrates 5 state-of-the-art deep learning approaches for analyzing provenance data.

### Primary Workflow

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Your SOC Data   │ ───> │ Pretrained PIDS  │ ───> │ Performance     │
│ (JSON Logs)     │      │ Models           │      │ Comparison      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                            │
                                                            ▼
                                                    ┌─────────────────┐
                                                    │ Deploy Best     │
                                                    │ Model to SOC    │
                                                    └─────────────────┘
```

---

## ✨ Key Features

### 🎯 **Evaluation-First Design**
- **Pretrained Models**: Use existing weights immediately - no training required
- **Quick Deployment**: Evaluate all models on your data in minutes
- **Performance Comparison**: Automatic comparison with statistical significance testing
- **One-Command Workflow**: `./scripts/run_evaluation.sh` does everything

### 📊 **Multi-Model Support**
- **5 State-of-the-Art Models**: MAGIC, Kairos, Orthrus, ThreaTrace, Continuum_FL
- **Consistent Interface**: All models through unified `BasePIDSModel` API
- **Automatic Registration**: Dynamic model discovery via decorator pattern
- **Pretrained Weights**: Ready-to-use checkpoints for all models

### � **Comprehensive Evaluation**
- **Multiple Metrics**: AUROC, AUPRC, F1-Score, Precision, Recall, Detection Rate
- **Statistical Analysis**: Significance testing for model comparison
- **Rich Visualizations**: ROC curves, precision-recall curves, confusion matrices
- **Detailed Reports**: JSON and text formats with all metrics

### 🔧 **Production-Ready**
- **CPU-First Design**: Runs on CPU by default (no GPU required)
- **GPU Support**: Automatic GPU detection and utilization when available
- **Large-Scale Data**: Handles 2GB+ JSON files with chunked loading
- **Checkpointing**: Save/resume training with early stopping (for retraining)
- **Logging**: Comprehensive logging for debugging and monitoring
- **Error Handling**: Graceful degradation and informative error messages

### 📦 **Easy to Use**
- **YAML Configurations**: All settings in human-readable configs
- **One-Command Setup**: Automated environment and dependency installation
- **Streamlined Workflow**: Evaluation script handles all steps automatically
- **Comprehensive Docs**: README.md and QUICKSTART.md cover all features

### 🔬 **Extensible Architecture** (Advanced)
- **Modular Design**: Separate data, models, training, evaluation
- **Plugin System**: Add new models with ~200 lines of code
- **Configurable**: Override any setting via YAML or command-line
- **Retraining Support**: Optional model training on custom datasets

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  (CLI Commands, Configuration Files, Experiment Scripts)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                   Framework Core Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Model     │  │   Dataset    │  │   Experiment     │   │
│  │  Registry   │  │   Loaders    │  │    Manager       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                    Model Layer                                │
│  ┌──────┐  ┌────────┐  ┌─────────┐  ┌───────────┐ ┌────┐  │
│  │MAGIC │  │Kairos  │  │Orthrus  │  │ThreaTrace │ │...│   │
│  └──────┘  └────────┘  └─────────┘  └───────────┘ └────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                     Data Layer                                │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐    │
│  │  Custom SOC  │  │   DARPA    │  │   StreamSpot     │    │
│  │     Data     │  │   TC E3/E5 │  │   (Scene-based)  │    │
│  └──────────────┘  └────────────┘  └──────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### Core Components

1. **BasePIDSModel**: Abstract interface all models implement
2. **ModelRegistry**: Dynamic model registration and discovery
3. **Dataset Loaders**: Handle JSON, DARPA TC, StreamSpot formats
4. **Training Pipeline**: Consistent training with early stopping, checkpointing
5. **Evaluation Pipeline**: Unified metrics computation and reporting
6. **Experiment Manager**: Orchestrate multi-model comparisons

---

## 🤖 Supported Models

| Model | Paper | Year | Architecture | Strengths |
|-------|-------|------|--------------|-----------|
| **MAGIC** | [USENIX Security '24](https://www.usenix.org/conference/usenixsecurity24) | 2024 | Masked Graph Autoencoder | Self-supervised, high accuracy |
| **Kairos** | [IEEE S&P '24](https://www.ieee-security.org/TC/SP2024/) | 2024 | Temporal GNN | Temporal modeling, link prediction |
| **Orthrus** | [USENIX Security '25](https://www.usenix.org/conference/usenixsecurity25) | 2025 | Multi-Decoder | Contrastive learning, robust |
| **ThreaTrace** | [IEEE TIFS '22](https://ieeexplore.ieee.org/document/9721562) | 2022 | GraphChi GNN | Scalable, disk-based processing |
| **Continuum_FL** | Federated PIDS | 2023 | Federated STGNN | Privacy-preserving, distributed |

### Model Selection Guide

- **Best Overall Accuracy**: MAGIC (AUROC ~0.92-0.95)
- **Temporal Data**: Kairos (explicit temporal modeling)
- **Large-Scale Graphs**: ThreaTrace (GraphChi disk-based)
- **Privacy-Sensitive**: Continuum_FL (federated learning)
- **Robust Features**: Orthrus (contrastive learning)

---

## 📊 Supported Datasets

### 1. Custom SOC Data
- **Format**: JSON (Elastic/ELK compatible)
- **Size**: Supports 2GB+ files
- **Types**: endpoint_file.json, endpoint_network.json, endpoint_process.json
- **Features**: Automatic graph construction from provenance events

### 2. DARPA Transparent Computing (TC)
- **Engagements**: E3, E5
- **Systems**: CADETS (BSD), THEIA (Linux), TRACE (Android), CLEARSCOPE, FiveDirections
- **Events**: 268K+ nodes, 6M+ edges (CADETS E3)
- **Ground Truth**: Attack scenarios with timestamps

### 3. StreamSpot
- **Type**: Scene-based provenance graphs
- **Categories**: YouTube, Gmail, VGame, Download, CNN, Drive-by-Download (attack)
- **Scenes**: 600 total (100 per category)
- **Features**: Graph-level classification

### 4. Unicorn Wget
- **Type**: Simulated attack scenarios
- **Variants**: wget, wget-long
- **Features**: Controlled attack injection

---

## 💻 Requirements

### System Requirements
- **OS**: Linux (Ubuntu 18.04+), macOS (10.14+), Windows (with WSL2)
- **CPU**: 4+ cores recommended (framework runs on CPU by default)
- **RAM**: 16GB+ recommended (8GB minimum)
- **GPU**: Optional - NVIDIA GPU with 8GB+ VRAM (auto-detected if available)
- **Storage**: 50GB+ free space

### Software Requirements
- **Conda**: Anaconda or Miniconda (required)
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **CUDA**: 11.6+ (optional, only for GPU acceleration)
- **Git**: For cloning repositories

**Note**: The framework defaults to CPU execution. GPU is automatically used if available and can be controlled via `CUDA_VISIBLE_DEVICES` environment variable or `--device` parameter.

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
cd /path/to/PIDS_Files
# Framework is already in PIDS_Comparative_Framework/
```

### Step 2: Automated Setup (Recommended)

```bash
cd PIDS_Comparative_Framework

# Run the automated setup script
./scripts/setup.sh
```

**What `setup.sh` does:**
1. Checks for Conda installation
2. Creates `pids_framework` conda environment (Python 3.10)
3. Installs PyTorch 1.12.1 with CUDA 11.6
4. Installs DGL 1.0.0 and PyTorch Geometric
5. Installs core dependencies
6. Creates necessary directories
7. Verifies installation

### Step 3: Install Model-Specific Dependencies

```bash
# Activate the environment
conda activate pids_framework

# Install dependencies for all models
./scripts/install_model_deps.sh --all

# OR install for specific models
./scripts/install_model_deps.sh --models magic kairos orthrus
```

### Step 4: Verify Installation

```bash
# Test model registry
python -c "from models import list_available_models; print(list_available_models())"

# Expected output:
# ['magic', 'magic_streamspot', 'magic_darpa', 'kairos', 'orthrus', 
#  'threatrace', 'continuum_fl', 'continuum_fl_streamspot', 'continuum_fl_darpa']
```

### Manual Installation (If Automated Fails)

<details>
<summary>Click to expand manual installation steps</summary>

```bash
# 1. Create conda environment
conda create -n pids_framework python=3.10 -y
conda activate pids_framework

# 2. Install PyTorch with CUDA
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y

# 3. Install DGL
conda install -c dglteam dgl-cuda11.6==1.0.0 -y

# 4. Install PyTorch Geometric
pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
pip install torch-geometric==2.1.0

# 5. Install core dependencies
pip install -r requirements.txt

# 6. Install model-specific dependencies
pip install -r requirements/magic.txt
pip install -r requirements/kairos.txt
pip install -r requirements/orthrus.txt
pip install -r requirements/threatrace.txt
pip install -r requirements/continuum_fl.txt
```

</details>

---

## ⚡ Quick Start

> **TL;DR**: Run `./scripts/run_evaluation.sh` to evaluate all pretrained models on your data! (CPU by default)

### Default Workflow: Evaluate Pretrained Models (3 Steps)

#### Step 1: Setup Environment

```bash
cd PIDS_Comparative_Framework
./scripts/setup.sh
conda activate pids_framework
```

#### Step 2: Prepare Your Data

```bash
# Ensure your SOC data is in JSON format at ../custom_dataset/
ls ../custom_dataset/
# Expected: endpoint_file.json, endpoint_network.json, endpoint_process.json
```

#### Step 3: Run Evaluation (CPU by default)

```bash
# Evaluate ALL pretrained models on your custom data (runs on CPU)
./scripts/run_evaluation.sh

# Results saved to: results/evaluation_YYYYMMDD_HHMMSS/
```

**That's it!** The script will:
1. ✅ Download/copy pretrained weights from existing checkpoints
2. ✅ Preprocess your custom SOC data
3. ✅ Evaluate all models on your data (CPU default, GPU auto-detected if available)
4. ✅ Generate comparison report with metrics (AUROC, AUPRC, F1, etc.)

### View Results

```bash
# Navigate to results directory
cd results/evaluation_YYYYMMDD_HHMMSS/

# View comparison report
cat comparison_report.json

# Check individual model performance
cat magic_evaluation.log
cat kairos_evaluation.log
```

### Evaluate Specific Model

```bash
# MAGIC only (CPU)
./scripts/run_evaluation.sh --model magic

# With custom data path
./scripts/run_evaluation.sh --model kairos --data-path /path/to/logs

# Force GPU usage (if available)
CUDA_VISIBLE_DEVICES=0 ./scripts/run_evaluation.sh --model magic
```

### Alternative: Manual Step-by-Step Evaluation

<details>
<summary>Click to expand manual evaluation steps</summary>

```bash
# 1. Download pretrained weights
python scripts/download_weights.py --copy-existing --all-models

# 2. Preprocess your data
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset/ \
    --output-dir data/custom_soc \
    --dataset-name custom_soc

# 3. Evaluate single model (CPU by default)
python experiments/evaluate.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --pretrained \
    --checkpoint-dir checkpoints \
    --device -1  # -1 for CPU (default), 0+ for GPU
    --output-dir results/magic_eval

# 4. Compare all models
python experiments/compare.py \
    --results-dir results/ \
    --dataset custom_soc \
    --output-file results/comparison.json
```

</details>

### Advanced: Retrain on Custom Data (Optional)

> ⚠️ **Note**: Retraining is an **advanced feature**. Start with pretrained evaluation first!

<details>
<summary>Click to expand retraining instructions</summary>

```bash
# Retrain MAGIC on your custom data
python experiments/train.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 100 \
    --batch-size 32 \
    --config configs/experiments/train_single.yaml

# Evaluate retrained model
python experiments/evaluate.py \
    --model magic \
    --checkpoint checkpoints/magic_custom_soc_best.pt \
    --dataset custom_soc

# Compare with pretrained baseline
python experiments/compare.py \
    --checkpoints checkpoints/magic_pretrained.pt checkpoints/magic_custom_soc_best.pt \
    --labels "Pretrained" "Retrained" \
    --dataset custom_soc
```

</details>

---

## 📖 Usage Examples

### Example 1: Evaluate All Models on Custom SOC Data (PRIMARY USE CASE)

```bash
# One command to evaluate everything
./scripts/run_evaluation.sh

# View results
cat results/evaluation_*/comparison_report.json
```

**Expected Output:**
```json
{
  "dataset": "custom_soc",
  "timestamp": "2025-10-13T10:30:00",
  "models": {
    "magic": {
      "auc_roc": 0.9234,
      "auc_pr": 0.8967,
      "f1": 0.8756,
      "precision": 0.8543,
      "recall": 0.8978
    },
    "kairos": {
      "auc_roc": 0.9156,
      "auc_pr": 0.8834,
      "f1": 0.8623,
      "precision": 0.8412,
      "recall": 0.8845
    },
    ...
  }
}
```

### Example 2: Evaluate Specific Model with Detailed Options

```bash
# Evaluate MAGIC with custom settings
./scripts/run_evaluation.sh \
    --model magic \
    --data-path /path/to/soc/logs \
    --output-dir results/magic_detailed \
    --skip-download  # If weights already exist
```

### Example 3: Compare Models Across Multiple Datasets

```bash
# Evaluate on DARPA CADETS E3
./scripts/run_evaluation.sh --dataset cadets_e3 --data-path data/darpa_cadets_e3/

# Evaluate on StreamSpot
./scripts/run_evaluation.sh --dataset streamspot --data-path data/streamspot/

# Evaluate on Custom SOC
./scripts/run_evaluation.sh --dataset custom_soc --data-path ../custom_dataset/

# Aggregate results
python experiments/aggregate_results.py \
    --results results/evaluation_*/comparison_report.json \
    --output results/cross_dataset_comparison.csv
```

### Example 4: Train MAGIC on DARPA CADETS E3 (Advanced - Retraining)

```bash
python experiments/train.py \
    --model magic_darpa \
    --dataset cadets_e3 \
    --epochs 500 \
    --batch-size 1 \
    --learning-rate 0.0005 \
    --device cuda
```

### Example 2: Fine-tune Pretrained Model

```yaml
# Create config: configs/experiments/finetune_magic.yaml
model:
  name: magic
  pretrained_checkpoint: checkpoints/magic/checkpoint-cadets-e3.pt

dataset:
  name: custom_soc
  config: configs/datasets/custom_soc.yaml

training:
  num_epochs: 50              # Fewer epochs
  learning_rate: 0.0001       # Lower learning rate
  freeze_encoder: false       # Fine-tune all layers
```

```bash
python experiments/train.py --config configs/experiments/finetune_magic.yaml
```

### Example 3: Evaluate All Models

```bash
# Evaluate all models on test set
python experiments/evaluate.py \
    --all-models \
    --dataset custom_soc \
    --output results/evaluation/
```

### Example 4: Custom Dataset Configuration

```yaml
# Edit configs/datasets/my_soc_data.yaml
dataset_name: my_soc_data
dataset_type: custom

data:
  root_dir: /path/to/my/data/
  files:
    - security_events.json
    - network_logs.json
    
format:
  type: json
  schema: elastic

graph:
  node_types: [process, file, network, registry]
  edge_types: [read, write, execute, connect]
  
labels:
  source: groundtruth
  label_file: Ground_Truth/my_labels.json
```

```bash
python scripts/preprocess_data.py --config configs/datasets/my_soc_data.yaml
python experiments/train.py --model magic --dataset my_soc_data
```

---

## ⚙️ Configuration

### Configuration Hierarchy

```
configs/
├── models/              # Model hyperparameters
│   ├── magic.yaml
│   ├── kairos.yaml
│   ├── orthrus.yaml
│   ├── threatrace.yaml
│   └── continuum_fl.yaml
├── datasets/            # Dataset specifications
│   ├── custom_soc.yaml
│   ├── cadets_e3.yaml
│   └── streamspot.yaml
└── experiments/         # Experiment templates
    ├── compare_all.yaml
    └── train_single.yaml
```

### Key Configuration Options

**Model Config Example** (`configs/models/magic.yaml`):
```yaml
model_name: magic
architecture:
  num_hidden: 256
  num_layers: 4
  mask_rate: 0.3
training:
  learning_rate: 0.0005
  num_epochs: 500
  batch_size: 1
evaluation:
  k_neighbors: 20
device: cuda
```

**Dataset Config Example** (`configs/datasets/custom_soc.yaml`):
```yaml
dataset_name: custom_soc
data:
  root_dir: ../custom_dataset/
  files: [endpoint_file.json, endpoint_network.json, endpoint_process.json]
format:
  type: json
  schema: elastic
graph:
  node_types: [file, process, network, registry]
  edge_types: [read, write, execute, connect]
preprocessing:
  normalize_features: true
  chunk_size: 10000
```

---

## 🧪 Experiments

### Experiment 1: Model Selection

**Goal**: Find the best model for your data

```bash
python experiments/compare.py \
    --config configs/experiments/compare_all.yaml \
    --dataset custom_soc \
    --output results/model_selection/
```

**Outputs**:
- `comparison_results.csv` - Metrics table
- `comparison_results.json` - Detailed results
- `plots/metrics_comparison.png` - Bar chart
- `plots/performance_vs_time.png` - Scatter plot
- `plots/radar_comparison.png` - Radar chart

### Experiment 2: Hyperparameter Tuning

```bash
# Create configs for different hyperparameters
for hidden in 128 256 512; do
    cp configs/models/magic.yaml configs/models/magic_h${hidden}.yaml
    sed -i "s/num_hidden: 256/num_hidden: ${hidden}/" configs/models/magic_h${hidden}.yaml
    
    python experiments/train.py \
        --model magic \
        --config configs/models/magic_h${hidden}.yaml \
        --dataset custom_soc
done
```

### Experiment 3: Cross-Dataset Evaluation

```bash
# Train on CADETS E3, test on custom data
python experiments/train.py --model magic --dataset cadets_e3
python experiments/evaluate.py --model magic --checkpoint checkpoints/magic/best.pt --dataset custom_soc

# Train on custom data, test on StreamSpot
python experiments/train.py --model magic --dataset custom_soc
python experiments/evaluate.py --model magic --checkpoint checkpoints/magic/best.pt --dataset streamspot
```

---

## 📊 Results

### Benchmark Results on DARPA TC

| Model | CADETS E3 | THEIA E3 | TRACE E3 | Avg. Training Time |
|-------|-----------|----------|----------|-------------------|
| MAGIC | 0.924 | 0.918 | 0.931 | 25 min |
| Kairos | 0.916 | 0.905 | 0.922 | 45 min |
| Orthrus | 0.909 | 0.897 | 0.915 | 35 min |
| ThreaTrace | 0.898 | 0.883 | 0.906 | 18 min |
| Continuum_FL | 0.911 | 0.901 | 0.918 | 30 min |

### Visualization Examples

The framework automatically generates:

1. **ROC Curves**: True Positive Rate vs False Positive Rate
2. **Precision-Recall Curves**: Precision vs Recall trade-offs
3. **Confusion Matrices**: Classification performance breakdown
4. **Training Curves**: Loss and metrics over epochs
5. **Comparison Charts**: Side-by-side model performance

---

## 🔧 Extending the Framework

### Adding a New Model

**Step 1: Create Model Wrapper** (`models/my_model_wrapper.py`)

```python
from models.base_model import BasePIDSModel, ModelRegistry

@ModelRegistry.register('my_model')
class MyModel(BasePIDSModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
        
    def forward(self, batch):
        # Forward pass
        pass
        
    def train_epoch(self, dataloader, optimizer, **kwargs):
        # Training loop
        pass
        
    def evaluate(self, dataloader, **kwargs):
        # Evaluation
        pass
        
    def get_embeddings(self, batch):
        # Extract embeddings
        pass
        
    def save_checkpoint(self, path, **kwargs):
        # Save model
        pass
        
    def load_checkpoint(self, path, **kwargs):
        # Load model
        pass
```

**Step 2: Create Configuration** (`configs/models/my_model.yaml`)

```yaml
model_name: my_model
architecture:
  hidden_dim: 256
  num_layers: 3
training:
  learning_rate: 0.001
  num_epochs: 100
```

**Step 3: Register Model** (`models/__init__.py`)

```python
try:
    from models.my_model_wrapper import MyModel
except ImportError as e:
    print(f"Warning: Could not import MyModel: {e}")
```

**Step 4: Use Your Model**

```bash
python experiments/train.py --model my_model --dataset custom_soc
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CPU vs GPU Usage

**Question**: How do I control whether the framework uses CPU or GPU?

**Answer**:
```bash
# Default: CPU (no GPU required)
./scripts/run_evaluation.sh

# Force GPU (if available)
CUDA_VISIBLE_DEVICES=0 ./scripts/run_evaluation.sh

# Manually specify in Python scripts
python experiments/train.py --device -1  # CPU (default)
python experiments/train.py --device 0   # GPU 0
python experiments/train.py --device 1   # GPU 1

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

**Note**: The framework defaults to CPU to ensure it works on all systems. GPU is automatically used when available and explicitly requested.

#### Issue 2: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Solution 1: Switch to CPU (default)
python experiments/train.py --device -1

# Solution 2: Reduce batch size in config
# Edit configs/models/MODEL.yaml:
training:
  batch_size: 8  # or smaller

# Solution 3: Use gradient checkpointing
training:
  gradient_checkpointing: true
```

#### Issue 3: PyTorch Import Error (MKL Symbol)

**Error**: `ImportError: .../libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`

**Cause**: This is a compatibility issue between PyTorch and Intel MKL threading layers.

**Quick Fix**:
```bash
conda activate pids_framework
./scripts/fix_pytorch_mkl.sh
```

**Manual Solutions**:
```bash
# Option 1: Set environment variable (recommended)
export MKL_THREADING_LAYER=GNU
python -c "import torch; print(torch.__version__)"

# Make it permanent for your environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export MKL_THREADING_LAYER=GNU' > $CONDA_PREFIX/etc/conda/activate.d/mkl_fix.sh

# Option 2: Reinstall compatible MKL
conda install "mkl<2024" -c conda-forge --force-reinstall

# Option 3: Use pip-installed PyTorch
conda uninstall pytorch torchvision torchaudio
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

**Verification**:
```bash
# Test PyTorch installation
python scripts/test_pytorch.py

# Or manually
python -c "import torch; print(f'PyTorch {torch.__version__} working!')"
```

📖 **Detailed Guide**: See [docs/PYTORCH_MKL_FIX.md](docs/PYTORCH_MKL_FIX.md) for comprehensive troubleshooting.

#### Issue 4: Other Import Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
conda activate pids_framework
pip install torch==1.12.1
```

#### Issue 5: JSON Files Too Large

**Error**: `MemoryError: Unable to allocate array`

**Solution**:
```bash
# Use chunked loading
python scripts/preprocess_data.py --chunk-size 5000
```

#### Issue 6: Model Not Found

**Error**: `KeyError: 'magic'`

**Solution**:
```bash
# Reinstall model dependencies
./scripts/install_model_deps.sh --models magic

# Verify model registration
python -c "from models import list_available_models; print(list_available_models())"
```

### Debug Mode

Enable detailed logging:
```bash
export PYTHONPATH=/path/to/PIDS_Comparative_Framework:$PYTHONPATH
export PIDS_LOG_LEVEL=DEBUG

python experiments/train.py --model magic --dataset custom_soc --debug
```

---

## 📚 Documentation

### Main Documentation Files

1. **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide with common workflows
2. **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - Complete framework overview
3. **[FRAMEWORK_GUIDE.md](FRAMEWORK_GUIDE.md)** - Detailed technical documentation
4. **[docs/getting_started.md](docs/getting_started.md)** - Beginner's tutorial
5. **[docs/installation.md](docs/installation.md)** - Detailed installation guide
6. **[docs/datasets.md](docs/datasets.md)** - Data format specifications
7. **[docs/models.md](docs/models.md)** - Model architecture details

### API Reference

See `FRAMEWORK_GUIDE.md` for complete API documentation.

### Help Commands

All scripts have built-in help:
```bash
python scripts/download_weights.py --help
python scripts/preprocess_data.py --help
python experiments/train.py --help
python experiments/evaluate.py --help
python experiments/compare.py --help
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd PIDS_Comparative_Framework

# Install development dependencies
pip install -r requirements/dev.txt

# Run tests
pytest tests/
```

---

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{pids_comparative_framework,
  title = {PIDS Comparative Framework: A Unified Platform for Provenance-based Intrusion Detection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/PIDS_Comparative_Framework}
}
```

### Cite Individual Models

**MAGIC**:
```bibtex
@inproceedings{magic2024,
  title={MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning},
  booktitle={USENIX Security},
  year={2024}
}
```

**Kairos**:
```bibtex
@inproceedings{kairos2024,
  title={Kairos: Practical Intrusion Detection and Investigation using Whole-system Provenance},
  booktitle={IEEE S&P},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This framework integrates multiple PIDS models, each with their own licenses:
- MAGIC: [License](../MAGIC/LICENSE)
- Kairos: [License](../kairos/LICENSE)
- Orthrus: [License](../orthrus/LICENSE)
- ThreaTrace: [License](../threaTrace/LICENSE)

---

## 📞 Support

### Getting Help

- **Documentation**: See [docs/](docs/) folder
- **Issues**: Check [Troubleshooting](#troubleshooting) section
- **Examples**: See [configs/experiments/](configs/experiments/) for templates

### Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

## 🎯 Roadmap

### Current Version: 1.0.0

- ✅ 5 integrated PIDS models
- ✅ Custom SOC data support
- ✅ DARPA TC dataset support
- ✅ Comprehensive evaluation metrics
- ✅ Model comparison framework

### Planned Features (v1.1.0)

- [ ] Web-based dashboard for real-time monitoring
- [ ] Automated hyperparameter optimization
- [ ] Ensemble model support
- [ ] Incremental learning for continuous deployment
- [ ] Integration with SIEM systems (Splunk, ELK)

### Future Enhancements (v2.0.0)

- [ ] Explainable AI features (attack path visualization)
- [ ] Active learning for label-efficient training
- [ ] Multi-host correlation
- [ ] Streaming inference for real-time detection

---

## 🌟 Acknowledgments

This framework builds upon the excellent work of:

- **MAGIC Team** - Masked graph autoencoder approach
- **Kairos Team** - Temporal provenance analysis
- **Orthrus Team** - Multi-decoder architecture
- **ThreaTrace Team** - Scalable graph processing
- **FedML Community** - Federated learning infrastructure

---

<div align="center">

**Made with ❤️ for the Security Research Community**

[⬆ Back to Top](#pids-comparative-framework)

</div>
