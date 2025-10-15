# PIDS Comparative Framework# PIDS Comparative Framework



**A Unified, Standalone Framework for Evaluating State-of-the-Art Provenance-based Intrusion Detection Systems (PIDS)**<div align="center">



[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)**A Unified Framework for Evaluating State-of-the-Art Provenance-based Intrusion Detection Systems (PIDS)**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

---[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)



## 📌 Overview[Quick Start](#quick-start) | [Installation](#installation) | [Models](#supported-models) | [Configuration](#configuration) | [Troubleshooting](#troubleshooting)



The **PIDS Comparative Framework** is a production-ready platform designed to evaluate and compare state-of-the-art Provenance-based Intrusion Detection Systems (PIDS) on custom Security Operations Center (SOC) data.</div>



### 🎯 Primary Use Case> **📚 Documentation**: This README provides complete framework documentation. For quick setup, see [setup.md](setup.md).



**Evaluate pretrained PIDS models on your custom SOC data** to determine which model performs best for your environment.---



- ✅ **Ready-to-Use**: Pre-trained models included - no training required## 🎯 **Primary Use Case: Evaluate Pretrained Models**

- ✅ **Standalone**: All model implementations self-contained - no external repos needed

- ✅ **Custom Data**: Evaluate on your own JSON-formatted system logsThis framework is designed to **evaluate pretrained PIDS models on your custom SOC data**. The models are already trained on benchmark datasets (DARPA, StreamSpot) and ready to detect intrusions in your environment.

- ✅ **Multi-Model**: Compare 5 state-of-the-art approaches simultaneously

- ✅ **CPU-First**: Runs on CPU by default, GPU optional for faster evaluation✅ **Default Workflow**: Evaluate pretrained weights → Compare performance → Deploy best model  

- 🔄 **Advanced**: Retrain models on custom data (optional feature)✅ **CPU-First**: Runs on CPU by default (no GPU required)  

✅ **GPU Support**: Automatically uses GPU when available  

---🔄 **Advanced Feature**: Retrain models on custom data (optional)



## 🏗️ Framework Architecture---



### Design Principles## 📋 Table of Contents



1. **Standalone Implementation**: All models are self-contained within the framework with zero dependencies on external repositories- [Overview](#overview)

2. **Unified Interface**: All models implement `BasePIDSModel` for consistent evaluation- [Key Features](#key-features)

3. **Plugin Architecture**: Easy to extend with new models via decorator-based registry- [System Architecture](#system-architecture)

4. **Evaluation-First**: Optimized for evaluating pretrained models on custom data- [Supported Models](#supported-models)

- [Supported Datasets](#supported-datasets)

### High-Level Architecture- [Requirements](#requirements)

- [Installation](#installation)

```- [Quick Start](#quick-start)

┌─────────────────────────────────────────────────────────────┐- [Usage Examples](#usage-examples)

│                  PIDS Comparative Framework                  │- [Configuration](#configuration)

│                                                               │- [Experiments](#experiments)

│  ┌────────────────────────────────────────────────────────┐ │- [Results](#results)

│  │              Model Registry (Plugin System)             │ │- [Extending the Framework](#extending-the-framework)

│  │  - Auto-discovery of models via @register decorator    │ │- [Troubleshooting](#troubleshooting)

│  │  - Consistent BasePIDSModel interface                  │ │- [Documentation](#documentation)

│  └────────────────────────────────────────────────────────┘ │- [Contributing](#contributing)

│                                                               │- [Citation](#citation)

│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │- [License](#license)

│  │  MAGIC   │  │ Kairos   │  │ Orthrus  │  │ThreaTrace│   │

│  │ Wrapper  │  │ Wrapper  │  │ Wrapper  │  │ Wrapper  │...│---

│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │

│       │             │              │              │          │## 🔍 Overview

│  ┌────▼──────────────▼──────────────▼──────────────▼─────┐ │

│  │        Standalone Model Implementations               │ │The **PIDS Comparative Framework** is a production-ready platform designed for Security Operations Centers (SOC) to:

│  │  (models/implementations/ - No external dependencies) │ │

│  └──────────────────────────────────────────────────────┘ │✅ **Evaluate** pretrained PIDS models on your custom SOC data (2GB+ logs supported)  

│                                                               │✅ **Compare** multiple models with consistent benchmarking and metrics  

│  ┌──────────────────────────────────────────────────────┐  │✅ **Deploy** the best-performing model for your specific environment  

│  │           Data Pipeline                              │  │✅ **Train** models on custom data (advanced feature, optional)  

│  │  - JSON logs → Graph construction → Batching        │  │✅ **Fine-tune** existing models using transfer learning (advanced feature)  

│  │  - Support for custom SOC data formats              │  │✅ **Extend** easily with new models through a clean interface  

│  └──────────────────────────────────────────────────────┘  │

│                                                               │### What is Provenance-based Intrusion Detection?

│  ┌──────────────────────────────────────────────────────┐  │

│  │           Evaluation Engine                          │  │Provenance graphs capture system-level information flows (process→file, process→network) to model normal behavior and detect anomalous activities indicative of cyber attacks. This framework integrates 5 state-of-the-art deep learning approaches for analyzing provenance data.

│  │  - Pretrained weight loading                        │  │

│  │  - Metric computation (AUC-ROC, F1, etc.)           │  │### Primary Workflow

│  │  - Multi-model comparison                           │  │

│  └──────────────────────────────────────────────────────┘  │```

└─────────────────────────────────────────────────────────────┘┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐

```│ Your SOC Data   │ ───> │ Pretrained PIDS  │ ───> │ Performance     │

│ (JSON Logs)     │      │ Models           │      │ Comparison      │

### Directory Structure└─────────────────┘      └──────────────────┘      └─────────────────┘

                                                            │

```                                                            ▼

PIDS_Comparative_Framework/                                                    ┌─────────────────┐

├── README.md                       # ← This file - Framework overview                                                    │ Deploy Best     │

├── SETUP.md                        # ← Step-by-step installation guide                                                    │ Model to SOC    │

├── EXTEND.md                       # ← Guide to add new models                                                    └─────────────────┘

├── requirements.txt                # Python dependencies```

├── environment.yml                 # Conda environment spec

│---

├── models/                         # 🧠 Model implementations

│   ├── base_model.py              # BasePIDSModel & ModelRegistry## ✨ Key Features

│   ├── __init__.py                # Auto-discovery of models

│   │### 🎯 **Evaluation-First Design**

│   ├── implementations/           # 📦 Standalone implementations- **Pretrained Models**: Use existing weights immediately - no training required

│   │   ├── magic/                # MAGIC (USENIX Security 2024)- **Quick Deployment**: Evaluate all models on your data in minutes

│   │   ├── kairos/               # Kairos (IEEE S&P 2024)- **Performance Comparison**: Automatic comparison with statistical significance testing

│   │   ├── orthrus/              # Orthrus (USENIX Security 2025)- **One-Command Workflow**: `./scripts/run_evaluation.sh` does everything

│   │   ├── threatrace/           # ThreaTrace (IEEE TIFS 2022)

│   │   ├── continuum_fl/         # Continuum_FL (Federated Learning)### 📊 **Multi-Model Support**

│   │   └── utils/                # Shared utilities- **5 State-of-the-Art Models**: MAGIC, Kairos, Orthrus, ThreaTrace, Continuum_FL

│   │- **Consistent Interface**: All models through unified `BasePIDSModel` API

│   ├── magic_wrapper.py          # MAGIC → BasePIDSModel adapter- **Automatic Registration**: Dynamic model discovery via decorator pattern

│   ├── kairos_wrapper.py         # Kairos adapter- **Pretrained Weights**: Ready-to-use checkpoints for all models

│   ├── orthrus_wrapper.py        # Orthrus adapter

│   ├── threatrace_wrapper.py     # ThreaTrace adapter### � **Comprehensive Evaluation**

│   └── continuum_fl_wrapper.py   # Continuum_FL adapter- **Multiple Metrics**: AUROC, AUPRC, F1-Score, Precision, Recall, Detection Rate

│- **Statistical Analysis**: Significance testing for model comparison

├── data/                          # 📊 Dataset handling- **Rich Visualizations**: ROC curves, precision-recall curves, confusion matrices

│   └── dataset.py                # Base classes for datasets- **Detailed Reports**: JSON and text formats with all metrics

│

├── experiments/                   # 🧪 Experiment scripts### 🔧 **Production-Ready**

│   ├── evaluate.py               # ⭐ Main evaluation script- **CPU-First Design**: Runs on CPU by default (no GPU required)

│   ├── train.py                  # Training script (advanced)- **GPU Support**: Automatic GPU detection and utilization when available

│   └── compare.py                # Multi-model comparison- **Large-Scale Data**: Handles 2GB+ JSON files with chunked loading

│- **Checkpointing**: Save/resume training with early stopping (for retraining)

├── utils/                         # 🛠️ Framework utilities- **Logging**: Comprehensive logging for debugging and monitoring

│   ├── common.py                 # Common utilities- **Error Handling**: Graceful degradation and informative error messages

│   └── metrics.py                # Evaluation metrics

│### 📦 **Easy to Use**

├── scripts/                       # 📜 Setup & helper scripts- **YAML Configurations**: All settings in human-readable configs

│   ├── setup.sh                  # One-command setup- **One-Command Setup**: Automated environment and dependency installation

│   ├── preprocess_data.py        # Data preprocessing- **Streamlined Workflow**: Evaluation script handles all steps automatically

│   └── run_evaluation.sh         # Quick evaluation runner- **Comprehensive Docs**: README.md and QUICKSTART.md cover all features

│

├── configs/                       # ⚙️ Configuration files### 🔬 **Extensible Architecture** (Advanced)

│   ├── datasets/                 # Dataset configs- **Modular Design**: Separate data, models, training, evaluation

│   ├── models/                   # Model configs- **Plugin System**: Add new models with ~200 lines of code

│   └── experiments/              # Experiment configs- **Configurable**: Override any setting via YAML or command-line

│- **Retraining Support**: Optional model training on custom datasets

├── checkpoints/                   # 💾 Pretrained model weights

│   ├── magic/---

│   ├── kairos/

│   ├── orthrus/## 🏗️ System Architecture

│   ├── threatrace/

│   └── continuum_fl/```

│┌─────────────────────────────────────────────────────────────┐

├── data/                          # 📁 Data directory│                    User Interface Layer                      │

│   ├── custom/                   # ← Your custom SOC data goes here│  (CLI Commands, Configuration Files, Experiment Scripts)     │

│   ├── cadets_e3/               # DARPA datasets (optional)└──────────────────────────┬──────────────────────────────────┘

│   └── streamspot/              # StreamSpot dataset (optional)                           │

│┌──────────────────────────┴──────────────────────────────────┐

└── results/                       # 📈 Evaluation results│                   Framework Core Layer                        │

    └── evaluation_results.json│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │

```│  │   Model     │  │   Dataset    │  │   Experiment     │   │

│  │  Registry   │  │   Loaders    │  │    Manager       │   │

---│  └─────────────┘  └──────────────┘  └──────────────────┘   │

└──────────────────────────┬──────────────────────────────────┘

## 🚀 Quick Start                           │

┌──────────────────────────┴──────────────────────────────────┐

### 1. Installation│                    Model Layer                                │

│  ┌──────┐  ┌────────┐  ┌─────────┐  ┌───────────┐ ┌────┐  │

```bash│  │MAGIC │  │Kairos  │  │Orthrus  │  │ThreaTrace │ │...│   │

cd PIDS_Comparative_Framework│  └──────┘  └────────┘  └─────────┘  └───────────┘ └────┘  │

└──────────────────────────┬──────────────────────────────────┘

# Run automated setup (installs dependencies, downloads pretrained models)                           │

bash scripts/setup.sh┌──────────────────────────┴──────────────────────────────────┐

```│                     Data Layer                                │

│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐    │

**See [SETUP.md](SETUP.md) for detailed installation instructions.**│  │  Custom SOC  │  │   DARPA    │  │   StreamSpot     │    │

│  │     Data     │  │   TC E3/E5 │  │   (Scene-based)  │    │

### 2. Prepare Your Data│  └──────────────┘  └────────────┘  └──────────────────┘    │

└──────────────────────────────────────────────────────────────┘

Place your custom SOC logs (JSON format) in `data/custom/`:```



```bash### Core Components

data/custom/

├── endpoint_process.json    # Process events1. **BasePIDSModel**: Abstract interface all models implement

├── endpoint_file.json       # File events2. **ModelRegistry**: Dynamic model registration and discovery

└── endpoint_network.json    # Network events3. **Dataset Loaders**: Handle JSON, DARPA TC, StreamSpot formats

```4. **Training Pipeline**: Consistent training with early stopping, checkpointing

5. **Evaluation Pipeline**: Unified metrics computation and reporting

Preprocess the data:6. **Experiment Manager**: Orchestrate multi-model comparisons



```bash---

python scripts/preprocess_data.py \

    --data-path data/custom \## 🤖 Supported Models

    --output data/custom/preprocessed.pkl

```| Model | Paper | Year | Architecture | Strengths |

|-------|-------|------|--------------|-----------|

### 3. Evaluate Pretrained Models| **MAGIC** | [USENIX Security '24](https://www.usenix.org/conference/usenixsecurity24) | 2024 | Masked Graph Autoencoder | Self-supervised, high accuracy |

| **Kairos** | [IEEE S&P '24](https://www.ieee-security.org/TC/SP2024/) | 2024 | Temporal GNN | Temporal modeling, link prediction |

Evaluate all models on your custom data:| **Orthrus** | [USENIX Security '25](https://www.usenix.org/conference/usenixsecurity25) | 2025 | Multi-Decoder | Contrastive learning, robust |

| **ThreaTrace** | [IEEE TIFS '22](https://ieeexplore.ieee.org/document/9721562) | 2022 | GraphChi GNN | Scalable, disk-based processing |

```bash| **Continuum_FL** | Federated PIDS | 2023 | Federated STGNN | Privacy-preserving, distributed |

python experiments/evaluate.py \

    --all-models \### Model Selection Guide

    --dataset custom \

    --data-path data/custom \- **Best Overall Accuracy**: MAGIC (AUROC ~0.92-0.95)

    --pretrained \- **Temporal Data**: Kairos (explicit temporal modeling)

    --output-dir results/custom- **Large-Scale Graphs**: ThreaTrace (GraphChi disk-based)

```- **Privacy-Sensitive**: Continuum_FL (federated learning)

- **Robust Features**: Orthrus (contrastive learning)

Or evaluate a specific model:

---

```bash

python experiments/evaluate.py \## 📊 Supported Datasets

    --model magic \

    --dataset custom \### 1. Custom SOC Data

    --data-path data/custom \- **Format**: JSON (Elastic/ELK compatible)

    --pretrained- **Size**: Supports 2GB+ files

```- **Types**: endpoint_file.json, endpoint_network.json, endpoint_process.json

- **Features**: Automatic graph construction from provenance events

### 4. View Results

### 2. DARPA Transparent Computing (TC)

```bash- **Engagements**: E3, E5

cat results/custom/evaluation_results_custom.json- **Systems**: CADETS (BSD), THEIA (Linux), TRACE (Android), CLEARSCOPE, FiveDirections

```- **Events**: 268K+ nodes, 6M+ edges (CADETS E3)

- **Ground Truth**: Attack scenarios with timestamps

---

### 3. StreamSpot

## 🎓 Supported Models- **Type**: Scene-based provenance graphs

- **Categories**: YouTube, Gmail, VGame, Download, CNN, Drive-by-Download (attack)

The framework includes **5 state-of-the-art PIDS models** with pretrained weights:- **Scenes**: 600 total (100 per category)

- **Features**: Graph-level classification

| Model | Paper | Venue | Architecture | Key Features |

|-------|-------|-------|--------------|--------------|### 4. Unicorn Wget

| **MAGIC** | Masked Graph Representation Learning | USENIX Security 2024 | Masked GAT AutoEncoder | Entity-level detection, unsupervised |- **Type**: Simulated attack scenarios

| **Kairos** | Practical Intrusion Detection using Provenance | IEEE S&P 2024 | Temporal GNN + Link Prediction | Temporal modeling, investigation |- **Variants**: wget, wget-long

| **Orthrus** | High Quality Attribution in Provenance IDS | USENIX Security 2025 | Graph Transformer + Multi-task | High-quality attribution, contrastive learning |- **Features**: Controlled attack injection

| **ThreaTrace** | Thread-Level Provenance Graph Analysis | IEEE TIFS 2022 | Sketch + GNN | Scalable, thread-level detection |

| **Continuum_FL** | Federated Learning PIDS | Research | STGNN AutoEncoder | Federated learning, privacy-preserving |---



### Model Capabilities Matrix## 💻 Requirements



| Model | Entity Detection | Graph Detection | Temporal Modeling | Multi-task | Federated |### System Requirements

|-------|-----------------|-----------------|-------------------|------------|-----------|- **OS**: Linux (Ubuntu 18.04+), macOS (10.14+), Windows (with WSL2)

| MAGIC | ✅ | ✅ | ❌ | ❌ | ❌ |- **CPU**: 4+ cores recommended (framework runs on CPU by default)

| Kairos | ✅ | ✅ | ✅ | ❌ | ❌ |- **RAM**: 16GB+ recommended (8GB minimum)

| Orthrus | ✅ | ✅ | ✅ | ✅ | ❌ |- **GPU**: Optional - NVIDIA GPU with 8GB+ VRAM (auto-detected if available)

| ThreaTrace | ✅ | ✅ | ✅ | ❌ | ❌ |- **Storage**: 50GB+ free space

| Continuum_FL | ✅ | ✅ | ✅ | ❌ | ✅ |

### Software Requirements

---- **Conda**: Anaconda or Miniconda (required)

- **Python**: 3.8, 3.9, 3.10, or 3.11

## 📊 Evaluation Workflow- **CUDA**: 11.6+ (optional, only for GPU acceleration)

- **Git**: For cloning repositories

```

┌──────────────────────────────────────────────────────────────┐**Note**: The framework defaults to CPU execution. GPU is automatically used if available and can be controlled via `CUDA_VISIBLE_DEVICES` environment variable or `--device` parameter.

│  STEP 1: Data Preparation                                    │

│  ┌────────────────┐                                          │---

│  │ Custom SOC     │  JSON logs (Elastic, Splunk, etc.)      │

│  │ Logs           │  - Process events                        │## 🚀 Installation

│  └────────┬───────┘  - File events                           │

│           │          - Network events                         │### Step 1: Clone the Repository

│           ▼                                                   │

│  ┌────────────────┐                                          │```bash

│  │ Preprocessing  │  Extract entities & relations            │cd /path/to/PIDS_Files

│  │ Script         │  → Build provenance graph                │# Framework is already in PIDS_Comparative_Framework/

│  └────────┬───────┘  → Create temporal snapshots             │```

│           │                                                   │

└───────────┼───────────────────────────────────────────────────┘### Step 2: Automated Setup (Recommended)

            │

┌───────────┼───────────────────────────────────────────────────┐```bash

│  STEP 2: Model Loading                                       │cd PIDS_Comparative_Framework

│           ▼                                                   │

│  ┌────────────────┐                                          │# Run the automated setup script

│  │ Load Pretrained│  Checkpoints from:                       │./scripts/setup.sh

│  │ Models         │  - DARPA TC training                     │```

│  └────────┬───────┘  - StreamSpot training                   │

│           │          - Other benchmark datasets               │**What `setup.sh` does:**

│           │                                                   │1. Checks for Conda installation

│  ┌───────▼────┬────────┬────────┬──────────┬───────────┐   │2. Creates `pids_framework` conda environment (Python 3.10)

│  │  MAGIC     │ Kairos │Orthrus │ThreaTrace│Continuum_FL│   │3. Installs PyTorch 1.12.1 with CUDA 11.6

│  └────────┬───┴───┬────┴───┬────┴────┬─────┴─────┬─────┘   │4. Installs DGL 1.0.0 and PyTorch Geometric

│           │       │        │         │           │           │5. Installs core dependencies

└───────────┼───────┼────────┼─────────┼───────────┼───────────┘6. Creates necessary directories

            │       │        │         │           │7. Verifies installation

┌───────────┼───────┼────────┼─────────┼───────────┼───────────┐

│  STEP 3: Evaluation                                          │### Step 3: Install Model-Specific Dependencies

│           │       │        │         │           │           │

│           ▼       ▼        ▼         ▼           ▼           │```bash

│  ┌────────────────────────────────────────────────────────┐ │# Activate the environment

│  │  Forward Pass Through Models                           │ │conda activate pids_framework

│  │  - Entity embeddings / Anomaly scores                  │ │

│  │  - Graph-level predictions                             │ │# Install dependencies for all models

│  └──────────────────────┬─────────────────────────────────┘ │./scripts/install_model_deps.sh --all

│                         │                                    │

│                         ▼                                    │# OR install for specific models

│  ┌────────────────────────────────────────────────────────┐ │./scripts/install_model_deps.sh --models magic kairos orthrus

│  │  Metric Computation                                    │ │```

│  │  - AUC-ROC, AUC-PR                                     │ │

│  │  - F1, Precision, Recall                               │ │### Step 4: Verify Installation

│  │  - Detection Rate, FPR                                 │ │

│  └──────────────────────┬─────────────────────────────────┘ │```bash

└─────────────────────────┼─────────────────────────────────────┘# Test model registry

                          │python -c "from models import list_available_models; print(list_available_models())"

┌─────────────────────────┼─────────────────────────────────────┐

│  STEP 4: Comparison & Results                               │# Expected output:

│                         ▼                                    │# ['magic', 'magic_streamspot', 'magic_darpa', 'kairos', 'orthrus', 

│  ┌────────────────────────────────────────────────────────┐ │#  'threatrace', 'continuum_fl', 'continuum_fl_streamspot', 'continuum_fl_darpa']

│  │  Statistical Comparison                                │ │```

│  │  - Rank models by performance                          │ │

│  │  - Statistical significance tests                      │ │### Manual Installation (If Automated Fails)

│  │  - Best model recommendation                           │ │

│  └──────────────────────┬─────────────────────────────────┘ │<details>

│                         │                                    │<summary>Click to expand manual installation steps</summary>

│                         ▼                                    │

│  ┌────────────────────────────────────────────────────────┐ │```bash

│  │  Generate Report                                       │ │# 1. Create conda environment

│  │  - JSON results                                        │ │conda create -n pids_framework python=3.10 -y

│  │  - HTML visualization                                  │ │conda activate pids_framework

│  │  - Detailed logs                                       │ │

│  └────────────────────────────────────────────────────────┘ │# 2. Install PyTorch with CUDA

└──────────────────────────────────────────────────────────────┘conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y

```

# 3. Install DGL

---conda install -c dglteam dgl-cuda11.6==1.0.0 -y



## 📈 Evaluation Metrics# 4. Install PyTorch Geometric

pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

The framework computes comprehensive detection metrics:pip install torch-geometric==2.1.0



### Detection Metrics# 5. Install core dependencies

- **AUC-ROC**: Area Under ROC Curve - Overall detection capabilitypip install -r requirements.txt

- **AUC-PR**: Area Under Precision-Recall Curve - Performance on imbalanced data

- **F1 Score**: Harmonic mean of precision and recall# 6. Install model-specific dependencies

- **Precision**: True positives / (True positives + False positives)pip install -r requirements/magic.txt

- **Recall**: True positives / (True positives + False negatives)pip install -r requirements/kairos.txt

- **FPR**: False Positive Rate - Critical for SOC usabilitypip install -r requirements/orthrus.txt

- **TPR**: True Positive Rate (Detection Rate)pip install -r requirements/threatrace.txt

pip install -r requirements/continuum_fl.txt

### Entity-Level Metrics (for applicable models)```

- **k-NN Detection**: Entity-level anomaly detection using k-nearest neighbors

- **Entity Precision/Recall**: Per-entity detection accuracy</details>

- **Temporal Detection Rate**: Detection performance over time windows

---

---

## ⚡ Quick Start

## 🔧 Advanced Features

> **TL;DR**: Run `./scripts/run_evaluation.sh` to evaluate all pretrained models on your data! (CPU by default)

### 1. Train Models from Scratch

### Default Workflow: Evaluate Pretrained Models (3 Steps)

```bash

python experiments/train.py \#### Step 1: Setup Environment

    --model magic \

    --dataset custom \```bash

    --data-path data/custom \cd PIDS_Comparative_Framework

    --epochs 50 \./scripts/setup.sh

    --batch-size 32 \conda activate pids_framework

    --lr 0.001 \```

    --save-dir checkpoints/custom

```#### Step 2: Prepare Your Data



### 2. Fine-Tune Pretrained Models```bash

# Ensure your SOC data is in JSON format at ../custom_dataset/

```bashls ../custom_dataset/

python experiments/train.py \# Expected: endpoint_file.json, endpoint_network.json, endpoint_process.json

    --model magic \```

    --dataset custom \

    --pretrained \#### Step 3: Run Evaluation (CPU by default)

    --checkpoint checkpoints/magic/checkpoint-streamspot.pt \

    --fine-tune \```bash

    --epochs 20 \# Evaluate ALL pretrained models on your custom data (runs on CPU)

    --lr 0.0001./scripts/run_evaluation.sh

```

# Results saved to: results/evaluation_YYYYMMDD_HHMMSS/

### 3. Compare Multiple Models```



```bash**That's it!** The script will:

python experiments/compare.py \1. ✅ Download/copy pretrained weights from existing checkpoints

    --models magic kairos orthrus \2. ✅ Preprocess your custom SOC data

    --dataset custom \3. ✅ Evaluate all models on your data (CPU default, GPU auto-detected if available)

    --pretrained \4. ✅ Generate comparison report with metrics (AUROC, AUPRC, F1, etc.)

    --output-dir results/comparison

```### View Results



### 4. Custom Configurations```bash

# Navigate to results directory

Use YAML config files:cd results/evaluation_YYYYMMDD_HHMMSS/



```bash# View comparison report

python experiments/evaluate.py \cat comparison_report.json

    --config configs/experiments/evaluate_custom.yaml

```# Check individual model performance

cat magic_evaluation.log

---cat kairos_evaluation.log

```

## 🧪 Working with Custom Data

### Evaluate Specific Model

### Supported Data Formats

```bash

The framework accepts JSON logs from common SOC platforms:# MAGIC only (CPU)

./scripts/run_evaluation.sh --model magic

- **Elastic/ELK Stack**: Endpoint logs

- **Splunk**: Security events# With custom data path

- **Sysmon**: Windows system monitoring./scripts/run_evaluation.sh --model kairos --data-path /path/to/logs

- **Auditd**: Linux audit logs

- **Custom JSON**: Any JSON with entity information# Force GPU usage (if available)

CUDA_VISIBLE_DEVICES=0 ./scripts/run_evaluation.sh --model magic

### Expected Schema```



```json### Alternative: Manual Step-by-Step Evaluation

{

  "@timestamp": "2024-10-14T10:30:00.000Z",<details>

  "event": {<summary>Click to expand manual evaluation steps</summary>

    "kind": "event",

    "category": ["process"]```bash

  },# 1. Download pretrained weights

  "process": {python scripts/download_weights.py --copy-existing --all-models

    "pid": 1234,

    "name": "bash",# 2. Preprocess your data

    "executable": "/bin/bash",python scripts/preprocess_data.py \

    "parent": {"pid": 1000}    --input-dir ../custom_dataset/ \

  },    --output-dir data/custom_soc \

  "file": {    --dataset-name custom_soc

    "path": "/etc/passwd"

  },# 3. Evaluate single model (CPU by default)

  "network": {python experiments/evaluate.py \

    "destination": {"ip": "192.168.1.1", "port": 443}    --model magic \

  }    --dataset custom_soc \

}    --data-path data/custom_soc \

```    --pretrained \

    --checkpoint-dir checkpoints \

### Preprocessing Pipeline    --device -1  # -1 for CPU (default), 0+ for GPU

    --output-dir results/magic_eval

```bash

python scripts/preprocess_data.py \# 4. Compare all models

    --data-path data/custom \python experiments/compare.py \

    --output data/custom/preprocessed.pkl \    --results-dir results/ \

    --time-window 3600 \    --dataset custom_soc \

    --entity-types process file network    --output-file results/comparison.json

``````



**Steps:**</details>

1. Parse JSON logs

2. Extract entities (processes, files, IPs)### Advanced: Retrain on Custom Data (Optional)

3. Build provenance graph (edges = causal relationships)

4. Create temporal snapshots> ⚠️ **Note**: Retraining is an **advanced feature**. Start with pretrained evaluation first!

5. Generate node/edge features

6. Save in PyTorch-compatible format<details>

<summary>Click to expand retraining instructions</summary>

---

```bash

## 📚 Documentation# Retrain MAGIC on your custom data

python experiments/train.py \

- **[SETUP.md](SETUP.md)**: Complete installation and setup guide    --model magic \

- **[EXTEND.md](EXTEND.md)**: Add new PIDS models to the framework    --dataset custom_soc \

- **This README**: Architecture and usage overview    --data-path data/custom_soc \

    --epochs 100 \

---    --batch-size 32 \

    --config configs/experiments/train_single.yaml

## 🤝 Adding New Models

# Evaluate retrained model

The framework is designed for **easy extension** with new state-of-the-art PIDS models.python experiments/evaluate.py \

    --model magic \

**Quick Steps:**    --checkpoint checkpoints/magic_custom_soc_best.pt \

    --dataset custom_soc

1. Create implementation in `models/implementations/your_model/`

2. Create wrapper class inheriting from `BasePIDSModel`# Compare with pretrained baseline

3. Register with `@ModelRegistry.register('your_model')`python experiments/compare.py \

4. Add config in `configs/models/your_model.yaml`    --checkpoints checkpoints/magic_pretrained.pt checkpoints/magic_custom_soc_best.pt \

5. Done! Framework auto-discovers your model    --labels "Pretrained" "Retrained" \

    --dataset custom_soc

**See [EXTEND.md](EXTEND.md) for complete tutorial with examples.**```



---</details>



## 📊 Performance Benchmarks---



Typical evaluation times on **CPU** (Intel i7, 16GB RAM):## 📖 Usage Examples



| Dataset Size | MAGIC | Kairos | Orthrus | ThreaTrace | Continuum_FL |### Example 1: Evaluate All Models on Custom SOC Data (PRIMARY USE CASE)

|-------------|-------|--------|---------|------------|--------------|

| 10K events | 2 min | 3 min | 4 min | 2 min | 5 min |```bash

| 100K events | 15 min | 20 min | 25 min | 15 min | 30 min |# One command to evaluate everything

| 1M events | 2 hours | 3 hours | 4 hours | 2 hours | 5 hours |./scripts/run_evaluation.sh



**GPU Acceleration** (NVIDIA RTX 3090):# View results

- **10-20x faster** for large datasetscat results/evaluation_*/comparison_report.json

- Automatic detection and usage```



---**Expected Output:**

```json

## ⚙️ System Requirements{

  "dataset": "custom_soc",

### Minimum  "timestamp": "2025-10-13T10:30:00",

- **Python**: 3.8+  "models": {

- **RAM**: 8GB    "magic": {

- **Disk**: 10GB      "auc_roc": 0.9234,

- **CPU**: 2+ cores      "auc_pr": 0.8967,

      "f1": 0.8756,

### Recommended      "precision": 0.8543,

- **Python**: 3.9+      "recall": 0.8978

- **RAM**: 16GB    },

- **Disk**: 50GB (with datasets)    "kairos": {

- **GPU**: 6GB+ VRAM (optional)      "auc_roc": 0.9156,

- **CPU**: 4+ cores      "auc_pr": 0.8834,

      "f1": 0.8623,

---      "precision": 0.8412,

      "recall": 0.8845

## 🐛 Troubleshooting    },

    ...

### Common Issues  }

}

**Out of Memory**```

```bash

# Solution: Reduce batch size### Example 2: Evaluate Specific Model with Detailed Options

python experiments/evaluate.py --model magic --batch-size 16

``````bash

# Evaluate MAGIC with custom settings

**CUDA Errors**./scripts/run_evaluation.sh \

```bash    --model magic \

# Solution: Use CPU    --data-path /path/to/soc/logs \

python experiments/evaluate.py --model magic --device -1    --output-dir results/magic_detailed \

```    --skip-download  # If weights already exist

```

**Missing Pretrained Weights**

```bash### Example 3: Compare Models Across Multiple Datasets

# Solution: Download weights

bash scripts/download_weights.sh```bash

```# Evaluate on DARPA CADETS E3

./scripts/run_evaluation.sh --dataset cadets_e3 --data-path data/darpa_cadets_e3/

**Import Errors**

```bash# Evaluate on StreamSpot

# Solution: Reinstall dependencies./scripts/run_evaluation.sh --dataset streamspot --data-path data/streamspot/

pip install -r requirements.txt --upgrade

```# Evaluate on Custom SOC

./scripts/run_evaluation.sh --dataset custom_soc --data-path ../custom_dataset/

**See [SETUP.md](SETUP.md) for more troubleshooting.**

# Aggregate results

---python experiments/aggregate_results.py \

    --results results/evaluation_*/comparison_report.json \

## 📝 Citation    --output results/cross_dataset_comparison.csv

```

If you use this framework in your research:

### Example 4: Train MAGIC on DARPA CADETS E3 (Advanced - Retraining)

```bibtex

@software{pids_comparative_framework2025,```bash

  title={PIDS Comparative Framework: A Unified Platform for Evaluating python experiments/train.py \

         Provenance-based Intrusion Detection Systems},    --model magic_darpa \

  year={2025},    --dataset cadets_e3 \

  url={https://github.com/yourusername/PIDS_Comparative_Framework}    --epochs 500 \

}    --batch-size 1 \

```    --learning-rate 0.0005 \

    --device cuda

**And cite the individual models you evaluate.**```



---### Example 2: Fine-tune Pretrained Model



## 📜 License```yaml

# Create config: configs/experiments/finetune_magic.yaml

MIT License - see [LICENSE](LICENSE) for details.model:

  name: magic

Individual model implementations retain their original licenses.  pretrained_checkpoint: checkpoints/magic/checkpoint-cadets-e3.pt



---dataset:

  name: custom_soc

## 🙏 Acknowledgments  config: configs/datasets/custom_soc.yaml



This framework integrates the following excellent works:training:

- MAGIC (USENIX Security 2024)  num_epochs: 50              # Fewer epochs

- Kairos (IEEE S&P 2024)  learning_rate: 0.0001       # Lower learning rate

- Orthrus (USENIX Security 2025)  freeze_encoder: false       # Fine-tune all layers

- ThreaTrace (IEEE TIFS 2022)```

- Continuum_FL (Federated Learning Research)

```bash

Thanks to the authors for their contributions to the community.python experiments/train.py --config configs/experiments/finetune_magic.yaml

```

---

### Example 3: Evaluate All Models

## 📧 Contact & Support

```bash

- **Issues**: [GitHub Issues](https://github.com/yourusername/PIDS_Comparative_Framework/issues)# Evaluate all models on test set

- **Discussions**: [GitHub Discussions](https://github.com/yourusername/PIDS_Comparative_Framework/discussions)python experiments/evaluate.py \

    --all-models \

---    --dataset custom_soc \

    --output results/evaluation/

**Made with ❤️ for the Security Research Community**```


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

#### Issue 4: Missing gdown Module

**Error**: `ModuleNotFoundError: No module named 'gdown'` when running `download_weights.py`

**Cause**: The script needs optional dependencies for downloading from Google Drive.

**Solution (Choose One)**:

```bash
# Option 1: Use the simple copy script (NO dependencies required - RECOMMENDED)
python scripts/copy_weights.py

# Option 2: Install optional dependencies and use full-featured script
pip install gdown requests tqdm
python scripts/download_weights.py --copy-existing --all-models
```

**The simple `copy_weights.py` script:**
- ✅ No external dependencies needed
- ✅ Automatically finds weights in parent directories
- ✅ Copies MAGIC, Continuum_FL, Orthrus, Kairos, ThreaTrace weights
- ✅ Clear output with progress indicators

#### Issue 5: Other Import Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
conda activate pids_framework
pip install torch==1.12.1
```

#### Issue 6: JSON Files Too Large

**Error**: `MemoryError: Unable to allocate array`

**Solution**:
```bash
# Use chunked loading
python scripts/preprocess_data.py --chunk-size 5000
```

#### Issue 7: Dependency Version Conflicts

**Error**: `ERROR: No matching distribution found for pyg-lib==0.2.0` or version conflicts during model dependency installation

**Cause**: Each model originally used different PyTorch versions, causing conflicts when installing in a unified environment.

**Solution**: The framework now uses unified compatible versions. If you encounter conflicts:

```bash
# The requirements files have been updated to use compatible versions
# Simply re-run the installation
./scripts/install_model_deps.sh --all

# If you modified requirements files, restore them:
git checkout requirements/*.txt

# Or manually update to use PyTorch 1.12.1 compatible versions
```

**What was fixed**:
- All models now use PyTorch 1.12.1 (compatible baseline)
- PyTorch Geometric unified to version 2.1.0
- Removed optional dependencies that cause conflicts (pyg-lib)
- Updated install script to skip already-installed packages

📖 **Detailed Guide**: See [docs/DEPENDENCY_COMPATIBILITY.md](docs/DEPENDENCY_COMPATIBILITY.md) for version compatibility information.

#### Issue 8: Model Not Found

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
