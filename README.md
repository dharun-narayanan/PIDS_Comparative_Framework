# PIDS Comparative Framework

<div align="center">

**An Extensible Framework for Provenance-based Intrusion Detection Systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8--3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Quick Start](#-quick-start) | [Installation](#-installation) | [Models](#-supported-models) | [Adding Models](#-adding-new-models) | [Documentation](#-documentation)

</div>

---

## 🎯 Overview

The **PIDS Comparative Framework** is a production-ready, extensible platform for evaluating state-of-the-art Provenance-based Intrusion Detection Systems on custom datasets.

### Key Features

✅ **Extensible Architecture** - Add new models by creating a YAML config file (no code needed)  
✅ **Shared Components** - Reusable encoder/decoder library eliminates code duplication  
✅ **Task-Based Pipeline** - Modular execution with automatic caching  
✅ **Custom Datasets** - Works with any preprocessed provenance data  
✅ **Pretrained Weights** - Use existing checkpoints or train from scratch  
✅ **Multi-Model Comparison** - Evaluate 5+ state-of-the-art models simultaneously  
✅ **Unsupervised Evaluation** - Works without ground truth labels  
✅ **Automatic Anomaly Analysis** - Top anomalies and ensemble consensus  
✅ **CPU-First** - Runs on CPU by default, GPU optional  

### What's New (October 2025 Restructuring)

🚀 **Complete architectural overhaul for extensibility:**

- **Shared Encoders/Decoders** - 11 reusable components (5 encoders + 6 decoders)
- **Model Builder** - Dynamic model construction from YAML configs
- **Per-Model Configs** - Each model has its own `configs/models/{model}.yaml` file  
- **No Wrappers Needed** - `GenericModel` works with any encoder-decoder combination
- **Add Models in Minutes** - Just create a config file, no Python code required
- **Task-Based Pipeline** - 9 modular tasks with automatic caching and dependency management

**Before**: Adding a new model required writing 300+ lines of Python (wrapper class, encoder, decoder)  
**After**: Copy `configs/models/template.yaml`, edit configuration, done! No Python code needed.

**Key Architecture Changes:**
- Removed all model wrapper files (replaced by ModelBuilder)
- Removed model-specific implementations (replaced by shared components)
- Unified evaluation through pipeline-based system
- YAML-driven configuration for maximum flexibility

---

## 📊 Workflow

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ Preprocessed    │ ───> │ Task Pipeline    │ ───> │ Model Builder   │
│ Provenance Data │      │ (9 modular tasks)│      │ (YAML → Model)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                            │
        ┌───────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Shared Components (models/shared_encoders.py + decoders.py) │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   GAT    │  │   SAGE   │  │   Trans  │  │   Time   │      │
│  │ Encoder  │  │ Encoder  │  │ former   │  │ Encoder  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   Edge   │  │   Node   │  │Contrast  │  │ Anomaly  │      │
│  │ Decoder  │  │ Decoder  │  │ Decoder  │  │ Decoder  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│ GenericModel    │ ───> │ Inference        │ ───> │ Metrics &       │
│ (unified API)   │      │ (cached tasks)   │      │ Comparison      │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

**Pipeline Tasks:**
1. `load_data` - Load preprocessed graphs
2. `preprocess` - Extract time windows
3. `build_model` - Construct model from YAML
4. `load_checkpoint` - Load pretrained weights
5. `prepare_dataloaders` - Create batched data loaders
6. `run_inference` - Execute model inference
7. `compute_predictions` - Process model outputs
8. `evaluate_metrics` - Calculate detection metrics
9. `calculate_metrics` - Final metric aggregation

---

## 🚀 Quick Start

### 1. Run Evaluation with Pretrained Weights

```bash
# Evaluate MAGIC model on custom dataset
python experiments/evaluate_pipeline.py \
  --models magic \
  --dataset custom_soc \
  --data-path data/preprocessed/custom_soc \
  --checkpoints-dir checkpoints

# Evaluate multiple models
python experiments/evaluate_pipeline.py \
  --models magic,kairos,orthrus \
  --dataset cadets \
  --data-path data/preprocessed/cadets \
  --device cuda
```

### 2. Add a New Model (No Code Required!)

```bash
# Copy template
cp configs/models/template.yaml configs/models/my_model.yaml

# Edit configuration (choose encoder, decoder, training params)
vim configs/models/my_model.yaml

# Run immediately!
python experiments/evaluate_pipeline.py \
  --models my_model \
  --dataset my_dataset \
  --data-path data/my_dataset
```

### 3. Use Model Programmatically

```python
from models.model_builder import ModelBuilder

# Initialize ModelBuilder
builder = ModelBuilder(config_dir="configs/models")

# Build model with pretrained weights
model = builder.build_model(
    model_name="magic",
    dataset_name="cadets",
    device="cuda"
)

# Run inference
with torch.no_grad():
    predictions = model.decode(model.encode(data))
```

---

## 🏗️ System Architecture

### New Extensible Architecture (October 2025)

```
┌─────────────────────────────────────────────────────────────────┐
│                   PIDS Comparative Framework                    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  configs/models/  (Per-Model YAML Configurations)          │ │
│  │  ├── magic.yaml         ├── orthrus.yaml                   │ │
│  │  ├── kairos.yaml        ├── threatrace.yaml                │ │
│  │  ├── continuum_fl.yaml  └── template.yaml                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  ModelBuilder (models/model_builder.py)                    │ │
│  │  - Load YAML config                                        │ │
│  │  - Construct model from shared components                  │ │
│  │  - Load pretrained weights with fallbacks                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Shared Components                                         │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  Encoders (shared_encoders.py)                       │  │ │
│  │  │  - GATEncoder        - GraphTransformerEncoder       │  │ │
│  │  │  - SAGEEncoder       - TimeEncoder                   │  │ │
│  │  │  - MultiEncoder      - Factory functions             │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  Decoders (shared_decoders.py)                       │  │ │
│  │  │  - EdgeDecoder       - ReconstructionDecoder         │  │ │
│  │  │  - NodeDecoder       - AnomalyDecoder                │  │ │
│  │  │  - ContrastiveDecoder- InnerProductDecoder           │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  GenericModel (wraps any encoder-decoder combination)      │ │
│  │  - Single/multi-encoder support                            │ │
│  │  - Single/multi-decoder support                            │ │
│  │  - Unified forward pass and inference API                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Task-Based Pipeline (pipeline/)                           │ │
│  │  9 modular tasks with automatic caching:                   │ │
│  │  1. load_data         2. preprocess        3. build_model  │ │
│  │  4. load_checkpoint   5. prepare_dataloaders               │ │
│  │  6. run_inference     7. compute_predictions               │ │
│  │  8. evaluate_metrics  9. calculate_metrics                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Benefits

✅ **No Wrappers** - `GenericModel` works with any configuration  
✅ **No Duplication** - Single implementation per encoder/decoder  
✅ **Easy Extension** - Add models via YAML, no Python needed  
✅ **Automatic Construction** - ModelBuilder handles everything  
✅ **Smart Checkpoints** - Tries multiple paths with graceful fallback  
✅ **Cached Execution** - Pipeline tasks cache intermediate results
✅ **Multi-Model Support** - Evaluate multiple models in one run

---

## � Supported Models

The framework currently includes 5 state-of-the-art PIDS models, all configurable via YAML:

| Model | Architecture | Best For | Config File |
|-------|-------------|----------|-------------|
| **MAGIC** | GAT + Edge/Reconstruction | General-purpose detection | `configs/models/magic.yaml` |
| **Kairos** | Transformer + Time Encoder | Temporal attack patterns | `configs/models/kairos.yaml` |
| **Orthrus** | Multi-encoder (Transformer + SAGE) | Multi-objective learning | `configs/models/orthrus.yaml` |
| **ThreaTrace** | Multi-encoder (GAT + SAGE) | Graph clustering-based | `configs/models/threatrace.yaml` |
| **Continuum_FL** | GAT + Federated Learning | Distributed/privacy-preserving | `configs/models/continuum_fl.yaml` |

### Adding Your Own Model

1. Copy the template: `cp configs/models/template.yaml configs/models/your_model.yaml`
2. Configure encoder (GAT, SAGE, Transformer, Time, or Multi-encoder)
3. Configure decoder (Edge, Node, Contrastive, Reconstruction, Anomaly, InnerProduct)
4. Set training/data/inference parameters
5. Add checkpoint paths for your datasets
6. Run: `python experiments/evaluate_pipeline.py --models your_model --dataset your_dataset`

**No Python code needed!** The ModelBuilder dynamically constructs your model from the YAML configuration.

---

## 📂 Directory Structure

```
PIDS_Comparative_Framework/
├── README.md                          # This file - Overview
├── SETUP.md                           # Complete setup guide
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── models/                            # 🧠 Model components (4 files)
│   ├── __init__.py                   # Module exports
│   ├── model_builder.py              # ModelBuilder + GenericModel (514 lines)
│   ├── shared_encoders.py            # 5 encoder types (532 lines)
│   └── shared_decoders.py            # 6 decoder types (651 lines)
│
├── configs/                           # ⚙️ Configuration files
│   └── models/                       # 🆕 Per-model YAML configs
│       ├── magic.yaml
│       ├── kairos.yaml
│       ├── orthrus.yaml
│       ├── threatrace.yaml
│       ├── continuum_fl.yaml
│       └── template.yaml             # 🆕 Template for new models
│
├── pipeline/                          # 🔄 Task-based pipeline (4 files)
│   ├── __init__.py                   # Module exports
│   ├── pipeline_builder.py           # Pipeline construction
│   ├── task_manager.py               # Task orchestration
│   └── task_registry.py              # 9 task definitions (730 lines)
│
├── experiments/                       # 🧪 Experiment scripts (2 files)
│   ├── evaluate_pipeline.py          # Main evaluation script (315 lines)
│   └── train.py                      # Reference training (366 lines)
│
├── data/                              # 📊 Dataset handling
│   └── dataset.py
│
├── data/                              # 📊 Dataset handling
│   └── dataset.py                    # Dataset loading utilities
│
├── utils/                             # 🛠️ Utilities
│   ├── common.py                     # Common utilities
│   ├── metrics.py                    # Evaluation metrics
│   └── visualization.py              # Result visualization
│
├── scripts/                           # 📜 Setup and preprocessing (5 files)
│   ├── setup.sh                      # Environment setup
│   ├── setup_models.py               # Model setup script (871 lines)
│   ├── preprocess_data.py            # Data preprocessing
│   ├── run_evaluation.sh             # Batch evaluation
│   └── verify_installation.py        # Installation verification
│
├── requirements/                      # 📦 Model-specific dependencies
│   ├── magic.txt
│   ├── kairos.txt
│   ├── orthrus.txt
│   ├── threatrace.txt
│   └── continuum_fl.txt
│
├── checkpoints/                       # 💾 Pretrained model weights
├── data/                              # � Preprocessed datasets
└── results/                           # 📈 Evaluation results
```

---

## 📚 Documentation

### Core Documentation
- **[README.md](README.md)** (this file) - Framework overview and quick start
- **[setup.md](setup.md)** - Complete setup guide with detailed instructions

### Configuration Templates
- **[configs/models/template.yaml](configs/models/template.yaml)** - Template for adding new models with all options documented
├── checkpoints/                   # 💾 Pretrained model weights
│   ├── magic/                    # MAGIC checkpoints
│   ├── kairos/                   # Kairos checkpoints
│   ├── orthrus/                  # Orthrus checkpoints
│   ├── threatrace/               # ThreaTrace checkpoints
│   └── continuum_fl/             # Continuum_FL checkpoints
│
├── data/                          # 📁 Data directory
│   ├── custom_soc/               # ← Your custom SOC data
│   ├── cadets_e3/                # DARPA datasets (optional)
│   └── streamspot/               # StreamSpot dataset (optional)
│
└── results/                       # 📈 Evaluation results
    └── evaluation_*/             # Timestamped result directories
```

---

## 🚀 Quick Start

### Prerequisites

- **Conda** (Anaconda or Miniconda) - [Install Conda](https://docs.conda.io/en/latest/miniconda.html)
- **Python 3.8-3.10** (installed via Conda)
- **10GB disk space** (for dependencies and pretrained weights)
- **Git** (for downloading some model weights)

### Installation (5 minutes)

```bash
# Clone the repository
cd /path/to/PIDS_Files/PIDS_Comparative_Framework

# Run automated setup (creates environment, installs dependencies, downloads weights)
./scripts/setup.sh

# Activate environment
conda activate pids_framework

# Verify installation
python scripts/verify_installation.py
```

**Setup script does:**
1. ✅ Creates conda environment from `environment.yml`
2. ✅ Installs PyTorch 1.12.1 with CUDA 11.6 support
3. ✅ Installs DGL 1.0.0 (Deep Graph Library)
4. ✅ Installs PyTorch Geometric + extensions (torch-scatter, torch-sparse, torch-cluster)
5. ✅ Applies MKL threading fix automatically
6. ✅ Creates directory structure
7. ✅ Verifies installation

**Time:** 10-15 minutes (depending on download speed)

### Prepare Your Data

```bash
# 1. Place your JSON logs in custom_dataset/ directory
mkdir -p ../custom_dataset
cp /path/to/your/*.json ../custom_dataset/

# 2. Preprocess data (converts JSON → graph format)
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc

# Output: custom_soc_graph.pkl, custom_soc_features.pt, custom_soc_metadata.json
```

### Evaluate Models

```bash
# Run complete evaluation workflow (automatic anomaly analysis included)
./scripts/run_evaluation.sh

# Or evaluate specific model
./scripts/run_evaluation.sh --model magic

# Or use preprocessed data directly
./scripts/run_evaluation.sh \
    --data-path data/custom_soc \
    --skip-preprocess
```

**The evaluation script automatically:**
1. Runs all models with any available pretrained weights
2. Calculates unsupervised metrics (Score Separation Ratio, anomaly counts)
3. Ranks models by separation ratio
4. Analyzes top anomalies for each model
5. Generates ensemble consensus report

### View Results

```bash
# Check results directory
ls results/evaluation_*/

# View unsupervised metrics comparison
cat results/evaluation_*/comparison_report.json

# View anomaly analysis
cat results/evaluation_*/magic_anomalies.json
cat results/evaluation_*/ensemble_consensus.json

# View per-model evaluation logs
cat results/evaluation_*/magic_evaluation.log
```

**Output includes:**
- Model rankings by separation ratio
- Critical and high-risk anomaly counts
- Top 1000 anomalies per model with scores, timestamps, entity info
- Ensemble consensus: anomalies flagged by multiple models
- Optional supervised metrics (if labels provided)

**That's it!** You've evaluated 5 PIDS models on your custom data with automatic anomaly analysis.

---

## 🧠 Supported Models

### 1. MAGIC (Masked Graph Autoencoder)
- **Paper:** USENIX Security 2024
- **Architecture:** DGL-based graph autoencoder with masking
- **Approach:** Unsupervised learning via masked node/edge reconstruction
- **Weights:** ✅ Auto-downloaded from GitHub
- **Best For:** Large-scale provenance graphs, general-purpose APT detection

### 2. Kairos (Temporal Provenance Analysis)
- **Paper:** IEEE S&P 2024
- **Architecture:** Temporal GNN with database backend
- **Approach:** Time-aware graph neural network with historical context
- **Weights:** ⚠️ Manual download from Google Drive required
- **Best For:** Long-term attack campaigns, temporal anomaly detection

### 3. Orthrus (Multi-Decoder Architecture)
- **Paper:** USENIX Security 2025
- **Architecture:** Contrastive learning with multiple decoders
- **Approach:** High-quality attribution through contrastive learning
- **Weights:** ✅ Auto-downloaded from GitHub or Zenodo
- **Best For:** Attack attribution, high-precision detection

### 4. ThreaTrace (Sketch-based Detection)
- **Paper:** IEEE TIFS 2022
- **Architecture:** Scalable sketch-based representation
- **Approach:** Efficient graph processing via sketching algorithms
- **Weights:** ✅ Auto-downloaded via git sparse-checkout (~500MB)
- **Best For:** Large-scale deployments, resource-constrained environments

### 5. Continuum_FL (Federated Learning)
- **Paper:** Federated Learning Conference
- **Architecture:** Federated learning with GAT and RNN
- **Approach:** Privacy-preserving distributed learning
- **Weights:** ✅ Auto-downloaded from GitHub
- **Best For:** Multi-site deployments, privacy-sensitive environments

---

## 📊 Supported Datasets

### DARPA TC (Transparent Computing)
- **Engagements:** E3, E5
- **Datasets:** CADETS, CLEARSCOPE, THEIA, TRACE
- **Events:** 100M+ system events
- **Format:** JSON (preprocessed available)

### StreamSpot
- **Source:** University of Illinois
- **Events:** 600+ application scenarios
- **Format:** Graph format
- **Use Case:** Anomaly detection benchmarks

### Custom SOC Data
- **Format:** JSON logs (Elastic/ELK or custom schema)
- **Events:** Process, file, network events
- **Size:** Supports 2GB+ files with chunked loading
- **Schema:** Flexible schema mapping

---

## ⚙️ Configuration

Models and datasets are configured via YAML files:

```yaml
# configs/models/magic.yaml
model:
  name: magic
  type: autoencoder
  architecture:
    encoder:
      hidden_dim: 128
      num_layers: 3
      dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100

evaluation:
  batch_size: 64
  k_neighbors: 5
  detection_level: entity
```

Override settings:
```bash
python experiments/evaluate_pipeline.py \
    --models magic \
    --dataset streamspot \
    --data-path data/preprocessed/streamspot \
    --checkpoints-dir checkpoints \
    --batch-size 16
```

---

## 🧪 Usage Examples

### Basic Evaluation

```bash
# Evaluate all models
./scripts/run_evaluation.sh

# Evaluate specific model
./scripts/run_evaluation.sh --model magic
```

### Advanced Evaluation

```bash
# Pipeline evaluation with custom options
python experiments/evaluate_pipeline.py \
    --models magic kairos \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --checkpoints-dir checkpoints \
    --batch-size 16 \
    --device 0 \
    --cache-dir .cache \
    --output-dir results/my_eval
```

### Preprocessing Options

```bash
# Custom time window (1 hour)
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --time-window 3600

# Filter event types
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --event-types process file

# Large dataset optimization
python scripts/preprocess_data.py \
    --input-dir ../custom_dataset \
    --output-dir data/custom_soc \
    --dataset-name custom_soc \
    --chunk-size 50000 \
    --verbose
```

### Training (Reference Only)

**Note:** The framework is primarily designed for evaluation with pretrained weights. Training functionality is provided for reference but requires additional setup.

```bash
# Train MAGIC on custom data (reference only)
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
    --device 0

# Resume training
python experiments/train.py \
    --model magic \
    --checkpoint checkpoints/magic/checkpoint-streamspot.pt \
    --resume
```

### Batch Processing

```bash
# Evaluate multiple datasets
for dataset in dataset1 dataset2 dataset3; do
    ./scripts/run_evaluation.sh \
        --data-path data/$dataset \
        --dataset $dataset \
        --output-dir results/${dataset}_evaluation
done
```

---

## 📈 Evaluation Metrics

### Unsupervised Anomaly Detection Metrics

The framework uses **unsupervised metrics** suitable for unlabeled data. All models are unsupervised anomaly detectors that output anomaly scores:

#### Primary Metrics
- **Score Separation Ratio** (std/mean): Measures how well the model separates normal vs. anomalous behavior
  - Higher ratio = better separation
  - Used for model ranking
  - Threshold-independent metric

- **Anomaly Score Distribution**: Statistical analysis of anomaly scores
  - Mean, median, standard deviation
  - Percentiles (75th, 90th, 95th, 99th)
  - Critical anomalies (>99th percentile)
  - High-risk anomalies (95-99th percentile)

#### Automatic Anomaly Analysis
The framework automatically identifies and analyzes:
- **Top 1000 Anomalies**: Highest scoring events
- **Temporal Patterns**: Time-based analysis
- **Entity Statistics**: Most suspicious processes/files/hosts
- **Attack Patterns**: Edge types, node characteristics
- **Ensemble Consensus**: Anomalies flagged by multiple models

### Supervised Metrics (Optional)
When ground truth labels are available:
- **AUROC** (Area Under ROC Curve): Overall detection performance
- **AUPRC** (Area Under Precision-Recall Curve): Performance with class imbalance
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives

**Note:** Supervised metrics are shown only if labels exist, but are not used for model ranking.

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Conda environment not activated
```bash
Error: Conda environment 'pids_framework' is not activated!

Solution:
conda activate pids_framework
```

#### 2. PyTorch import fails
```bash
OSError: cannot allocate memory in static TLS block

Solution: MKL fix is automatic, but if it persists:
export MKL_THREADING_LAYER=GNU
python -c "import torch; print(torch.__version__)"
```

#### 3. Out of memory
```bash
RuntimeError: CUDA out of memory

Solution: Reduce batch size or use CPU
python experiments/evaluate_pipeline.py --models magic --batch-size 4 --device -1
```

#### 4. Preprocessed data not found
```bash
Error: No preprocessed data found

Solution: Check filename
ls data/custom_soc/custom_soc_graph.pkl
# Or re-run preprocessing with correct dataset name
```

#### 5. Model checkpoint not found
```bash
Error: Checkpoint not found

Solution: Re-download checkpoints
python scripts/setup_models.py --all --force-download
```

**For detailed troubleshooting, see [setup.md](setup.md#troubleshooting)**

---

## 📖 Documentation

- **[setup.md](setup.md)** - Complete installation and usage guide

### Command Reference

| Script | Purpose | Documentation |
|--------|---------|---------------|
| `setup.sh` | Environment setup | [setup.md](setup.md#installation) |
| `setup_models.py` | Download weights | [setup.md](setup.md#model-specific-setup) |
| `preprocess_data.py` | Data preprocessing | [setup.md](setup.md#preparing-custom-data) |
| `run_evaluation.sh` | Evaluation workflow | [setup.md](setup.md#running-evaluation) |
| `verify_installation.py` | Installation checks | [setup.md](setup.md#verification) |
| `analyze_anomalies.py` | Anomaly analysis | Automatically called by `run_evaluation.sh` |

---

## 🔬 Extending the Framework

### Adding a New Model

The framework uses a **YAML-based configuration system** with `ModelBuilder` for easy model addition:

**Step 1: Create Model Configuration**

```yaml
# configs/models/your_model.yaml
model_name: your_model

architecture:
  encoder:
    type: gat  # or sage, graph_transformer, time_encoder
    config:
      input_dim: 128
      hidden_dim: 256
      num_layers: 3
      
  decoder:
    type: edge  # or node, contrastive, reconstruction, anomaly
    config:
      hidden_dim: 256
      output_dim: 2

training:
  batch_size: 64
  learning_rate: 0.001
  num_epochs: 100

data:
  detection_level: entity  # or edge, both
  k_neighbors: 5
```

**Step 2: Add Checkpoint Configuration**

```yaml
checkpoints:
  streamspot:
    path: checkpoints/your_model/streamspot.pt
  darpa_cadets:
    path: checkpoints/your_model/cadets.pt
```

**Step 3: Use It!**

```bash
python experiments/evaluate_pipeline.py \
    --models your_model \
    --dataset streamspot \
    --data-path data/preprocessed/streamspot
```

**That's it!** No Python wrapper code needed. ModelBuilder dynamically constructs your model from the YAML configuration.

**For custom architectures**: Add new encoder/decoder classes to `models/shared_encoders.py` or `models/shared_decoders.py`, then reference them in your YAML config.

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Contribution Areas

- 🆕 Add new PIDS models (just add a YAML config!)
- 📊 Add new datasets
- 🧪 Add new evaluation metrics
- 🏗️ Add new encoder/decoder architectures
- 📝 Improve documentation
- 🐛 Fix bugs
- ⚡ Performance optimizations

---

## 📚 Citation

### Cite This Framework

```bibtex
@software{pids_comparative_framework_2025,
  title = {PIDS Comparative Framework: A Unified Platform for Provenance-based Intrusion Detection},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/PIDS_Comparative_Framework}
}
```

### Cite Individual Models

**MAGIC:**
```bibtex
@inproceedings{magic2024,
  title={MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning},
  booktitle={USENIX Security},
  year={2024}
}
```

**Kairos:**
```bibtex
@inproceedings{kairos2024,
  title={Kairos: Practical Intrusion Detection and Investigation using Whole-system Provenance},
  booktitle={IEEE S&P},
  year={2024}
}
```

**Orthrus:**
```bibtex
@inproceedings{orthrus2025,
  title={Orthrus: High Quality Attribution in Provenance-based Intrusion Detection},
  booktitle={USENIX Security},
  year={2025}
}
```

**ThreaTrace:**
```bibtex
@article{threatrace2022,
  title={Enabling Refinable Cross-Host Attack Investigation with Efficient Data Flow Tagging and Tracking},
  journal={IEEE TIFS},
  year={2022}
}
```

**Continuum_FL:**
```bibtex
@inproceedings{continuum_fl,
  title={Federated Learning for Provenance-based Intrusion Detection},
  booktitle={Federated Learning Conference},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This framework integrates multiple PIDS models, each with their own licenses:
- **MAGIC**: Check [MAGIC repository](https://github.com/FDUDSDE/MAGIC)
- **Kairos**: Check [Kairos repository](https://github.com/ubc-provenance/kairos)
- **Orthrus**: Check [Orthrus repository](https://github.com/ubc-provenance/orthrus)
- **ThreaTrace**: Check [ThreaTrace repository](https://github.com/Provenance-IDS/threaTrace)
- **Continuum_FL**: Check [Continuum_FL repository](https://github.com/kamelferrahi/Continuum_FL)

---

## 🌟 Acknowledgments

This framework builds upon the excellent work of:

- **MAGIC Team** (FDUDSDE) - Masked graph autoencoder approach
- **Kairos Team** (UBC Provenance) - Temporal provenance analysis
- **Orthrus Team** (UBC Provenance) - High-quality attribution
- **ThreaTrace Team** - Scalable sketch-based detection
- **Continuum_FL Team** - Federated learning for PIDS

We thank the authors for making their models available and for advancing the field of provenance-based intrusion detection.

---

## 📞 Support

### Getting Help

- **Documentation**: See [setup.md](setup.md)
- **Troubleshooting**: Check [setup.md Troubleshooting](setup.md#troubleshooting) section
- **Issues**: Open an issue on GitHub with detailed description
- **Examples**: See `configs/` for configuration templates

### Contact

For questions or issues:
- **GitHub Issues**: https://github.com/yourusername/PIDS_Comparative_Framework/issues
- **Email**: your.email@example.com

---

## 🎯 Roadmap

### Current Version: 1.0.0 ✅

- ✅ 5 integrated PIDS models
- ✅ Custom SOC data support
- ✅ DARPA TC dataset support
- ✅ Comprehensive evaluation metrics
- ✅ Model comparison framework
- ✅ CPU and GPU support
- ✅ Automated setup and workflow

### Planned Features (v1.1.0) 🔄

- [ ] Web-based dashboard for real-time monitoring
- [ ] Automated hyperparameter optimization
- [ ] Ensemble model support
- [ ] Incremental learning for continuous deployment
- [ ] Integration with SIEM systems (Splunk, ELK, QRadar)
- [ ] REST API for model serving

### Future Enhancements (v2.0.0) 🚀

- [ ] Explainable AI features (attack path visualization)
- [ ] Active learning for label-efficient training
- [ ] Multi-host correlation analysis
- [ ] Streaming inference for real-time detection
- [ ] Attack scenario simulation
- [ ] Adversarial robustness testing

---

<div align="center">

**Made with ❤️ for the Security Research Community**

If you find this framework useful, please ⭐ star the repository!

[⬆ Back to Top](#pids-comparative-framework)

</div>
