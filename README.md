# PIDS Comparative Framework# PIDS Comparative Framework



A unified framework for evaluating Provenance-based Intrusion Detection Systems (PIDS) across multiple models and datasets.<div align="center">



## Overview**An Extensible Framework for Provenance-based Intrusion Detection Systems**



This framework provides a modular and extensible platform for:[![Python 3.8+](https://img.shields.io/badge/python-3.8--3.10-blue.svg)](https://www.python.org/downloads/)

- Training and evaluating multiple PIDS models (MAGIC, Kairos, Orthrus, ThreaTrace, Continuum_FL)[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

- Supporting custom datasets with standardized preprocessing[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

- Comparing model performance across different datasets

- Reproducible experiments with consistent evaluation metrics[Quick Start](#-quick-start) | [Installation](#-installation) | [Models](#-supported-models) | [Adding Models](#-adding-new-models) | [Documentation](#-documentation)



## Key Features</div>



### Configuration-Driven Model Building---

- **ModelBuilder**: Dynamically constructs models from YAML configurations

- **Shared Components**: Reusable encoders and decoders across all models## 🎯 Overview

- **Zero Code Changes**: Add new models by creating YAML config files

The **PIDS Comparative Framework** is a production-ready, extensible platform for evaluating state-of-the-art Provenance-based Intrusion Detection Systems on custom datasets.

### Modular Pipeline System

The framework implements a 9-task pipeline with automatic caching:### Key Features

1. **load_data**: Load raw provenance data

2. **preprocess**: Clean and normalize data✅ **Extensible Architecture** - Add new models by creating a YAML config file (no code needed)  

3. **build_graphs**: Construct provenance graphs✅ **Shared Components** - Reusable encoder/decoder library eliminates code duplication  

4. **extract_features**: Generate node/edge features✅ **Task-Based Pipeline** - Modular execution with automatic caching  

5. **split_data**: Create train/val/test splits✅ **Custom Datasets** - Works with any preprocessed provenance data  

6. **prepare_model_input**: Format data for model consumption✅ **Pretrained Weights** - Use existing checkpoints or train from scratch  

7. **run_inference**: Execute model predictions✅ **Multi-Model Comparison** - Evaluate 5+ state-of-the-art models simultaneously  

8. **process_predictions**: Post-process model outputs✅ **CPU-First** - Runs on CPU by default, GPU optional  

9. **calculate_metrics**: Compute evaluation metrics

### What's New (October 2025 Restructuring)

### Supported Models

1. **MAGIC**: Multi-level Anomaly Graph Intrusion Clustering🚀 **Complete architectural overhaul for extensibility:**

2. **Kairos**: Temporal Graph Neural Network

3. **Orthrus**: Dual-headed detection system- **Shared Encoders/Decoders** - 11 reusable components (5 encoders + 6 decoders)

4. **ThreaTrace**: Thread-level provenance analysis- **Model Builder** - Dynamic model construction from YAML configs

5. **Continuum_FL**: Federated learning approach- **Per-Model Configs** - Each model has its own `configs/models/{model}.yaml` file  

- **No Wrappers Needed** - `GenericModel` works with any encoder-decoder combination

### Supported Datasets- **Add Models in Minutes** - Just create a config file, no Python code required

- **StreamSpot**: Streaming system call graphs- **Task-Based Pipeline** - 9 modular tasks with automatic caching and dependency management

- **DARPA TC**: Transparent Computing dataset (Cadets, ClearScope, Theia, Trace)

- **Unicorn**: Attack scenario dataset**Before**: Adding a new model required writing 300+ lines of Python (wrapper class, encoder, decoder)  

- **Custom datasets**: Easy integration via YAML configuration**After**: Copy `configs/models/template.yaml`, edit configuration, done! No Python code needed.



## Installation**Key Architecture Changes:**

- Removed all model wrapper files (replaced by ModelBuilder)

### Quick Start- Removed model-specific implementations (replaced by shared components)

```bash- Unified evaluation through pipeline-based system

# Clone the repository- YAML-driven configuration for maximum flexibility

git clone <repository_url>

cd PIDS_Comparative_Framework---



# Install dependencies## 📊 Workflow

pip install -r requirements.txt

```

# Or using conda┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐

conda env create -f environment.yml│ Preprocessed    │ ───> │ Task Pipeline    │ ───> │ Model Builder   │

conda activate pids_framework│ Provenance Data │      │ (9 modular tasks)│      │ (YAML → Model)  │

```└─────────────────┘      └──────────────────┘      └─────────────────┘

                                                            │

### Setup Models        ┌───────────────────────────────────────────────────┘

```bash        │

# Download and setup all model implementations        ▼

python scripts/setup_models.py --all┌──────────────────────────────────────────────────────────────┐

│  Shared Components (models/shared_encoders.py + decoders.py) │

# Or setup specific models│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │

python scripts/setup_models.py --models magic kairos│  │   GAT    │  │   SAGE   │  │   Trans  │  │   Time   │    │

│  │ Encoder  │  │ Encoder  │  │ former   │  │ Encoder  │    │

# Verify installation│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │

python scripts/verify_installation.py│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │

```│  │   Edge   │  │   Node   │  │Contrast  │  │ Anomaly  │    │

│  │ Decoder  │  │ Decoder  │  │ Decoder  │  │ Decoder  │    │

## Usage│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │

└──────────────────────────────────────────────────────────────┘

### Running Evaluation Pipeline        │

        ▼

The primary way to use the framework is through the evaluation pipeline:┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐

│ GenericModel    │ ───> │ Inference        │ ───> │ Metrics &       │

```bash│ (unified API)   │      │ (cached tasks)   │      │ Comparison      │

# Evaluate a specific model on a dataset└─────────────────┘      └──────────────────┘      └─────────────────┘

python experiments/evaluate_pipeline.py \```

    --model magic \

    --dataset streamspot \**Pipeline Tasks:**

    --split test1. `load_data` - Load preprocessed graphs

2. `preprocess` - Extract time windows

# Run with specific checkpoint3. `build_model` - Construct model from YAML

python experiments/evaluate_pipeline.py \4. `load_checkpoint` - Load pretrained weights

    --model kairos \5. `prepare_dataloaders` - Create batched data loaders

    --dataset cadets \6. `run_inference` - Execute model inference

    --checkpoint checkpoints/kairos_best.pt7. `compute_predictions` - Process model outputs

8. `evaluate_metrics` - Calculate detection metrics

# Force re-computation (skip cache)9. `calculate_metrics` - Final metric aggregation

python experiments/evaluate_pipeline.py \

    --model orthrus \---

    --dataset theia \

    --skip-cache## 🚀 Quick Start

```

### 1. Run Evaluation with Pretrained Weights

### Training Models

```bash

⚠️ **Important**: The `train.py` script is a **reference implementation** provided as an example of how to integrate models with the framework. Each PIDS model has its own specific training requirements, hyperparameters, and procedures that are optimized for that model.# Evaluate MAGIC model on custom dataset

python experiments/evaluate_pipeline.py \

**For actual training**, use the original training scripts included with each model's implementation (located in their respective directories like `MAGIC/`, `kairos/`, etc.).  --models magic \

  --dataset custom_soc \

The reference training script can be used as a starting point:  --data-path data/preprocessed/custom_soc \

  --checkpoints-dir checkpoints

```bash

# Reference training example (modify based on model requirements)# Evaluate multiple models

python experiments/train.py \python experiments/evaluate_pipeline.py \

    --model magic \  --models magic,kairos,orthrus \

    --dataset streamspot \  --dataset cadets \

    --epochs 50 \  --data-path data/preprocessed/cadets \

    --batch-size 32  --device cuda

``````



### Adding Custom Datasets### 2. Add a New Model (No Code Required!)



1. **Create dataset configuration**:```bash

```yaml# Copy template

# configs/datasets/my_dataset.yamlcp configs/models/template.yaml configs/models/my_model.yaml

name: my_dataset

type: provenance_graph# Edit configuration (choose encoder, decoder, training params)

data_dir: data/my_datasetvim configs/models/my_model.yaml



preprocessing:# Run immediately!

  node_types: [process, file, socket]python experiments/evaluate_pipeline.py \

  edge_types: [read, write, execute]  --models my_model \

    --dataset my_dataset \

features:  --data-path data/my_dataset

  node_features:```

    - name

    - timestamp### 3. Use Model Programmatically

    - attributes

``````python

from models.model_builder import ModelBuilder

2. **Run pipeline**:

```bash# Initialize ModelBuilder

python experiments/evaluate_pipeline.py \builder = ModelBuilder(config_dir="configs/models")

    --model magic \

    --dataset my_dataset \# Build model with pretrained weights

    --config configs/datasets/my_dataset.yamlmodel = builder.build_model(

```    model_name="magic",

    dataset_name="cadets",

## Project Structure    device="cuda"

)

```

PIDS_Comparative_Framework/# Run inference

├── configs/                    # Configuration fileswith torch.no_grad():

│   ├── datasets/              # Dataset configurations    predictions = model.decode(model.encode(data))

│   ├── experiments/           # Experiment configurations```

│   └── models/                # Model architecture definitions (YAML)

│       ├── magic.yaml         # MAGIC model configuration---

│       ├── kairos.yaml        # Kairos model configuration

│       ├── orthrus.yaml       # Orthrus model configuration## 🏗️ System Architecture

│       ├── threatrace.yaml    # ThreaTrace model configuration

│       ├── continuum_fl.yaml  # Continuum_FL model configuration### New Extensible Architecture (October 2025)

│       └── template.yaml      # Template for new models

│```

├── models/                    # Core model system (4 files)┌─────────────────────────────────────────────────────────────────┐

│   ├── __init__.py           # Module exports│                   PIDS Comparative Framework                     │

│   ├── model_builder.py      # ModelBuilder + GenericModel class│                                                                   │

│   ├── shared_encoders.py    # 5 reusable encoder types│  ┌────────────────────────────────────────────────────────────┐ │

│   └── shared_decoders.py    # 6 reusable decoder types│  │  configs/models/  (Per-Model YAML Configurations)          │ │

││  │  ├── magic.yaml         ├── orthrus.yaml                   │ │

├── pipeline/                  # Pipeline system (4 files)│  │  ├── kairos.yaml        ├── threatrace.yaml                │ │

│   ├── __init__.py           # Module exports│  │  ├── continuum_fl.yaml  └── template.yaml                  │ │

│   ├── pipeline_builder.py   # Pipeline orchestration│  └────────────────────────────────────────────────────────────┘ │

│   ├── task_manager.py       # Task execution and caching│                            ↓                                      │

│   └── task_registry.py      # Task definitions (9 tasks)│  ┌────────────────────────────────────────────────────────────┐ │

││  │  ModelBuilder (models/model_builder.py)                    │ │

├── experiments/               # Evaluation scripts (2 files)│  │  - Load YAML config                                         │ │

│   ├── evaluate_pipeline.py  # Main evaluation script│  │  - Construct model from shared components                   │ │

│   └── train.py              # Reference training implementation│  │  - Load pretrained weights with fallbacks                   │ │

││  └────────────────────────────────────────────────────────────┘ │

├── data/                      # Data handling│                            ↓                                      │

│   └── dataset.py            # Dataset loading utilities│  ┌────────────────────────────────────────────────────────────┐ │

││  │  Shared Components                                          │ │

├── utils/                     # Utility functions│  │  ┌──────────────────────────────────────────────────────┐  │ │

│   ├── common.py             # Common utilities│  │  │  Encoders (shared_encoders.py)                       │  │ │

│   ├── metrics.py            # Evaluation metrics│  │  │  - GATEncoder        - GraphTransformerEncoder       │  │ │

│   └── visualization.py      # Result visualization│  │  │  - SAGEEncoder       - TimeEncoder                    │  │ │

││  │  │  - MultiEncoder      - Factory functions              │  │ │

├── scripts/                   # Setup and preprocessing (5 files)│  │  └──────────────────────────────────────────────────────┘  │ │

│   ├── setup.sh              # Environment setup│  │  ┌──────────────────────────────────────────────────────┐  │ │

│   ├── setup_models.py       # Model setup script│  │  │  Decoders (shared_decoders.py)                       │  │ │

│   ├── preprocess_data.py    # Data preprocessing│  │  │  - EdgeDecoder       - ReconstructionDecoder         │  │ │

│   ├── run_evaluation.sh     # Batch evaluation│  │  │  - NodeDecoder       - AnomalyDecoder                │  │ │

│   └── verify_installation.py # Installation verification│  │  │  - ContrastiveDecoder- InnerProductDecoder           │  │ │

││  │  └──────────────────────────────────────────────────────┘  │ │

├── requirements/              # Model-specific dependencies│  └────────────────────────────────────────────────────────────┘ │

│   ├── magic.txt│                            ↓                                      │

│   ├── kairos.txt│  ┌────────────────────────────────────────────────────────────┐ │

│   ├── orthrus.txt│  │  GenericModel (wraps any encoder-decoder combination)     │ │

│   ├── threatrace.txt│  │  - Single/multi-encoder support                            │ │

│   └── continuum_fl.txt│  │  - Single/multi-decoder support                            │ │

││  │  - Unified forward pass and inference API                  │ │

├── README.md                  # This file│  └────────────────────────────────────────────────────────────┘ │

├── SETUP.md                   # Detailed setup guide│                            ↓                                      │

├── requirements.txt           # Python dependencies│  ┌────────────────────────────────────────────────────────────┐ │

└── environment.yml            # Conda environment│  │  Task-Based Pipeline (pipeline/)                           │ │

```│  │  9 modular tasks with automatic caching:                   │ │

│  │  1. load_data         2. preprocess        3. build_model    │ │

## Architecture│  │  4. load_checkpoint   5. prepare_dataloaders               │ │

│  │  6. run_inference     7. compute_predictions               │ │

### ModelBuilder System│  │  8. evaluate_metrics  9. calculate_metrics                 │ │

│  └────────────────────────────────────────────────────────────┘ │

The framework uses a configuration-driven approach to construct models dynamically from YAML files:└─────────────────────────────────────────────────────────────────┘

```

```python

from models import ModelBuilder### Key Benefits



# Initialize builder with config directory✅ **No Wrappers** - `GenericModel` works with any configuration  

builder = ModelBuilder(config_dir="configs/models")✅ **No Duplication** - Single implementation per encoder/decoder  

✅ **Easy Extension** - Add models via YAML, no Python needed  

# Build model from YAML configuration✅ **Automatic Construction** - ModelBuilder handles everything  

model = builder.build_model("magic")✅ **Smart Checkpoints** - Tries multiple paths with graceful fallback  

✅ **Cached Execution** - Pipeline tasks cache intermediate results

# Model is constructed with shared components and ready for use✅ **Multi-Model Support** - Evaluate multiple models in one run

outputs = model(input_data)

```---



### Model Configuration (YAML)## � Supported Models



Models are defined declaratively in YAML files:The framework currently includes 5 state-of-the-art PIDS models, all configurable via YAML:



```yaml| Model | Architecture | Best For | Config File |

# configs/models/magic.yaml|-------|-------------|----------|-------------|

name: magic| **MAGIC** | GAT + Edge/Reconstruction | General-purpose detection | `configs/models/magic.yaml` |

type: graph_anomaly_detection| **Kairos** | Transformer + Time Encoder | Temporal attack patterns | `configs/models/kairos.yaml` |

| **Orthrus** | Multi-encoder (Transformer + SAGE) | Multi-objective learning | `configs/models/orthrus.yaml` |

encoder:| **ThreaTrace** | Multi-encoder (GAT + SAGE) | Graph clustering-based | `configs/models/threatrace.yaml` |

  type: graphsage| **Continuum_FL** | GAT + Federated Learning | Distributed/privacy-preserving | `configs/models/continuum_fl.yaml` |

  hidden_dims: [128, 256, 512]

  num_layers: 3### Adding Your Own Model

  dropout: 0.2

1. Copy the template: `cp configs/models/template.yaml configs/models/your_model.yaml`

decoder:2. Configure encoder (GAT, SAGE, Transformer, Time, or Multi-encoder)

  type: anomaly_detection3. Configure decoder (Edge, Node, Contrastive, Reconstruction, Anomaly, InnerProduct)

  hidden_dims: [512, 256, 128]4. Set training/data/inference parameters

  output_dim: 15. Add checkpoint paths for your datasets

6. Run: `python experiments/evaluate_pipeline.py --models your_model --dataset your_dataset`

hyperparameters:

  learning_rate: 0.001**No Python code needed!** The ModelBuilder dynamically constructs your model from the YAML configuration.

  batch_size: 32

```---



### Shared Components## 📂 Directory Structure



**Encoders** (`shared_encoders.py` - 532 lines):```

- `MLPEncoder`: Multi-layer perceptronPIDS_Comparative_Framework/

- `GraphSAGEEncoder`: GraphSAGE convolutions├── README.md                          # This file - Overview

- `GATEncoder`: Graph Attention Networks├── EXTENSIBLE_ARCHITECTURE.md         # 🆕 Extensibility guide

- `TransformerEncoder`: Transformer-based encoding├── RESTRUCTURING_COMPLETE.md          # 🆕 Restructuring summary

- `RNNEncoder`: Recurrent neural network├── ARCHITECTURE_EXTRACTION_SUMMARY.md # Architecture details

├── TASK_ARCHITECTURE.md               # Task pipeline design

**Decoders** (`shared_decoders.py` - 651 lines):├── requirements.txt                   # Python dependencies

- `MLPDecoder`: Multi-layer perceptron│

- `AttentionDecoder`: Attention-based decoding├── models/                            # 🧠 Model components

- `GraphDecoder`: Graph reconstruction│   ├── shared_encoders.py            # 🆕 Shared encoder library

- `SequenceDecoder`: Sequential output│   ├── shared_decoders.py            # 🆕 Shared decoder library

- `ClassificationHead`: Classification tasks│   ├── model_builder.py              # 🆕 Dynamic model construction

- `AnomalyDetectionHead`: Anomaly scoring│   ├── base_model.py                 # Base classes

│   │

### Pipeline Tasks│   └── implementations/              # Legacy model-specific code

│       ├── magic/

Each task in the pipeline is self-contained and cacheable:│       ├── kairos/

│       ├── orthrus/

```python│       ├── threatrace/

from pipeline import PipelineBuilder│       └── continuum_fl/

│

# Create pipeline for specific model and dataset├── configs/                           # ⚙️ Configuration files

pipeline = PipelineBuilder.build(│   └── models/                       # 🆕 Per-model YAML configs

    model_name="magic",│       ├── magic.yaml

    dataset_name="streamspot",│       ├── kairos.yaml

    config=config│       ├── orthrus.yaml

)│       ├── threatrace.yaml

│       ├── continuum_fl.yaml

# Execute pipeline (uses cached results when available)│       └── template.yaml             # 🆕 Template for new models

results = pipeline.run(tasks=['load_data', 'preprocess', 'build_graphs'])│

```├── pipeline/                          # � Task-based pipeline

│   ├── task_manager.py               # Task orchestration

## Evaluation Metrics│   ├── task_registry.py              # 9 modular tasks

│   ├── pipeline_builder.py           # Pipeline construction

The framework computes standard PIDS metrics:│   ├── TASK_ARCHITECTURE.md

- **Accuracy**: Overall detection accuracy│   └── QUICKSTART.md

- **Precision**: True positive rate│

- **Recall**: Coverage of actual attacks├── experiments/                       # 🧪 Experiment scripts

- **F1-Score**: Harmonic mean of precision and recall│   ├── evaluate_pipeline.py          # Task-based evaluation (NEW)

- **AUC-ROC**: Area under ROC curve│   └── train.py                      # Training script

- **AP**: Average Precision│

- **False Positive Rate**: FP / (FP + TN)├── data/                              # 📊 Dataset handling

│   └── dataset.py

## Adding New Models│

├── utils/                             # 🛠️ Utilities

To add a new PIDS model to the framework:│   ├── common.py

│   └── metrics.py

1. **Create model configuration file** in `configs/models/`:│

```yaml├── checkpoints/                       # 💾 Pretrained weights

# configs/models/my_model.yaml│   ├── magic/

name: my_model│   ├── kairos/

type: graph_anomaly_detection│   ├── orthrus/

│   ├── threatrace/

encoder:│   └── continuum_fl/

  type: gat  # Choose from: mlp, graphsage, gat, transformer, rnn│

  hidden_dims: [128, 256]└── artifacts/                         # 📦 Pipeline artifacts (cached)

  num_layers: 2    ├── magic/

  num_heads: 4  # For GAT/Transformer    ├── kairos/

  dropout: 0.1    └── ... (task outputs)

```

decoder:

  type: anomaly_detection  # Choose from: mlp, attention, graph, sequence, classification, anomaly_detection---

  hidden_dims: [256, 128]

  output_dim: 1## 📚 Documentation

  

hyperparameters:### Core Documentation

  learning_rate: 0.001- **[README.md](README.md)** (this file) - Framework overview and quick start

  batch_size: 32- **[EXTENSIBLE_ARCHITECTURE.md](EXTENSIBLE_ARCHITECTURE.md)** - Complete guide to extensibility, adding models, and configuration

  weight_decay: 1e-5- **[RESTRUCTURING_COMPLETE.md](RESTRUCTURING_COMPLETE.md)** - Detailed implementation summary of October 2025 restructuring

```

### Pipeline Documentation

2. **Test the model**:- **[TASK_ARCHITECTURE.md](pipeline/TASK_ARCHITECTURE.md)** - Task-based pipeline design and 9-task breakdown

```bash- **[QUICKSTART.md](pipeline/QUICKSTART.md)** - Quick start guide for pipeline usage

# Verify config is valid

python scripts/verify_installation.py### Architecture Documentation

- **[ARCHITECTURE_EXTRACTION_SUMMARY.md](ARCHITECTURE_EXTRACTION_SUMMARY.md)** - Details on shared component extraction

# Run evaluation

python experiments/evaluate_pipeline.py --model my_model --dataset streamspot### Configuration Templates

```- **[configs/models/template.yaml](configs/models/template.yaml)** - Template for adding new models with all options documented



That's it! No Python code needed. The ModelBuilder automatically constructs the model from your YAML configuration using shared components.---

├── checkpoints/                   # 💾 Pretrained model weights

## Citation│   ├── magic/                    # MAGIC checkpoints

│   ├── kairos/                   # Kairos checkpoints

If you use this framework in your research, please cite:│   ├── orthrus/                  # Orthrus checkpoints

│   ├── threatrace/               # ThreaTrace checkpoints

```bibtex│   └── continuum_fl/             # Continuum_FL checkpoints

@article{pids_framework,│

  title={PIDS Comparative Framework: A Unified Platform for Provenance-based Intrusion Detection},├── data/                          # 📁 Data directory

  author={Your Name},│   ├── custom_soc/               # ← Your custom SOC data

  journal={arXiv preprint},│   ├── cadets_e3/                # DARPA datasets (optional)

  year={2024}│   └── streamspot/               # StreamSpot dataset (optional)

}│

```└── results/                       # 📈 Evaluation results

    └── evaluation_*/             # Timestamped result directories

## Contributing```



We welcome contributions! Please:---

- Add new models by creating YAML configuration files

- Support new datasets via YAML configurations## 🚀 Quick Start

- Implement new pipeline tasks in `task_registry.py`

- Improve shared encoders/decoders### Prerequisites

- Enhance documentation

- **Conda** (Anaconda or Miniconda) - [Install Conda](https://docs.conda.io/en/latest/miniconda.html)

## License- **Python 3.8-3.10** (installed via Conda)

- **10GB disk space** (for dependencies and pretrained weights)

This project is licensed under the MIT License - see the LICENSE file for details.- **Git** (for downloading some model weights)



## Documentation### Installation (5 minutes)



- **[README.md](README.md)** - This file (overview and usage)```bash

- **[SETUP.md](SETUP.md)** - Detailed setup and installation guide# Clone the repository

cd /path/to/PIDS_Files/PIDS_Comparative_Framework

## Contact

# Run automated setup (creates environment, installs dependencies, downloads weights)

For questions or issues, please open an issue on GitHub or contact the maintainers../scripts/setup.sh


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
# Run evaluation on all models (automatic)
./scripts/run_evaluation.sh

# Or evaluate specific model
./scripts/run_evaluation.sh --model magic

# Or use preprocessed data directly
./scripts/run_evaluation.sh \
    --data-path data/custom_soc \
    --skip-preprocess
```

### View Results

```bash
# Check results directory
ls results/evaluation_*/

# View comparison report
cat results/evaluation_*/comparison_report.json

# View per-model results
cat results/evaluation_*/magic_evaluation.log
```

**That's it!** You've evaluated 5 PIDS models on your custom data.

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

### Detection Metrics
- **AUROC** (Area Under ROC Curve): Overall detection performance
- **AUPRC** (Area Under Precision-Recall Curve): Performance with class imbalance
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **Detection Rate**: Percentage of attacks detected

### Statistical Analysis
- **Significance Testing**: Paired t-tests for model comparison
- **Confidence Intervals**: 95% confidence intervals for metrics
- **Cross-Validation**: K-fold validation support

### Visualization
- **ROC Curves**: True positive rate vs false positive rate
- **Precision-Recall Curves**: Precision vs recall tradeoff
- **Confusion Matrices**: Classification breakdown
- **Feature Importance**: Top contributing features (model-dependent)

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

**For detailed troubleshooting, see [Setup.md](Setup.md#troubleshooting)**

---

## 📖 Documentation

- **[Setup.md](Setup.md)** - Complete installation and usage guide
- **[EXTEND.md](EXTEND.md)** - Guide to add new models
- **[SCRIPT_ANALYSIS.md](SCRIPT_ANALYSIS.md)** - Script analysis and maintenance guide

### Command Reference

| Script | Purpose | Documentation |
|--------|---------|---------------|
| `setup.sh` | Environment setup | [Setup.md](Setup.md#installation) |
| `setup_models.py` | Download weights | [Setup.md](Setup.md#model-specific-setup) |
| `preprocess_data.py` | Data preprocessing | [Setup.md](Setup.md#preparing-custom-data) |
| `run_evaluation.sh` | Evaluation workflow | [Setup.md](Setup.md#running-evaluation) |
| `verify_installation.py` | Installation checks | [Setup.md](Setup.md#verification) |
| `verify_implementation.py` | Framework verification | [Setup.md](Setup.md#verification) |

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

- **Documentation**: See [Setup.md](Setup.md) and [EXTEND.md](EXTEND.md)
- **Troubleshooting**: Check [Setup.md Troubleshooting](Setup.md#troubleshooting) section
- **Issues**: Open an issue on GitHub with detailed description
- **Examples**: See `configs/experiments/` for configuration templates

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
