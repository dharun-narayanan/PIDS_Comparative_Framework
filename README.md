# PIDS Comparative Framework

<div align="center">

**A Unified Framework for Evaluating State-of-the-Art Provenance-based Intrusion Detection Systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8--3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Quick Start](#-quick-start) | [Installation](#-installation) | [Models](#-supported-models) | [Usage](#-usage) | [Troubleshooting](#-troubleshooting)

</div>

---

## 🎯 Overview

The **PIDS Comparative Framework** is a production-ready platform that enables Security Operations Centers (SOC) to evaluate and compare state-of-the-art Provenance-based Intrusion Detection Systems (PIDS) on custom data.

### Primary Use Case

**Evaluate pretrained PIDS models on your custom SOC data** to determine which model performs best for your environment.

✅ **Ready-to-Use**: Pretrained models included - no training required  
✅ **Standalone**: All model implementations self-contained  
✅ **Custom Data**: Works with your JSON-formatted system logs  
✅ **Multi-Model**: Compare 5 state-of-the-art approaches simultaneously  
✅ **CPU-First**: Runs on CPU by default, GPU optional  
🔄 **Advanced**: Retrain models on custom data (optional)

### What is Provenance-based Intrusion Detection?

Provenance graphs capture system-level information flows (process→file, process→network) to model normal behavior and detect anomalous activities indicative of cyber attacks. This framework integrates 5 state-of-the-art deep learning approaches for analyzing provenance data.

---

## 📊 Workflow

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

### 🎯 Evaluation-First Design
- **Pretrained Models**: Use existing weights immediately - no training required
- **Quick Deployment**: Evaluate all models on your data in minutes
- **Performance Comparison**: Automatic comparison with statistical significance testing
- **One-Command Workflow**: `./scripts/run_evaluation.sh` handles everything

### 📊 Multi-Model Support
- **5 State-of-the-Art Models**: MAGIC, Kairos, Orthrus, ThreaTrace, Continuum_FL
- **Consistent Interface**: All models through unified `BasePIDSModel` API
- **Automatic Registration**: Dynamic model discovery via decorator pattern
- **Pretrained Weights**: Ready-to-use checkpoints for all models

### 🔬 Comprehensive Evaluation
- **Multiple Metrics**: AUROC, AUPRC, F1-Score, Precision, Recall, Detection Rate
- **Statistical Analysis**: Significance testing for model comparison
- **Rich Visualizations**: ROC curves, precision-recall curves, confusion matrices
- **Detailed Reports**: JSON and text formats with all metrics

### 🔧 Production-Ready
- **CPU-First Design**: Runs on CPU by default (no GPU required)
- **GPU Support**: Automatic GPU detection and utilization when available
- **Large-Scale Data**: Handles 2GB+ JSON files with chunked loading
- **Checkpointing**: Save/resume training with early stopping (for retraining)
- **Comprehensive Logging**: Debug, info, warning, and error levels
- **Error Handling**: Graceful degradation and informative error messages

### 📦 Easy to Use
- **YAML Configurations**: All settings in human-readable configs
- **One-Command Setup**: Automated environment and dependency installation
- **Streamlined Workflow**: Evaluation script handles all steps automatically
- **Comprehensive Documentation**: README, Setup.md, and EXTEND.md cover everything

### 🔬 Extensible Architecture (Advanced)
- **Modular Design**: Separate data, models, training, evaluation components
- **Plugin System**: Add new models with ~200 lines of code
- **Configurable**: Override any setting via YAML or command-line
- **Retraining Support**: Optional model training on custom datasets

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PIDS Comparative Framework                  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Model Registry (Plugin System)                │ │
│  │  Auto-discovery via @register decorator                 │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  MAGIC   │  │ Kairos   │  │ Orthrus  │  │ThreaTrace│   │
│  │ Wrapper  │  │ Wrapper  │  │ Wrapper  │  │ Wrapper  │...│
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │              │          │
│  ┌────▼─────────────▼──────────────▼──────────────▼─────┐  │
│  │      Standalone Model Implementations                 │  │
│  │  (No external dependencies)                           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Data Pipeline                            │  │
│  │  JSON logs → Graph → Batching → Evaluation           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure

```
PIDS_Comparative_Framework/
├── README.md                       # This file - Framework overview
├── Setup.md                        # Complete installation & usage guide
├── EXTEND.md                       # Guide to add new models
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment specification
│
├── models/                         # 🧠 Model implementations
│   ├── base_model.py              # BasePIDSModel & ModelRegistry
│   ├── __init__.py                # Auto-discovery of models
│   │
│   ├── implementations/           # 📦 Standalone implementations
│   │   ├── magic/                # MAGIC (Graph Autoencoder)
│   │   ├── kairos/               # Kairos (Temporal GNN)
│   │   ├── orthrus/              # Orthrus (Contrastive Learning)
│   │   ├── threatrace/           # ThreaTrace (Sketch-based)
│   │   ├── continuum_fl/         # Continuum_FL (Federated Learning)
│   │   └── utils/                # Shared utilities
│   │
│   ├── magic_wrapper.py          # MAGIC → BasePIDSModel adapter
│   ├── kairos_wrapper.py         # Kairos adapter
│   ├── orthrus_wrapper.py        # Orthrus adapter
│   ├── threatrace_wrapper.py     # ThreaTrace adapter
│   └── continuum_fl_wrapper.py   # Continuum_FL adapter
│
├── data/                          # 📊 Dataset handling
│   └── dataset.py                # Base classes for datasets
│
├── experiments/                   # 🧪 Experiment scripts
│   ├── evaluate.py               # ⭐ Main evaluation script
│   ├── train.py                  # Training script (advanced)
│   └── compare.py                # Multi-model comparison
│
├── utils/                         # 🛠️ Framework utilities
│   ├── common.py                 # Common utilities
│   └── metrics.py                # Evaluation metrics
│
├── scripts/                       # 📜 Setup & helper scripts
│   ├── setup.sh                  # One-command environment setup
│   ├── setup_models.py           # Download pretrained weights
│   ├── preprocess_data.py        # Data preprocessing
│   ├── run_evaluation.sh         # Complete evaluation workflow
│   ├── verify_installation.py    # Installation verification
│   └── verify_implementation.py  # Framework verification
│
├── configs/                       # ⚙️ Configuration files
│   ├── datasets/                 # Dataset configs
│   ├── models/                   # Model configs
│   └── experiments/              # Experiment configs
│
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
python experiments/evaluate.py \
    --model magic \
    --config configs/models/magic_custom.yaml \
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
# Direct evaluation with options
python experiments/evaluate.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --pretrained \
    --batch-size 16 \
    --detection-level entity \
    --k-neighbors 5 \
    --device 0 \
    --save-predictions \
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

### Training (Advanced)

```bash
# Train MAGIC on custom data
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
python experiments/evaluate.py --model magic --batch-size 4 --device -1
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

The framework makes it easy to add new PIDS models:

```python
# models/your_model_wrapper.py
from models.base_model import BasePIDSModel, ModelRegistry

@ModelRegistry.register('your_model')
class YourModel(BasePIDSModel):
    def __init__(self, config):
        super().__init__(config)
        # Your implementation
    
    def forward(self, batch):
        # Forward pass
        pass
    
    def evaluate(self, dataloader, **kwargs):
        # Evaluation logic
        pass
```

**See [EXTEND.md](EXTEND.md) for complete guide** (~200 lines of code to add a model)

---

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Contribution Areas

- 🆕 Add new PIDS models
- 📊 Add new datasets
- 🧪 Add new evaluation metrics
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
