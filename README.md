# PIDS Comparative Framework

<div align="center">

**A Production-Ready, Extensible Platform for Provenance-based Intrusion Detection Systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8--3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Overview](#-overview) | [Architecture](#-architecture-overview) | [Models](#-supported-models) | [Quick Start](SETUP.md#-quick-start) | [Full Setup](SETUP.md)

</div>

---

## Overview

The **PIDS Comparative Framework** is a unified platform for evaluating state-of-the-art Provenance-based Intrusion Detection Systems (PIDS) on **custom Security Operations Center (SOC) data** and **DARPA TC datasets**. Built with extensibility and ease-of-use at its core, the framework enables rapid prototyping, comprehensive evaluation, and deployment of graph-based intrusion detection models.

### Why PIDS Comparative Framework?

**Unified Evaluation Platform**
- Compare state-of-the-art PIDS models on your own SOC data **and DARPA datasets**
- Standardized preprocessing, inference, and evaluation pipeline
- **Universal semantic parser** automatically detects and handles multiple data formats
- Reproducible results with consistent metrics across models

**Multi-Format Data Support**
- ✅ **DARPA TC (CDM v18)** - JSON and binary AVRO formats
- ✅ **Custom SOC Logs** - Elastic/ELK stack, NDJSON
- ✅ **Custom JSON** - Configurable schema mapping
- **Automatic format detection** - No manual format specification needed

**Plug-and-Play Extensibility**
- Add new models via YAML configuration files (zero Python code)
- 17 reusable components: 8 encoders + 9 decoders
- Mix-and-match architecture components to create custom models
- Smart caching of intermediate results for fast iteration

**Comprehensive Analysis**
- Unsupervised anomaly detection metrics (score separation, percentiles)
- Optional supervised metrics when ground truth available (AUROC, F1)
- Ensemble consensus detection across multiple models
- Automatic extraction and ranking of top anomalies

---

## Architecture Overview

The framework follows a **modular architecture** with three key systems working together:

### 1. YAML-Driven Model System

Models are defined declaratively in YAML files, eliminating the need for model-specific Python code:

```yaml
# configs/models/my_model.yaml
name: "my_model"
architecture:
  encoder:
    type: "gat"
    in_channels: 128
    hidden_channels: 256
    out_channels: 128
  decoder:
    type: "edge"
    in_dim: 128
    hidden_dim: 64
    out_dim: 2
```

**Key Components:**
- **ModelBuilder** - Dynamically constructs models from YAML configs
- **GenericModel** - Universal wrapper for any encoder-decoder combination
- **Shared Encoders** (8 types) - GAT, SAGE, Transformer, GIN, GLSTM, Time, Linear, Multi
- **Shared Decoders** (9 types) - Edge, Node, Contrastive, Reconstruction, Anomaly, InnerProduct, NodLink, EdgeLinear, CustomEdgeMLP

**Benefits:**
- Add new models in minutes (not hours)
- No Python wrapper code required
- Easy hyperparameter tuning
- Consistent component interfaces

### 2. Task-Based Pipeline System

The pipeline orchestrates 9 modular tasks that transform raw logs into actionable threat intelligence:

```
Raw JSON Logs → Preprocessing → Graph Construction → Feature Extraction 
→ Model Inference → Anomaly Detection → Threat Analysis
```

**Pipeline Tasks:**
1. **load_preprocessed_data** - Load graph from pickle
2. **construct_time_windows** - Create temporal batches  
3. **graph_transformation** - Convert to model format
4. **feature_extraction** - Extract node/edge features
5. **featurization_inference** - Apply model-specific transforms
6. **batch_construction** - Create PyTorch Geometric batches
7. **model_inference** - Run pretrained models
8. **calculate_metrics** - Compute detection metrics
9. **attack_tracing** - Trace attack paths (optional)

**Key Features:**
- Intelligent artifact caching (skip completed tasks)
- Automatic dependency resolution
- Parallel execution where possible
- Resumable from any task
- Per-task logging and error handling

### 3. Data Processing Engine

Handles diverse provenance data formats with robust preprocessing:

**Supported Formats:**
- **DARPA TC CDM (v18)** - JSON and binary AVRO formats
- Elastic/ELK Stack JSON logs
- Newline-delimited JSON (NDJSON)
- JSON arrays
- Custom JSON schemas

**Universal Semantic Parser:**
- **Automatic format detection** - Three specialized parsers: DARPACDMParser, ElasticParser, CustomJSONParser
- **Entity extraction** - Processes, files, network connections, memory objects, sockets, registry keys
- **Event parsing** - 20+ event types (exec, fork, read, write, connect, open, close, etc.)
- **Unified representation** - Converts all formats to common Event and Entity model
- **Binary AVRO support** - Native support for DARPA TC binary files (requires avro-python3 or fastavro)

**Processing Pipeline:**
```
JSON/AVRO Events → Semantic Parser → Entity Extraction → Graph Construction → Feature Engineering
```

**Features:**
- Chunked loading for 2GB+ files
- Automatic schema detection
- Provenance graph construction
- Temporal windowing support
- Entity deduplication and normalization
---

## Supported Models

All models are unsupervised anomaly detectors pretrained on standard PIDS benchmarks:

| Model | Description | Key Technique | Pretrained Datasets |
|-------|-------------|---------------|---------------------|
| **MAGIC** | Masked Graph Autoencoder | Graph masking + reconstruction | StreamSpot, CADETS, THEIA, TRACE, Wget |
| **Kairos** | Whole-system Provenance | Temporal GNN + sketching | CADETS (E3/E5), CLEARSCOPE (E3/E5), THEIA (E3/E5), StreamSpot |
| **Orthrus** | Multi-Decoder Contrastive | Contrastive learning | CADETS E3, CLEARSCOPE (E3/E5), THEIA (E3/E5) |
| **ThreaTrace** | Scalable Sketch-based | Locality-sensitive hashing | DARPA TC, StreamSpot, Unicorn SC (140+ models) |
| **Continuum_FL** | Federated Learning PIDS | Federated GNN | StreamSpot, CADETS E3, THEIA E3, TRACE E3, CLEARSCOPE E3 |

**Total Pretrained Weights:** 20+ checkpoints across 5 models covering major PIDS benchmarks.

---

## ⚙️ System Workflow

The framework provides an end-to-end pipeline from raw SOC logs to actionable threat intelligence:

```text
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Raw Provenance Logs                                 │
│  • DARPA TC (CDM v18) - JSON/Binary AVRO                    │
│  • Elastic/ELK Stack JSON                                   │
│  • NDJSON (newline-delimited)                               │
│  • JSON arrays                                              │
│  • Custom JSON formats                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SEMANTIC PARSER (Auto-Detection)                           │
│  • DARPACDMParser    → DARPA TC datasets                    │
│  • ElasticParser     → Elastic/ELK logs                     │
│  • CustomJSONParser  → Generic JSON                         │
│                                                             │
│  Output: Unified Event & Entity representation              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  PREPROCESSING (scripts/preprocess_data.py)                 │
│  • Universal semantic parser (auto-detects format)          │
│  • Parse provenance events (JSON/NDJSON/binary AVRO)        │
│  • Extract entities (processes, files, sockets)             │
│  • Build provenance graph (entities=nodes, events=edges)    │
│  • Extract features (entity types, attributes)              │
│  • Save as pickle + metadata JSON                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  TASK PIPELINE (9 Modular, Cacheable Tasks)                 │
│                                                             │
│  1. load_preprocessed_data   → Load graph pickle            │
│  2. construct_time_windows   → Temporal batching            │
│  3. graph_transformation     → Format conversion            │
│  4. feature_extraction       → Node/edge features           │
│  5. featurization_inference  → Model-specific transforms    │
│  6. batch_construction       → PyG Data objects             │
│  7. model_inference          → Run pretrained models        │
│  8. calculate_metrics        → Anomaly detection metrics    │
│  9. attack_tracing           → Attack path analysis         │
│                                                             │
│  Artifacts cached in: artifacts/{model}/{task}/             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  MODEL EXECUTION (ModelBuilder + GenericModel)              │
│                                                             │
│  YAML Config ──→ ModelBuilder ──→ GenericModel              │
│                       │                                     │
│                       ├──→ Encoder(s) ──→ Embeddings        │
│                       │    (GAT, SAGE, Transformer, etc.)   │
│                       │                                     │
│                       └──→ Decoder(s) ──→ Predictions       │
│                            (Edge, Node, Contrastive, etc.)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Evaluation Results                                 │
│                                                             │
│  results/evaluation_{timestamp}/                            │
│  ├── {model}_metrics.json     (Detection metrics)          │
│  ├── {model}_anomalies.json   (Top-K anomalies)           │
│  ├── ensemble_consensus.json  (Multi-model agreement)      │
│  └── comparison_report.json   (Model rankings)             │
│                                                             │
│  artifacts/{model}/            (Cached intermediate data)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Ready to evaluate PIDS models on your provenance data?** See the [Complete Setup Guide](SETUP.md) for detailed instructions.

### 3-Step Quickstart

**Option A: Evaluate on Custom SOC Data**
```bash
# 1. Setup environment
./scripts/setup.sh

# 2. Download pretrained weights
python scripts/download_checkpoints.py --all

# 3. Preprocess and evaluate on your SOC data
./scripts/run_evaluation.sh \
  --data-path ../custom_dataset \
  --dataset custom_soc \
  --model all
```

**Option B: Evaluate on DARPA TC Datasets**
```bash
# 1. Preprocess DARPA dataset (auto-detects JSON/binary AVRO format)
python scripts/preprocess_data.py \
  --input-dir ../DARPA/ta1-cadets-e3-official-1.json \
  --dataset-name cadets_e3 \
  --dataset-type darpa

# 2. Evaluate models
./scripts/run_evaluation.sh \
  --data-path data/darpa \
  --dataset cadets_e3 \
  --dataset-type darpa \
  --skip-preprocess
```

**Option C: Quick Test with Binary AVRO Files**
```bash
# Process binary DARPA files and run quick evaluation
# The framework automatically detects and handles binary AVRO format
./scripts/run_evaluation.sh \
  --data-path ../DARPA/ta1-theia-e3-official-1r.bin \
  --dataset theia_e3 \
  --max-events 100000 \
  --model magic

# Alternative: Process entire dataset without sampling
./scripts/run_evaluation.sh \
  --data-path ../DARPA/ta1-theia-e3-official-1r.json \
  --dataset theia_e3 \
  --model magic
```

**Note:** DARPA datasets often come in directories with files like `*.bin`, `*.bin.1`, `*.bin.2`, etc. The framework automatically finds and processes all files in the directory.

### Universal Preprocessing

The framework now includes a **unified preprocessing script** that handles all data formats with automatic detection:

```bash
# Universal preprocessor - auto-detects format and dataset type
python scripts/preprocess_data.py \
  --input-dir <path_to_data> \
  --dataset-name <dataset_name>

# Supported formats (auto-detected):
# - JSON (Elastic/ELK logs)
# - NDJSON (New Line Delimited JSON)
# - Binary AVRO (DARPA TC CDM v18)
# - Custom JSON schemas

# Supported dataset types (auto-detected):
# - custom_soc (Elastic/ELK logs)
# - darpa (DARPA TC CDM format)
# - custom (any JSON format)
```

**Key Features:**
- ✅ **Automatic format detection** - No need to specify JSON vs binary
- ✅ **Multi-format support** - JSON, NDJSON, binary AVRO
- ✅ **Semantic parsing** - Intelligent schema detection
- ✅ **Large dataset handling** - Sampling support for testing
- ✅ **Comprehensive statistics** - Detailed preprocessing metrics

For comprehensive setup instructions, data preprocessing, troubleshooting, and advanced usage, refer to:
- **[SETUP.md](SETUP.md)** - General setup and installation
- **[DARPA_DATASETS.md](docs/DARPA_DATASETS.md)** - DARPA TC dataset guide
- **[QUICKSTART_DARPA.md](QUICKSTART_DARPA.md)** - Quick start for DARPA datasets

---

## Programmatic Usage

The framework provides flexible APIs for custom integration and automation.

### Option 1: Using ModelBuilder Directly

```python
from models.model_builder import ModelBuilder
import torch

# Create model builder
builder = ModelBuilder(config_dir="configs/models")

# Build and load pretrained model
model = builder.build_and_load(
    model_name="magic",
    dataset_name="streamspot",
    device="cpu"
)

# Run inference on graph data
model.eval()
with torch.no_grad():
    # Encode graph to embeddings
    embeddings = model.encode(batch)
    
    # Decode to predictions
    predictions = model.decode(embeddings, batch, inference=True)
    
    # Or use full forward pass
    output = model(batch, inference=True)
```

### Option 2: Using Complete Pipeline

```python
from pipeline.pipeline_builder import PipelineBuilder

# Create pipeline configuration
config = {
    'data': {
        'path': 'data/custom_soc',
        'dataset': 'custom_soc'
    },
    'artifact_dir': 'artifacts',
    'device': 'cpu'
}

# Build and execute pipeline
builder = PipelineBuilder(config)
results = builder.build_and_execute(
    model_name="magic",
    tasks=None,  # Run all tasks
    force_restart=False  # Use cached artifacts
)

# Access results
metrics = results['calculate_metrics']
predictions = results['model_inference']['predictions']
scores = results['model_inference']['scores']

print(f"Score Separation Ratio: {metrics['score_separation_ratio']:.4f}")
print(f"Critical Anomalies: {metrics['anomaly_counts']['critical_99.9']}")
```

### Option 3: Custom Task Pipeline

```python
from pipeline.pipeline_builder import PipelineBuilder

config = {...}  # Your configuration
builder = PipelineBuilder(config)

# Build pipeline with custom tasks
task_manager = builder.build_pipeline(
    model_name="magic",
    tasks=['load_preprocessed_data', 'model_inference', 'calculate_metrics']
)

# Execute specific tasks
results = task_manager.execute_pipeline()

# Save execution metadata
task_manager.save_execution_metadata('artifacts/magic/metadata.json')
```

### Option 4: Adding a New Model via YAML

No Python code required - just create a YAML configuration:

```yaml
# configs/models/my_model.yaml
name: "my_model"

architecture:
  encoder:
    type: "gat"
    in_channels: 128
    hidden_channels: 256
    out_channels: 128
    num_layers: 3
    num_heads: 8
    dropout: 0.1
    
  decoder:
    type: "edge"
    in_dim: 128
    hidden_dim: 256
    out_dim: 2
    dropout: 0.1

checkpoint:
  pretrained:
    enabled: true
    path: "checkpoints/my_model/{dataset}.pt"
    strict: false

data:
  window:
    size: 3600  # 1 hour windows
  features:
    node_feat_dim: 128
    edge_feat_dim: 64
```

Then evaluate immediately:

```bash
python experiments/evaluate_pipeline.py \
  --models my_model \
  --data-path data/custom_soc \
  --dataset custom_soc
```

---

## 📁 Directory Structure

```text
PIDS_Comparative_Framework/
│
├── models/                                             # Core model components
│   ├── model_builder.py                                # ModelBuilder + GenericModel
│   ├── shared_encoders.py                              # 8 reusable encoder types
│   ├── shared_decoders.py                              # 9 reusable decoder types
│   └── __init__.py
│
├── configs/                                            # YAML configuration files
│   ├── models/                                         # Per-model configurations
│   │   ├── magic.yaml, kairos.yaml, orthrus.yaml
│   │   ├── threatrace.yaml, continuum_fl.yaml
│   │   └── template.yaml                               # Template for new models
│   ├── datasets/                                       # Dataset configurations
│   └── experiments/                                    # Experiment presets
│
├── pipeline/                                           # Task-based pipeline system
│   ├── pipeline_builder.py                             # Pipeline orchestration
│   ├── task_manager.py                                 # Task execution + caching
│   ├── task_registry.py                                # 9 task implementations
│   └── __init__.py
│
├── experiments/                                        # Main evaluation scripts
│   ├── evaluate_pipeline.py                            # Primary evaluation script
│   └── train.py                                        # Reference training script
│
├── scripts/                                            # Setup and utility scripts
│   ├── analyze_anomalies.py                            # Anomaly analysis tools
│   ├── download_checkpoints.py                         # Download pretrained weights
│   ├── preprocess_data.py                              # SOC data preprocessing
│   ├── run_evaluation.sh                               # End-to-end workflow
│   ├── setup.sh                                        # One-command environment setup
│   ├── verify_installation.py                          # Installation check
│   └── visualize_attacks.sh                            # Attack visualization workflow
│
├── data/                                               # Data management
│   ├── dataset.py                                      # Dataset loading utilities
│   └── {dataset_name}/                                 # Preprocessed datasets
│       ├── graph.pkl                                   # Graph structure
│       └── metadata.json                               # Dataset statistics
│
├── utils/                                              # Common utilities
│   ├── common.py                                       # Logging, config loading
│   ├── metrics.py                                      # Detection metrics
│   └── visualize_attack_graphs.py                      # Attack graph visualization
│
├── checkpoints/                                        # Pretrained model weights
│   └── {model}/                                        # Per-model checkpoints
│       └── {dataset}.pt                                # Model weights
│
├── artifacts/                                          # Cached pipeline results
│   └── {model}/                                        # Per-model artifacts
│       ├── {task}/                                     # Per-task outputs
│       └── execution_metadata.json
│
└── results/                                            # Evaluation outputs
    └── evaluation_{timestamp}/
        ├── {model}_metrics.json
        ├── {model}_anomalies.json
        ├── ensemble_consensus.json
        └── comparison_report.json
```

---
## Architecture Details

### 1. Model Construction System

**ModelBuilder** dynamically constructs models from YAML configurations:

- **Dynamic Construction**: Builds models from YAML configs at runtime
- **Component Registry**: Maps encoder/decoder types to implementations  
- **Smart Checkpointing**: Tries multiple checkpoint paths with graceful fallbacks
- **Dimension Adaptation**: Automatically handles input dimension mismatches

**GenericModel** provides a universal interface for all models:

- **Universal Interface**: Works with any encoder-decoder combination
- **Multi-Component Support**: Single/multi encoder, single/multi decoder
- **Unified API**: `encode()`, `decode()`, `forward()` methods
- **Device Management**: Automatic CPU/GPU handling

### 2. Shared Component Library

**8 Reusable Encoders:**

- `GATEncoder` - Graph Attention Networks (multi-head attention)
- `SAGEEncoder` - GraphSAGE (neighborhood aggregation)
- `GraphTransformerEncoder` - Transformer-based graph encoding
- `TimeEncoder` - Temporal graph encoder with time embeddings
- `GINEncoder` - Graph Isomorphism Network
- `GLSTMEncoder` - Graph LSTM for sequential patterns
- `LinearEncoder` - Simple linear projection baseline
- `MultiEncoder` - Combine multiple encoders (concat, mean, max, attention)

**9 Reusable Decoders:**

- `EdgeDecoder` - Edge-level classification/prediction
- `NodeDecoder` - Node-level classification  
- `ContrastiveDecoder` - Temporal contrastive learning
- `ReconstructionDecoder` - Feature reconstruction (autoencoders)
- `AnomalyDecoder` - Anomaly score prediction
- `InnerProductDecoder` - Graph autoencoder-style decoder
- `NodLinkDecoder` - Node-link prediction
- `EdgeLinearDecoder` - Linear edge prediction
- `CustomEdgeMLPDecoder` - Custom MLP for complex edge tasks

### 3. Task Pipeline System

**PipelineBuilder** orchestrates 9 modular tasks:
- `EdgeDecoder` - Edge classification/prediction
- `NodeDecoder` - Node classification
- `ContrastiveDecoder` - Temporal contrastive learning
- `ReconstructionDecoder` - Feature reconstruction
- `AnomalyDecoder` - Anomaly score prediction
- `InnerProductDecoder` - Graph autoencoder-style decoder
- `NodLinkDecoder` - Node-link prediction
- `EdgeLinearDecoder` - Linear edge prediction
- `CustomEdgeMLPDecoder` - Custom MLP for edge prediction

#### 4. **Task Pipeline System**

**9 Modular Tasks:**
1. **load_preprocessed_data** - Load graph from preprocessed pickle
2. **construct_time_windows** - Create temporal batches
3. **graph_transformation** - Convert to DGL/PyG format
4. **feature_extraction** - Extract node/edge features
5. **featurization_inference** - Model-specific feature transforms
6. **batch_construction** - Create DataLoaders
7. **model_inference** - Run model predictions
8. **calculate_metrics** - Compute detection metrics
9. **attack_tracing** (optional) - Trace attack paths

**Pipeline Features:**
- Automatic dependency resolution
- Artifact caching (skip completed tasks)
- Parallel execution where possible
- Error recovery and logging
- Progress tracking
### Design Philosophy

**Separation of Concerns**
- Models defined declaratively in YAML
- Core components (encoders/decoders) implemented once, reused everywhere
- Pipeline tasks handle data flow, not model logic

**Plug-and-Play Architecture**
- Add encoders/decoders → Available to all models
- Add tasks → Usable in any pipeline
- Add models → Just create YAML config

---

## Pipeline Tasks

The framework's task-based pipeline provides modularity, caching, and flexibility. Each task is self-contained and produces cached artifacts.

### Task Overview

| Task | Description | Input | Output | Cacheable |
|------|-------------|-------|--------|-----------|
| **1. load_preprocessed_data** | Load graph from pickle file | Preprocessed data path | Graph data dictionary | ✅ |
| **2. construct_time_windows** | Create temporal batches | Graph data | List of time windows | ✅ |
| **3. graph_transformation** | Convert to model format | Graph data | DGL/PyG graphs | ✅ |
| **4. feature_extraction** | Extract node/edge features | Graph data | Feature tensors | ✅ |
| **5. featurization_inference** | Model-specific transforms | Features + model | Transformed features | ✅ |
| **6. batch_construction** | Create DataLoaders | Features + graphs | DataLoaders | ✅ |
| **7. model_inference** | Run model predictions | Model + DataLoaders | Predictions + scores | ✅ |
| **8. calculate_metrics** | Compute metrics | Predictions + labels | Metrics dictionary | ✅ |
| **9. attack_tracing** | Trace attack paths | Graph + predictions | Attack subgraphs | ⚠️ |

### Task Details

#### 1. load_preprocessed_data
**Purpose**: Load preprocessed provenance graphs from pickle files

**Configuration**:
```yaml
data_path: "data/custom_soc"
dataset_name: "custom_soc"
```

**Output**:
```python
{
    'graph_data': {...},      # NetworkX graph or dict
    'num_nodes': 50000,
    'num_edges': 150000,
    'stats': {...}
}
```

#### 2. construct_time_windows
**Purpose**: Segment graphs into temporal windows for streaming analysis

**Configuration**:
```yaml
window_size: 3600             # 1 hour windows
overlap: 0.1                  # 10% overlap
min_events: 100               # Minimum events per window
```

**Output**:
```python
[
    {'window_id': 0, 'start': t0, 'end': t1, 'graph': g0},
    {'window_id': 1, 'start': t1, 'end': t2, 'graph': g1},
    ...
]
```

#### 3. graph_transformation
**Purpose**: Convert graphs to model-specific formats (DGL, PyTorch Geometric)

**Supported Formats**:
- NetworkX → DGL
- NetworkX → PyG (PyTorch Geometric)
- Edge list → DGL/PyG
- Custom provenance format

**Output**: Model-ready graph objects

#### 4. feature_extraction
**Purpose**: Extract and normalize node/edge features

**Features Extracted**:
- **Node features**: Entity type, degree, betweenness, timestamps
- **Edge features**: Event type, duration, frequency
- **Temporal features**: Time of day, day of week
- **Statistical features**: In/out degree, clustering coefficient

**Normalization**: Z-score, min-max, or log scaling

#### 5. featurization_inference
**Purpose**: Apply model-specific feature transformations

**Examples**:
- Embedding lookups for categorical features
- Positional encodings for temporal data
- Feature masking for autoencoders
- Data augmentation

#### 6. batch_construction
**Purpose**: Create efficient DataLoaders for training/inference

**Configuration**:
```yaml
batch_size: 32
num_workers: 4
shuffle: false                # Usually false for evaluation
pin_memory: true
```

#### 7. model_inference
**Purpose**: Run model predictions and generate anomaly scores

**Process**:
1. Load pretrained checkpoint
2. Set model to eval mode
3. Run inference on batches
4. Collect predictions and scores
5. Post-process outputs

**Output**:
```python
{
    'predictions': np.array([...]),      # Class predictions
    'scores': np.array([...]),           # Anomaly scores
    'embeddings': np.array([...]),       # Optional embeddings
    'edge_scores': np.array([...])       # Edge-level scores
}
```

#### 8. calculate_metrics
**Purpose**: Compute comprehensive evaluation metrics

**Metrics**:

*Unsupervised (always computed):*
- Score Separation Ratio (std/mean)
- Score distribution statistics
- Anomaly counts by percentile

*Supervised (if labels available):*
- AUROC, AUPRC
- Precision, Recall, F1-Score
- Confusion matrix
- FPR, TPR at various thresholds

**Output**:
```python
{
    'unsupervised': {
        'separation_ratio': 0.45,
        'mean_score': 0.23,
        'std_score': 0.10,
        'critical_anomalies': 150,
        'high_risk_anomalies': 750
    },
    'supervised': {  # If labels available
        'auroc': 0.92,
        'auprc': 0.88,
        'f1': 0.85,
        'precision': 0.87,
        'recall': 0.83
    }
}
```

#### 9. attack_tracing (Optional)
**Purpose**: Trace attack paths from detected anomalies

**Process**:
1. Extract high-scoring events
2. Perform backward/forward provenance traversal
3. Identify attack subgraphs
4. Generate attack narratives

**Output**: Attack subgraphs with highlighted malicious paths

### Pipeline Execution

**Sequential Execution**:
```python
from pipeline.pipeline_builder import PipelineBuilder

pipeline = PipelineBuilder(model_name="magic", dataset_name="cadets")

# Add tasks in order
pipeline.add_task("load_preprocessed_data")
pipeline.add_task("construct_time_windows")
pipeline.add_task("graph_transformation")
pipeline.add_task("feature_extraction")
pipeline.add_task("batch_construction")
pipeline.add_task("model_inference")
pipeline.add_task("calculate_metrics")

# Execute
results = pipeline.execute()
```

**Cached Execution** (Skip completed tasks):
```python
# On first run: all tasks execute
results1 = pipeline.execute()

# On second run: loads from cache
results2 = pipeline.execute()  # Much faster!

# Force re-execution
results3 = pipeline.execute(force_recompute=True)
```

### Caching Strategy

**Artifact Structure**:
```
artifacts/{model_name}/
├── execution_metadata.json          # Execution timestamps
├── load_preprocessed_data/
│   └── output.pkl                   # Cached graph
├── construct_time_windows/
│   └── output.pkl                   # Cached windows
├── graph_transformation/
│   └── output.pkl                   # Transformed graphs
├── feature_extraction/
│   └── output.pkl                   # Extracted features
├── featurization_inference/
│   └── output.pkl                   # Transformed features
├── batch_construction/
│   └── output.pkl                   # DataLoaders (serialized)
├── model_inference/
│   └── output.pkl                   # Predictions + scores
└── calculate_metrics/
    └── output.pkl                   # Metrics
```

**Cache Management**:
- Automatic cache invalidation on config changes
- Manual cache clearing: `rm -rf artifacts/{model_name}/`
- Selective task re-execution

---
## Adding Your Own Model

**Step 1: Create Configuration**
```bash
cp configs/models/template.yaml configs/models/your_model.yaml
```

**Step 2: Configure Architecture**
```yaml
name: "your_model"

architecture:
  encoder:
    type: "gat"              # Options: gat, sage, transformer, time, gin, glstm, linear
    in_dim: 128
    hidden_dim: 256
    out_dim: 128
    num_layers: 3
  
  decoder:
    type: "edge"             # Options: edge, node, contrastive, reconstruction, 
                             # anomaly, inner_product, nodlink, edge_linear
    in_dim: 128
    hidden_dim: 256
    out_dim: 2

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100

checkpoints:
  your_dataset:
    path: "checkpoints/your_model/checkpoint.pt"
```

**Step 3: Run Evaluation**
```bash
python experiments/evaluate_pipeline.py \
  --models your_model \
  --dataset your_dataset \
  --data-path data/your_dataset
```

**No Python wrapper needed!** The `ModelBuilder` dynamically constructs your model.

---

## 📊 Attack Graph Visualization

After running model evaluations, you can visualize and compare attack graphs across multiple models using the `visualize_attack_graphs.py` utility.

### Features

- **Interactive Multi-Model Comparison** - Compare attack graphs from all models side-by-side
- **Clean Node Visualization** - Hover over nodes to see detailed information
- **Evaluation Reference** - Track which evaluation run's artifacts you're visualizing
- **Attack Path Reconstruction** - Backward/forward provenance traversal from anomalies
- **Entity Clustering** - Group related entities and events
- **Export Formats** - HTML (interactive), JSON (summary), GraphML (for Gephi/Cytoscape)
- **Auto-Open Browser** - Automatically opens visualization in your default browser

### Quick Usage

```bash
**Quick Usage

```bash
# Visualize using most recent evaluation (auto-detected)
./scripts/visualize_attacks.sh

# Visualize specific evaluation results
./scripts/visualize_attacks.sh \
  --evaluation-dir results/evaluation_20251105_003744

# Visualize with custom settings
./scripts/visualize_attacks.sh \
  --evaluation-dir results/evaluation_20251105_003744 \
  --threshold 99.0 \
  --top-k 50 \
  --top-paths 20

# Visualize specific models (using Python script directly)
python utils/visualize_attack_graphs.py \
  --artifacts-dir artifacts \
  --evaluation-dir results/evaluation_20251105_003744 \
  --models magic kairos orthrus

# Specify output directory
./scripts/visualize_attacks.sh \
  --evaluation-dir results/evaluation_20251105_003744 \
  --output-dir results/my_attack_visualization

# Remote server mode - start HTTP server for VS Code Remote / SSH access
./scripts/visualize_attacks.sh --serve
```

**Note:** The `--evaluation-dir` flag specifies which evaluation results to use. If not provided, the script automatically uses the most recent evaluation found in `results/`. The `--serve` flag starts an HTTP server on port 8000, which VS Code can automatically forward for remote access.
```

**Note:** The `--evaluation-dir` flag is for reference/logging only. Attack graphs are always built from the artifacts in the `--artifacts-dir` directory, which contains the model inference outputs and graph transformations.

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--evaluation-dir` | Most recent | Path to evaluation results directory (auto-detects most recent if not specified) |
| `--artifacts-dir` | `artifacts` | Directory containing model artifacts (graphs, inference outputs) |
| `--models` | All 5 models | Space-separated list of models to visualize |
| `--output-dir` | `results/attack_graph_visualization` | Output directory for visualizations |
| `--threshold-percentile` | `99.0` | Percentile threshold for anomalies (90-99.9) |
| `--top-k` | `100` | Number of top anomalies to include |
| `--top-paths` | `15` | Number of attack paths to extract |
| `--cluster-by` | `entity` | Clustering strategy: `entity`, `temporal`, or `path` |
| `--no-browser` | false | Skip auto-opening browser (useful for headless servers) |
| `--serve` | false | Start HTTP server on port 8000 for remote access (e.g., VS Code Remote) |

### Generated Output

The visualization script creates:

```
results/attack_graph_visualization/
├── attack_graph_viewer.html          # Interactive visualization (auto-opens)
├── attack_summary.json               # Attack statistics and paths
├── magic_attack_graph.graphml        # GraphML for external tools
├── kairos_attack_graph.graphml
├── orthrus_attack_graph.graphml
├── threatrace_attack_graph.graphml
└── continuum_fl_attack_graph.graphml
```

### Interactive Viewer Features

The HTML viewer provides:
- **Model Tabs** - Switch between different model visualizations
- **Attack Path Highlighting** - View top attack paths with severity scores
- **Entity Tooltips** - Hover over nodes to see detailed entity information
- **Provenance Edges** - Colored edges showing dependency relationships
- **Attack Statistics** - Summary metrics for each model
- **Zoom/Pan Controls** - Interactive graph navigation

### Integration with External Tools

Export GraphML files to:
- **Gephi** - Advanced graph visualization and analysis
- **Cytoscape** - Network analysis and visualization
- **Neo4j** - Graph database import for querying
- **Python NetworkX** - Custom analysis scripts

### Example Workflow

```bash
# Step 1: Run evaluation
./scripts/run_evaluation.sh --data-path data/custom_soc

# Step 2: Visualize results
python utils/visualize_attack_graphs.py \
  --artifacts-dir artifacts \
  --output-dir results/attack_viz \
  --threshold-percentile 99.0

# Step 3: View interactive HTML (auto-opens in browser)
# File: results/attack_viz/attack_graph_viewer.html

# Step 4: Import GraphML into Gephi for advanced analysis
# File: results/attack_viz/*_attack_graph.graphml
```

### Programmatic Usage

```python
from utils.visualize_attack_graphs import AttackGraphReconstructor, MultiModelVisualizer
import pickle
from pathlib import Path

# Load preprocessed graph and model scores
with open('data/custom_soc/graph.pkl', 'rb') as f:
    graph_data = pickle.load(f)

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

---
## 📚 Resources

### Documentation
- **[Complete Setup Guide](SETUP.md)** - Installation, configuration, troubleshooting
- **[Model Configurations](configs/models/)** - YAML configurations for all models
- **[Dataset Configurations](configs/datasets/)** - Dataset preprocessing configs

### Key Scripts
- `scripts/setup.sh` - Automated environment setup
- `scripts/preprocess_data.py` - Universal data preprocessing (SOC/DARPA/custom, JSON/NDJSON/AVRO)
- `scripts/download_checkpoints.py` - Download pretrained weights
- `scripts/run_evaluation.sh` - End-to-end evaluation workflow
- `experiments/evaluate_pipeline.py` - Model evaluation
- `scripts/analyze_anomalies.py` - Anomaly analysis tools
- `scripts/visualize_attacks.sh` - Attack visualization workflow
- `utils/visualize_attack_graphs.py` - Interactive attack graph visualization

### Citation & References

**Models Integrated:**
- **MAGIC**: Han et al. "MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning" (2023)
- **Kairos**: Hossain et al. "SLEUTH: Real-time Attack Scenario Reconstruction from COTS Audit Data" (USENIX Security 2017)
- **Orthrus**: Hossain et al. "Combating Dependence Explosion in Forensic Analysis" (IEEE S&P 2020)
- **ThreaTrace**: Wang et al. "You Are What You Do: Hunting Stealthy Malware via Data Provenance Analysis" (NDSS 2020)
- **Continuum_FL**: Feranhi et al. "Continuum: A Platform for Cost-Aware, Low-Latency Continual Learning" (SoCC 2021)

---

## Next Steps & Contributions

We welcome contributions! Areas of interest:

**High Priority:**
- Add new PIDS models (via YAML config)
- Implement new encoders/decoders
- Add log format parsers
- Improve documentation

**Medium Priority:**
- Web visualization dashboard
- SIEM integrations
- Performance optimizations
- Additional metrics

---

<!-- ## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- -->

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/PIDS_Comparative_Framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/PIDS_Comparative_Framework/discussions)

---

**Built with ❤️ for the cybersecurity research community**
