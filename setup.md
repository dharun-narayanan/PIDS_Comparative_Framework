# PIDS Comparative Framework - Setup Guide# PIDS Framework - Complete Setup Guide



Complete step-by-step instructions to install, configure, and run the PIDS Comparative Framework.## 🎯 Overview



---This framework evaluates **5 pretrained PIDS models** on your custom SOC data:

- **MAGIC** - Masked Graph Autoencoder

## 📋 Table of Contents- **Kairos** - Temporal GNN

- **Orthrus** - Multi-Decoder Contrastive Learning

1. [Prerequisites](#prerequisites)- **ThreaTrace** - Scalable Graph Processing

2. [Installation Methods](#installation-methods)- **Continuum_FL** - Federated Learning PIDS

3. [Quick Setup (Recommended)](#quick-setup-recommended)

4. [Manual Setup](#manual-setup)**Default workflow**: Setup environment → Setup models & weights → Preprocess data → Evaluate models → Compare performance

5. [Downloading Pretrained Models](#downloading-pretrained-models)

6. [Preparing Custom Data](#preparing-custom-data)---

7. [Running Your First Evaluation](#running-your-first-evaluation)

8. [Configuration](#configuration)## 📋 Prerequisites

9. [Troubleshooting](#troubleshooting)

10. [Advanced Setup](#advanced-setup)Before starting, ensure you have:

- ✅ **Conda** installed (Anaconda or Miniconda)

---- ✅ **Python 3.8+** (will be installed via conda)

- ✅ **Graphviz (system binary)** - required for visualization (optional but recommended). 

## ✅ PrerequisitesOn macOS:

```bash

### System Requirements# Install Graphviz via Homebrew

brew install graphviz

**Minimum:**# Verify

- OS: Linux, macOS, or Windows (WSL2 recommended)dot -V

- Python: 3.8 or higher```

- RAM: 8GBOn Debian/Ubuntu:

- Disk Space: 10GB free```bash

- Internet connection (for setup)sudo apt-get update && sudo apt-get install -y graphviz

```

**Recommended:**- ✅ **Git** (for cloning repositories)

- OS: Ubuntu 20.04+ or macOS 12+- ✅ **50GB+** free disk space

- Python: 3.9 or 3.10- ✅ **16GB+** RAM (8GB minimum)

- RAM: 16GB- ⚙️ **GPU** (optional - framework runs on CPU by default)

- Disk Space: 50GB free (with datasets)

- GPU: NVIDIA GPU with 6GB+ VRAM (optional, for faster evaluation)**Check Conda installation:**

```bash

### Software Prerequisitesconda --version

# If not installed, download from: https://docs.conda.io/en/latest/miniconda.html

1. **Python 3.8+**```

   ```bash

   python3 --version  # Should be 3.8 or higher---

   ```

## 🚀 Streamlined Setup (3 Simple Steps)

2. **pip** (Python package manager)

   ```bashThe framework has been optimized to minimize setup steps. You now need only **3 commands** to get started!

   pip --version

   ```### Step 1: Navigate to Framework Directory



3. **Git** (to clone the repository)```bash

   ```bash# Navigate to the framework

   git --versioncd ../PIDS_Comparative_Framework

   ```

# Verify you're in the correct directory

4. **CUDA Toolkit** (optional, for GPU support)ls

   ```bash# Expected output: README.md, setup.md, scripts/, models/, etc.

   nvidia-smi  # Check if GPU is available```

   ```

---

---

### Step 2: Run Complete Environment Setup

## 🚀 Installation Methods

**This single script does everything:**

You can install the framework using one of three methods:- ✅ Creates conda environment

- ✅ Installs all core dependencies

1. **Quick Setup Script** (Recommended) - Automated one-command setup- ✅ Applies PyTorch MKL fix automatically

2. **Conda Environment** - Isolated environment with all dependencies- ✅ Creates directory structure

3. **Manual pip Install** - For advanced users- ✅ Verifies installation



---```bash

# Make setup script executable (if not already)

## ⚡ Quick Setup (Recommended)chmod +x scripts/setup.sh



### Option 1: Automated Setup Script# Run the complete setup script

./scripts/setup.sh

```bash```

# Clone the repository

git clone https://github.com/yourusername/PIDS_Comparative_Framework.git**What this does:**

cd PIDS_Comparative_Framework1. ✅ Checks for Conda installation

2. ✅ Creates `pids_framework` conda environment (Python 3.10)

# Run the setup script (installs everything)3. ✅ Installs PyTorch 1.12.1 with CUDA 11.6

bash scripts/setup.sh4. ✅ Installs DGL 1.0.0 (Deep Graph Library)

```5. ✅ Installs PyTorch Geometric 2.1.0

6. ✅ Applies MKL threading fix automatically (no separate step needed!)

**What the setup script does:**7. ✅ Creates necessary directories

1. ✅ Checks Python version8. ✅ Verifies installation

2. ✅ Creates virtual environment

3. ✅ Installs all dependencies**Expected output:**

4. ✅ Downloads pretrained model weights```

5. ✅ Sets up directory structureStep 1/7: Checking for Conda installation...

6. ✅ Verifies installation✓ Conda is installed: conda 23.x.x



**Expected output:**Step 2/7: Creating pids_framework environment...

```✓ Environment created successfully

✓ Python 3.9.7 found

✓ Creating virtual environment...Step 3/7: Initializing Conda for shell...

✓ Installing PyTorch...✓ Conda initialized

✓ Installing framework dependencies...

✓ Downloading pretrained weights...Step 4/7: Activating environment...

✓ Verifying installation...✓ Environment activated: pids_framework



Setup complete! 🎉Step 5/7: Applying PyTorch MKL threading fix...

✓ MKL threading fix applied (will auto-activate with environment)

To activate the environment:✓ PyTorch is working!

    source venv/bin/activate

Step 6/7: Creating directory structure...

To test the installation:✓ Directory structure created

    python scripts/verify_installation.py

```Step 7/7: Verifying installation...

✓ All core dependencies installed successfully

### Activate the Environment

============================================

```bash✓ Setup completed successfully!

# Linux/macOS============================================

source venv/bin/activate```



# Windows (if using WSL or Git Bash)**If setup fails**, the script includes automatic fixes for common issues. See [Troubleshooting](#troubleshooting) section if problems persist.

source venv/Scripts/activate

```---



### Verify Installation### Step 3: Activate Conda Environment



```bash```bash

python scripts/verify_installation.py# Activate the environment

```conda activate pids_framework



**Expected output:**# Verify activation

```echo $CONDA_DEFAULT_ENV

✓ Python version: 3.9.7# Should output: pids_framework

✓ PyTorch version: 2.0.1```

✓ CUDA available: Yes (CUDA 11.8)

✓ All required packages installed**Important:** You must activate this environment every time you use the framework.

✓ Model implementations found: 5

✓ Pretrained weights found: 5---



Installation verified successfully! ✓### Step 4: Setup Models and Pretrained Weights

```

**This unified script handles everything:**

---- ✅ Installs model-specific dependencies

- ✅ **Downloads pretrained weights from GitHub repositories** (primary method)

## 🐍 Manual Setup- ✅ Falls back to local directories if GitHub download fails



### Step 1: Clone Repository```bash

# Setup ALL models (recommended - downloads from GitHub!)

```bashpython scripts/setup_models.py --all

git clone https://github.com/yourusername/PIDS_Comparative_Framework.git

cd PIDS_Comparative_Framework# OR setup specific models only

```python scripts/setup_models.py --models magic kairos orthrus



### Step 2: Create Virtual Environment# List available models and their GitHub repos

python scripts/setup_models.py --list

#### Using venv (Python standard)```



```bash**How it works:**

# Create virtual environment1. **Primary Method**: Downloads weights directly from official GitHub repositories:

python3 -m venv venv   - **MAGIC**: https://github.com/FDUDSDE/MAGIC

   - **Continuum_FL**: https://github.com/kamelferrahi/Continuum_FL

# Activate it   - **ThreaTrace**: https://github.com/threaTrace-detector/threaTrace

source venv/bin/activate  # Linux/macOS   - **Orthrus**: https://github.com/ubc-provenance/orthrus

# OR   - **Kairos**: https://github.com/ubc-provenance/kairos

venv\Scripts\activate  # Windows

```2. **Fallback Method**: If GitHub download fails, searches local directories:

   - `../MAGIC/checkpoints/`

#### Using Conda (Recommended for researchers)   - `../Continuum_FL/checkpoints/`

   - `../orthrus/weights/`

```bash   - `../kairos/DARPA/`

# Create conda environment from yaml   - `../threaTrace/example_models/`

conda env create -f environment.yml

**Expected output:**

# Activate it```

conda activate pids_framework================================================================================

```  PIDS Framework - Model Setup (GitHub Download)

================================================================================

### Step 3: Install DependenciesSetting up models: magic, kairos, orthrus, threatrace, continuum_fl

Strategy: Download from GitHub → Fallback to local if needed

```bash

# Upgrade pip================================================================================

pip install --upgrade pip  Setting up MAGIC

================================================================================

# Install PyTorch (CPU version)Description: Masked Graph Autoencoder for APT Detection

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpuGitHub: https://github.com/FDUDSDE/MAGIC



# For GPU support (CUDA 11.8)Installing dependencies for MAGIC...

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118✓ MAGIC dependencies installed



# Install framework dependenciesDownloading weights for MAGIC...

pip install -r requirements.txt   Repository: https://github.com/FDUDSDE/MAGIC

```   📥 streamspot: MAGIC trained on StreamSpot dataset

   Downloading with curl: checkpoint-streamspot.pt

### Step 4: Install Framework in Development Mode   ✓ Downloaded: checkpoint-streamspot.pt

   📥 cadets: MAGIC trained on DARPA CADETS

```bash   ✓ Downloaded: checkpoint-cadets.pt

pip install -e .   ...

```✓ Downloaded 5 checkpoint(s) from GitHub/official sources



### Step 5: Verify Installation...



```bash================================================================================

python -c "import torch; print(f'PyTorch {torch.__version__}')"  Setup Summary

python -c "from models import ModelRegistry; print(f'Models: {ModelRegistry.list_models()}')"================================================================================

```✓ Dependencies installed for 5 model(s)

✓ Downloaded 18 checkpoint(s) from GitHub/official sources

---✓ Copied 2 checkpoint(s) from local fallback



## 💾 Downloading Pretrained ModelsCheckpoints saved to: checkpoints/



### Automatic DownloadNext steps:

  1. Verify weights: ls -lh checkpoints/*/

```bash  2. Preprocess data: python scripts/preprocess_data.py

# Download all pretrained weights  3. Run evaluation: ./scripts/run_evaluation.sh

bash scripts/download_weights.sh```

```

**Note on Special Cases:**

### Manual Download- **Kairos**: Pre-trained weights available from [Google Drive](https://drive.google.com/drive/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C) (manual download required)

- **Orthrus**: Pre-trained weights available from [Zenodo](https://zenodo.org/records/14641605) (automatic download via curl/wget)

If automatic download fails, download manually:

The script will automatically use `curl` or `wget` (whichever is available) to download weights.

1. **Create checkpoints directory**

   ```bash---

   mkdir -p checkpoints/{magic,kairos,orthrus,threatrace,continuum_fl}

   ```### Step 5: Prepare Your Custom Dataset



2. **Download from releases** (or your model hosting)Place your SOC provenance data in the `custom_dataset` directory.

   ```bash

   # Example for MAGIC```bash

   wget https://github.com/yourusername/PIDS_Models/releases/download/v1.0/magic_checkpoints.zip# Ensure you're in the framework directory

   unzip magic_checkpoints.zip -d checkpoints/magic/cd ~/PIDS/PIDS_Comparative_Framework  # Adjust path as needed

   ```

# Check if custom_dataset directory exists (it should be a sibling directory)

3. **Verify checkpoints**ls -la ../custom_dataset/

   ```bash

   ls -R checkpoints/# Expected files:

   ```# endpoint_file.json       - File system events (read, write, create, delete)

# endpoint_network.json    - Network events (connect, send, receive)

Expected structure:# endpoint_process.json    - Process events (create, execute, terminate)

``````

checkpoints/

├── magic/**If custom_dataset doesn't exist or is empty:**

│   ├── checkpoint-cadets.pt

│   ├── checkpoint-streamspot.pt```bash

│   └── checkpoint-theia.pt# Navigate to parent directory

├── kairos/cd ..

│   ├── checkpoint-cadets-e3.pt

│   └── checkpoint-clearscope-e3.pt# Create custom_dataset directory

├── orthrus/mkdir -p custom_dataset

│   └── checkpoint-cadets-e3.pt

├── threatrace/# Copy your SOC logs (adjust source path)

│   ├── checkpoint-cadets.ptcp /path/to/your/soc/logs/*.json custom_dataset/

│   └── checkpoint-streamspot.pt

└── continuum_fl/# Verify files are present

    ├── checkpoint-cadets-e3.ptls -lh custom_dataset/

    └── checkpoint-streamspot.pt# Expected: At least one .json file with provenance events

```

# Return to framework directory

---cd PIDS_Comparative_Framework

```

## 📊 Preparing Custom Data

**Required Data Format:**

### Step 1: Collect Your SOC Logs- JSON files containing provenance events

- Events should include: timestamp, event_type, entity information (process, file, or network)

Export logs from your SOC platform in JSON format:- See existing `custom_dataset/` files for format examples



**Supported platforms:**---

- Elastic/ELK Stack

- Splunk### Step 6: Preprocess the Dataset

- Sysmon

- Linux auditdBefore evaluation, the data must be preprocessed to convert JSON logs into graph structures.

- Custom JSON logs

```bash

### Step 2: Organize Data Files# Activate environment (if not already active)

conda activate pids_framework

```bash

# Create data directory# Preprocess custom dataset

mkdir -p data/custompython scripts/preprocess_data.py \

    --input-dir ../custom_dataset \

# Place your JSON files    --output-dir data/custom_soc \

data/custom/    --dataset-name custom_soc

├── endpoint_process.json    # Process events

├── endpoint_file.json       # File events# Expected output:

└── endpoint_network.json    # Network events# ════════════════════════════════════════════════════════

```#   Preprocessing Dataset: custom_soc

# ════════════════════════════════════════════════════════

### Step 3: Validate Data Format# Input directory: ../custom_dataset

# Output directory: data/custom_soc

Check if your data matches the expected schema:# 

# Step 1/4: Loading JSON files...

```bash# ✓ Loaded endpoint_file.json (1234 events)

python scripts/validate_data.py --data-path data/custom# ✓ Loaded endpoint_network.json (567 events)

```# ✓ Loaded endpoint_process.json (890 events)

# Total events: 2691

**Example valid event:**# 

```json# Step 2/4: Building provenance graph...

{# ✓ Created 450 nodes (processes, files, sockets)

  "@timestamp": "2024-10-14T10:30:00.000Z",# ✓ Created 2241 edges (dependencies)

  "event": {# 

    "kind": "event",# Step 3/4: Extracting features...

    "category": ["process"]# ✓ Node features extracted (128-dim)

  },# ✓ Edge features extracted (64-dim)

  "process": {# 

    "pid": 1234,# Step 4/4: Saving preprocessed data...

    "name": "bash",# ✓ Saved graph: data/custom_soc/graph.bin

    "executable": "/bin/bash",# ✓ Saved features: data/custom_soc/features.pt

    "command_line": "bash -c 'ls -la'",# ✓ Saved metadata: data/custom_soc/metadata.json

    "parent": {# 

      "pid": 1000,# ✓ Preprocessing completed successfully!

      "name": "systemd"```

    }

  },**Verify preprocessed data:**

  "user": {

    "name": "root",```bash

    "id": "0"# Check output directory

  },ls -lh data/custom_soc/

  "host": {

    "name": "web-server-01"# Expected files:

  }# graph.bin         - DGL graph structure

}# features.pt       - Node/edge features (PyTorch tensor)

```# metadata.json     - Dataset statistics and info

```

### Step 4: Preprocess Data

**Common preprocessing options:**

```bash

python scripts/preprocess_data.py \```bash

    --data-path data/custom \# Preprocess with custom window size (temporal graph segmentation)

    --output data/custom/preprocessed.pkl \python scripts/preprocess_data.py \

    --time-window 3600 \    --input-dir ../custom_dataset \

    --entity-types process file network \    --output-dir data/custom_soc \

    --verbose    --dataset-name custom_soc \

```    --window-size 3600  # 1-hour time windows



**Parameters:**# Preprocess with specific event types only

- `--data-path`: Input directory with JSON filespython scripts/preprocess_data.py \

- `--output`: Output file (pickle format)    --input-dir ../custom_dataset \

- `--time-window`: Time window in seconds (default: 3600 = 1 hour)    --output-dir data/custom_soc \

- `--entity-types`: Entity types to extract    --dataset-name custom_soc \

- `--verbose`: Show detailed progress    --event-types process file  # Exclude network events



**Expected output:**# Get help on all options

```python scripts/preprocess_data.py --help

Loading JSON files...```

✓ Loaded 50000 process events

✓ Loaded 30000 file events---

✓ Loaded 20000 network events

### Step 7: Run Evaluation

Building provenance graph...

✓ Extracted 15000 unique entitiesNow you're ready to evaluate all pretrained models on your preprocessed data!

✓ Constructed 35000 edges

✓ Created 100 temporal snapshots```bash

# Run evaluation on ALL models (recommended)

Saving preprocessed data..../scripts/run_evaluation.sh --data-path data/custom_soc

✓ Saved to data/custom/preprocessed.pkl

# Expected output:

Preprocessing complete!# ════════════════════════════════════════════════════════

- Total events: 100000#     PIDS Framework - Model Evaluation

- Unique entities: 15000# ════════════════════════════════════════════════════════

- Edges: 35000# 

- Time range: 2024-10-01 to 2024-10-14# Configuration:

```#   Models:        magic, kairos, orthrus, threatrace, continuum_fl

#   Dataset:       custom_soc

### Step 5: Verify Preprocessed Data#   Data Path:     data/custom_soc

#   Device:        CPU

```bash#   Output:        results/evaluation_20251014_143000

python scripts/inspect_data.py --data-path data/custom/preprocessed.pkl#

```# Evaluating MAGIC...

# ✓ Loaded checkpoint: checkpoints/magic/checkpoint-cadets.pt

---# ✓ Evaluation completed (AUROC: 0.9245, F1: 0.8710)

#

## 🎯 Running Your First Evaluation# Evaluating Kairos...

# ✓ Loaded checkpoint: checkpoints/kairos/model_cadets.pt

### Quick Test with Single Model# ✓ Evaluation completed (AUROC: 0.9156, F1: 0.8523)

#

```bash# Evaluating Orthrus...

# Evaluate MAGIC model on custom data# ✓ Loaded checkpoint: checkpoints/orthrus/provdetector_main.pkl

python experiments/evaluate.py \# ✓ Evaluation completed (AUROC: 0.9087, F1: 0.8402)

    --model magic \#

    --dataset custom \# Evaluating ThreaTrace...

    --data-path data/custom \# ✓ Loaded checkpoint: checkpoints/threatrace/darpatc/cadets/

    --pretrained \# ✓ Evaluation completed (AUROC: 0.8956, F1: 0.8234)

    --output-dir results/test#

```# Evaluating Continuum_FL...

# ✓ Loaded checkpoint: checkpoints/continuum_fl/checkpoint-cadets-e3.pt

**Expected output:**# ✓ Evaluation completed (AUROC: 0.9123, F1: 0.8601)

```#

==================================================# ════════════════════════════════════════════════════════

PIDS Model Evaluation#   Evaluation Summary

==================================================# ════════════════════════════════════════════════════════

Device: cpu# ✓ All 5 models evaluated successfully

Models to evaluate: ['magic']# ✓ Results saved to: results/evaluation_20251014_143000

Dataset: custom# ✓ Comparison report: results/evaluation_20251014_143000/comparison.json

```

Loading dataset...

✓ Test dataset: 100 samples**Evaluate specific model only:**



==================================================```bash

Evaluating magic# Evaluate MAGIC only (fastest)

==================================================./scripts/run_evaluation.sh --model magic --data-path data/custom_soc

✓ Loading checkpoint: checkpoints/magic/checkpoint-streamspot.pt

Running evaluation...# Evaluate Kairos only

./scripts/run_evaluation.sh --model kairos --data-path data/custom_soc

Results for magic:

--------------------------------------------------# Evaluate multiple specific models

auc_roc: 0.8523./scripts/run_evaluation.sh --models magic kairos orthrus --data-path data/custom_soc

auc_pr: 0.7891```

f1: 0.7654

precision: 0.8123**GPU acceleration (if available):**

recall: 0.7234

Evaluation time: 2m 34s```bash

# Check GPU availability

Evaluation finished!python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

```

# Run on GPU

### Evaluate All Models./scripts/run_evaluation.sh --data-path data/custom_soc --device 0



```bash# Specify GPU device (if multiple GPUs)

python experiments/evaluate.py \CUDA_VISIBLE_DEVICES=1 ./scripts/run_evaluation.sh --data-path data/custom_soc

    --all-models \```

    --dataset custom \

    --data-path data/custom \---

    --pretrained \

    --output-dir results/all_models### Step 8: View Results

```

Check the evaluation results and comparison report.

### Compare Models

```bash

```bash# Navigate to results directory (use actual timestamp)

python experiments/compare.py \cd results/evaluation_20251014_143000

    --models magic kairos orthrus \

    --dataset custom \# View comparison report (JSON format)

    --data-path data/custom \cat comparison.json

    --pretrained \

    --output-dir results/comparison# Example output:

```# {

#   "dataset": "custom_soc",

---#   "timestamp": "2025-10-14T14:30:00Z",

#   "models": {

## ⚙️ Configuration#     "magic": {

#       "auroc": 0.9245,

### Using Config Files#       "auprc": 0.8532,

#       "f1": 0.8710,

Create a custom configuration file:#       "precision": 0.8456,

#       "recall": 0.8973,

```yaml#       "fpr": 0.0823,

# configs/experiments/my_evaluation.yaml#       "checkpoint": "checkpoints/magic/checkpoint-cadets.pt"

#     },

models:#     "kairos": {

  - magic#       "auroc": 0.9156,

  - kairos#       "auprc": 0.8421,

  - orthrus#       ...

#     }

dataset:#   },

  name: custom#   "best_model": "magic",

  path: data/custom#   "best_auroc": 0.9245

  batch_size: 32# }

  num_workers: 4

# View individual model results

evaluation:cat magic_results.json

  pretrained: truecat kairos_results.json

  detection_level: entity

  k_neighbors: 5# Check detailed logs

tail -n 100 magic_evaluation.log

system:tail -n 100 kairos_evaluation.log

  device: -1  # -1 for CPU, 0+ for GPU

  seed: 42# View predictions (if saved)

head -n 20 predictions.csv

output:```

  dir: results/my_eval

  save_predictions: true**Understanding the results:**

```

| Metric | Description | Good Value |

Run with config:|--------|-------------|------------|

| **AUROC** | Area Under ROC Curve | > 0.90 |

```bash| **AUPRC** | Area Under Precision-Recall | > 0.80 |

python experiments/evaluate.py --config configs/experiments/my_evaluation.yaml| **F1** | Harmonic mean of precision/recall | > 0.85 |

```| **Precision** | Correct detections / Total detections | > 0.80 |

| **Recall** | Detected attacks / Total attacks | > 0.85 |

### Model-Specific Configuration| **FPR** | False Positive Rate | < 0.10 |



Each model can be configured individually:**Next steps based on results:**



```yaml- **AUROC > 0.90**: ✅ Excellent! Model ready for deployment

# configs/models/magic.yaml- **AUROC 0.80-0.90**: ⚠️ Good, consider fine-tuning for better performance

- **AUROC < 0.80**: ❌ Poor fit, consider retraining on labeled data (see Advanced section)

architecture:

  num_hidden: 256---

  num_layers: 4

  n_dim: 128## 📊 Complete Workflow Summary

  e_dim: 64

  mask_rate: 0.3Here's the complete workflow from start to finish:

  negative_slope: 0.2

  alpha_l: 2```bash

# 1. Setup environment (one-time)

training:./scripts/setup.sh

  batch_size: 32conda activate pids_framework

  learning_rate: 0.001

  epochs: 50# 2. Setup models and weights (one-time)

python scripts/setup_models.py --all

evaluation:

  k_neighbors: 5# 3. Prepare your data (one-time per dataset)

```# Copy your SOC logs to ../custom_dataset/

ls -la ../custom_dataset/*.json

---

# 4. Preprocess data (one-time per dataset)

## 🐛 Troubleshootingpython scripts/preprocess_data.py \

    --input-dir ../custom_dataset \

### Issue 1: Import Errors    --output-dir data/custom_soc \

    --dataset-name custom_soc

**Problem:**

```# 5. Run evaluation (main workflow)

ImportError: No module named 'torch'./scripts/run_evaluation.sh --data-path data/custom_soc

```

# 6. View results

**Solution:**cat results/evaluation_*/comparison.json

```bash```

# Ensure virtual environment is activated

source venv/bin/activate  # or conda activate pids_framework---



# Reinstall PyTorch## 🔧 Advanced Options

pip install torch torchvision torchaudio

```### Preprocessing Options



---```bash

# Custom time window (default: 3600 seconds)

### Issue 2: CUDA Out of Memorypython scripts/preprocess_data.py \

    --input-dir ../custom_dataset \

**Problem:**    --output-dir data/custom_soc \

```    --dataset-name custom_soc \

RuntimeError: CUDA out of memory    --window-size 7200  # 2-hour windows

```

# Filter specific event types

**Solutions:**python scripts/preprocess_data.py \

    --input-dir ../custom_dataset \

1. **Reduce batch size:**    --output-dir data/custom_soc \

   ```bash    --dataset-name custom_soc \

   python experiments/evaluate.py --model magic --batch-size 16    --event-types process file  # Exclude network events

   ```

# See all options

2. **Use CPU instead:**python scripts/preprocess_data.py --help

   ```bash```

   python experiments/evaluate.py --model magic --device -1

   ```### Evaluation Options



3. **Clear GPU cache:**```bash

   ```python# Evaluate specific models only

   import torch./scripts/run_evaluation.sh --models magic kairos --data-path data/custom_soc

   torch.cuda.empty_cache()

   ```# Use GPU (if available)

./scripts/run_evaluation.sh --data-path data/custom_soc --device 0

---

# Custom output directory

### Issue 3: Pretrained Weights Not Found./scripts/run_evaluation.sh --data-path data/custom_soc --output-dir results/my_eval



**Problem:**# See all options

```./scripts/run_evaluation.sh --help

FileNotFoundError: Checkpoint not found: checkpoints/magic/checkpoint-cadets.pt```

```

---

**Solutions:**

## 🔄 Advanced: Retraining Models (Optional)

1. **Download weights:**

   ```bash> ⚠️ **Only needed if pretrained models achieve AUROC < 0.80 on your data**

   bash scripts/download_weights.sh

   ```### Prerequisites

- Labeled attack data (ground truth)

2. **Specify checkpoint manually:**- Sufficient training data (thousands of events)

   ```bash- Time (training can take hours)

   python experiments/evaluate.py \

       --model magic \### Quick Retraining Guide

       --checkpoint /path/to/your/checkpoint.pt

   ``````bash

# 1. Prepare labeled data

3. **Skip pretrained (use random init):**cat > data/custom_soc/ground_truth.json << EOF

   ```bash{

   python experiments/evaluate.py --model magic --dataset custom  "attacks": [

   # (omit --pretrained flag)    {

   ```      "start_time": "2025-10-14T08:00:00Z",

      "end_time": "2025-10-14T09:00:00Z",

---      "attack_type": "apt",

      "entities": [1234, 1235, 1236]

### Issue 4: Data Preprocessing Fails    }

  ]

**Problem:**}

```EOF

JSONDecodeError: Expecting value: line 1 column 1

```# 2. Preprocess with labels

python scripts/preprocess_data.py \

**Solutions:**    --input-dir ../custom_dataset \

    --output-dir data/custom_soc \

1. **Check JSON format:**    --dataset-name custom_soc \

   ```bash    --labels-file data/custom_soc/ground_truth.json

   python -m json.tool data/custom/endpoint_process.json

   ```# 3. Train model (CPU)

python experiments/train.py \

2. **Validate data:**    --model magic \

   ```bash    --dataset custom_soc \

   python scripts/validate_data.py --data-path data/custom    --data-path data/custom_soc \

   ```    --epochs 100 \

    --batch-size 32 \

3. **Check file encoding:**    --device -1

   ```bash

   file data/custom/endpoint_process.json# 4. Evaluate retrained model

   # Should be: UTF-8 Unicode textpython experiments/evaluate.py \

   ```    --model magic \

    --checkpoint checkpoints/magic_custom_soc_best.pt \

---    --dataset custom_soc \

    --data-path data/custom_soc

### Issue 5: Slow Evaluation```



**Problem:** Evaluation takes too long**Fine-tuning (recommended over training from scratch):**

```bash

**Solutions:**python experiments/train.py \

    --model magic \

1. **Use GPU:**    --pretrained checkpoints/magic/checkpoint-cadets.pt \

   ```bash    --dataset custom_soc \

   python experiments/evaluate.py --model magic --device 0    --data-path data/custom_soc \

   ```    --epochs 50 \

    --learning-rate 0.0001 \

2. **Reduce data size:**    --device -1

   ```bash```

   python scripts/preprocess_data.py --data-path data/custom --max-events 10000

   ```---



3. **Increase batch size (if memory allows):**## 🐛 Troubleshooting

   ```bash

   python experiments/evaluate.py --model magic --batch-size 64### Common Issues

   ```

#### 1. Conda Not Found

4. **Use multiple workers:**```bash

   ```bash# Install Miniconda

   python experiments/evaluate.py --model magic --num-workers 8wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

   ```bash Miniconda3-latest-Linux-x86_64.sh

# Restart terminal and retry

---```



### Issue 6: Permission Denied#### 2. Environment Not Activated

```bash

**Problem:**conda activate pids_framework

```# Verify: echo $CONDA_DEFAULT_ENV

PermissionError: [Errno 13] Permission denied```

```

#### 3. Custom Dataset Not Found

**Solution:**```bash

```bash# Create and populate directory

# Fix permissionsmkdir -p ../custom_dataset

chmod +x scripts/*.shcp /path/to/your/logs/*.json ../custom_dataset/

chmod +x scripts/*.pyls -la ../custom_dataset/

```

# Run with appropriate permissions

sudo python experiments/evaluate.py  # (not recommended)#### 4. Pretrained Weights Missing

# OR```bash

# Fix directory permissions# Re-run model setup

chown -R $USER:$USER .python scripts/setup_models.py --all

```

# Or copy from local directories

---python scripts/setup_models.py --models magic --copy-local

```

## 🔬 Advanced Setup

#### 5. Out of Memory

### 1. Multi-GPU Support```bash

# Use CPU instead of GPU

```bash./scripts/run_evaluation.sh --device -1

# Use specific GPU

CUDA_VISIBLE_DEVICES=0 python experiments/evaluate.py --model magic --device 0# Or reduce batch size in preprocessing

python scripts/preprocess_data.py --batch-size 16

# Use multiple GPUs (if model supports it)```

CUDA_VISIBLE_DEVICES=0,1 python experiments/evaluate.py --model magic

```#### 6. PyTorch MKL Error

**Error:** `undefined symbol: iJIT_NotifyEvent`

### 2. Distributed Evaluation

**Solution:** The setup script already applies the fix automatically. If error persists:

```bash```bash

# For very large datasets, use distributed evaluationconda activate pids_framework

python -m torch.distributed.launch \export MKL_THREADING_LAYER=GNU

    --nproc_per_node=4 \# Test: python -c "import torch; print(torch.__version__)"

    experiments/evaluate.py \```

    --model magic \

    --dataset custom \#### 7. Import Errors

    --distributed```bash

```# Ensure environment is activated

conda activate pids_framework

### 3. Docker Setup

# Reinstall dependencies

```bash./scripts/setup.sh

# Build Docker image```

docker build -t pids_framework .

#### 8. Permission Denied

# Run in container```bash

docker run --gpus all \chmod +x scripts/*.sh

    -v $(pwd)/data:/app/data \./scripts/run_evaluation.sh

    -v $(pwd)/results:/app/results \```

    pids_framework \

    python experiments/evaluate.py --all-models --dataset custom### Get Help

```

```bash

### 4. Cluster Deployment (Slurm)# Script help

python scripts/preprocess_data.py --help

```bash./scripts/run_evaluation.sh --help

# Submit job to Slurm clusterpython scripts/setup_models.py --help

sbatch scripts/slurm_evaluate.sh

```# Check installation

python -c "import torch; import dgl; import torch_geometric; print('All imports OK')"

Example `slurm_evaluate.sh`:```

```bash

#!/bin/bash---

#SBATCH --job-name=pids_eval

#SBATCH --nodes=1## 📚 Additional Resources

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=8- **README.md** - Complete framework documentation

#SBATCH --mem=32G- **configs/** - Configuration files for models and datasets

#SBATCH --time=04:00:00- **scripts/** - All utility scripts with `--help` options

#SBATCH --gres=gpu:1- **experiments/** - Training, evaluation, and comparison scripts



module load python/3.9### Get Help

source venv/bin/activate

```bash

python experiments/evaluate.py \# Script help

    --all-models \./scripts/run_evaluation.sh --help

    --dataset custom \python scripts/download_weights.py --help

    --pretrained \python scripts/preprocess_data.py --help

    --output-dir results/clusterpython experiments/train.py --help

```python experiments/evaluate.py --help

python experiments/compare.py --help

---

# Model information

## ✅ Post-Setup Checklistpython -c "from models import list_available_models; print(list_available_models())"



- [ ] Python 3.8+ installed# Check installation

- [ ] Virtual environment created and activatedpython scripts/verify_installation.py

- [ ] All dependencies installed (`pip list` shows torch, dgl, etc.)```

- [ ] Pretrained weights downloaded

- [ ] Custom data prepared and preprocessed---

- [ ] Test evaluation runs successfully

- [ ] Configuration files created## 🎯 Quick Reference

- [ ] GPU detected (if applicable)

### Essential Workflow

---

```bash

## 📚 Next Steps# ONE-TIME SETUP (Steps 1-4)

# 1. Setup environment

1. **Run your first evaluation** - See "Running Your First Evaluation" above./scripts/setup.sh

2. **Read the main README** - [README.md](README.md) for architecture detailsconda activate pids_framework

3. **Explore configurations** - Check `configs/` directory

4. **Add a new model** - See [EXTEND.md](EXTEND.md)# 2. Setup models

python scripts/setup_models.py --all

---

# 3. Prepare data (copy your SOC logs)

## 📧 Getting Helpls -la ../custom_dataset/*.json



If you encounter issues not covered here:# 4. Preprocess data

python scripts/preprocess_data.py \

1. Check the [README.md](README.md)    --input-dir ../custom_dataset \

2. Search [GitHub Issues](https://github.com/yourusername/PIDS_Comparative_Framework/issues)    --output-dir data/custom_soc \

3. Open a new issue with:    --dataset-name custom_soc

   - Your setup (OS, Python version, GPU)

   - Error messages# MAIN WORKFLOW (Steps 5-6)

   - Steps to reproduce# 5. Run evaluation

./scripts/run_evaluation.sh --data-path data/custom_soc

---

# 6. View results

**Setup complete! Ready to evaluate PIDS models! 🎉**cat results/evaluation_*/comparison.json

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
