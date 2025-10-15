# PIDS Comparative Framework - Script Analysis

Analysis of all scripts to identify essential vs redundant scripts.

**Analysis Date:** October 14, 2025  
**Analyzed Scripts:** 8 scripts in `scripts/` directory

---

## 📊 Script Classification

### ✅ Essential Scripts (Keep)

These scripts are critical for framework operation and should be retained:

#### 1. **setup.sh** (309 lines)
- **Purpose:** Complete automated environment setup
- **Status:** ✅ ESSENTIAL
- **Functionality:**
  - Creates conda environment from environment.yml
  - Installs PyTorch 1.12.1 with CUDA 11.6
  - Installs DGL 1.0.0
  - Installs PyTorch Geometric and extensions
  - Applies MKL threading fix automatically
  - Creates directory structure
  - Verifies installation
- **User Need:** PRIMARY - First script users run for setup
- **Dependencies:** Conda, environment.yml
- **Rationale:** Core setup script, cannot be removed

#### 2. **setup_models.py** (871 lines)
- **Purpose:** Download pretrained weights and install model-specific dependencies
- **Status:** ✅ ESSENTIAL
- **Functionality:**
  - Downloads pretrained weights from GitHub for 5 models
  - Handles multiple download methods (curl, wget, git sparse-checkout, gdown)
  - Local fallback if weights exist in external directories
  - Installs model-specific dependencies
  - Force re-download option
  - Download-only or install-only modes
- **User Need:** PRIMARY - Required to get pretrained weights
- **Arguments:** `--all`, `--models`, `--force-download`, `--no-install`, `--download-only`, `--no-copy`, `--list`
- **Rationale:** Essential for evaluation workflow, downloads 500MB+ of weights

#### 3. **preprocess_data.py** (435 lines)
- **Purpose:** Convert JSON logs to graph format for model evaluation
- **Status:** ✅ ESSENTIAL
- **Functionality:**
  - Loads JSON logs with chunked reading (memory efficient)
  - Supports Elastic/custom schemas
  - Builds provenance graphs (DGL/PyG format)
  - Extracts node and edge features
  - Saves preprocessed data (graph.pkl, features.pt, metadata.json)
  - Temporal windowing support
  - Event type filtering
- **User Need:** PRIMARY - Required to evaluate on custom data
- **Arguments:** `--input-dir`, `--output-dir`, `--dataset-name`, `--schema`, `--time-window`, `--event-types`, `--chunk-size`, `--verbose`
- **Rationale:** Core data preparation step, cannot skip for custom data

#### 4. **run_evaluation.sh** (286 lines)
- **Purpose:** Complete evaluation workflow orchestration
- **Status:** ✅ ESSENTIAL
- **Functionality:**
  - Checks conda environment activation
  - Calls setup_models.py if weights missing
  - Auto-detects preprocessed data
  - Runs preprocessing if needed
  - Evaluates all 5 models (or specific models)
  - Generates comparison report
  - Creates timestamped result directories
- **User Need:** PRIMARY - Main entry point for evaluation
- **Arguments:** `--model`, `--dataset`, `--data-path`, `--skip-download`, `--skip-preprocess`, `--output-dir`
- **Rationale:** Primary user-facing script, orchestrates entire workflow

#### 5. **verify_installation.py** (373 lines)
- **Purpose:** Verify framework installation with 9 comprehensive checks
- **Status:** ✅ ESSENTIAL
- **Functionality:**
  - Checks Python version (3.8-3.10 required)
  - Verifies core dependencies (torch, numpy, pandas, etc.)
  - Tests deep learning frameworks (PyTorch, DGL, PyG)
  - Validates model integrations (imports for all 5 models)
  - Checks directory structure (12 directories)
  - Verifies configuration files (10+ YAML configs)
  - Tests script executability (7 scripts)
  - Checks documentation files (4 markdown files)
  - Validates external model directories (5 optional)
- **User Need:** PRIMARY - Troubleshooting and validation
- **Arguments:** None (automatic checks)
- **Rationale:** Essential for troubleshooting, called by setup.sh

#### 6. **verify_implementation.py** (461 lines) - **UPDATED VERSION**
- **Purpose:** Comprehensive framework verification (updated October 2025)
- **Status:** ✅ ESSENTIAL - **RECENTLY UPDATED**
- **Functionality:**
  - Tests model implementations (imports for all 5 models)
  - Verifies ModelRegistry functionality
  - Checks framework components
  - Tests checkpoint availability
  - Validates configuration files
  - Verifies directory structure
  - Comprehensive reporting with 8 test categories
- **User Need:** PRIMARY - Development and validation
- **Arguments:** None (automatic checks)
- **Rationale:** Updated comprehensive verification, supersedes older checks
- **Note:** This is the NEW version created in the recent updates

---

### ⚠️ Redundant Scripts (Consider Removing)

These scripts have overlapping functionality or are no longer needed:

#### 7. **test_pytorch.py** (202 lines)
- **Purpose:** Test PyTorch installation and diagnose issues
- **Status:** ⚠️ REDUNDANT
- **Functionality:**
  - Checks Python environment
  - Tests PyTorch import
  - Tests CUDA availability
  - Tests dependencies (numpy, scipy, sklearn, etc.)
  - Tests PyTorch Geometric and DGL
  - Provides MKL fix suggestions
- **User Need:** SECONDARY - Debugging only
- **Overlap:** 90% overlap with `verify_installation.py` which does the same checks
- **Rationale for Removal:**
  - `verify_installation.py` performs all the same checks
  - `verify_installation.py` is called by setup.sh automatically
  - Having two similar scripts confuses users
  - The MKL fix is now automatic in setup.sh
  - Maintenance burden to keep both scripts in sync
- **Recommendation:** **REMOVE** or merge into verify_installation.py

#### 8. **extract_model_implementations.py** (164 lines)
- **Purpose:** One-time script to extract model implementations from external repos
- **Status:** ⚠️ ONE-TIME SCRIPT (not needed for users)
- **Functionality:**
  - Creates standalone model implementations from external repos
  - Extracts files from Kairos, Orthrus, Continuum_FL, ThreaTrace
  - Creates __init__.py and README.md for each model
  - Provides adaptation checklist
- **User Need:** NONE - This was a development-time script
- **Rationale for Removal:**
  - Models are already extracted and integrated
  - Users never need to run this
  - Only useful for framework developers during initial setup
  - Confuses users who see it in scripts/
  - Not documented in Setup.md or README.md
- **Recommendation:** **MOVE to dev_tools/** or **DELETE**
- **Alternative:** Create a `dev_tools/` directory for development scripts

---

## 📋 Summary

### Essential Scripts (6)
| Script | Lines | Purpose | User Need |
|--------|-------|---------|-----------|
| setup.sh | 309 | Environment setup | PRIMARY |
| setup_models.py | 871 | Download weights | PRIMARY |
| preprocess_data.py | 435 | Data preprocessing | PRIMARY |
| run_evaluation.sh | 286 | Evaluation workflow | PRIMARY |
| verify_installation.py | 373 | Installation verification | PRIMARY |
| verify_implementation.py | 461 | Framework verification | PRIMARY |

**Total: 2,735 lines of essential code**

### Redundant Scripts (2)
| Script | Lines | Reason for Removal |
|--------|-------|--------------------|
| test_pytorch.py | 202 | 90% overlap with verify_installation.py |
| extract_model_implementations.py | 164 | One-time dev script, not for users |

**Total: 366 lines to remove**

---

## 🎯 Recommendations

### Immediate Actions

1. **Delete `test_pytorch.py`**
   - Functionality covered by `verify_installation.py`
   - Reduces confusion
   - Saves 202 lines

2. **Move or Delete `extract_model_implementations.py`**
   - Option A: Delete entirely (models already extracted)
   - Option B: Move to `dev_tools/` directory for developers
   - Saves 164 lines from user-facing scripts/

3. **Update Documentation**
   - Remove references to `test_pytorch.py` from any docs
   - Update Setup.md script reference section (remove test_pytorch.py)
   - Update README.md to not mention extract_model_implementations.py

### Scripts Directory Structure (After Cleanup)

```
scripts/
├── setup.sh                     # ✅ Environment setup
├── setup_models.py              # ✅ Download weights
├── preprocess_data.py           # ✅ Data preprocessing
├── run_evaluation.sh            # ✅ Evaluation workflow
├── verify_installation.py       # ✅ Installation checks
└── verify_implementation.py     # ✅ Framework verification (updated)

Total: 6 essential scripts (2,735 lines)
```

### Optional: Development Tools Directory

If you want to preserve development scripts:

```
dev_tools/
├── extract_model_implementations.py  # Model extraction tool
├── benchmark_performance.py          # Performance benchmarking
└── README.md                         # Developer documentation
```

---

## 🔍 Detailed Analysis

### Why Remove test_pytorch.py?

**Functional Overlap Analysis:**

| Check | test_pytorch.py | verify_installation.py |
|-------|----------------|----------------------|
| Python version | ✅ | ✅ |
| PyTorch import | ✅ | ✅ |
| CUDA availability | ✅ | ✅ |
| Dependencies | ✅ | ✅ |
| PyTorch Geometric | ✅ | ✅ |
| DGL | ✅ | ✅ |
| MKL environment | ✅ | ✅ (implicitly) |
| Framework structure | ❌ | ✅ |
| Model imports | ❌ | ✅ |
| Config files | ❌ | ✅ |

**Verdict:** `verify_installation.py` does everything `test_pytorch.py` does, plus more. The 202 lines in test_pytorch.py are redundant.

### Why Remove/Move extract_model_implementations.py?

**Usage Analysis:**
- **When created:** Initial framework development (one-time)
- **When used:** Never used by end users
- **Target audience:** Framework developers only
- **Current status:** Models already extracted and integrated
- **User confusion:** Users see it in scripts/ and wonder if they need to run it
- **Documentation:** Not mentioned in Setup.md or README.md (correctly omitted)

**Verdict:** This is a development tool, not a user-facing script. Should be removed from scripts/ or moved to dev_tools/.

---

## ✅ Post-Cleanup Benefits

1. **Clearer User Experience**
   - Only 6 scripts instead of 8
   - No confusion about which verification script to use
   - No mysterious extract script

2. **Easier Maintenance**
   - Fewer scripts to maintain
   - No duplicate functionality to keep in sync
   - Clear purpose for each script

3. **Better Documentation**
   - Each script serves a unique purpose
   - No redundant documentation needed
   - Clearer command reference in Setup.md

4. **Reduced Size**
   - 366 fewer lines in scripts/
   - 13% reduction in script code

---

## 📝 Implementation Plan

### Step 1: Backup (Safety)
```bash
cd PIDS_Comparative_Framework
mkdir -p archive/deprecated_scripts
cp scripts/test_pytorch.py archive/deprecated_scripts/
cp scripts/extract_model_implementations.py archive/deprecated_scripts/
```

### Step 2: Remove Redundant Scripts
```bash
rm scripts/test_pytorch.py
rm scripts/extract_model_implementations.py
```

### Step 3: Verify Remaining Scripts
```bash
ls -la scripts/
# Should show only 6 scripts + verify_implementation.py
```

### Step 4: Update Documentation
- Update README.md (remove mentions of deleted scripts)
- Update Setup.md command reference (remove test_pytorch.py section)
- Update EXTEND.md if it references these scripts

### Step 5: Test
```bash
# Verify all essential workflows still work
./scripts/setup.sh
python scripts/verify_installation.py
python scripts/verify_implementation.py
```

---

## 🎓 Lessons for Future

### When to Keep a Script
- ✅ Unique functionality not covered elsewhere
- ✅ User-facing (part of standard workflow)
- ✅ Documented in main documentation
- ✅ Called by other scripts or users directly

### When to Remove a Script
- ❌ Overlaps >80% with another script
- ❌ One-time development tool
- ❌ Not documented (indicates it's not user-facing)
- ❌ Confuses users
- ❌ Not maintained

---

## 📊 Impact Assessment

### Removed Lines by Category

| Category | Lines Removed |
|----------|---------------|
| Redundant verification | 202 |
| Development tools | 164 |
| **Total** | **366** |

### User Impact
- ✅ **Positive:** Clearer, simpler scripts directory
- ✅ **Positive:** No confusion about which script to use
- ✅ **Positive:** Easier to maintain
- ⚠️ **Neutral:** No loss of functionality
- ❌ **Negative:** None

---

## 🎯 Conclusion

**Recommendation: Remove both redundant scripts**

1. **Delete `test_pytorch.py`** - Functionality fully covered by verify_installation.py
2. **Delete `extract_model_implementations.py`** - One-time dev tool, models already extracted

**Result:** Cleaner, more maintainable framework with no loss of functionality.

**Final Scripts Directory:** 6 essential scripts (2,735 lines) serving distinct purposes.

---

**Analysis by:** GitHub Copilot  
**Date:** October 14, 2025  
**Status:** Ready for implementation
