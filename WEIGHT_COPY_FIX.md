# Weight Copy Issue - FIXED ✅

## Problem

When running:
```bash
python scripts/download_weights.py --copy-existing --all-models
```

You got:
```
Traceback (most recent call last):
  File ".../scripts/download_weights.py", line 15, in <module>
    import gdown
ModuleNotFoundError: No module named 'gdown'
```

## Root Cause

The `download_weights.py` script was importing `gdown` (Google Drive downloader) at the top of the file, even when you just wanted to copy local files (which doesn't need gdown).

## Solution Applied ✅

### 1. Fixed `download_weights.py`
- Made `gdown` import conditional (only imports when actually downloading from Google Drive)
- Made `requests` and `tqdm` imports conditional too
- Fixed logging to not require missing `utils.common.setup_logger` function

### 2. Created New Simple Script: `copy_weights.py`
A dependency-free script that just copies weights from existing directories:
- No external dependencies needed (pure Python stdlib)
- Clear output with emojis
- Handles all 5 models (MAGIC, Continuum_FL, Orthrus, Kairos, ThreaTrace)
- Automatic detection of source directories

---

## 🚀 Use the Simple Script NOW

```bash
cd ~/PIDS/PIDS_Comparative_Framework
conda activate pids_framework

# Use the new simple script (no dependencies required!)
python scripts/copy_weights.py
```

This will:
1. ✅ Find MAGIC checkpoints in `../MAGIC/checkpoints/`
2. ✅ Find Continuum_FL checkpoints in `../Continuum_FL/checkpoints/`
3. ✅ Find Orthrus weights in `../orthrus/weights/`
4. ✅ Find Kairos weights in `../kairos/DARPA/`
5. ✅ Find ThreaTrace models in `../threaTrace/models/`
6. ✅ Copy all to framework's `checkpoints/` directory

---

## Expected Output

```
================================================================================
  PIDS Framework - Copy Existing Pretrained Weights
================================================================================

📦 Checking MAGIC weights...
  ✓ Copied: checkpoint-cadets-e3.pt -> magic/checkpoint-cadets-e3.pt
  ✓ Copied: checkpoint-streamspot.pt -> magic/checkpoint-streamspot.pt
  ✓ Copied: checkpoint-theia-e3.pt -> magic/checkpoint-theia-e3.pt
  ✓ Copied: checkpoint-trace-e3.pt -> magic/checkpoint-trace-e3.pt
  ✓ Copied 4 MAGIC checkpoint(s)

📦 Checking Continuum_FL weights...
  ✓ Copied: checkpoint-cadets-e3.pt -> continuum_fl/checkpoint-cadets-e3.pt
  ✓ Copied: checkpoint-clearscope-e3.pt -> continuum_fl/checkpoint-clearscope-e3.pt
  ✓ Copied: checkpoint-streamspot.pt -> continuum_fl/checkpoint-streamspot.pt
  ✓ Copied 3 Continuum_FL checkpoint(s)

📦 Checking Orthrus weights...
  ✓ Copied: CADETS_E3.pkl -> orthrus/CADETS_E3.pkl
  ✓ Copied: THEIA_E3.pkl -> orthrus/THEIA_E3.pkl
  ✓ Copied 2 Orthrus checkpoint(s)

📦 Checking Kairos weights...
  ⚠️  Source directory not found: ../kairos/DARPA

📦 Checking ThreaTrace weights...
  ⚠️  Source directory not found: ../threaTrace/models

================================================================================
✅ Successfully copied 9 checkpoint(s)
📁 Checkpoints saved to: /home/ortsoc-admin/PIDS/PIDS_Comparative_Framework/checkpoints

Next steps:
  1. Verify checkpoints: ls -lR checkpoints/
  2. Run evaluation: ./scripts/run_evaluation.sh
================================================================================
```

---

## Verify Checkpoints Were Copied

```bash
# Check what was copied
ls -lR checkpoints/

# Expected structure:
# checkpoints/
#   magic/
#     checkpoint-cadets-e3.pt
#     checkpoint-streamspot.pt
#     checkpoint-theia-e3.pt
#     checkpoint-trace-e3.pt
#   continuum_fl/
#     checkpoint-cadets-e3.pt
#     checkpoint-streamspot.pt
#     ...
#   orthrus/
#     CADETS_E3.pkl
#     THEIA_E3.pkl
#     ...
```

---

## If You Still Want to Use download_weights.py

The original script now works too, but requires optional dependencies:

```bash
# Install optional dependencies for downloading from URLs
pip install gdown requests tqdm

# Then you can use:
python scripts/download_weights.py --copy-existing --all-models

# Or download from Google Drive/URLs:
python scripts/download_weights.py --model magic --variant streamspot
```

But **you don't need these dependencies** if you're just copying local files with `copy_weights.py`!

---

## Summary

✅ **Problem**: `gdown` import error  
✅ **Fixed**: Made imports conditional in `download_weights.py`  
✅ **Created**: New simple `copy_weights.py` script (no dependencies)  
✅ **Use**: `python scripts/copy_weights.py` (works now!)  

---

## Next Steps

1. ✅ Copy weights: `python scripts/copy_weights.py`
2. ✅ Verify: `ls -lR checkpoints/`
3. ✅ Preprocess data: `python scripts/preprocess_data.py --input-dir ../custom_dataset/`
4. ✅ Run evaluation: `./scripts/run_evaluation.sh`

**Your next command:**
```bash
python scripts/copy_weights.py
```
