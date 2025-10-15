# Quick Fix Summary & Next Steps

## ✅ Fixes Applied Successfully

I've fixed the following issues in your PIDS Comparative Framework:

### 1. **Device Handling Errors** ✅
- **Issue**: ThreaTrace and Continuum_FL crashed with "Device index must not be negative"
- **Fix**: Added safe device initialization that falls back to CPU if GPU specification is invalid
- **Files**: `models/base_model.py`, `models/threatrace_wrapper.py`, `models/continuum_fl_wrapper.py`

### 2. **Import Errors** ✅
- **Issue**: Kairos and Orthrus couldn't import due to missing `prepare_*_batch` functions
- **Fix**: Added the missing batch preparation functions
- **Files**: `models/implementations/kairos/__init__.py`, `models/implementations/orthrus/__init__.py`

### 3. **Compare Script Arguments** ✅
- **Issue**: `compare.py` didn't accept `--results-dir`, `--output-file`, `--generate-plots`
- **Fix**: Added these CLI arguments to the script
- **File**: `experiments/compare.py`

---

## ⚠️ Remaining Issues to Investigate

### **Empty Dataset Problem** (0 events loaded)

Your preprocessed data file has **0 events**. This is why evaluations skip without running.

**To diagnose**, I've created a diagnostic tool:

```bash
cd PIDS_Comparative_Framework

# Run diagnostic
python scripts/diagnose_data.py data/custom_soc

# This will show you:
# - Whether source JSON files exist and have data
# - What's in the preprocessed pickle file
# - Why the dataset loader might be failing
```

---

## 🚀 Next Steps

### **Option 1: Fix Your Custom Data** (Recommended if you want to use your SOC data)

1. **Check source data exists**:
   ```bash
   ls -lh data/custom_soc/*.json
   ```

2. **Run diagnostic**:
   ```bash
   python scripts/diagnose_data.py data/custom_soc
   ```

3. **Re-run preprocessing** (if source files have data):
   ```bash
   python scripts/preprocess_data.py \
       --dataset custom_soc \
       --data-path data/custom_soc \
       --output-dir data/custom_soc \
       --verbose
   ```

4. **Re-run evaluation**:
   ```bash
   ./scripts/run_evaluation.sh \
       --data-path data/custom_soc \
       --skip-preprocess
   ```

### **Option 2: Test with Sample Data** (Verify framework works)

```bash
# Download DARPA CADETS dataset (if not already present)
# Then run evaluation
./scripts/run_evaluation.sh \
    --dataset cadets \
    --data-path data/darpa/cadets \
    --skip-download
```

### **Option 3: Train Models on Your Data** (If you have data but no checkpoints)

```bash
# Train MAGIC on your custom data
python experiments/train.py \
    --model magic \
    --dataset custom_soc \
    --data-path data/custom_soc \
    --epochs 50 \
    --batch-size 32

# Then evaluate
./scripts/run_evaluation.sh \
    --model magic \
    --data-path data/custom_soc \
    --skip-preprocess
```

---

## 📝 Expected Data Format

Your JSON files should look like:

**endpoint_process.json**:
```json
[
  {
    "timestamp": "2024-01-01T00:00:00Z",
    "event_type": "PROCESS_START",
    "subject": {"uuid": "proc-123", "pid": 1234},
    "object": {"path": "/bin/bash"},
    ...
  },
  ...
]
```

If your format is different, you may need to:
1. Transform it to match the expected schema
2. Create a custom dataset loader (see `EXTEND.md`)

---

## 🔍 Debugging Commands

```bash
# Check if preprocessed file exists and size
ls -lh data/custom_soc/custom_soc_graph.pkl

# Check source JSON files
for f in data/custom_soc/*.json; do 
    echo "$f: $(jq length $f 2>/dev/null || echo 'not valid JSON') entries"
done

# View preprocessing config
cat configs/datasets/custom_soc.yaml

# Check model configs
ls -lh configs/models/

# View evaluation logs
tail -f results/evaluation_*/magic_evaluation.log
```

---

## 📚 Documentation References

- **Full fixes details**: See `FIXES_APPLIED.md`
- **Extending framework**: See `EXTEND.md`
- **Setup instructions**: See `SETUP.md`
- **Script analysis**: See `SCRIPT_ANALYSIS.md`

---

## ✉️ Need Help?

If you're still stuck:

1. **Run diagnostic**: `python scripts/diagnose_data.py data/custom_soc`
2. **Check logs**: Look in `results/evaluation_*/`
3. **Verify environment**: 
   ```bash
   conda activate pids_framework
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

---

## 🎯 Summary

**What's Fixed**: Device errors, import errors, CLI arguments ✅
**What Needs Investigation**: Empty dataset (0 events) ⚠️
**Next Action**: Run `python scripts/diagnose_data.py data/custom_soc`

Good luck! 🚀
