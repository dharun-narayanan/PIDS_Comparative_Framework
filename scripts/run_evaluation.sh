#!/bin/bash
# run_evaluation.sh - Default workflow: Evaluate pretrained models on custom dataset
#
# This is the PRIMARY use case for the PIDS Comparative Framework:
# 1. Download pretrained weights
# 2. Preprocess your custom SOC data
# 3. Evaluate all models on your data
# 4. Compare performance metrics
#
# Usage:
#   ./scripts/run_evaluation.sh                    # Evaluate all models on custom data
#   ./scripts/run_evaluation.sh --model magic      # Evaluate specific model
#   ./scripts/run_evaluation.sh --help             # Show all options

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default parameters
MODEL="all"
DATASET="custom_soc"
DATASET_TYPE="auto"  # auto, darpa, custom_soc, custom
DATA_PATH="../custom_dataset"
DATA_FORMAT="auto"  # auto, json, ndjson, bin, avro
SKIP_DOWNLOAD=false
SKIP_PREPROCESS=false
MAX_EVENTS=""  # Empty means all events
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/evaluation_${TIMESTAMP}"
ARTIFACT_DIR="artifacts/artifacts_${TIMESTAMP}"  # Use timestamped artifacts directory

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dataset-type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --data-format)
            DATA_FORMAT="$2"
            shift 2
            ;;
        --max-events)
            MAX_EVENTS="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Default workflow: Evaluate pretrained models on your custom SOC data or DARPA datasets"
            echo ""
            echo "Options:"
            echo "  --model MODEL            Model to evaluate (magic, kairos, orthrus, threatrace, continuum_fl, all)"
            echo "                           Default: all"
            echo "  --dataset DATASET        Dataset name (e.g., custom_soc, cadets_e3, theia_e3)"
            echo "                           Default: custom_soc"
            echo "  --dataset-type TYPE      Dataset type (auto, darpa, custom_soc, custom)"
            echo "                           Default: auto (auto-detect from path)"
            echo "  --data-path PATH         Path to preprocessed data (e.g., data/custom_soc)"
            echo "                           OR path to JSON/binary source files (e.g., ../DARPA/ta1-cadets-e3-official-1.json)"
            echo "                           Script will auto-detect if data is already preprocessed"
            echo "                           Default: ../custom_dataset"
            echo "  --data-format FORMAT     Data format (auto, json, ndjson, bin, avro)"
            echo "                           Default: auto (auto-detect)"
            echo "  --max-events NUM         Maximum events to process per file (for testing/sampling)"
            echo "                           Default: process all events"
            echo "  --skip-download          Skip downloading pretrained weights"
            echo "  --skip-preprocess        Skip data preprocessing (use if already preprocessed)"
            echo "  --output-dir DIR         Output directory for results"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate all models on custom SOC data"
            echo "  $0 --data-path ../custom_dataset --dataset custom_soc"
            echo ""
            echo "  # Evaluate DARPA CADETS dataset"
            echo "  $0 --data-path ../DARPA/ta1-cadets-e3-official-1.json --dataset cadets_e3 --dataset-type darpa"
            echo ""
            echo "  # Evaluate DARPA THEIA with binary AVRO files"
            echo "  $0 --data-path ../DARPA/ta1-theia-e3-official-1r.bin --dataset theia_e3 --data-format bin"
            echo ""
            echo "  # Quick test with sample of DARPA data"
            echo "  $0 --data-path ../DARPA/ta1-trace-e3-official-1.json --dataset trace_e3 --max-events 10000"
            echo ""
            echo "  # Evaluate specific model on already preprocessed data"
            echo "  $0 --model magic --data-path data/darpa/cadets_e3 --skip-preprocess"
            echo ""
            echo "  # Full workflow (download weights, preprocess, evaluate all)"
            echo "  $0"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}    PIDS Comparative Framework - Evaluation Workflow${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Model(s):      ${GREEN}${MODEL}${NC}"
echo -e "  Dataset:       ${GREEN}${DATASET}${NC}"
echo -e "  Dataset Type:  ${GREEN}${DATASET_TYPE}${NC}"
echo -e "  Data Path:     ${GREEN}${DATA_PATH}${NC}"
echo -e "  Data Format:   ${GREEN}${DATA_FORMAT}${NC}"
if [[ -n "$MAX_EVENTS" ]]; then
    echo -e "  Max Events:    ${GREEN}${MAX_EVENTS}${NC}"
fi
echo -e "  Output Dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo ""

# Check if conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "pids_framework" ]]; then
    echo -e "${RED}Error: Conda environment 'pids_framework' is not activated!${NC}"
    echo -e "${YELLOW}Please run: conda activate pids_framework${NC}"
    exit 1
fi

# Check if data path exists (can be directory or parent directory of preprocessed files)
if [[ ! -d "$DATA_PATH" ]]; then
    # Check if it's a parent directory containing the graph file
    PARENT_DIR=$(dirname "$DATA_PATH")
    if [[ ! -d "$PARENT_DIR" ]]; then
        echo -e "${RED}Error: Data path does not exist: $DATA_PATH${NC}"
        echo -e "${YELLOW}Please ensure your data is available at the specified path.${NC}"
        echo -e "${YELLOW}Note: DATA_PATH should be either:${NC}"
        echo -e "${YELLOW}  - A directory with source files (for preprocessing)${NC}"
        echo -e "${YELLOW}  - A directory containing preprocessed .pkl files${NC}"
        exit 1
    fi
fi

echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}Step 1/5: Setting Up Artifacts Directory${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"

# Create timestamped artifacts directory for this evaluation
echo -e "${BLUE}Creating artifacts directory: ${ARTIFACT_DIR}${NC}"
mkdir -p "$ARTIFACT_DIR"
echo -e "${GREEN}✓ Artifacts directory created${NC}"

echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}Step 2/5: Setting up Models and Pretrained Weights${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

if [[ "$SKIP_DOWNLOAD" == false ]]; then
    echo -e "${BLUE}Setting up models and copying pretrained weights...${NC}"
    
    if [[ "$MODEL" == "all" ]]; then
        python scripts/download_checkpoints.py --all
    else
        python scripts/download_checkpoints.py --models "$MODEL"
    fi
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Model weights setup successfully${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Some weights may not be available${NC}"
        echo -e "${YELLOW}  Continuing with available weights...${NC}"
    fi
else
    echo -e "${YELLOW}Skipping model setup (--skip-download specified)${NC}"
fi

echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}Step 3/5: Checking Preprocessed Data${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

# Determine output subdirectory based on dataset type
if [[ "$DATASET_TYPE" == "darpa" ]] || [[ "$DATA_PATH" =~ "DARPA" ]] || [[ "$DATA_PATH" =~ "ta1-" ]]; then
    PREPROCESSED_DATA_PATH="data/darpa"
    ACTUAL_DATASET_TYPE="darpa"
elif [[ "$DATASET_TYPE" == "custom_soc" ]] || [[ "$DATASET" == "custom_soc" ]]; then
    PREPROCESSED_DATA_PATH="data/custom_soc"
    ACTUAL_DATASET_TYPE="custom_soc"
else
    PREPROCESSED_DATA_PATH="data/processed"
    ACTUAL_DATASET_TYPE="custom"
fi

# Check if data is already preprocessed (has .pkl or .pt files)
# Check in multiple locations: exact path, with dataset name, and parent directory
if [[ -f "${DATA_PATH}/${DATASET}_graph.pkl" ]] || [[ -f "${DATA_PATH}/graph.pkl" ]]; then
    echo -e "${GREEN}✓ Preprocessed data found in specified path${NC}"
    PREPROCESSED_DATA_PATH="$DATA_PATH"
    echo -e "${BLUE}  Using: ${PREPROCESSED_DATA_PATH}/${DATASET}_graph.pkl or graph.pkl${NC}"
elif [[ -f "${PREPROCESSED_DATA_PATH}/${DATASET}_graph.pkl" ]] || [[ -f "${PREPROCESSED_DATA_PATH}/graph.pkl" ]]; then
    echo -e "${GREEN}✓ Preprocessed data found${NC}"
    echo -e "${BLUE}  Using: ${PREPROCESSED_DATA_PATH}/${DATASET}_graph.pkl${NC}"
elif [[ -f "$(dirname ${DATA_PATH})/${DATASET}_graph.pkl" ]]; then
    echo -e "${GREEN}✓ Preprocessed data found in parent directory${NC}"
    PREPROCESSED_DATA_PATH="$(dirname ${DATA_PATH})"
    echo -e "${BLUE}  Using: ${PREPROCESSED_DATA_PATH}/${DATASET}_graph.pkl${NC}"
elif [[ "$SKIP_PREPROCESS" == false ]]; then
    echo -e "${YELLOW}⚠ Preprocessed data not found${NC}"
    echo -e "${BLUE}Preprocessing your data using unified preprocessor...${NC}"
    echo -e "${BLUE}This may take several minutes for large datasets (2GB+)${NC}"
    
    # Build preprocessing command with flags
    # Check if DATA_PATH is a file or directory
    if [[ -f "$DATA_PATH" ]]; then
        # Single file - use --input-files
        PREPROCESS_CMD="python scripts/preprocess_data.py --input-files \"$DATA_PATH\" --output-dir data --dataset-name \"$DATASET\""
    else
        # Directory - use --input-dir
        PREPROCESS_CMD="python scripts/preprocess_data.py --input-dir \"$DATA_PATH\" --output-dir data --dataset-name \"$DATASET\""
    fi
    
    # Add dataset type if specified
    if [[ "$DATASET_TYPE" != "auto" ]]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --dataset-type \"$DATASET_TYPE\""
    fi
    
    # Add format if specified
    if [[ "$DATA_FORMAT" != "auto" ]]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --format \"$DATA_FORMAT\""
    fi
    
    # Add max events if specified
    if [[ -n "$MAX_EVENTS" ]]; then
        PREPROCESS_CMD="$PREPROCESS_CMD --max-events-per-file $MAX_EVENTS"
    fi
    
    echo -e "${BLUE}Running: $PREPROCESS_CMD${NC}"
    eval $PREPROCESS_CMD
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Data preprocessing completed${NC}"
    else
        echo -e "${RED}Error: Data preprocessing failed${NC}"
        echo -e "${YELLOW}Hint: Ensure DATA_PATH points to directory with data files${NC}"
        echo -e "${YELLOW}  Current: ${DATA_PATH}${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: No preprocessed data found and --skip-preprocess was specified${NC}"
    echo -e "${YELLOW}Please either:${NC}"
    echo -e "${YELLOW}  1. Run preprocessing first: python scripts/preprocess_data.py --input-dir \"${DATA_PATH}\" --output-dir data --dataset-name \"${DATASET}\"${NC}"
    echo -e "${YELLOW}  2. Provide correct --data-path pointing to preprocessed data${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}Step 4/5: Running Model Evaluation${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

echo -e "${BLUE}Evaluating model(s) on your custom dataset...${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

if [[ "$MODEL" == "all" ]]; then
    # Evaluate all models
    echo -e "${BLUE}Evaluating all available models...${NC}"
    
    for model_name in magic kairos orthrus threatrace continuum_fl; do
        echo ""
        echo -e "${YELLOW}Evaluating ${model_name}...${NC}"
        
        python experiments/evaluate_pipeline.py \
            --models "$model_name" \
            --dataset "$DATASET" \
            --data-path "${PREPROCESSED_DATA_PATH}" \
            --checkpoints-dir checkpoints \
            --artifact-dir "$ARTIFACT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            2>&1 | tee "$OUTPUT_DIR/${model_name}_evaluation.log"
        
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓ ${model_name} evaluation completed${NC}"
        else
            echo -e "${RED}✗ ${model_name} evaluation failed (check log for details)${NC}"
        fi
    done
else
    # Evaluate single model
    echo -e "${BLUE}Evaluating ${MODEL}...${NC}"
    
    python experiments/evaluate_pipeline.py \
        --models "$MODEL" \
        --dataset "$DATASET" \
        --data-path "${PREPROCESSED_DATA_PATH}" \
        --checkpoints-dir checkpoints \
        --artifact-dir "$ARTIFACT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/${MODEL}_evaluation.log"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Evaluation completed${NC}"
    else
        echo -e "${RED}Error: Evaluation failed${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}Step 5/5: Analyzing Anomaly Detection Results${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

echo -e "${BLUE}Running anomaly analysis on detection results...${NC}"

# Create anomaly analysis directory
ANOMALY_DIR="${OUTPUT_DIR}/anomaly_analysis"
mkdir -p "$ANOMALY_DIR"

if [[ "$MODEL" == "all" ]]; then
    echo -e "${BLUE}Analyzing anomalies for all models...${NC}"
    
    # Analyze each model
    for model_name in magic kairos orthrus threatrace continuum_fl; do
        if [[ -d "${ARTIFACT_DIR}/${model_name}/model_inference" ]]; then
            echo -e "${YELLOW}Analyzing ${model_name}...${NC}"
            
            python scripts/analyze_anomalies.py \
                --model "$model_name" \
                --top-k 100 \
                --artifacts-dir "$ARTIFACT_DIR" \
                --data-path "${PREPROCESSED_DATA_PATH}" \
                --dataset "$DATASET" \
                --output-dir "$ANOMALY_DIR" \
                2>&1 | tee "$ANOMALY_DIR/${model_name}_analysis.log"
            
            if [[ $? -eq 0 ]]; then
                echo -e "${GREEN}✓ ${model_name} analysis completed${NC}"
            else
                echo -e "${YELLOW}⚠ ${model_name} analysis failed${NC}"
            fi
        fi
    done
    
    # Generate ensemble analysis
    echo ""
    echo -e "${YELLOW}Generating ensemble consensus analysis...${NC}"
    python scripts/analyze_anomalies.py \
        --ensemble \
        --top-k 100 \
        --artifacts-dir "$ARTIFACT_DIR" \
        --data-path "${PREPROCESSED_DATA_PATH}" \
        --dataset "$DATASET" \
        --output-dir "$ANOMALY_DIR" \
        2>&1 | tee "$ANOMALY_DIR/ensemble_analysis.log"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Ensemble analysis completed${NC}"
    fi
else
    # Analyze single model
    if [[ -d "${ARTIFACT_DIR}/${MODEL}/model_inference" ]]; then
        echo -e "${BLUE}Analyzing anomalies for ${MODEL}...${NC}"
        
        python scripts/analyze_anomalies.py \
            --model "$MODEL" \
            --top-k 100 \
            --artifacts-dir "$ARTIFACT_DIR" \
            --data-path "${PREPROCESSED_DATA_PATH}" \
            --dataset "$DATASET" \
            --output-dir "$ANOMALY_DIR" \
            2>&1 | tee "$ANOMALY_DIR/analysis.log"
        
        if [[ $? -eq 0 ]]; then
            echo -e "${GREEN}✓ Anomaly analysis completed${NC}"
        else
            echo -e "${YELLOW}⚠ Analysis failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ No inference results found for ${MODEL}${NC}"
    fi
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ EVALUATION AND ANALYSIS COMPLETED SUCCESSFULLY!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Results saved to:${NC} ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "${BLUE}Artifacts saved to:${NC} ${GREEN}${ARTIFACT_DIR}${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. View evaluation:       cat ${OUTPUT_DIR}/evaluation_results_${DATASET}.json"
echo -e "  2. View metadata:         cat ${OUTPUT_DIR}/evaluation_metadata.json"
echo -e "  3. View anomalies:        cat ${ANOMALY_DIR}/magic_analysis.json"
echo -e "  4. Check consensus:       cat ${ANOMALY_DIR}/ensemble_analysis.json"
echo -e "  5. Review logs:           tail ${OUTPUT_DIR}/*.log"
echo -e "  6. Visualize results:     ./scripts/visualize_attacks.sh --evaluation-dir ${OUTPUT_DIR}"
echo ""
if [[ "$MODEL" == "all" ]]; then
    echo -e "${BLUE}Performance Summary:${NC}"
    echo -e "  Evaluation metrics:  ${OUTPUT_DIR}/evaluation_results_${DATASET}.json"
    echo -e "  Anomaly analyses:    ${ANOMALY_DIR}/*.json"
    echo ""
    echo -e "${BLUE}Top performing models (by score separation):${NC}"
    python3 -c "
import json, sys
try:
    with open('${OUTPUT_DIR}/evaluation_results_${DATASET}.json', 'r') as f:
        results = json.load(f)
    models = []
    for r in results:
        if r.get('success'):
            m = r.get('metrics', {})
            sep = m.get('score_separation_ratio', 0)
            models.append((r['model'], sep))
    models.sort(key=lambda x: x[1], reverse=True)
    for i, (name, sep) in enumerate(models[:3], 1):
        print(f'  {i}. {name:<20} Separation: {sep:.4f}')
except Exception as e:
    pass
"
    echo ""
fi
echo -e "${YELLOW}Optional: To investigate specific anomalies, see:${NC}"
echo -e "  cat ${ANOMALY_DIR}/<model>_analysis.json | jq '.top_anomalies[:10]'"
echo ""
