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
DATA_PATH="../custom_dataset"
SKIP_DOWNLOAD=false
SKIP_PREPROCESS=false
OUTPUT_DIR="results/evaluation_$(date +%Y%m%d_%H%M%S)"

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
        --data-path)
            DATA_PATH="$2"
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
            echo "Default workflow: Evaluate pretrained models on your custom SOC data"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model to evaluate (magic, kairos, orthrus, threatrace, continuum_fl, all)"
            echo "                         Default: all"
            echo "  --dataset DATASET      Dataset name (default: custom_soc)"
            echo "  --data-path PATH       Path to your data (default: ../custom_dataset)"
            echo "  --skip-download        Skip downloading pretrained weights"
            echo "  --skip-preprocess      Skip data preprocessing"
            echo "  --output-dir DIR       Output directory for results"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Evaluate all models"
            echo "  $0 --model magic                      # Evaluate MAGIC only"
            echo "  $0 --data-path /path/to/logs          # Use different data path"
            echo "  $0 --skip-download --skip-preprocess  # Skip setup steps"
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
echo -e "  Data Path:     ${GREEN}${DATA_PATH}${NC}"
echo -e "  Output Dir:    ${GREEN}${OUTPUT_DIR}${NC}"
echo ""

# Check if conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV}" ]] || [[ "${CONDA_DEFAULT_ENV}" != "pids_framework" ]]; then
    echo -e "${RED}Error: Conda environment 'pids_framework' is not activated!${NC}"
    echo -e "${YELLOW}Please run: conda activate pids_framework${NC}"
    exit 1
fi

# Check if data path exists
if [[ ! -d "$DATA_PATH" ]]; then
    echo -e "${RED}Error: Data path does not exist: $DATA_PATH${NC}"
    echo -e "${YELLOW}Please ensure your SOC data is available at the specified path.${NC}"
    exit 1
fi

echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}Step 1/4: Setting up Models and Pretrained Weights${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

if [[ "$SKIP_DOWNLOAD" == false ]]; then
    echo -e "${BLUE}Setting up models and copying pretrained weights...${NC}"
    
    if [[ "$MODEL" == "all" ]]; then
        python scripts/setup_models.py --all --no-install
    else
        python scripts/setup_models.py --models "$MODEL" --no-install
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
echo -e "${CYAN}Step 2/4: Preprocessing Custom Dataset${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

if [[ "$SKIP_PREPROCESS" == false ]]; then
    echo -e "${BLUE}Preprocessing your SOC data...${NC}"
    echo -e "${BLUE}This may take several minutes for large datasets (2GB+)${NC}"
    
    python scripts/preprocess_data.py \
        --input-dir "$DATA_PATH" \
        --output-dir "data/${DATASET}" \
        --dataset-name "$DATASET"
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Data preprocessing completed${NC}"
    else
        echo -e "${RED}Error: Data preprocessing failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping preprocessing (--skip-preprocess specified)${NC}"
fi

echo ""
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
echo -e "${CYAN}Step 3/4: Running Model Evaluation${NC}"
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
        
        python experiments/evaluate.py \
            --model "$model_name" \
            --dataset "$DATASET" \
            --data-path "data/${DATASET}" \
            --pretrained \
            --checkpoint-dir checkpoints \
            --output-dir "$OUTPUT_DIR" \
            --save-predictions \
            --detection-level both \
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
    
    python experiments/evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --data-path "data/${DATASET}" \
        --pretrained \
        --checkpoint-dir checkpoints \
        --output-dir "$OUTPUT_DIR" \
        --save-predictions \
        --detection-level both \
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
echo -e "${CYAN}Step 4/4: Generating Comparison Report${NC}"
echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

if [[ "$MODEL" == "all" ]]; then
    echo -e "${BLUE}Generating comparative analysis...${NC}"
    
    python experiments/compare.py \
        --results-dir "$OUTPUT_DIR" \
        --dataset "$DATASET" \
        --output-file "$OUTPUT_DIR/comparison_report.json" \
        --generate-plots
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✓ Comparison report generated${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Could not generate comparison report${NC}"
    fi
else
    echo -e "${YELLOW}Skipping comparison (single model evaluation)${NC}"
fi

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ EVALUATION COMPLETED SUCCESSFULLY!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Results saved to:${NC} ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Review results:        ls ${OUTPUT_DIR}"
echo -e "  2. View comparison:       cat ${OUTPUT_DIR}/comparison_report.json"
echo -e "  3. Check model logs:      tail ${OUTPUT_DIR}/*.log"
echo ""
if [[ "$MODEL" == "all" ]]; then
    echo -e "${BLUE}Performance Summary:${NC}"
    echo -e "  Check ${OUTPUT_DIR}/comparison_report.json for detailed metrics"
    echo ""
fi
echo -e "${YELLOW}Optional: To retrain models on your custom data, see:${NC}"
echo -e "  python experiments/train.py --help"
echo ""
