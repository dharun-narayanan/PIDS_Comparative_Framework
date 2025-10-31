#!/bin/bash
# Script to visualize attack graphs from all models

set -e

# Default parameters
THRESHOLD_PERCENTILE=99
TOP_K=100
TOP_PATHS=15
CLUSTER_BY="entity"
OUTPUT_DIR="results/attack_graph_visualization"
NO_BROWSER=""
SERVE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold)
            THRESHOLD_PERCENTILE="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --top-paths)
            TOP_PATHS="$2"
            shift 2
            ;;
        --cluster-by)
            CLUSTER_BY="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-browser)
            NO_BROWSER="--no-browser"
            shift
            ;;
        --serve)
            SERVE="--serve"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --threshold PERCENTILE     Percentile threshold for anomalies (default: 99)"
            echo "  --top-k NUMBER            Number of top anomalies to include (default: 100)"
            echo "  --top-paths NUMBER        Number of top attack paths to extract (default: 15)"
            echo "  --cluster-by METHOD       Clustering strategy: entity, temporal, path (default: entity)"
            echo "  --output-dir PATH         Output directory (default: results/attack_graph_visualization)"
            echo "  --no-browser              Skip auto-opening browser (useful for remote servers)"
            echo "  --serve                   Start HTTP server for remote access (recommended for VS Code Remote)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "Attack Graph Visualization Script"
echo "============================================"
echo "Threshold: ${THRESHOLD_PERCENTILE}%"
echo "Top-K: ${TOP_K}"
echo "Top Paths: ${TOP_PATHS}"
echo "Clustering: ${CLUSTER_BY}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"
echo ""

# Run visualization
python utils/visualize_attack_graphs.py \
    --threshold-percentile ${THRESHOLD_PERCENTILE} \
    --top-k ${TOP_K} \
    --top-paths ${TOP_PATHS} \
    --cluster-by ${CLUSTER_BY} \
    --output-dir ${OUTPUT_DIR} \
    ${NO_BROWSER} \
    ${SERVE}

echo ""
echo "============================================"
echo "Visualization Complete!"
echo "============================================"
echo ""
echo "Generated files:"
echo "  • Interactive HTML: ${OUTPUT_DIR}/attack_graph_viewer.html"
echo "  • Attack Summary: ${OUTPUT_DIR}/attack_summary.json"
echo "  • GraphML files: ${OUTPUT_DIR}/*_attack_graph.graphml"
echo ""
echo "To view the visualization:"
echo "  1. In VS Code: Right-click the HTML file and select 'Open with Live Server' or 'Open in Simple Browser'"
echo "  2. Command line: xdg-open ${OUTPUT_DIR}/attack_graph_viewer.html"
echo "  3. Manual: Copy the file to your local machine and open in browser"
echo ""
