#!/bin/bash

# Install model-specific dependencies
# Usage: ./install_model_deps.sh --models magic kairos orthrus
#        ./install_model_deps.sh --all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REQUIREMENTS_DIR="$PROJECT_DIR/requirements"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Available models
AVAILABLE_MODELS=("magic" "kairos" "orthrus" "threatrace" "continuum_fl")

# Parse arguments
MODELS=()
INSTALL_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --models)
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all              Install dependencies for all models"
            echo "  --models MODEL...  Install dependencies for specific models"
            echo "  --help            Show this help message"
            echo ""
            echo "Available models: ${AVAILABLE_MODELS[*]}"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If --all is set, install for all models
if [ "$INSTALL_ALL" = true ]; then
    MODELS=("${AVAILABLE_MODELS[@]}")
fi

# If no models specified, show help
if [ ${#MODELS[@]} -eq 0 ]; then
    print_error "No models specified"
    echo "Use --all to install all models or --models MODEL... to install specific models"
    echo "Available models: ${AVAILABLE_MODELS[*]}"
    exit 1
fi

print_info "Installing dependencies for models: ${MODELS[*]}"
echo ""

# Install dependencies for each model
for model in "${MODELS[@]}"; do
    # Check if model is available
    if [[ ! " ${AVAILABLE_MODELS[@]} " =~ " ${model} " ]]; then
        print_warning "Unknown model: $model (skipping)"
        continue
    fi
    
    REQ_FILE="$REQUIREMENTS_DIR/${model}.txt"
    
    if [ ! -f "$REQ_FILE" ]; then
        print_warning "Requirements file not found for $model: $REQ_FILE (skipping)"
        continue
    fi
    
    print_info "Installing dependencies for $model..."
    
    # Special handling for models with specific requirements
    case $model in
        magic)
            print_info "Installing MAGIC dependencies..."
            pip install -r "$REQ_FILE"
            ;;
        kairos)
            print_info "Installing Kairos dependencies..."
            # Install base dependencies (skip torch-geometric extensions, already installed)
            pip install --no-deps torch-geometric==2.1.0 || print_warning "PyG already installed"
            pip install networkx==2.8.8 python-louvain==0.16 graphviz==0.20.1
            pip install psycopg2-binary==2.9.6 sqlalchemy==2.0.15
            ;;
        orthrus)
            print_info "Installing Orthrus dependencies..."
            # Install base dependencies (skip torch-geometric extensions, already installed)
            pip install --no-deps torch-geometric==2.1.0 || print_warning "PyG already installed"
            pip install networkx==2.8.8 psycopg2-binary==2.9.6 sqlalchemy==2.0.15
            ;;
        threatrace)
            print_info "Installing ThreaTrace dependencies..."
            # ThreaTrace works with the common PyG version
            pip install --no-deps torch-geometric==2.1.0 || print_warning "PyG already installed"
            # Additional dependencies are already covered by framework base
            ;;
        continuum_fl)
            print_info "Installing Continuum FL dependencies..."
            # Install MPI support
            if ! command -v mpirun &> /dev/null; then
                print_warning "MPI not found. Installing via conda..."
                conda install -c conda-forge mpi4py openmpi -y || print_warning "MPI installation failed - federated learning may not work"
            else
                print_info "MPI already installed"
                pip install mpi4py==3.1.3 || print_warning "mpi4py installation failed"
            fi
            # Install PyG (already done by framework)
            pip install --no-deps torch-geometric==2.1.0 || print_warning "PyG already installed"
            # Install additional dependencies
            pip install wandb==0.15.0 redis==4.5.4 paho-mqtt==1.6.1
            ;;
        *)
            print_info "Installing $model dependencies..."
            pip install -r "$REQ_FILE"
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        print_info "✓ Successfully installed dependencies for $model"
    else
        print_error "✗ Failed to install dependencies for $model"
    fi
    
    echo ""
done

print_info "Model dependencies installation complete!"
echo ""
echo "Installed models: ${MODELS[*]}"
echo ""
echo "To verify installation, run:"
echo "  python -c 'from models import ModelRegistry; print(ModelRegistry.list_models())'"
