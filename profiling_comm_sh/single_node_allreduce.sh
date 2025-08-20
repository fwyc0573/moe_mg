#!/bin/bash

# Single-Node AllReduce Performance Test Script
# Tests AllReduce communication across 2, 4, and 8 GPUs on a single machine
# Tensor sizes: 2KB to 8GB (powers of 2)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMM_TEST_SCRIPT="${SCRIPT_DIR}/pytorch_comm_test.py"
RESULTS_DIR="${SCRIPT_DIR}/single_node_allreduce_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test parameters
MIN_SIZE="2K"
MAX_SIZE="8G"
STEP_FACTOR="2"
ITERATIONS="20"
WARMUP_ITERS="5"

# PyTorch/NCCL environment
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Function to format bytes to human readable
format_size() {
    local bytes=$1
    if [ $bytes -ge $((1024*1024*1024)) ]; then
        echo "$((bytes / (1024*1024*1024)))G"
    elif [ $bytes -ge $((1024*1024)) ]; then
        echo "$((bytes / (1024*1024)))M"
    elif [ $bytes -ge 1024 ]; then
        echo "$((bytes / 1024))K"
    else
        echo "${bytes}B"
    fi
}

# Setup results directory
mkdir -p "$RESULTS_DIR"

# Function to run single-node AllReduce test
run_single_node_allreduce_test() {
    local gpu_count=$1
    local gpu_list=""
    
    # Generate GPU list (0,1,2,...)
    for ((i=0; i<gpu_count; i++)); do
        if [ $i -eq 0 ]; then
            gpu_list="$i"
        else
            gpu_list="$gpu_list,$i"
        fi
    done
    
    echo "=========================================="
    echo "Single-Node AllReduce Test: ${gpu_count} GPUs"
    echo "GPU List: $gpu_list"
    echo "=========================================="
    
    local log_file="${RESULTS_DIR}/allreduce_${gpu_count}gpu_${TIMESTAMP}.txt"
    local csv_file="${RESULTS_DIR}/allreduce_${gpu_count}gpu_${TIMESTAMP}.csv"
    
    echo "Running ${gpu_count}-GPU AllReduce test..."
    echo "Log file: $log_file"
    echo "CSV file: $csv_file"
    
    # Set CUDA_VISIBLE_DEVICES for this test
    export CUDA_VISIBLE_DEVICES="$gpu_list"
    
    torchrun --nproc_per_node=$gpu_count \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$COMM_TEST_SCRIPT" \
        --operation "allreduce" \
        --min_size "$MIN_SIZE" \
        --max_size "$MAX_SIZE" \
        --step_factor "$STEP_FACTOR" \
        --iterations "$ITERATIONS" \
        --warmup_iters "$WARMUP_ITERS" \
        --log_file "$log_file" \
        --csv_file "$csv_file" \
        --test_mode "single_node" \
        2>&1 | tee "${log_file}.console"
    
    echo "${gpu_count}-GPU AllReduce test completed."
    echo "Results saved to: $log_file"
    echo "CSV data saved to: $csv_file"
    echo ""
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --2gpu          Run 2-GPU test only"
    echo "  --4gpu          Run 4-GPU test only"
    echo "  --8gpu          Run 8-GPU test only"
    echo "  --all           Run all GPU configurations (default)"
    echo "  --help          Show this help message"
    echo ""
    echo "Test Parameters:"
    echo "  MIN_SIZE:       $MIN_SIZE"
    echo "  MAX_SIZE:       $MAX_SIZE"
    echo "  STEP_FACTOR:    $STEP_FACTOR"
    echo "  ITERATIONS:     $ITERATIONS"
    echo "  WARMUP_ITERS:   $WARMUP_ITERS"
    echo ""
    echo "Results will be saved to: $RESULTS_DIR"
}

# Parse command line arguments
TEST_MODE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --2gpu)
            TEST_MODE="2gpu"
            shift
            ;;
        --4gpu)
            TEST_MODE="4gpu"
            shift
            ;;
        --8gpu)
            TEST_MODE="8gpu"
            shift
            ;;
        --all)
            TEST_MODE="all"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
echo "Single-Node AllReduce Performance Test"
echo "======================================"
echo "Timestamp: $TIMESTAMP"
echo "Test mode: $TEST_MODE"
echo "Results directory: $RESULTS_DIR"
echo ""

# Check if Python script exists
if [[ ! -f "$COMM_TEST_SCRIPT" ]]; then
    echo "Error: pytorch_comm_test.py not found in $SCRIPT_DIR"
    echo "Please ensure the Python script is in the profiling_comm_sh directory."
    exit 1
fi

# Check GPU availability
nvidia-smi > /dev/null 2>&1 || {
    echo "Error: nvidia-smi not found or GPUs not available"
    exit 1
}

# Run tests based on mode
case $TEST_MODE in
    "2gpu")
        run_single_node_allreduce_test 2
        ;;
    "4gpu")
        run_single_node_allreduce_test 4
        ;;
    "8gpu")
        run_single_node_allreduce_test 8
        ;;
    "all")
        run_single_node_allreduce_test 2
        echo "Waiting 5 seconds before next test..."
        sleep 5
        run_single_node_allreduce_test 4
        echo "Waiting 5 seconds before next test..."
        sleep 5
        run_single_node_allreduce_test 8
        ;;
esac

echo ""
echo "=========================================="
echo "All single-node AllReduce tests completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="
