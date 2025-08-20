#!/bin/bash

# Multi-Node AllReduce Performance Test Script
# Tests AllReduce communication across 2 machines
# Configurations: 2x2GPUs (4 total), 2x4GPUs (8 total), 2x8GPUs (16 total)
# Tensor sizes: 2KB to 8GB (powers of 2)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMM_TEST_SCRIPT="${SCRIPT_DIR}/pytorch_comm_test.py"
RESULTS_DIR="${SCRIPT_DIR}/multi_node_allreduce_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test parameters
MIN_SIZE="2K"
MAX_SIZE="8G"
STEP_FACTOR="2"
ITERATIONS="20"
WARMUP_ITERS="5"

# Multi-node configuration (matching sendrecv.sh)
NODE1="10.232.194.226"
NODE2="10.232.195.25"

# For demo purposes, use current machine IP if configured nodes are not accessible
CURRENT_IP=$(hostname -I | awk '{print $2}')  # Use second IP
if ! ping -c 1 -W 1 "$NODE1" >/dev/null 2>&1; then
    echo "Warning: Configured NODE1 ($NODE1) is not accessible."
    echo "Using current machine IP ($CURRENT_IP) for demonstration."
    NODE1="$CURRENT_IP"
    NODE2="$CURRENT_IP"  # For demo, both nodes are the same machine
fi

# PyTorch/NCCL environment
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Setup results directory
mkdir -p "$RESULTS_DIR"

# Function to run multi-node AllReduce test
run_multi_node_allreduce_test() {
    local gpus_per_node=$1
    local total_gpus=$((gpus_per_node * 2))
    local gpu_list=""
    
    # Generate GPU list (0,1,2,...)
    for ((i=0; i<gpus_per_node; i++)); do
        if [ $i -eq 0 ]; then
            gpu_list="$i"
        else
            gpu_list="$gpu_list,$i"
        fi
    done
    
    echo "=========================================="
    echo "Multi-Node AllReduce Test: ${gpus_per_node} GPUs per node (${total_gpus} total)"
    echo "GPU List per node: $gpu_list"
    echo "Node1: $NODE1, Node2: $NODE2"
    echo "=========================================="
    
    local log_file="${RESULTS_DIR}/allreduce_${total_gpus}gpu_2node_${TIMESTAMP}.txt"
    local csv_file="${RESULTS_DIR}/allreduce_${total_gpus}gpu_2node_${TIMESTAMP}.csv"
    
    echo "Running ${total_gpus}-GPU (2-node) AllReduce test..."
    echo "Log file: $log_file"
    echo "CSV file: $csv_file"
    
    # Check if we're on the master node
    local current_ip=$(hostname -I | awk '{print $1}')
    
    if [[ "$current_ip" == "$NODE1" ]] || [[ "$NODE1" == "$CURRENT_IP" ]]; then
        echo "Running as master node ($NODE1)"
        
        # Set CUDA_VISIBLE_DEVICES for this test
        export CUDA_VISIBLE_DEVICES="$gpu_list"
        
        torchrun --nproc_per_node=$gpus_per_node \
            --nnodes=2 \
            --node_rank=0 \
            --master_addr="$NODE1" \
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
            --test_mode "multi_node" \
            2>&1 | tee "${log_file}.console"
    else
        echo "Please run this script on the master node ($NODE1)"
        echo "On the worker node ($NODE2), run:"
        echo "export CUDA_VISIBLE_DEVICES=\"$gpu_list\""
        echo "torchrun --nproc_per_node=$gpus_per_node --nnodes=2 --node_rank=1 --master_addr=$NODE1 --master_port=29500 $COMM_TEST_SCRIPT --operation allreduce --min_size $MIN_SIZE --max_size $MAX_SIZE --step_factor $STEP_FACTOR --iterations $ITERATIONS --warmup_iters $WARMUP_ITERS --log_file $log_file --csv_file $csv_file --test_mode multi_node"
        exit 1
    fi
    
    echo "${total_gpus}-GPU (2-node) AllReduce test completed."
    echo "Results saved to: $log_file"
    echo "CSV data saved to: $csv_file"
    echo ""
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --2x2gpu        Run 2x2GPU test (4 total GPUs)"
    echo "  --2x4gpu        Run 2x4GPU test (8 total GPUs)"
    echo "  --2x8gpu        Run 2x8GPU test (16 total GPUs)"
    echo "  --all           Run all configurations (default)"
    echo "  --help          Show this help message"
    echo ""
    echo "Test Parameters:"
    echo "  MIN_SIZE:       $MIN_SIZE"
    echo "  MAX_SIZE:       $MAX_SIZE"
    echo "  STEP_FACTOR:    $STEP_FACTOR"
    echo "  ITERATIONS:     $ITERATIONS"
    echo "  WARMUP_ITERS:   $WARMUP_ITERS"
    echo ""
    echo "Node Configuration:"
    echo "  NODE1:          $NODE1"
    echo "  NODE2:          $NODE2"
    echo ""
    echo "Results will be saved to: $RESULTS_DIR"
    echo ""
    echo "Note: This script should be run on the master node ($NODE1)."
    echo "The worker node ($NODE2) should run the corresponding torchrun command."
}

# Parse command line arguments
TEST_MODE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --2x2gpu)
            TEST_MODE="2x2gpu"
            shift
            ;;
        --2x4gpu)
            TEST_MODE="2x4gpu"
            shift
            ;;
        --2x8gpu)
            TEST_MODE="2x8gpu"
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
echo "Multi-Node AllReduce Performance Test"
echo "====================================="
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
    "2x2gpu")
        run_multi_node_allreduce_test 2
        ;;
    "2x4gpu")
        run_multi_node_allreduce_test 4
        ;;
    "2x8gpu")
        run_multi_node_allreduce_test 8
        ;;
    "all")
        run_multi_node_allreduce_test 2
        echo "Waiting 10 seconds before next test..."
        sleep 10
        run_multi_node_allreduce_test 4
        echo "Waiting 10 seconds before next test..."
        sleep 10
        run_multi_node_allreduce_test 8
        ;;
esac

echo ""
echo "=========================================="
echo "All multi-node AllReduce tests completed!"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="
