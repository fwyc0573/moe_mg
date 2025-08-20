#!/bin/bash

# Robust Multi-Node AllReduce Test Script
# Enhanced version with better error handling, timeouts, and debugging

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMM_TEST_SCRIPT="${SCRIPT_DIR}/pytorch_comm_test.py"
RESULTS_DIR="${SCRIPT_DIR}/multi_node_allreduce_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Network configuration
MASTER_IP="10.232.194.226"
WORKER_IP="10.232.195.25"
BASE_PORT="29500"

# Test parameters (reduced for initial testing)
MIN_SIZE="2K"
MAX_SIZE="1M"  # Reduced from 8G for initial testing
STEP_FACTOR="2"
ITERATIONS="5"  # Reduced from 20
WARMUP_ITERS="2"  # Reduced from 5

# Enhanced NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_TIMEOUT=60000
export NCCL_CONNECT_TIMEOUT=60000
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Function to find available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    
    while [[ $port -lt $((start_port + 100)) ]]; do
        if ! netstat -tuln | grep ":$port " > /dev/null; then
            echo $port
            return 0
        fi
        ((port++))
    done
    
    echo "No available port found starting from $start_port" >&2
    return 1
}

# Function to get current node role
get_node_role() {
    local current_ip=$(hostname -I | awk '{print $1}')
    if [[ "$current_ip" == "$MASTER_IP" ]]; then
        echo "master"
    elif [[ "$current_ip" == "$WORKER_IP" ]]; then
        echo "worker"
    else
        echo "unknown"
    fi
}

# Function to test network connectivity
test_connectivity() {
    local target_ip=$1
    local port=$2
    
    echo "Testing connectivity to $target_ip:$port..."
    
    if timeout 5 bash -c "echo >/dev/tcp/$target_ip/$port" 2>/dev/null; then
        echo "✅ Connection successful"
        return 0
    else
        echo "❌ Connection failed"
        return 1
    fi
}

# Function to run master node
run_master_node() {
    local gpus_per_node=$1
    local port=$2
    
    echo "=========================================="
    echo "Starting Master Node (Rank 0)"
    echo "=========================================="
    echo "GPUs per node: $gpus_per_node"
    echo "Master IP: $MASTER_IP"
    echo "Port: $port"
    echo "Waiting for worker node connection..."
    echo ""
    
    local gpu_list=""
    for ((i=0; i<gpus_per_node; i++)); do
        if [ $i -eq 0 ]; then
            gpu_list="$i"
        else
            gpu_list="$gpu_list,$i"
        fi
    done
    
    export CUDA_VISIBLE_DEVICES="$gpu_list"
    
    local log_file="${RESULTS_DIR}/allreduce_$((gpus_per_node*2))gpu_2node_${TIMESTAMP}.txt"
    local csv_file="${RESULTS_DIR}/allreduce_$((gpus_per_node*2))gpu_2node_${TIMESTAMP}.csv"
    
    echo "Log file: $log_file"
    echo "CSV file: $csv_file"
    echo ""
    
    # Add timeout to the entire torchrun command
    timeout 600 torchrun \
        --nproc_per_node=$gpus_per_node \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr="$MASTER_IP" \
        --master_port="$port" \
        --max_restarts=3 \
        --rdzv_timeout=300 \
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
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Master node test completed successfully"
    elif [[ $exit_code -eq 124 ]]; then
        echo "❌ Master node test timed out (10 minutes)"
    else
        echo "❌ Master node test failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Function to run worker node
run_worker_node() {
    local gpus_per_node=$1
    local port=$2
    
    echo "=========================================="
    echo "Starting Worker Node (Rank 1)"
    echo "=========================================="
    echo "GPUs per node: $gpus_per_node"
    echo "Master IP: $MASTER_IP"
    echo "Port: $port"
    echo ""
    
    # Test connectivity to master before starting
    if ! test_connectivity "$MASTER_IP" "$port"; then
        echo "❌ Cannot connect to master node. Aborting."
        return 1
    fi
    
    local gpu_list=""
    for ((i=0; i<gpus_per_node; i++)); do
        if [ $i -eq 0 ]; then
            gpu_list="$i"
        else
            gpu_list="$gpu_list,$i"
        fi
    done
    
    export CUDA_VISIBLE_DEVICES="$gpu_list"
    
    local log_file="${RESULTS_DIR}/allreduce_$((gpus_per_node*2))gpu_2node_${TIMESTAMP}_worker.txt"
    
    echo "Worker log file: $log_file"
    echo ""
    
    # Add timeout to the entire torchrun command
    timeout 600 torchrun \
        --nproc_per_node=$gpus_per_node \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr="$MASTER_IP" \
        --master_port="$port" \
        --max_restarts=3 \
        --rdzv_timeout=300 \
        "$COMM_TEST_SCRIPT" \
        --operation "allreduce" \
        --min_size "$MIN_SIZE" \
        --max_size "$MAX_SIZE" \
        --step_factor "$STEP_FACTOR" \
        --iterations "$ITERATIONS" \
        --warmup_iters "$WARMUP_ITERS" \
        --test_mode "multi_node" \
        2>&1 | tee "$log_file"
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Worker node test completed successfully"
    elif [[ $exit_code -eq 124 ]]; then
        echo "❌ Worker node test timed out (10 minutes)"
    else
        echo "❌ Worker node test failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Function to display usage
usage() {
    echo "Robust Multi-Node AllReduce Test"
    echo "================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --master        Run as master node (rank 0)"
    echo "  --worker        Run as worker node (rank 1)"
    echo "  --gpus N        Number of GPUs per node (default: 2)"
    echo "  --port N        Master port (default: auto-detect)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  Master node: $0 --master --gpus 2"
    echo "  Worker node: $0 --worker --gpus 2"
    echo ""
    echo "Environment Variables:"
    echo "  MASTER_IP: $MASTER_IP"
    echo "  WORKER_IP: $WORKER_IP"
}

# Parse command line arguments
NODE_ROLE=""
GPUS_PER_NODE="2"
CUSTOM_PORT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --master)
            NODE_ROLE="master"
            shift
            ;;
        --worker)
            NODE_ROLE="worker"
            shift
            ;;
        --gpus)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        --port)
            CUSTOM_PORT="$2"
            shift 2
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

# Auto-detect node role if not specified
if [[ -z "$NODE_ROLE" ]]; then
    NODE_ROLE=$(get_node_role)
    if [[ "$NODE_ROLE" == "unknown" ]]; then
        echo "Cannot auto-detect node role. Please specify --master or --worker"
        usage
        exit 1
    fi
    echo "Auto-detected node role: $NODE_ROLE"
fi

# Find available port
if [[ -n "$CUSTOM_PORT" ]]; then
    MASTER_PORT="$CUSTOM_PORT"
else
    MASTER_PORT=$(find_available_port "$BASE_PORT")
    if [[ $? -ne 0 ]]; then
        echo "Failed to find available port"
        exit 1
    fi
fi

echo "Using port: $MASTER_PORT"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check if Python script exists
if [[ ! -f "$COMM_TEST_SCRIPT" ]]; then
    echo "Error: pytorch_comm_test.py not found in $SCRIPT_DIR"
    exit 1
fi

# Run based on node role
case $NODE_ROLE in
    "master")
        run_master_node "$GPUS_PER_NODE" "$MASTER_PORT"
        ;;
    "worker")
        run_worker_node "$GPUS_PER_NODE" "$MASTER_PORT"
        ;;
    *)
        echo "Invalid node role: $NODE_ROLE"
        usage
        exit 1
        ;;
esac
