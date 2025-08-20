#!/bin/bash

# Test script for the fixed multi-node AllReduce
# Use this to verify the CUDA device fix works

echo "Testing Fixed Multi-Node AllReduce"
echo "=================================="

# Configuration
MASTER_IP="10.232.194.226"
WORKER_IP="10.232.195.25"
PORT="29501"  # Use different port to avoid conflicts

# Reduced test parameters for quick verification
MIN_SIZE="2K"
MAX_SIZE="128K"  # Much smaller for quick test
STEP_FACTOR="4"
ITERATIONS="3"   # Reduced iterations
WARMUP_ITERS="1"

# Get current node IP
CURRENT_IP=$(hostname -I | awk '{print $1}')

echo "Current IP: $CURRENT_IP"
echo "Master IP: $MASTER_IP"
echo "Worker IP: $WORKER_IP"
echo "Port: $PORT"
echo ""

# Determine node role
if [[ "$CURRENT_IP" == "$MASTER_IP" ]]; then
    NODE_ROLE="master"
    NODE_RANK="0"
elif [[ "$CURRENT_IP" == "$WORKER_IP" ]]; then
    NODE_ROLE="worker"
    NODE_RANK="1"
else
    echo "Unknown node. Please run this on master ($MASTER_IP) or worker ($WORKER_IP)"
    exit 1
fi

echo "Detected role: $NODE_ROLE (rank $NODE_RANK)"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_TIMEOUT=30000
export NCCL_CONNECT_TIMEOUT=30000

# Create results directory
mkdir -p multi_node_allreduce_results

# Generate unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="multi_node_allreduce_results/test_fixed_${NODE_ROLE}_${TIMESTAMP}.txt"
CSV_FILE="multi_node_allreduce_results/test_fixed_${NODE_ROLE}_${TIMESTAMP}.csv"

echo "Starting $NODE_ROLE node test..."
echo "Log file: $LOG_FILE"
echo "CSV file: $CSV_FILE"
echo ""

if [[ "$NODE_ROLE" == "master" ]]; then
    echo "=== MASTER NODE COMMAND ==="
    echo "Waiting for worker node to connect..."
    echo ""
fi

# Run the test
torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_IP" \
    --master_port="$PORT" \
    --rdzv_timeout=60 \
    pytorch_comm_test.py \
    --operation allreduce \
    --min_size "$MIN_SIZE" \
    --max_size "$MAX_SIZE" \
    --step_factor "$STEP_FACTOR" \
    --iterations "$ITERATIONS" \
    --warmup_iters "$WARMUP_ITERS" \
    --log_file "$LOG_FILE" \
    --csv_file "$CSV_FILE" \
    --test_mode multi_node \
    2>&1 | tee "${LOG_FILE}.console"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ $NODE_ROLE node test completed successfully!"
    echo "Results saved to: $LOG_FILE"
    if [[ "$NODE_ROLE" == "master" ]]; then
        echo "CSV data saved to: $CSV_FILE"
    fi
else
    echo "❌ $NODE_ROLE node test failed with exit code $EXIT_CODE"
    echo "Check the log file for details: $LOG_FILE"
fi
echo "=========================================="
