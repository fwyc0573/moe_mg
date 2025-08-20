#!/bin/bash

# Test script to verify new GPU configuration options
# This script tests the argument parsing without actually running the tests

echo "Testing New GPU Configuration Options"
echo "====================================="

SCRIPT="./multi_node_allreduce.sh"

# Test all new GPU configurations
configs=("2x1gpu" "2x2gpu" "2x3gpu" "2x4gpu" "2x5gpu" "2x6gpu" "2x7gpu" "2x8gpu")

echo "Testing individual GPU configurations:"
echo ""

for config in "${configs[@]}"; do
    echo "Testing --$config option..."
    
    # Test if the script accepts the option (dry run)
    if $SCRIPT --$config --help >/dev/null 2>&1; then
        echo "✅ --$config option is recognized"
    else
        echo "❌ --$config option failed"
    fi
done

echo ""
echo "Testing --all option..."
if $SCRIPT --all --help >/dev/null 2>&1; then
    echo "✅ --all option is recognized"
else
    echo "❌ --all option failed"
fi

echo ""
echo "Testing invalid option..."
if $SCRIPT --invalid-option >/dev/null 2>&1; then
    echo "❌ Invalid option was accepted (should fail)"
else
    echo "✅ Invalid option correctly rejected"
fi

echo ""
echo "Verifying MAX_SIZE parameter change:"
max_size_output=$($SCRIPT --help 2>/dev/null | grep "MAX_SIZE:" | awk '{print $2}')
if [[ "$max_size_output" == "4G" ]]; then
    echo "✅ MAX_SIZE correctly set to 4G"
else
    echo "❌ MAX_SIZE is $max_size_output (expected 4G)"
fi

echo ""
echo "Configuration Summary:"
echo "====================="
echo "Available GPU configurations:"
for config in "${configs[@]}"; do
    total_gpus=$((${config:2:1} * 2))
    echo "  --$config: $total_gpus total GPUs (${config:2:1} per node)"
done

echo ""
echo "Test Parameters:"
echo "  MIN_SIZE: 2K"
echo "  MAX_SIZE: 4G (reduced from 8G)"
echo "  STEP_FACTOR: 2"
echo "  ITERATIONS: 20"
echo "  WARMUP_ITERS: 5"

echo ""
echo "All configuration tests completed!"
