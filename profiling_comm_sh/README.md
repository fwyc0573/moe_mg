# PyTorch NCCL Communication Performance Testing Suite

This directory contains a **self-contained, independent** testing suite for PyTorch NCCL communication performance across different GPU configurations and tensor sizes. All required files are included in this directory - no external dependencies on other project directories.

## Overview

This **independent testing suite** includes:
- **Single-node tests**: 2, 4, 8 GPUs on one machine
- **Multi-node tests**: 2x2, 2x4, 2x8 GPUs across two machines
- **Send/Recv and AllReduce operations**: Complete communication testing
- **Comprehensive tensor size range**: 2KB to 8GB (powers of 2)
- **Detailed performance metrics**: Latency, algorithm bandwidth, bus bandwidth
- **Self-contained**: All scripts and Python code included in this directory

## Files Structure

```
profiling_comm_sh/
├── README.md                      # This file
├── pytorch_comm_test.py           # Independent PyTorch communication test script
├── run_all_allreduce_tests.sh     # Master script to run all tests
├── single_node_allreduce.sh       # Single-node AllReduce tests
├── multi_node_allreduce.sh        # Multi-node AllReduce tests
├── test_allreduce_functionality.sh # Quick functionality test
├── single_node_allreduce_results/ # Results from single-node tests
└── multi_node_allreduce_results/  # Results from multi-node tests
```

## Quick Start

### Run All Tests
```bash
# Run both single-node and multi-node tests
./run_all_allreduce_tests.sh --all

# Run only single-node tests
./run_all_allreduce_tests.sh --single-node

# Run only multi-node tests
./run_all_allreduce_tests.sh --multi-node
```

### Single-Node Tests
```bash
# Run all single-node configurations (2, 4, 8 GPUs)
./single_node_allreduce.sh --all

# Run specific GPU count
./single_node_allreduce.sh --2gpu
./single_node_allreduce.sh --4gpu
./single_node_allreduce.sh --8gpu
```

### Multi-Node Tests
```bash
# Run all multi-node configurations (2x2, 2x4, 2x8 GPUs)
./multi_node_allreduce.sh --all

# Run specific configuration
./multi_node_allreduce.sh --2x2gpu  # 4 total GPUs
./multi_node_allreduce.sh --2x4gpu  # 8 total GPUs
./multi_node_allreduce.sh --2x8gpu  # 16 total GPUs
```

## Test Configurations

### Single-Node Tests
| Configuration | GPUs | Description |
|---------------|------|-------------|
| 2GPU | 2 | AllReduce across 2 GPUs on single machine |
| 4GPU | 4 | AllReduce across 4 GPUs on single machine |
| 8GPU | 8 | AllReduce across 8 GPUs on single machine |

### Multi-Node Tests
| Configuration | Total GPUs | Description |
|---------------|------------|-------------|
| 2x2GPU | 4 | AllReduce across 2 machines, 2 GPUs each |
| 2x4GPU | 8 | AllReduce across 2 machines, 4 GPUs each |
| 2x8GPU | 16 | AllReduce across 2 machines, 8 GPUs each |

### Test Parameters
- **Tensor sizes**: 2KB, 4KB, 8KB, 16KB, 32KB, 64KB, 128KB, 256KB, 512KB, 1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB, 128MB, 256MB, 512MB, 1GB, 2GB, 4GB, 8GB
- **Iterations**: 20 per tensor size
- **Warmup iterations**: 5 per tensor size
- **Data type**: float32

## Multi-Node Setup

For multi-node tests, you need to coordinate between two machines:

### Node Configuration
Edit the IP addresses in `multi_node_allreduce.sh`:
```bash
NODE1="10.232.194.226"  # Master node IP
NODE2="10.232.195.25"   # Worker node IP
```

### Running Multi-Node Tests

1. **On Master Node (NODE1)**:
   ```bash
   ./multi_node_allreduce.sh --2x2gpu
   ```

2. **On Worker Node (NODE2)** (run simultaneously):
   ```bash
   export CUDA_VISIBLE_DEVICES="0,1"
   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \
     --master_addr=10.232.194.226 --master_port=29500 \
     ../sendrecv/pytorch_sendrecv.py --operation allreduce \
     --min_size 2K --max_size 8G --step_factor 2 \
     --iterations 20 --warmup_iters 5 --test_mode multi_node
   ```

## Output Files

Each test generates three types of output files:

### 1. Log File (.txt)
- Detailed execution log with timestamps
- Example: `allreduce_4gpu_20241220_143022.txt`

### 2. CSV File (.csv)
- Performance data in CSV format
- Columns: size_bytes, size_str, count_elements, type, redop, root, time_us, algbw_gbps, busbw_gbps, errors
- Example: `allreduce_4gpu_20241220_143022.csv`

### 3. Console Output (.console)
- Complete console output including NCCL debug information
- Example: `allreduce_4gpu_20241220_143022.txt.console`

## Performance Metrics

### Algorithm Bandwidth (algbw_gbps)
- Effective bandwidth from application perspective
- Formula: `(tensor_size * 8) / (time_us * 1000)` GB/s

### Bus Bandwidth (busbw_gbps)
- Actual network/interconnect bandwidth utilization
- Formula: `algbw * 2 * (world_size - 1) / world_size` GB/s

### Time (time_us)
- Average communication time in microseconds
- Measured over multiple iterations after warmup

## Environment Requirements

### Software
- PyTorch with NCCL support
- CUDA toolkit
- Python 3.7+

### Hardware
- NVIDIA GPUs with CUDA support
- NVLink or PCIe interconnect for single-node
- InfiniBand or Ethernet for multi-node

### Environment Variables
The scripts automatically set:
```bash
export NCCL_IB_DISABLE=0      # Enable InfiniBand
export NCCL_DEBUG=INFO        # Enable NCCL debugging
export NCCL_DEBUG_SUBSYS=ALL  # Debug all subsystems
```

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Reduce max tensor size or use fewer GPUs
   - Check available GPU memory with `nvidia-smi`

2. **Multi-Node Connection Issues**
   - Verify network connectivity between nodes
   - Check firewall settings for port 29500
   - Ensure both nodes have the same NCCL version

3. **CUDA Device Error**
   - Verify CUDA_VISIBLE_DEVICES setting
   - Check GPU availability with `nvidia-smi`

### Debug Mode
For additional debugging, modify the scripts to include:
```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
```

## Performance Analysis

### Expected Trends
- **Small tensors**: Latency-dominated, performance decreases with more GPUs
- **Large tensors**: Bandwidth-dominated, performance improves with more GPUs
- **Crossover point**: Typically around 128KB-1MB depending on hardware

### Optimization Tips
- Use larger batch sizes to amortize communication overhead
- Consider gradient accumulation for small effective batch sizes
- Pipeline parallelism can help hide communication latency

## Support

For issues or questions:
1. Check the console output files for detailed error messages
2. Verify hardware and software requirements
3. Review NCCL documentation for advanced configuration options
