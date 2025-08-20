#!/usr/bin/env python3
"""
PyTorch NCCL Communication Performance Test

This script implements comprehensive communication testing using PyTorch's NCCL APIs
including Send/Recv and AllReduce operations with accurate latency measurement and
nccl-tests compatible output format.

Features:
- Point-to-point Send/Recv testing
- AllReduce collective communication testing
- Single-node and multi-node configurations
- Comprehensive tensor size ranges (2KB to 8GB)

Based on the configuration parameters from sendrecv.sh and following the project's
coding conventions.
"""

import os
import sys
import time
import argparse
import csv
import socket
from datetime import datetime
from typing import List, Tuple, Dict, Any

import torch
import torch.distributed as dist


class NCCLSendRecvTester:
    """NCCL Send/Recv performance tester with nccl-tests compatible output"""
    
    def __init__(self, rank: int, world_size: int, args: argparse.Namespace):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = torch.device(f'cuda:{rank}')
        
        # Initialize logging
        self.log_file = None
        if args.log_file and rank == 0:
            self.log_file = open(args.log_file, 'w')
        
        # Initialize CSV output
        self.csv_file = None
        self.csv_writer = None
        if args.csv_file and rank == 0:
            self.csv_file = open(args.csv_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # Write CSV header
            self.csv_writer.writerow([
                'size_bytes', 'size_str', 'count_elements', 'type', 'redop', 'root',
                'time_us', 'algbw_gbps', 'busbw_gbps', 'errors'
            ])
    
    def log(self, message: str, print_to_console: bool = True):
        """Log message to file and optionally console"""
        if print_to_console:
            print(message)
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()
    
    def parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '2K', '1M', '1G') to bytes"""
        size_str = size_str.upper().strip()
        if size_str.endswith('K') or size_str.endswith('KB'):
            return int(size_str.rstrip('KB')) * 1024
        elif size_str.endswith('M') or size_str.endswith('MB'):
            return int(size_str.rstrip('MB')) * 1024 * 1024
        elif size_str.endswith('G') or size_str.endswith('GB'):
            return int(size_str.rstrip('GB')) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def format_size(self, bytes_val: int) -> str:
        """Format bytes to human readable string"""
        if bytes_val >= 1024 * 1024 * 1024:
            return f"{bytes_val // (1024 * 1024 * 1024)}G"
        elif bytes_val >= 1024 * 1024:
            return f"{bytes_val // (1024 * 1024)}M"
        elif bytes_val >= 1024:
            return f"{bytes_val // 1024}K"
        else:
            return f"{bytes_val}B"
    
    def generate_test_sizes(self) -> List[int]:
        """Generate test sizes based on min_size, max_size, and step_factor"""
        min_bytes = self.parse_size(self.args.min_size)
        max_bytes = self.parse_size(self.args.max_size)
        step_factor = self.args.step_factor
        
        sizes = []
        current_size = min_bytes
        while current_size <= max_bytes:
            sizes.append(current_size)
            current_size *= step_factor
        
        return sizes
    
    def warmup(self, tensor_size: int):
        """Perform warmup iterations"""
        if self.rank == 0:
            self.log(f"  Warming up with {self.args.warmup_iters} iterations...", False)
        
        # Create tensors
        send_tensor = torch.randn(tensor_size // 4, dtype=torch.float32, device=self.device)
        recv_tensor = torch.zeros_like(send_tensor)
        
        for _ in range(self.args.warmup_iters):
            if self.rank == 0:
                # Rank 0 sends to rank 1
                dist.send(send_tensor, dst=1)
                dist.recv(recv_tensor, src=1)
            else:
                # Rank 1 receives from rank 0 and sends back
                dist.recv(recv_tensor, src=0)
                dist.send(send_tensor, dst=0)
            
            # Synchronize
            torch.cuda.synchronize()
            dist.barrier()
    
    def measure_sendrecv_latency(self, tensor_size: int) -> float:
        """Measure send/recv latency for given tensor size"""
        # Create tensors
        send_tensor = torch.randn(tensor_size // 4, dtype=torch.float32, device=self.device)
        recv_tensor = torch.zeros_like(send_tensor)
        
        # Synchronize before measurement
        torch.cuda.synchronize()
        dist.barrier()
        
        # Measure time
        start_time = time.perf_counter()
        
        if self.rank == 0:
            # Rank 0 sends to rank 1
            dist.send(send_tensor, dst=1)
            dist.recv(recv_tensor, src=1)
        else:
            # Rank 1 receives from rank 0 and sends back
            dist.recv(recv_tensor, src=0)
            dist.send(send_tensor, dst=0)
        
        # Synchronize after communication
        torch.cuda.synchronize()
        dist.barrier()
        
        end_time = time.perf_counter()
        
        # Return latency in microseconds
        return (end_time - start_time) * 1e6
    
    def calculate_bandwidth(self, tensor_size: int, time_us: float) -> Tuple[float, float]:
        """Calculate algorithm and bus bandwidth"""
        # Algorithm bandwidth: data transferred / time
        algbw_gbps = (tensor_size * 2) / (time_us * 1e-6) / (1024**3)  # 2x for send+recv
        
        # Bus bandwidth (same as algorithm bandwidth for point-to-point)
        busbw_gbps = algbw_gbps
        
        return algbw_gbps, busbw_gbps
    
    def run_test_size(self, tensor_size: int) -> Dict[str, Any]:
        """Run test for a specific tensor size"""
        # Warmup
        self.warmup(tensor_size)
        
        # Measure multiple iterations
        latencies = []
        for i in range(self.args.iterations):
            latency = self.measure_sendrecv_latency(tensor_size)
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Calculate bandwidth
        algbw_gbps, busbw_gbps = self.calculate_bandwidth(tensor_size, avg_latency)
        
        return {
            'size_bytes': tensor_size,
            'size_str': self.format_size(tensor_size),
            'count_elements': tensor_size // 4,  # float32 = 4 bytes
            'avg_latency_us': avg_latency,
            'min_latency_us': min_latency,
            'max_latency_us': max_latency,
            'algbw_gbps': algbw_gbps,
            'busbw_gbps': busbw_gbps,
            'errors': 0  # No error checking in this implementation
        }
    
    def print_header(self):
        """Print nccl-tests compatible header"""
        if self.rank != 0:
            return

        hostname = socket.gethostname()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get NCCL version info
        try:
            import torch.distributed as dist
            nccl_version = "PyTorch NCCL backend"
        except:
            nccl_version = "Unknown"

        self.log("# nccl-tests compatible output")
        self.log(f"# Hostname: {hostname}")
        self.log(f"# Timestamp: {timestamp}")
        self.log(f"# Test mode: {self.args.test_mode}")
        self.log(f"# World size: {self.world_size}")
        self.log(f"# Iterations: {self.args.iterations}")
        self.log(f"# Warmup iterations: {self.args.warmup_iters}")
        self.log(f"# NCCL version: {nccl_version}")

        # Add topology information
        if self.args.test_mode == "multi_node":
            self.log(f"# Multi-node configuration: {self.world_size} processes across nodes")
        else:
            self.log(f"# Single-node configuration: {self.world_size} processes on single node")

        self.log("#")
        self.log("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s" % (
            "size", "count", "type", "redop", "root", "time", "algbw", "busbw", "#wrong"
        ))
        self.log("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s" % (
            "(B)", "(elements)", "", "", "", "(us)", "(GB/s)", "(GB/s)", ""
        ))
    
    def print_result(self, result: Dict[str, Any]):
        """Print test result in nccl-tests compatible format"""
        if self.rank != 0:
            return
        
        # Format time string
        time_us = result['avg_latency_us']
        if time_us >= 10000.0:
            time_str = f"{time_us:7.0f}"
        elif time_us >= 100.0:
            time_str = f"{time_us:7.1f}"
        else:
            time_str = f"{time_us:7.2f}"
        
        # Print result
        self.log("  %10d  %12d  %8s  %6s  %6s  %7s  %6.2f  %6.2f  %5s" % (
            result['size_bytes'],
            result['count_elements'],
            "float",
            "none",
            "none",
            time_str,
            result['algbw_gbps'],
            result['busbw_gbps'],
            "N/A"
        ))
        
        # Write to CSV
        if self.csv_writer:
            self.csv_writer.writerow([
                result['size_bytes'],
                result['size_str'],
                result['count_elements'],
                "float",
                "none",
                "none",
                result['avg_latency_us'],
                result['algbw_gbps'],
                result['busbw_gbps'],
                result['errors']
            ])
    
    def run_tests(self):
        """Run all tests"""
        if self.rank == 0:
            self.log("Starting PyTorch NCCL Send/Recv performance tests...")
        
        # Print header
        self.print_header()
        
        # Generate test sizes
        test_sizes = self.generate_test_sizes()
        
        if self.rank == 0:
            self.log(f"Testing {len(test_sizes)} sizes from {self.format_size(test_sizes[0])} to {self.format_size(test_sizes[-1])}")
        
        # Run tests for each size
        for tensor_size in test_sizes:
            result = self.run_test_size(tensor_size)
            self.print_result(result)
        
        if self.rank == 0:
            self.log("#")
            self.log("# Test completed successfully")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.log_file:
            self.log_file.close()
        if self.csv_file:
            self.csv_file.close()


class NCCLAllReduceTester:
    """NCCL AllReduce performance tester with nccl-tests compatible output"""

    def __init__(self, rank: int, world_size: int, args: argparse.Namespace):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = torch.device(f'cuda:{rank}')

        # Initialize logging
        self.log_file = None
        if args.log_file and rank == 0:
            self.log_file = open(args.log_file, 'w')

        # Initialize CSV output
        self.csv_file = None
        self.csv_writer = None
        if args.csv_file and rank == 0:
            self.csv_file = open(args.csv_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # Write CSV header
            self.csv_writer.writerow([
                'size_bytes', 'size_str', 'count_elements', 'type', 'redop', 'root',
                'time_us', 'algbw_gbps', 'busbw_gbps', 'errors'
            ])

    def log(self, message: str):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] Rank {self.rank}: {message}"
        print(log_msg)
        if self.log_file:
            self.log_file.write(log_msg + '\n')
            self.log_file.flush()

    def parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '2K', '1M', '1G') to bytes"""
        size_str = size_str.upper().strip()
        if size_str.endswith('K') or size_str.endswith('KB'):
            return int(size_str.rstrip('KB')) * 1024
        elif size_str.endswith('M') or size_str.endswith('MB'):
            return int(size_str.rstrip('MB')) * 1024 * 1024
        elif size_str.endswith('G') or size_str.endswith('GB'):
            return int(size_str.rstrip('GB')) * 1024 * 1024 * 1024
        else:
            return int(size_str)

    def format_size(self, bytes_val: int) -> str:
        """Format bytes to human readable string"""
        if bytes_val >= 1024 * 1024 * 1024:
            return f"{bytes_val // (1024 * 1024 * 1024)}G"
        elif bytes_val >= 1024 * 1024:
            return f"{bytes_val // (1024 * 1024)}M"
        elif bytes_val >= 1024:
            return f"{bytes_val // 1024}K"
        else:
            return f"{bytes_val}B"

    def generate_test_sizes(self) -> List[int]:
        """Generate test sizes based on min_size, max_size, and step_factor"""
        min_bytes = self.parse_size(self.args.min_size)
        max_bytes = self.parse_size(self.args.max_size)
        step_factor = self.args.step_factor

        sizes = []
        current_size = min_bytes
        while current_size <= max_bytes:
            sizes.append(current_size)
            current_size *= step_factor

        return sizes

    def warmup(self, tensor_size: int):
        """Warmup runs to stabilize performance"""
        tensor = torch.randn(tensor_size // 4, dtype=torch.float32, device=self.device)

        for _ in range(self.args.warmup_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

    def calculate_bandwidth(self, tensor_size: int, time_us: float) -> Tuple[float, float]:
        """Calculate algorithm and bus bandwidth in GB/s"""
        # AllReduce algorithm bandwidth: total data moved / time
        # For AllReduce, each rank sends and receives (world_size-1)/world_size of the data
        algbw_gbps = (tensor_size * 8) / (time_us * 1000)  # Convert to GB/s

        # Bus bandwidth for AllReduce: 2 * (world_size - 1) / world_size * algbw
        # This accounts for the fact that AllReduce requires 2*(N-1) communication steps
        busbw_gbps = algbw_gbps * 2 * (self.world_size - 1) / self.world_size

        return algbw_gbps, busbw_gbps

    def run_test_size(self, tensor_size: int) -> Dict[str, Any]:
        """Run AllReduce test for a specific tensor size"""
        # Warmup
        self.warmup(tensor_size)

        # Measure multiple iterations
        times = []
        tensor = torch.randn(tensor_size // 4, dtype=torch.float32, device=self.device)

        for _ in range(self.args.iterations):
            # Synchronize before measurement
            torch.cuda.synchronize()
            dist.barrier()

            # Measure time
            start_time = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append((end_time - start_time) * 1_000_000)  # Convert to microseconds

        # Calculate statistics
        avg_time_us = sum(times) / len(times)
        algbw_gbps, busbw_gbps = self.calculate_bandwidth(tensor_size, avg_time_us)

        return {
            'size_bytes': tensor_size,
            'size_str': self.format_size(tensor_size),
            'count_elements': tensor_size // 4,  # float32 = 4 bytes
            'type': 'float',
            'redop': 'sum',
            'root': 'N/A',
            'time_us': avg_time_us,
            'algbw_gbps': algbw_gbps,
            'busbw_gbps': busbw_gbps,
            'errors': 0
        }

    def print_result(self, result: Dict[str, Any]):
        """Print test result in nccl-tests format"""
        if self.rank == 0:
            # Print to console
            print(f"  {result['size_str']:>8}  {result['count_elements']:>10}     {result['type']:>5}     {result['redop']:>3}    {result['root']:>4}  "
                  f"{result['time_us']:>8.2f}  {result['algbw_gbps']:>7.2f}  {result['busbw_gbps']:>7.2f}  {result['errors']:>6}")

            # Write to CSV
            if self.csv_writer:
                self.csv_writer.writerow([
                    result['size_bytes'],
                    result['size_str'],
                    result['count_elements'],
                    result['type'],
                    result['redop'],
                    result['root'],
                    result['time_us'],
                    result['algbw_gbps'],
                    result['busbw_gbps'],
                    result['errors']
                ])

    def run_tests(self):
        """Run all AllReduce tests"""
        if self.rank == 0:
            self.log("Starting PyTorch NCCL AllReduce performance tests...")
            self.log(f"World size: {self.world_size}")
            self.log(f"Test mode: {self.args.test_mode}")

        # Print header
        if self.rank == 0:
            print("#")
            print("# NCCL AllReduce Performance Test")
            print(f"# Size       Count      Type   RedOp   Root     Time   AlgBw   BusBw  Error")
            print(f"# (B)        (elements)                          (us)  (GB/s)  (GB/s)")
            print("#")

        # Generate test sizes
        test_sizes = self.generate_test_sizes()

        if self.rank == 0:
            self.log(f"Testing {len(test_sizes)} sizes from {self.format_size(test_sizes[0])} to {self.format_size(test_sizes[-1])}")

        # Run tests for each size
        for tensor_size in test_sizes:
            result = self.run_test_size(tensor_size)
            self.print_result(result)

        if self.rank == 0:
            self.log("#")
            self.log("# AllReduce test completed successfully")

    def cleanup(self):
        """Cleanup resources"""
        if self.log_file:
            self.log_file.close()
        if self.csv_file:
            self.csv_file.close()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch NCCL Communication Performance Test')

    # Test parameters
    parser.add_argument('--min_size', type=str, default='2K', help='Minimum tensor size')
    parser.add_argument('--max_size', type=str, default='2G', help='Maximum tensor size')
    parser.add_argument('--step_factor', type=int, default=2, help='Step factor for size progression')
    parser.add_argument('--iterations', type=int, default=20, help='Number of test iterations')
    parser.add_argument('--warmup_iters', type=int, default=5, help='Number of warmup iterations')

    # Output files
    parser.add_argument('--log_file', type=str, help='Log file path')
    parser.add_argument('--csv_file', type=str, help='CSV output file path')

    # Test mode
    parser.add_argument('--test_mode', type=str, choices=['single_node', 'multi_node'],
                       default='single_node', help='Test mode')

    # Communication operation
    parser.add_argument('--operation', type=str, choices=['sendrecv', 'allreduce'],
                       default='sendrecv', help='Communication operation to test')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Initialize distributed training
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Validate world size based on operation
    if args.operation == 'sendrecv':
        if world_size != 2:
            if rank == 0:
                print(f"Error: Send/Recv test requires exactly 2 processes, got {world_size}")
            sys.exit(1)
    elif args.operation == 'allreduce':
        if world_size < 2:
            if rank == 0:
                print(f"Error: AllReduce test requires at least 2 processes, got {world_size}")
            sys.exit(1)

    # Set CUDA device
    torch.cuda.set_device(rank)

    try:
        # Create tester based on operation and run tests
        if args.operation == 'sendrecv':
            tester = NCCLSendRecvTester(rank, world_size, args)
        elif args.operation == 'allreduce':
            tester = NCCLAllReduceTester(rank, world_size, args)
        else:
            if rank == 0:
                print(f"Error: Unknown operation {args.operation}")
            sys.exit(1)

        tester.run_tests()
        tester.cleanup()

    except Exception as e:
        if rank == 0:
            print(f"Error during testing: {e}")
        sys.exit(1)

    finally:
        # Cleanup distributed training
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
