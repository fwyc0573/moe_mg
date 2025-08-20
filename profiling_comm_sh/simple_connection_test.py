#!/usr/bin/env python3
"""
Simple Multi-Node Connection Test

This script tests basic PyTorch distributed connection without NCCL complexity.
Use this to verify that the basic distributed setup works before running AllReduce tests.
"""

import os
import sys
import time
import socket
import argparse
from datetime import datetime

import torch
import torch.distributed as dist

def log_message(rank, message):
    """Log message with timestamp and rank"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Rank {rank}: {message}")
    sys.stdout.flush()

def test_basic_connection(rank, world_size, master_addr, master_port):
    """Test basic distributed connection"""
    log_message(rank, f"Starting connection test...")
    log_message(rank, f"World size: {world_size}")
    log_message(rank, f"Master addr: {master_addr}")
    log_message(rank, f"Master port: {master_port}")
    
    try:
        # Initialize process group with timeout
        log_message(rank, "Initializing process group...")
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout
        )
        
        log_message(rank, "✅ Process group initialized successfully!")
        
        # Test barrier synchronization
        log_message(rank, "Testing barrier synchronization...")
        dist.barrier()
        log_message(rank, "✅ Barrier synchronization successful!")
        
        # Test simple tensor operation
        log_message(rank, "Testing simple tensor operation...")
        device = torch.device(f'cuda:{rank}')
        tensor = torch.ones(10, device=device) * (rank + 1)
        
        log_message(rank, f"Before AllReduce: {tensor[:5].tolist()}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        log_message(rank, f"After AllReduce: {tensor[:5].tolist()}")
        
        # Expected result: sum of (1 + 2 + ... + world_size) = world_size * (world_size + 1) / 2
        expected_sum = world_size * (world_size + 1) // 2
        if abs(tensor[0].item() - expected_sum) < 1e-6:
            log_message(rank, "✅ AllReduce test successful!")
        else:
            log_message(rank, f"❌ AllReduce test failed! Expected {expected_sum}, got {tensor[0].item()}")
        
        # Final barrier
        dist.barrier()
        log_message(rank, "✅ All tests completed successfully!")
        
        return True
        
    except Exception as e:
        log_message(rank, f"❌ Connection test failed: {e}")
        return False
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            log_message(rank, "Process group destroyed")

def test_network_connectivity(master_addr, master_port):
    """Test basic network connectivity"""
    print(f"Testing network connectivity to {master_addr}:{master_port}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((master_addr, int(master_port)))
        sock.close()
        
        if result == 0:
            print("✅ Network connection successful")
            return True
        else:
            print(f"❌ Network connection failed (error code: {result})")
            return False
    except Exception as e:
        print(f"❌ Network test error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple Multi-Node Connection Test')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    parser.add_argument('--rank', type=int, required=True, help='Node rank')
    parser.add_argument('--world_size', type=int, required=True, help='Total number of nodes')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank (GPU index)')
    
    args = parser.parse_args()
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        print(f"Using CUDA device: {args.local_rank}")
    else:
        print("CUDA not available!")
        sys.exit(1)
    
    # Test network connectivity first (only for worker nodes)
    if args.rank > 0:
        if not test_network_connectivity(args.master_addr, args.master_port):
            print("Network connectivity test failed. Check firewall and network configuration.")
            sys.exit(1)
    
    # Run connection test
    success = test_basic_connection(args.rank, args.world_size, args.master_addr, args.master_port)
    
    if success:
        print("🎉 Connection test completed successfully!")
        sys.exit(0)
    else:
        print("💥 Connection test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
