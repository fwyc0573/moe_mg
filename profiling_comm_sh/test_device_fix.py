#!/usr/bin/env python3
"""
Quick test to verify CUDA device assignment fix
"""

import os
import torch
import torch.distributed as dist

def test_device_assignment():
    """Test the device assignment logic"""
    
    print("Testing CUDA device assignment logic...")
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    
    # Simulate different rank scenarios
    test_cases = [
        # (rank, expected_local_rank, description)
        (0, 0, "Single node rank 0"),
        (1, 1, "Single node rank 1"),
        (2, 0, "Multi-node worker rank 2 -> local 0"),
        (3, 1, "Multi-node worker rank 3 -> local 1"),
        (4, 0, "Multi-node worker rank 4 -> local 0"),
        (5, 1, "Multi-node worker rank 5 -> local 1"),
    ]
    
    for rank, expected_local, description in test_cases:
        # Test the logic from our fix
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        
        print(f"Rank {rank}: local_rank = {local_rank} (expected {expected_local}) - {description}")
        
        if local_rank == expected_local:
            print("  ✅ Correct")
        else:
            print("  ❌ Incorrect")
        
        # Test if we can create a device object
        try:
            device = torch.device(f'cuda:{local_rank}')
            print(f"  Device: {device}")
        except Exception as e:
            print(f"  Error creating device: {e}")
        
        print()

def test_with_local_rank_env():
    """Test with LOCAL_RANK environment variable set"""
    print("Testing with LOCAL_RANK environment variable...")
    
    # Simulate torchrun setting LOCAL_RANK
    test_local_ranks = [0, 1]
    
    for local_rank in test_local_ranks:
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        # Test different global ranks
        for global_rank in [0, 1, 2, 3]:
            detected_local = int(os.environ.get('LOCAL_RANK', global_rank % torch.cuda.device_count()))
            print(f"LOCAL_RANK={local_rank}, global_rank={global_rank} -> detected_local={detected_local}")
            
            if detected_local == local_rank:
                print("  ✅ Using LOCAL_RANK correctly")
            else:
                print("  ❌ LOCAL_RANK not used correctly")
    
    # Clean up
    if 'LOCAL_RANK' in os.environ:
        del os.environ['LOCAL_RANK']

if __name__ == "__main__":
    print("CUDA Device Assignment Test")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    test_device_assignment()
    test_with_local_rank_env()
    
    print("Test completed!")
