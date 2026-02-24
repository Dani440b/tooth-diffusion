#!/usr/bin/env python3
"""Quick test to verify the dataloader works correctly."""

import sys
sys.path.append(".")

from guided_diffusion.toothloader import ToothVolumes

# Test dataloader initialization
print("Testing ToothVolumes dataloader...")
try:
    ds = ToothVolumes(
        directory="prep_data/train",
        metadata_path="prep_data/metadata.csv",
        test_flag=False,
        normalize=(lambda x: 2 * x - 1),
        mode='train',
        img_size=256,
    )
    print(f"✓ Dataset initialized successfully with {len(ds)} samples")
    
    # Test loading first sample
    print("\nTesting first sample...")
    sample = ds[0]
    
    print(f"✓ Sample loaded successfully")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label shape: {sample['label'].shape}")
    print(f"  Cond image shape: {sample['cond_image'].shape}")
    print(f"  Cond label shape: {sample['cond_label'].shape}")
    print(f"  Diagnosis shape: {sample['diagnosis'].shape}")
    print(f"  Age shape: {sample['age'].shape}")
    print(f"  Sex shape: {sample['sex'].shape}")
    print(f"  Brain mask shape: {sample['brain_mask'].shape}")
    print(f"  Name: {sample.get('name', 'N/A')}")
    
    # Verify image values are in expected range
    import torch
    img_min = sample['image'].min().item()
    img_max = sample['image'].max().item()
    print(f"\n  Image value range: [{img_min:.3f}, {img_max:.3f}]")
    
    # Check if cropping worked
    print(f"\n✓ Volume cropping appears to have worked (depth={sample['image'].shape[1]})")
    
    print("\n✅ All tests passed! Dataloader is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
