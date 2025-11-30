"""Test all imports and basic functionality"""
import sys
import traceback

print("Testing imports...")

try:
    print("1. Testing config...")
    from config import *
    print(f"   ✓ Config loaded: {NUM_CLASSES} classes")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    traceback.print_exc()

try:
    print("2. Testing preprocessing...")
    from preprocessing import load_and_preprocess_audio_enhanced
    print("   ✓ Preprocessing loaded")
except Exception as e:
    print(f"   ✗ Preprocessing failed: {e}")
    traceback.print_exc()

try:
    print("3. Testing model...")
    from model import EnhancedHybridModel
    model = EnhancedHybridModel()
    print(f"   ✓ Model loaded: {model.get_num_parameters():,} parameters")
except Exception as e:
    print(f"   ✗ Model failed: {e}")
    traceback.print_exc()

try:
    print("4. Testing dataset...")
    from dataset import create_data_loaders
    print("   ✓ Dataset module loaded")
except Exception as e:
    print(f"   ✗ Dataset failed: {e}")
    traceback.print_exc()

try:
    print("5. Testing augmentation...")
    from augmentation import AudioAugmentor
    print("   ✓ Augmentation loaded")
except Exception as e:
    print(f"   ✗ Augmentation failed (this is optional)")

try:
    print("6. Testing torch...")
    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   ✓ Torch loaded, device: {device}")
except Exception as e:
    print(f"   ✗ Torch failed: {e}")

print("\nAll tests complete!")