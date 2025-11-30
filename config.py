"""
Configuration file for Speech Emotion Recognition
"""

# Audio Parameters
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512

# Model Parameters
BATCH_SIZE = 64  # Increased for better training
EPOCHS = 150  # More epochs for convergence
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 25  # More patience

# Emotions
EMOTIONS = ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
NUM_CLASSES = len(EMOTIONS)

# Paths
DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# Enhanced prosodic features dimension (if using enhanced preprocessing)
PROSODIC_DIM = 13  # Will be 80+ if you use enhanced features later

# Training enhancements
USE_FOCAL_LOSS = True
USE_LABEL_SMOOTHING = False  # Use focal loss OR label smoothing, not both
USE_MIXUP = True
GRADIENT_CLIP = 1.0
USE_SWA = True  # Stochastic Weight Averaging

# Augmentation
AUGMENT_PROB = 0.8
SPEC_AUGMENT = True

# Device - Auto-detect best available
import torch
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"🖥️  Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
    print("🖥️  Using Apple Silicon MPS")
else:
    DEVICE = 'cpu'
    print("🖥️  Using CPU")