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
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# Emotions
EMOTIONS = ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
NUM_CLASSES = len(EMOTIONS)

# Paths
DATA_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# Device
DEVICE = 'mps'  # Will be set automatically in training script