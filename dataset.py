"""
Dataset loader for Speech Emotion Recognition
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from preprocessing import load_and_preprocess_audio_enhanced as load_and_preprocess_audio
from config import *


class EmotionDataset(Dataset):
    """
    Dataset for emotion recognition
    
    Expected directory structure:
    data/raw/
        happiness/
            audio1.wav
            audio2.wav
        sadness/
            audio1.wav
        ...
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
        
        # Load all audio file paths
        self._load_samples()
        
    def _load_samples(self):
        """Scan directory and create list of (file_path, emotion_label)"""
        for emotion in EMOTIONS:
            emotion_dir = os.path.join(self.data_dir, emotion)
            
            if not os.path.exists(emotion_dir):
                print(f"Warning: Directory {emotion_dir} not found. Skipping.")
                continue
                
            for audio_file in os.listdir(emotion_dir):
                if audio_file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    file_path = os.path.join(emotion_dir, audio_file)
                    label = self.emotion_to_idx[emotion]
                    self.samples.append((file_path, label))
        
        print(f"Loaded {len(self.samples)} audio samples")
        
        # Print distribution
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\nEmotion distribution:")
        for emotion, idx in self.emotion_to_idx.items():
            count = label_counts.get(idx, 0)
            print(f"  {emotion}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        try:
            # Extract features
            mel_spec, prosodic = load_and_preprocess_audio(file_path)
            
            # Convert to tensors
            # Add channel dimension to mel_spec: (1, n_mels, time)
            mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)
            prosodic = torch.FloatTensor(prosodic)
            label = torch.LongTensor([label])[0]
            
            # Apply transforms if any
            if self.transform:
                mel_spec = self.transform(mel_spec)
            
            return mel_spec, prosodic, label
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__((idx + 1) % len(self.samples))


def create_data_loaders(data_dir, batch_size=BATCH_SIZE, train_split=0.8, random_seed=42):
    """
    Create train and validation data loaders
    
    Args:
        data_dir: Directory containing emotion folders
        batch_size: Batch size
        train_split: Fraction of data for training
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    full_dataset = EmotionDataset(data_dir)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for MPS compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    return train_loader, val_loader


# Test the dataset
if __name__ == "__main__":
    # Test loading
    dataset = EmotionDataset(DATA_DIR)
    
    if len(dataset) > 0:
        mel_spec, prosodic, label = dataset[0]
        print(f"\nSample shapes:")
        print(f"  Mel-spectrogram: {mel_spec.shape}")
        print(f"  Prosodic features: {prosodic.shape}")
        print(f"  Label: {label}")
        print(f"  Emotion: {EMOTIONS[label]}")