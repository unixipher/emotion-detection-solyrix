"""
Advanced data augmentation techniques for audio
"""
import numpy as np
import librosa
import torch
from config import *


class AudioAugmentor:
    """Comprehensive audio augmentation"""
    
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        
    def add_noise(self, audio, noise_factor=0.005):
        """Add white noise"""
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        return augmented
    
    def shift_time(self, audio, shift_max=0.2):
        """Shift audio in time"""
        shift = np.random.randint(int(self.sr * shift_max))
        direction = np.random.choice([-1, 1])
        return np.roll(audio, shift * direction)
    
    def change_pitch(self, audio, n_steps=None):
        """Random pitch shift"""
        if n_steps is None:
            n_steps = np.random.randint(-3, 4)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def change_speed(self, audio, speed_factor=None):
        """Change speed/tempo"""
        if speed_factor is None:
            speed_factor = np.random.uniform(0.85, 1.15)
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def add_reverb(self, audio):
        """Simple reverb effect"""
        # Create impulse response
        impulse = np.zeros(int(self.sr * 0.1))
        impulse[0] = 1
        for i in range(1, len(impulse), 1000):
            impulse[i] = 0.5 ** (i / 1000)
        
        # Convolve
        reverb = np.convolve(audio, impulse, mode='same')
        return 0.7 * audio + 0.3 * reverb
    
    def change_volume(self, audio, factor=None):
        """Change volume"""
        if factor is None:
            factor = np.random.uniform(0.7, 1.3)
        return audio * factor
    
    def add_background_noise(self, audio, noise_audio=None):
        """Add background noise from another sample"""
        if noise_audio is None:
            noise_audio = np.random.randn(len(audio))
        
        # Mix with SNR between 5-15 dB
        snr_db = np.random.uniform(5, 15)
        
        audio_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        snr = 10 ** (snr_db / 10)
        scale = np.sqrt(audio_power / (snr * noise_power))
        
        return audio + scale * noise_audio
    
    def spec_augment(self, mel_spec, num_mask=2, freq_masking=0.15, time_masking=0.15):
        """SpecAugment: mask time and frequency"""
        mel_spec = mel_spec.copy()
        n_mels, n_frames = mel_spec.shape
        
        # Frequency masking
        for _ in range(num_mask):
            f = np.random.randint(0, int(freq_masking * n_mels))
            f0 = np.random.randint(0, n_mels - f)
            mel_spec[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(num_mask):
            t = np.random.randint(0, int(time_masking * n_frames))
            t0 = np.random.randint(0, n_frames - t)
            mel_spec[:, t0:t0+t] = 0
        
        return mel_spec
    
    def augment_audio(self, audio, augment_prob=0.8):
        """Apply random augmentations"""
        if np.random.random() > augment_prob:
            return audio
        
        augmentations = []
        
        # Randomly select augmentations
        if np.random.random() > 0.5:
            augmentations.append(lambda x: self.add_noise(x, np.random.uniform(0.002, 0.008)))
        
        if np.random.random() > 0.5:
            augmentations.append(self.shift_time)
        
        if np.random.random() > 0.6:
            augmentations.append(self.change_pitch)
        
        if np.random.random() > 0.6:
            augmentations.append(self.change_speed)
        
        if np.random.random() > 0.7:
            augmentations.append(self.add_reverb)
        
        if np.random.random() > 0.5:
            augmentations.append(self.change_volume)
        
        # Apply augmentations
        for aug in augmentations:
            try:
                audio = aug(audio)
            except:
                continue
        
        return audio
    
    def augment_mel(self, mel_spec, augment_prob=0.5):
        """Apply SpecAugment to mel-spectrogram"""
        if np.random.random() > augment_prob:
            return mel_spec
        
        return self.spec_augment(mel_spec)


# Modified Dataset class with augmentation
class AugmentedEmotionDataset(torch.utils.data.Dataset):
    """Dataset with online augmentation"""
    
    def __init__(self, data_dir, augmentor=None, training=True):
        from dataset import EmotionDataset
        
        self.base_dataset = EmotionDataset(data_dir)
        self.augmentor = augmentor if augmentor else AudioAugmentor()
        self.training = training
    
    def __len__(self):
        # Optionally increase dataset size with augmentations
        return len(self.base_dataset) * (3 if self.training else 1)
    
    def __getitem__(self, idx):
        # Get base sample
        base_idx = idx % len(self.base_dataset)
        mel_spec, prosodic, label = self.base_dataset[base_idx]
        
        # Apply augmentation during training
        if self.training and idx >= len(self.base_dataset):
            # Convert back to numpy for augmentation
            mel_np = mel_spec.squeeze(0).numpy()
            
            # Apply mel augmentation
            mel_np = self.augmentor.augment_mel(mel_np)
            
            # Convert back to tensor
            mel_spec = torch.FloatTensor(mel_np).unsqueeze(0)
        
        return mel_spec, prosodic, label


# Usage example
def create_augmented_loaders(data_dir, batch_size=BATCH_SIZE):
    """Create loaders with augmentation"""
    augmentor = AudioAugmentor()
    
    full_dataset = AugmentedEmotionDataset(data_dir, augmentor, training=True)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader