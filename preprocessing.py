"""
Enhanced feature extraction with more comprehensive features
"""
import librosa
import numpy as np
from config import *


def extract_enhanced_prosodic_features(audio, sr):
    """
    Extract comprehensive prosodic and voice quality features
    Returns 40+ features instead of 13
    """
    features = []
    
    # 1. Pitch Features (F0)
    f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), 
                     fmax=librosa.note_to_hz('C7'), sr=sr)
    f0_voiced = f0[f0 > librosa.note_to_hz('C2')]
    
    if len(f0_voiced) > 0:
        features.extend([
            np.mean(f0_voiced),
            np.std(f0_voiced),
            np.max(f0_voiced),
            np.min(f0_voiced),
            np.median(f0_voiced),
            np.percentile(f0_voiced, 25),
            np.percentile(f0_voiced, 75),
        ])
    else:
        features.extend([0] * 7)
    
    # 2. Energy/Intensity Features
    energy = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    features.extend([
        np.mean(energy),
        np.std(energy),
        np.max(energy),
        np.min(energy),
        np.median(energy),
    ])
    
    # 3. Zero Crossing Rate (voice quality)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
    features.extend([
        np.mean(zcr),
        np.std(zcr),
        np.max(zcr),
    ])
    
    # 4. Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )[0]
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )
    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio, hop_length=HOP_LENGTH
    )[0]
    
    features.extend([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_flatness),
        np.std(spectral_flatness),
    ])
    
    # Add spectral contrast (7 bands)
    for i in range(spectral_contrast.shape[0]):
        features.append(np.mean(spectral_contrast[i]))
    
    # 5. MFCC Statistics (first 13 coefficients)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    for i in range(mfcc.shape[0]):
        features.extend([
            np.mean(mfcc[i]),
            np.std(mfcc[i]),
        ])
    
    # 6. Chroma Features (12-dimensional)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=HOP_LENGTH)
    features.extend([
        np.mean(chroma[i]) for i in range(chroma.shape[0])
    ])
    
    # 7. Tempo
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features.append(tempo)
    
    # 8. Harmonic and Percussive Components
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    features.extend([
        np.mean(librosa.feature.rms(y=y_harmonic)[0]),
        np.mean(librosa.feature.rms(y=y_percussive)[0]),
    ])
    
    return np.array(features, dtype=np.float32)


def extract_multi_resolution_mel(audio, sr):
    """
    Extract mel-spectrograms at multiple time resolutions
    Returns stacked spectrograms
    """
    # Original resolution
    mel1 = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=2048
    )
    mel1_db = librosa.power_to_db(mel1, ref=np.max)
    
    # Higher time resolution (shorter hop)
    mel2 = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH//2, n_fft=2048
    )
    mel2_db = librosa.power_to_db(mel2, ref=np.max)
    
    # Lower time resolution (longer hop)
    mel3 = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH*2, n_fft=2048
    )
    mel3_db = librosa.power_to_db(mel3, ref=np.max)
    
    # Resize all to same time dimension
    target_frames = mel1_db.shape[1]
    
    if mel2_db.shape[1] != target_frames:
        mel2_db = librosa.util.fix_length(mel2_db, size=target_frames, axis=1)
    
    if mel3_db.shape[1] != target_frames:
        mel3_db = librosa.util.fix_length(mel3_db, size=target_frames, axis=1)
    
    # Stack as 3 channels
    multi_res_mel = np.stack([mel1_db, mel2_db, mel3_db], axis=0)
    
    return multi_res_mel


def extract_gammatone_spectrogram(audio, sr, n_bands=64):
    """
    Extract gammatone spectrogram (more perceptually relevant)
    Gammatone filters model human auditory system better
    """
    # Note: librosa doesn't have gammatone directly
    # Using mel as approximation, but you could use python_speech_features
    # or implement custom gammatone filterbank
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_bands, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel_spec, ref=np.max)
def extract_mel_spectrogram(audio, sr):
    """Extract mel-spectrogram"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=2048
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def load_and_preprocess_audio(file_path, duration=DURATION, sr=SAMPLE_RATE):
    """
    Load and preprocess audio with robust error handling
    """
    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Check if audio is valid
        if len(audio) == 0:
            raise ValueError("Empty audio file")
        
        # Pad or truncate to fixed length
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract features with error handling
        mel_spec = extract_mel_spectrogram(audio, sr)
        prosodic = extract_prosodic_features(audio, sr)
        
        # Validate feature shapes
        if mel_spec.shape[0] != N_MELS:
            raise ValueError(f"Invalid mel-spec shape: {mel_spec.shape}")
        
        if len(prosodic) != 13:  # Or whatever your prosodic dim is
            raise ValueError(f"Invalid prosodic shape: {len(prosodic)}")
        
        return mel_spec, prosodic
        
    except Exception as e:
        # Return None to signal error to dataset
        print(f"Error in preprocessing {file_path}: {e}")
        return None, None


def extract_prosodic_features(audio, sr):
    """Extract prosodic features with robust error handling"""
    try:
        # Pitch (F0) using YIN algorithm
        f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), 
                         fmax=librosa.note_to_hz('C7'), sr=sr)
        
        # Remove unvoiced frames
        f0_voiced = f0[f0 > librosa.note_to_hz('C2')]
        
        if len(f0_voiced) > 0:
            pitch_mean = np.mean(f0_voiced)
            pitch_std = np.std(f0_voiced)
            pitch_max = np.max(f0_voiced)
            pitch_min = np.min(f0_voiced)
        else:
            pitch_mean = pitch_std = pitch_max = pitch_min = 0
        
        # Energy/Intensity
        energy = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        energy_max = np.max(energy)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                              hop_length=HOP_LENGTH)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, 
                                                            hop_length=HOP_LENGTH)[0]
        
        # Combine all prosodic features
        prosodic_features = np.array([
            pitch_mean, pitch_std, pitch_max, pitch_min,
            energy_mean, energy_std, energy_max,
            zcr_mean, zcr_std,
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_rolloff), np.std(spectral_rolloff)
        ], dtype=np.float32)
        
        # Replace any NaN or Inf values with 0
        prosodic_features = np.nan_to_num(prosodic_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return prosodic_features
        
    except Exception as e:
        print(f"Error extracting prosodic features: {e}")
        # Return zeros if extraction fails
        return np.zeros(13, dtype=np.float32)


# Update config.py to handle variable prosodic feature size
# The enhanced version returns ~80+ features vs original 13


def normalize_features(features):
    """Normalize features to zero mean and unit variance"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (features - mean) / std


# For preprocessing entire dataset with normalization
class FeatureNormalizer:
    """Fit normalizer on training data and apply to all"""
    def __init__(self):
        self.prosodic_mean = None
        self.prosodic_std = None
        
    def fit(self, train_loader):
        """Compute mean and std from training data"""
        all_prosodic = []
        
        for _, prosodic, _ in train_loader:
            all_prosodic.append(prosodic.numpy())
        
        all_prosodic = np.vstack(all_prosodic)
        self.prosodic_mean = np.mean(all_prosodic, axis=0)
        self.prosodic_std = np.std(all_prosodic, axis=0)
        self.prosodic_std[self.prosodic_std == 0] = 1
        
    def transform(self, prosodic):
        """Normalize prosodic features"""
        if self.prosodic_mean is None:
            return prosodic
        
        return (prosodic - self.prosodic_mean) / self.prosodic_std
    
    def save(self, path):
        """Save normalizer"""
        np.savez(path, mean=self.prosodic_mean, std=self.prosodic_std)
    
    def load(self, path):
        """Load normalizer"""
        data = np.load(path)
        self.prosodic_mean = data['mean']
        self.prosodic_std = data['std']