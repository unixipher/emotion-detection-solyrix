"""
Audio preprocessing and feature extraction
"""
import librosa
import numpy as np
from config import *


def extract_prosodic_features(audio, sr):
    """Extract prosodic features: pitch, energy, jitter, etc."""
    
    # Pitch (F0) using YIN algorithm
    f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), 
                     fmax=librosa.note_to_hz('C7'), sr=sr)
    
    # Remove unvoiced frames (f0 == fmin)
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
    
    # Zero Crossing Rate (voice quality indicator)
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
    ])
    
    return prosodic_features


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
    Load audio file and extract both mel-spectrogram and prosodic features
    
    Returns:
        mel_spec: Mel-spectrogram (N_MELS x TIME)
        prosodic: Prosodic features vector (13 features)
    """
    # Load audio
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    
    # Pad or truncate to fixed length
    target_length = sr * duration
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    # Extract features
    mel_spec = extract_mel_spectrogram(audio, sr)
    prosodic = extract_prosodic_features(audio, sr)
    
    return mel_spec, prosodic


def augment_audio(audio, sr):
    """Apply data augmentation"""
    augmentations = []
    
    # Original
    augmentations.append(audio)
    
    # Pitch shift
    audio_pitch_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    audio_pitch_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
    augmentations.extend([audio_pitch_up, audio_pitch_down])
    
    # Time stretch
    audio_faster = librosa.effects.time_stretch(audio, rate=1.1)
    audio_slower = librosa.effects.time_stretch(audio, rate=0.9)
    augmentations.extend([audio_faster, audio_slower])
    
    # Add noise
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise
    augmentations.append(audio_noise)
    
    return augmentations