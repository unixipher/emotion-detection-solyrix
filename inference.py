"""
Inference script for Speech Emotion Recognition
"""
import torch
import numpy as np
from preprocessing import load_and_preprocess_audio
from model import HybridEmotionModel
from config import *


class EmotionPredictor:
    def __init__(self, model_path, device=None):
        # Auto-detect device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.backends.mps.is_available() else 'cpu')
        
        # Load model
        self.model = HybridEmotionModel()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Using device: {self.device}")
    
    def predict(self, audio_file):
        """
        Predict emotion from audio file
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            emotion: Predicted emotion string
            probabilities: Dictionary of emotion probabilities
        """
        # Extract features
        mel_spec, prosodic = load_and_preprocess_audio(audio_file)
        
        # Convert to tensors and add batch dimension
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        prosodic = torch.FloatTensor(prosodic).unsqueeze(0)  # (1, 13)
        
        # Move to device
        mel_spec = mel_spec.to(self.device)
        prosodic = prosodic.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(mel_spec, prosodic)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
        
        # Get emotion and probabilities
        emotion = EMOTIONS[predicted_idx]
        probs_dict = {
            EMOTIONS[i]: float(probabilities[0][i].cpu().numpy()) 
            for i in range(NUM_CLASSES)
        }
        
        return emotion, probs_dict
    
    def predict_batch(self, audio_files):
        """
        Predict emotions for multiple audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of (emotion, probabilities) tuples
        """
        results = []
        for audio_file in audio_files:
            emotion, probs = self.predict(audio_file)
            results.append((emotion, probs))
        return results


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file>")
        print("Example: python inference.py test_audio.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_path = f"{MODEL_DIR}/best_model.pth"
    
    # Create predictor
    predictor = EmotionPredictor(model_path)
    
    # Predict
    print(f"\nAnalyzing: {audio_file}")
    emotion, probabilities = predictor.predict(audio_file)
    
    print(f"\n🎭 Predicted Emotion: {emotion.upper()}")
    print(f"\n📊 Confidence scores:")
    
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for emo, prob in sorted_probs:
        bar = "█" * int(prob * 50)
        print(f"  {emo:12s}: {prob:6.2%} {bar}")


if __name__ == "__main__":
    main()