"""
Push-to-Talk Inference Script
Records for fixed duration (3s) after keypress, then runs inference.
"""
import argparse
import os
import sys
import numpy as np
import sounddevice as sd
import torch
import time

# Use existing feature extraction
from preprocessing import extract_mel_spectrogram, extract_prosodic_features
from model import HybridEmotionModel
from config import SAMPLE_RATE, DURATION, EMOTIONS, NUM_CLASSES

def print_prediction(emotion, probs):
    """Format and print the prediction results nicely."""
    # Sort probabilities
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print(f"🎭 Predicted Emotion: {emotion.upper()}")
    print("📊 Probabilities:")
    for emo, p in items:
        # visual bar
        bar_len = int(p * 30)
        bar = "█" * bar_len
        print(f"  {emo:12s}: {p*100:6.2f}% {bar}")
    print("="*60 + "\n")

def record_audio(duration, sr, device_index=None):
    """Record audio for a fixed duration."""
    print(f"🔴 Recording for {duration} seconds...", end="", flush=True)
    
    # sd.rec records in the background, returns a numpy array
    recording = sd.rec(
        int(duration * sr), 
        samplerate=sr, 
        channels=1, 
        dtype='float32',
        device=device_index
    )
    
    # Wait until recording is finished
    sd.wait()
    print(" Done.")
    
    # Flatten to 1D array (samples,)
    return recording.flatten()

def main():
    parser = argparse.ArgumentParser(description="Push-to-Talk Emotion Detection")
    parser.add_argument("model_path", nargs="?", default="models/best_model.pth", type=str)
    parser.add_argument("--device", type=str, default=None, help="PyTorch device (cuda/mps/cpu)")
    parser.add_argument("--device-index", type=int, default=None, help="Audio device index")
    args = parser.parse_args()

    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if args.device:
        device = torch.device(args.device)
    print(f"✓ Using computation device: {device}")

    # 2. Locate and Load Model
    if not os.path.exists(args.model_path):
        # Fallback check in models/ folder
        alt_path = os.path.join("models", os.path.basename(args.model_path))
        if os.path.exists(alt_path):
            args.model_path = alt_path
        else:
            print(f"❌ Model not found at {args.model_path}")
            return

    print("⏳ Loading model...")
    model = HybridEmotionModel()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")

    print(f"\nConfiguration: {DURATION}s clips @ {SAMPLE_RATE}Hz")
    print("-" * 50)
    print("INSTRUCTIONS:")
    print("1. Press [ENTER] to start recording.")
    print("2. Speak clearly for 3 seconds.")
    print("3. Wait for prediction.")
    print("4. Press [Ctrl+C] to exit.")
    print("-" * 50)

    # 3. Main Interaction Loop
    try:
        while True:
            # Wait for user input
            input("Press [ENTER] to record... ")

            # Record
            audio_chunk = record_audio(DURATION, SAMPLE_RATE, args.device_index)

            # Analyze
            try:
                # Feature Extraction
                mel_spec = extract_mel_spectrogram(audio_chunk, SAMPLE_RATE)
                prosodic = extract_prosodic_features(audio_chunk, SAMPLE_RATE)

                # Prepare Tensors
                # Mel shape: (1, 1, n_mels, time)
                mel_t = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
                # Prosodic shape: (1, 13)
                pro_t = torch.FloatTensor(prosodic).unsqueeze(0).to(device)

                # Inference
                with torch.no_grad():
                    outputs = model(mel_t, pro_t)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

                # Results
                probs_dict = {EMOTIONS[i]: float(probs[i]) for i in range(NUM_CLASSES)}
                pred_idx = int(np.argmax(probs))
                pred_emotion = EMOTIONS[pred_idx]

                print_prediction(pred_emotion, probs_dict)

            except Exception as e:
                print(f"❌ Error during analysis: {e}")

    except KeyboardInterrupt:
        print("\n\n🔵 Exiting...")

if __name__ == "__main__":
    main()