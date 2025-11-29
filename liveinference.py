"""
Real-time microphone inference using HybridEmotionModel

Usage:
    python liveinference.py models/best_model.pth --hop 0.8

Notes:
- Requires: sounddevice
    pip install sounddevice
- Uses config.SAMPLE_RATE and config.DURATION
- Uses existing model: HybridEmotionModel in model.py
- Uses feature extractors from preprocessing.py
"""
import argparse
import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch

from preprocessing import extract_mel_spectrogram, extract_prosodic_features
from model import HybridEmotionModel
from config import SAMPLE_RATE, DURATION, EMOTIONS, NUM_CLASSES

# Small helper function to format output
def print_prediction(emotion, probs):
    # Get full sorted probabilities
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    print("\n" + "="*60)
    print(f"🎭 Predicted Emotion: {emotion.upper()}")
    print("📊 Probabilities:")
    for emo, p in items:
        bar = "█" * int(p * 30)
        print(f"  {emo:12s}: {p*100:6.2f}% {bar}")
    print("="*60)

class RealtimeEmotionDetector:
    def __init__(self, model_path, device=None, duration=DURATION, hop=1.0):
        self.model_path = model_path
        self.duration = duration
        self.hop = hop
        self.sr = SAMPLE_RATE
        self.full_length = int(self.sr * self.duration)
        self.hop_length = int(self.sr * hop)
        self.buffer = np.zeros(self.full_length, dtype=np.float32)
        # Counter to track when we should push a new chunk based on hop size
        self._samples_since_last_push = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if device is None else torch.device(device)

        # Load model
        self.model = HybridEmotionModel()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.q = queue.Queue()
        self.running = False
        # Keyboard listener control
        self._kb_thread = None
        # Stop key default: Ctrl+G (ASCII BEL, '\x07')
        self.stop_key = '\x07'

        print(f"Using device: {self.device}")
        print(f"Model loaded from: {model_path}")
        print(f"Sampling rate: {self.sr}, Duration: {self.duration}s, Hop: {self.hop}s")

    def audio_callback(self, indata, frames, time_info, status):
        """Callback called by sounddevice for each audio block."""
        if status:
            print("Sounddevice status:", status)
        # indata shape: (frames, channels)
        samples = indata[:, 0].astype(np.float32)
        # Append to buffer (sliding window)
        self.buffer = np.concatenate((self.buffer, samples))
        # If buffer too long, keep only last full_length
        if len(self.buffer) > self.full_length:
            self.buffer = self.buffer[-self.full_length:]
        # Update sample counter and push based on hop_length
        self._samples_since_last_push += frames
        while self._samples_since_last_push >= self.hop_length:
            if len(self.buffer) >= self.full_length:
                self.q.put(self.buffer.copy())
            self._samples_since_last_push -= self.hop_length

    def process_worker(self):
        """Process queued buffers and run model inference."""
        while self.running:
            try:
                audio_chunk = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # Extract features (mel spectrogram and prosodic) directly from the audio buffer
                mel_spec = extract_mel_spectrogram(audio_chunk, self.sr)
                prosodic = extract_prosodic_features(audio_chunk, self.sr)

                # Convert to tensors and shapes expected by model
                mel_t = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,n_mels,time)
                pro_t = torch.FloatTensor(prosodic).unsqueeze(0).to(self.device)  # (1,13)

                with torch.no_grad():
                    outputs = self.model(mel_t, pro_t)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

                # Build dict of emotion->prob
                probs_dict = {EMOTIONS[i]: float(probs[i]) for i in range(NUM_CLASSES)}
                pred_idx = int(np.argmax(probs))
                pred_emotion = EMOTIONS[pred_idx]

                # Print
                print_prediction(pred_emotion, probs_dict)
            except Exception as e:
                print("Error during processing:", e)

    def start(self, device_index=None):
        """Start microphone stream and processing thread."""
        # Reset buffer
        self.buffer = np.zeros(self.full_length, dtype=np.float32)
        self.running = True
        thread = threading.Thread(target=self.process_worker, daemon=True)
        thread.start()

        # Start keyboard listener (to detect Ctrl+G)
        self._kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kb_thread.start()

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sr,
                blocksize=self.hop_length,
                callback=self.audio_callback,
                device=device_index
            ):
                print("🔴 Listening... Press Ctrl+G (Ctrl+G) to stop or Ctrl+C to interrupt.")
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            thread.join(timeout=1.0)
            # Join keyboard thread if running
            if self._kb_thread is not None and self._kb_thread.is_alive():
                # Trigger so it exits
                try:
                    # On POSIX, we can write to stdin if not interactive; otherwise just wait shortly
                    pass
                except Exception:
                    pass
                self._kb_thread.join(timeout=1.0)

    def _keyboard_listener(self):
        """Listens for the configured stop_key (Ctrl+G default) and stops the detector.

        Uses `msvcrt` on Windows and `termios`/`tty` on POSIX systems to read single keystrokes
        without waiting for newline.
        """
        try:
            # Windows: use msvcrt
            import msvcrt
            while self.running:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    # msvcrt returns bytes
                    if ch == self.stop_key.encode():
                        print("\n🔵 Ctrl+G detected — stopping listening.")
                        self.running = False
                        break
                time.sleep(0.05)
        except Exception:
            # POSIX / fallback
            try:
                import sys, select, tty, termios
                fd = sys.stdin.fileno()
                if not sys.stdin.isatty():
                    # Not a TTY; nothing to listen
                    return
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while self.running:
                        r, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch == self.stop_key:
                                print("\n🔵 Ctrl+G detected — stopping listening.")
                                self.running = False
                                break
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                # As a last resort, just wait until self.running becomes False
                while self.running:
                    time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description="Realtime emotion detection from microphone")
    parser.add_argument("model_path", nargs="?", default="models/best_model.pth", type=str,
                        help="Path to the model checkpoint (e.g., models/best_model.pth). Default: models/best_model.pth")
    parser.add_argument("--hop", type=float, default=0.9, help="Hop in seconds between predictions (default 0.9)")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device (cuda/mps/cpu)")
    parser.add_argument("--device-index", type=int, default=None, help="Sound device index or None (auto)")
    parser.add_argument("--list-devices", action="store_true", help="List available sounddevice devices and exit")
    parser.add_argument("--stop-key", type=str, default="\x07",
                        help="Control character to stop listening. Default: Ctrl+G (\x07). Use quoted char like '\\x07' or 'q' to stop on q")
    args = parser.parse_args()

    # If user asked to list devices, show them and exit
    if args.list_devices:
        try:
            devices = sd.query_devices()
            print("Available sound input/output devices (index: name):")
            for i, d in enumerate(devices):
                print(f"{i:2d}: {d['name']}")
        except Exception as e:
            print("Could not list devices:", e)
        return

    # If model_path specified doesn't exist, and user passed a basename, try models/<basename>
    if not os.path.exists(args.model_path):
        # try in models/ directory next to the script
        alt = os.path.join(os.path.dirname(__file__), "models", os.path.basename(args.model_path))
        if os.path.exists(alt):
            args.model_path = alt
        else:
            # also try relative to repo root (working dir)
            if os.path.exists(os.path.join(os.getcwd(), args.model_path)):
                args.model_path = os.path.join(os.getcwd(), args.model_path)
            else:
                print(f"Model checkpoint not found at: {args.model_path}")
                print("Pass the correct path to your checkpoint, e.g.:\n  python liveinference.py models/best_model.pth")
                return

    detector = RealtimeEmotionDetector(args.model_path, device=args.device, duration=DURATION, hop=args.hop)
    # If user provided a different stop-key, set it here
    if args.stop_key:
        detector.stop_key = args.stop_key
    detector.start(device_index=args.device_index)


if __name__ == "__main__":
    main()