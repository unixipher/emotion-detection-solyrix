"""
Find and remove corrupted audio files
"""
import os
import librosa
from tqdm import tqdm
from config import DATA_DIR, EMOTIONS, SAMPLE_RATE, DURATION

def check_file(file_path):
    """Check if audio file can be loaded"""
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) == 0:
            return False
        return True
    except Exception as e:
        return False

def clean_dataset(data_dir):
    """Find and remove/list corrupted files"""
    corrupted = []
    
    print("Scanning dataset for corrupted files...")
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
        
        files = [f for f in os.listdir(emotion_dir) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        print(f"\nChecking {emotion} ({len(files)} files)...")
        for file in tqdm(files, desc=emotion):
            file_path = os.path.join(emotion_dir, file)
            if not check_file(file_path):
                corrupted.append(file_path)
                print(f"  ❌ Corrupted: {file_path}")
    
    print(f"\n{'='*60}")
    print(f"Found {len(corrupted)} corrupted files")
    print(f"{'='*60}")
    
    if corrupted:
        print("\nCorrupted files:")
        for f in corrupted:
            print(f"  {f}")
        
        response = input("\nRemove these files? (yes/no): ")
        if response.lower() == 'yes':
            for f in corrupted:
                os.remove(f)
                print(f"  Removed: {f}")
            print(f"\n✓ Removed {len(corrupted)} corrupted files")
        else:
            print("\n⚠️  Files not removed. They will be skipped during training.")
    else:
        print("\n✓ No corrupted files found!")

if __name__ == "__main__":
    clean_dataset(DATA_DIR)