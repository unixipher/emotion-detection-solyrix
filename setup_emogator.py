"""
Download and organize EmoGator dataset for emotion recognition
"""
import os
import shutil
import subprocess
from pathlib import Path

# Emotion mapping: 30 EmoGator emotions -> 6 basic emotions
EMOTION_MAPPING = {
    # Happiness cluster
    'Adoration': 'happiness',
    'Amusement': 'happiness',
    'Contentment': 'happiness',
    'Ecstasy': 'happiness',
    'Elation': 'happiness',
    'Relief': 'happiness',
    'Serenity': 'happiness',
    'Triumph': 'happiness',
    'Pride': 'happiness',
    'Surprise (Positive)': 'happiness',
    
    # Sadness cluster
    'Sadness': 'sadness',
    'Disappointment': 'sadness',
    'Distress': 'sadness',
    'Sympathy': 'sadness',
    'Guilt': 'sadness',
    'Shame': 'sadness',
    
    # Fear cluster
    'Fear': 'fear',
    'Confusion': 'fear',
    'Embarrassment': 'fear',
    
    # Anger cluster
    'Anger': 'anger',
    'Contempt': 'anger',
    
    # Surprise cluster
    'Surprise (Negative)': 'surprise',
    'Awe': 'surprise',
    'Realization': 'surprise',
    
    # Disgust cluster
    'Disgust': 'disgust',
    'Pain': 'disgust',
    
    # Neutral/Ambiguous (can be distributed or excluded)
    'Neutral': 'surprise',
    'Interest': 'surprise',
    'Desire': 'happiness',
    'Romantic Love': 'happiness',
}

# EmoGator emotion categories (01-30)
EMOGATOR_EMOTIONS = [
    'Adoration', 'Amusement', 'Anger', 'Awe', 'Confusion', 
    'Contempt', 'Contentment', 'Desire', 'Disappointment', 'Disgust',
    'Distress', 'Ecstasy', 'Elation', 'Embarrassment', 'Fear',
    'Guilt', 'Interest', 'Neutral', 'Pain', 'Pride',
    'Realization', 'Relief', 'Romantic Love', 'Sadness', 'Serenity',
    'Shame', 'Surprise (Negative)', 'Surprise (Positive)', 'Sympathy', 'Triumph'
]


def find_emogator_directory():
    """Try to find EmoGator directory in common locations"""
    possible_paths = [
        'EmoGator/data/mp3',  # Current directory
        '../EmoGator/data/mp3',  # Parent directory
        os.path.expanduser('~/EmoGator/data/mp3'),  # Home directory
        '/Users/mrinmoyhalder/EmoGator/data/mp3',  # Explicit path from user
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def clone_emogator():
    """Clone EmoGator repository"""
    print("📥 Downloading EmoGator dataset...")
    print("⚠️  This is ~4-5 GB and may take several minutes!")
    
    if os.path.exists('EmoGator'):
        print("✓ EmoGator directory already exists. Skipping clone.")
        return
    
    try:
        subprocess.run([
            'git', 'clone', 
            'https://github.com/fredbuhl/EmoGator.git'
        ], check=True)
        print("✓ Download complete!")
    except subprocess.CalledProcessError:
        print("❌ Error: Git clone failed. Do you have git installed?")
        print("   Install git: brew install git (on Mac)")
        print("   Or download manually from: https://github.com/fredbuhl/EmoGator")
        exit(1)


def organize_dataset(source_dir=None, target_dir='data/raw'):
    """
    Organize EmoGator files into emotion folders
    
    File format: NNNNNN-EE-I.mp3
    - NNNNNN: contributor ID (000001-000357)
    - EE: emotion category (01-30)
    - I: instance (1, 2, or 3)
    """
    print("\n📁 Organizing files into emotion folders...")
    
    # Try to find source directory if not provided
    if source_dir is None:
        source_dir = find_emogator_directory()
    
    if source_dir is None or not os.path.exists(source_dir):
        print(f"❌ Error: Could not find EmoGator/data/mp3 directory!")
        print("   Searched in:")
        print("     - EmoGator/data/mp3")
        print("     - ../EmoGator/data/mp3")
        print("     - ~/EmoGator/data/mp3")
        print("\n   Please run this script from the directory containing your project files,")
        print("   or make sure EmoGator is downloaded to one of the above locations.")
        return False
    
    print(f"✓ Using source directory: {source_dir}")
    
    # Create target directories
    for emotion in ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust']:
        emotion_dir = os.path.join(target_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        print(f"  Created: {emotion_dir}")
    
    # Statistics
    stats = {emotion: 0 for emotion in ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust']}
    total_files = 0
    skipped = 0
    
    print("\n📦 Processing files...")
    
    # Get all MP3 files
    all_files = [f for f in os.listdir(source_dir) if f.endswith('.mp3')]
    print(f"  Found {len(all_files)} MP3 files to process")
    
    # Process each MP3 file
    for filename in all_files:
        total_files += 1
        
        # Parse filename: NNNNNN-EE-I.mp3
        parts = filename.replace('.mp3', '').split('-')
        if len(parts) != 3:
            print(f"⚠️  Skipping malformed filename: {filename}")
            skipped += 1
            continue
        
        contributor_id, emotion_code, instance = parts
        
        # Convert emotion code to 0-indexed
        emotion_idx = int(emotion_code) - 1
        
        if emotion_idx < 0 or emotion_idx >= len(EMOGATOR_EMOTIONS):
            print(f"⚠️  Invalid emotion code in {filename}")
            skipped += 1
            continue
        
        # Get emotion name and mapped category
        original_emotion = EMOGATOR_EMOTIONS[emotion_idx]
        target_emotion = EMOTION_MAPPING.get(original_emotion)
        
        if not target_emotion:
            print(f"⚠️  No mapping for {original_emotion}, skipping {filename}")
            skipped += 1
            continue
        
        # Copy file to target directory
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, target_emotion, filename)
        
        shutil.copy2(source_path, target_path)
        stats[target_emotion] += 1
        
        if total_files % 5000 == 0:
            print(f"  Processed {total_files}/{len(all_files)} files...")
    
    # Print statistics
    print("\n✓ Organization complete!")
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total files processed: {total_files}")
    print(f"  Files organized: {sum(stats.values())}")
    print(f"  Files skipped: {skipped}")
    print(f"\n  Distribution by emotion:")
    for emotion in ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust']:
        count = stats[emotion]
        print(f"    {emotion:12s}: {count:5d} samples")
    
    return True


def verify_dataset(data_dir='data/raw'):
    """Verify the organized dataset"""
    print(f"\n🔍 Verifying dataset in {data_dir}...")
    
    emotions = ['happiness', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
    total = 0
    
    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        if os.path.exists(emotion_dir):
            count = len([f for f in os.listdir(emotion_dir) if f.endswith('.mp3')])
            total += count
            print(f"  {emotion:12s}: {count:5d} files")
        else:
            print(f"  {emotion:12s}: ❌ Directory not found")
    
    print(f"\n  Total: {total} files")
    
    if total > 0:
        print("\n✅ Dataset ready for training!")
        print("\nNext steps:")
        print("  1. Run: python train.py")
        print("  2. Wait for training to complete")
        print("  3. Use: python inference.py your_audio.mp3")
    else:
        print("\n❌ No files found. Please check the setup.")


def main():
    import sys
    
    print("=" * 60)
    print("EmoGator Dataset Setup")
    print("=" * 60)
    
    # Check if source path is provided as argument
    source_dir = None
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        print(f"Using provided source: {source_dir}")
    
    # Step 1: Skip cloning if directory exists
    emogator_path = find_emogator_directory()
    if emogator_path:
        print(f"✓ Found EmoGator at: {emogator_path}")
    else:
        print("⚠️  EmoGator not found in common locations.")
        clone_emogator()
    
    # Step 2: Organize into emotion folders
    success = organize_dataset(source_dir)
    
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        return
    
    # Step 3: Verify
    verify_dataset()
    
    print("\n" + "=" * 60)
    print("Setup Complete! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()