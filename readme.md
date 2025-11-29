# Speech Emotion Recognition (Hybrid CNN)

A lightweight hybrid CNN model that detects emotions from speech audio, including differentiating between crying in different emotional contexts (joy, sadness, anger, disgust).

## 🎯 Features

- **Hybrid Architecture**: Combines mel-spectrogram CNN with prosodic features
- **Small Model**: ~1-2M parameters, perfect for deployment
- **Emotion Classes**: Happiness, Sadness, Fear, Anger, Surprise, Disgust
- **Optimized for M2 Mac**: Uses MPS acceleration

## 📁 Project Structure

```
emotion-detection/
├── config.py              # All configuration parameters
├── preprocessing.py       # Audio feature extraction
├── model.py              # Hybrid CNN architecture
├── dataset.py            # Dataset loader
├── train.py              # Training script
├── inference.py          # Prediction script
├── requirements.txt      # Dependencies
│
├── data/
│   └── raw/              # Place your audio files here
│       ├── happiness/
│       ├── sadness/
│       ├── fear/
│       ├── anger/
│       ├── surprise/
│       └── disgust/
│
├── models/               # Saved models (created automatically)
└── results/              # Training plots and reports (created automatically)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your audio files in the following structure:

```
data/raw/
├── happiness/
│   ├── audio1.wav
│   ├── audio2.wav
├── sadness/
│   ├── audio1.wav
├── fear/
├── anger/
├── surprise/
└── disgust/
```

**Supported formats**: `.wav`, `.mp3`, `.flac`, `.ogg`

### 3. Train the Model

```bash
python train.py
```

This will:
- Load and preprocess audio files
- Train the hybrid CNN model
- Save the best model to `models/best_model.pth`
- Generate training plots in `results/`
- Show classification metrics

### 4. Make Predictions

```bash
python inference.py path/to/audio.wav
```

Example output:
```
🎭 Predicted Emotion: SADNESS

📊 Confidence scores:
  sadness     : 87.34% ████████████████████████████████████████████
  fear        : 8.21%  ████
  anger       : 2.45%  █
  happiness   : 1.23%  
  disgust     : 0.54%  
  surprise    : 0.23%  
```

## 🎓 Datasets to Use

For training, download these free datasets:

1. **RAVDESS** (Recommended): https://zenodo.org/record/1188976
   - 1,440 audio files
   - 8 emotions, 24 actors
   - High quality

2. **CREMA-D**: https://github.com/CheyneyComputerScience/CREMA-D
   - 7,442 audio files
   - 6 emotions, 91 actors

3. **TESS**: https://tspace.library.utoronto.ca/handle/1807/24487
   - 2,800 audio files
   - 7 emotions, 2 actresses

4. **EMO-DB**: http://emodb.bilderbar.info/
   - 535 audio files
   - German language

### Organizing Downloaded Datasets

After downloading, you'll need to organize files into emotion folders. Here's a helper script:

```python
# organize_dataset.py
import os
import shutil

def organize_ravdess(source_dir, target_dir='data/raw'):
    """
    RAVDESS filename format: 03-01-06-01-02-01-12.wav
    Third number is emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 
                            05=angry, 06=fearful, 07=disgust, 08=surprised
    """
    emotion_map = {
        '03': 'happiness',
        '04': 'sadness', 
        '05': 'anger',
        '06': 'fear',
        '07': 'disgust',
        '08': 'surprise'
    }
    
    for filename in os.listdir(source_dir):
        if filename.endswith('.wav'):
            emotion_code = filename.split('-')[2]
            if emotion_code in emotion_map:
                emotion = emotion_map[emotion_code]
                dest_dir = os.path.join(target_dir, emotion)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(
                    os.path.join(source_dir, filename),
                    os.path.join(dest_dir, filename)
                )
    print("✓ Dataset organized!")

# Usage:
# organize_ravdess('path/to/downloaded/ravdess')
```

## ⚙️ Configuration

Edit `config.py` to adjust:

- **Audio parameters**: Sample rate, duration, MFCC/mel settings
- **Training parameters**: Batch size, epochs, learning rate
- **Model parameters**: Can be modified in `model.py`

## 📊 Model Architecture

```
Input: Audio File (3 seconds)
    ↓
Feature Extraction
    ├─→ Mel-Spectrogram (128x130) ──→ CNN (4 conv layers) ──→ 256 features
    └─→ Prosodic Features (13)    ──→ Dense layers       ──→ 128 features
                                                              ↓
                                                        Concatenate (384)
                                                              ↓
                                                        Fusion Layers
                                                              ↓
                                                        Output (6 classes)
```

**Key Components**:
- **CNN Branch**: Processes mel-spectrograms to learn spectral patterns
- **Prosodic Branch**: Processes pitch, energy, voice quality features
- **Fusion**: Combines both to make final prediction

**Why this works for crying differentiation**:
- Joy crying: Higher pitch variability, rising intonation
- Sadness crying: Lower pitch, descending contours, breathy voice
- Anger crying: High intensity, harsh voice quality
- Disgust crying: Nasal quality, specific formants

## 🔍 Testing the Model

```python
from inference import EmotionPredictor

# Load model
predictor = EmotionPredictor('models/best_model.pth')

# Single prediction
emotion, probs = predictor.predict('test_audio.wav')
print(f"Emotion: {emotion}")

# Batch prediction
results = predictor.predict_batch(['audio1.wav', 'audio2.wav'])
```

## 📈 Expected Performance

With proper training data:
- **Overall Accuracy**: 70-85%
- **Crying differentiation**: 65-75% (depends on dataset quality)
- **Inference time**: <100ms on M2 Mac
- **Model size**: ~8MB

## 🐛 Troubleshooting

**Issue**: "No audio files found"
- Check that audio files are in correct folder structure
- Verify file extensions (.wav, .mp3, etc.)

**Issue**: "MPS not available"
- Update to latest PyTorch: `pip install --upgrade torch`
- Falls back to CPU automatically

**Issue**: Low accuracy
- Need more training data (minimum 100-200 samples per emotion)
- Try training for more epochs
- Check if emotions are balanced in dataset

## 📝 Notes

- Model expects 3-second audio clips (configurable)
- Automatically pads/truncates audio to fixed length
- Uses data augmentation (pitch shift, time stretch, noise)
- Early stopping prevents overfitting

## 🎯 Next Steps

1. Collect more "crying" samples across different emotions
2. Fine-tune on your specific use case
3. Experiment with model architecture in `model.py`
4. Add real-time inference from microphone
5. Export to ONNX for deployment

## 📧 Need Help?

Check the code comments or modify parameters in `config.py` to suit your needs!