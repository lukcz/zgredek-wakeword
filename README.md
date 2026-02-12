# 🧦 Wake Word Trainer

All-in-one training script for custom [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) models compatible with Home Assistant Voice PE and ESPHome.

**Just run it and wait** — everything is automatic, including downloading all required datasets.

## Features

- 🎤 **Automatic sample generation** using Edge TTS (Microsoft's neural voices)
- 📥 **Auto-downloads all training data** (~5GB, cached for future runs)
- 🌍 **Multi-language support**: Polish, English, German, Spanish, French
- 🧠 **Smart augmentation**: Room acoustics, background noise, music
- 📦 **Ready-to-use output**: `.tflite` model + `.json` manifest for ESPHome
- 🔧 **Auto-patches** TensorFlow compatibility issues

## Requirements

### System
- **WSL2 Ubuntu** (recommended) or native Linux
- Python 3.10+
- ~10GB free disk space
- FFmpeg

### Installation

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install ffmpeg -y

# 2. Create Python environment
python3 -m venv ~/wakeword-env
source ~/wakeword-env/bin/activate

# 3. Install Python packages
pip install edge-tts datasets==2.14.0 soundfile librosa pyyaml requests tqdm
pip install 'numpy<2' 'pyarrow>=12,<15' tensorflow==2.16.1

# 4. Clone and install micro-wake-word
git clone https://github.com/OHF-Voice/micro-wake-word.git
cd micro-wake-word && pip install -e .
cd ..

# 5. Download the trainer script
wget https://raw.githubusercontent.com/lukcz/zgredek-wakeword/main/train_wakeword.py
```

## Quick Start

### Basic Usage

```bash
# Polish wake word
python train_wakeword.py "Hej Zgredek"

# English wake word  
python train_wakeword.py "Hey Jarvis" --lang en

# With custom variations
python train_wakeword.py "Hey Computer" --lang en --variations "Hey PC,Yo Computer"
```

### Output

After training completes, you'll find:
```
./wakeword_training/
├── hej_zgredek.tflite    # Model file (~100KB)
├── hej_zgredek.json      # ESPHome manifest
├── generated_samples/     # TTS audio files
├── augmented_features/    # Processed spectrograms
└── trained_model/         # Training checkpoints
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wake_word` | (required) | The wake word phrase to train |
| `--variations` | "" | Comma-separated alternative phrasings |
| `--lang` | "pl" | Language: pl, en, de, es, fr |
| `--output_dir` | "./wakeword_training" | Output directory |
| `--steps` | 10000 | Training steps |
| `--batch_size` | 128 | Batch size for training |
| `--author` | "custom" | Author name for manifest |
| `--probability_cutoff` | 0.7 | Detection threshold (0.0-1.0) |
| `--data_dir` | None | Path to existing augmentation data |
| `--skip_generate` | False | Skip TTS sample generation |
| `--skip_download` | False | Skip dataset downloads |

## Examples

### Polish Wake Word (Recommended Settings)
```bash
python train_wakeword.py "Hej Zgredek" \
    --variations "Hej Zgredku,hej zgredek,Heej Zgredek" \
    --steps 15000 \
    --author "myname"
```

### English Wake Word with High Accuracy
```bash
python train_wakeword.py "Hey Assistant" \
    --lang en \
    --steps 20000 \
    --probability_cutoff 0.8
```

### Quick Test Run (Lower Quality)
```bash
python train_wakeword.py "Test Word" \
    --lang en \
    --steps 5000
```

### Reuse Downloaded Data
```bash
# First run downloads everything
python train_wakeword.py "Wake Word One" --output_dir ./training1

# Second run reuses the data (much faster!)
python train_wakeword.py "Wake Word Two" --data_dir ./training1 --output_dir ./training2
```

## Training Time

| Phase | First Run | Subsequent Runs |
|-------|-----------|-----------------|
| Sample generation | 5-10 min | 5-10 min |
| Dataset download | 30-60 min | Skipped (cached) |
| Feature generation | 10-20 min | 10-20 min |
| Model training | 10-30 min | 10-30 min |
| **Total** | **1-2 hours** | **25-60 min** |

*Times based on modern CPU. Training uses CPU by default (GPU often has compatibility issues with new hardware).*

## Parameter Trade-offs

### Training Steps (`--steps`)
| Value | Quality | Time | Use Case |
|-------|---------|------|----------|
| 5000 | Low | ~5 min | Quick testing |
| 10000 | Good | ~15 min | Default, most use cases |
| 15000 | Better | ~25 min | Production models |
| 20000+ | Best | ~35+ min | Maximum accuracy |

### Probability Cutoff (`--probability_cutoff`)
| Value | False Positives | False Negatives | Use Case |
|-------|-----------------|-----------------|----------|
| 0.5 | Many | Few | Very responsive, noisy |
| 0.7 | Balanced | Balanced | **Default, recommended** |
| 0.8 | Few | Some | Quiet environments |
| 0.9 | Rare | Many | High-precision needed |

### Language-Specific Tips

| Language | Recommended Variations | Notes |
|----------|----------------------|-------|
| Polish | Add vocative case (e.g., "Zgredku") | Edge TTS handles Polish well |
| English | Add informal variants | Multiple accents available |
| German | Include formal/informal | Good voice quality |

## Deploying to ESPHome

### 1. Upload to GitHub Releases

Create a new release on your GitHub repo and upload both files:
- `your_wake_word.tflite`
- `your_wake_word.json`

### 2. Update ESPHome Config

```yaml
micro_wake_word:
  models:
    - model: https://github.com/YOUR_USER/your-repo/releases/download/v1.0.0/your_wake_word.json
      probability_cutoff: 0.7
      sliding_window_size: 5
    - model: okay_nabu  # Backup wake word (optional)
```

### 3. Flash and Test

```bash
esphome run your-voice-pe.yaml
```

## Troubleshooting

### "Module not found" errors
```bash
pip install edge-tts datasets==2.14.0 soundfile librosa pyyaml requests tqdm
pip install 'numpy<2' 'pyarrow>=12,<15' tensorflow==2.16.1
```

### Training crashes with GPU errors
The script uses CPU by default. If you want to try GPU:
```bash
python train_wakeword.py "Wake Word" --cpu_only=False
```
Note: RTX 40xx/50xx series may have compatibility issues with TensorFlow.

### "numpy.ndarray has no attribute numpy"
The script auto-patches this. If it persists:
```bash
# Manual fix
sed -i 's/\.numpy()//' /path/to/micro-wake-word/microwakeword/train.py
```

### Downloads fail or timeout
```bash
# Use existing data from another training
python train_wakeword.py "New Word" --data_dir /path/to/previous/training
```

### Model not responding to wake word
1. Lower `probability_cutoff` to 0.5-0.6
2. Add more phrase variations
3. Increase training steps to 15000+
4. Check if TTS pronunciation matches your speech

### Too many false activations
1. Raise `probability_cutoff` to 0.8-0.9
2. Increase `negative_class_weight` in code
3. Use a more distinctive wake word (2+ syllables)

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SAMPLE GENERATION                                        │
│    Edge TTS → Multiple voices, speeds, pitches              │
│    "Hej Zgredek" → 2000+ audio variations                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. AUGMENTATION                                             │
│    + Room acoustics (MIT RIRs)                              │
│    + Background music (FMA)                                 │
│    + Environmental sounds (AudioSet)                        │
│    + Noise, EQ, pitch shifts                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TRAINING                                                 │
│    MixedNet CNN architecture                                │
│    Positive samples + Negative samples                      │
│    10000 steps → 99%+ accuracy                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EXPORT                                                   │
│    Quantized TFLite model (~100KB)                          │
│    JSON manifest for ESPHome                                │
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture

The script uses the **MixedNet** architecture optimized for wake word detection:

| Layer | Filters | Description |
|-------|---------|-------------|
| Conv1D | 32 | Initial feature extraction |
| MixConv Block 1 | 64 | Kernel sizes: [5] |
| MixConv Block 2 | 64 | Kernel sizes: [7, 11] |
| MixConv Block 3 | 64 | Kernel sizes: [9, 15] |
| MixConv Block 4 | 64 | Kernel sizes: [23] |
| Output | 1 | Wake word probability |

**Model size**: ~100KB (quantized)  
**Parameters**: ~26,000  
**Inference**: ~10ms per frame on ESP32-S3

## Datasets Used

| Dataset | Size | Purpose |
|---------|------|---------|
| Generated samples | ~500MB | Positive wake word examples |
| MIT RIRs | ~500MB | Room acoustics simulation |
| FMA (music) | ~2GB | Background music augmentation |
| AudioSet | ~2GB | Environmental sounds |
| Negative datasets | ~1GB | Speech/noise that's NOT the wake word |

All datasets are automatically downloaded on first run and cached for future use.

## Credits

- [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) by Kevin Ahrendt
- [Edge TTS](https://github.com/rany2/edge-tts) for sample generation
- [ESPHome](https://esphome.io/) for device integration
- Training data from MIT, FMA, AudioSet, and HuggingFace

## License

MIT License - Use freely for personal and commercial projects.

---

Made with 🧦 by [Zgredek](https://github.com/lukcz)
