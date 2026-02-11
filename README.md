# Zgredek Wake Word

Custom Polish wake word "Zgredek" for Home Assistant Voice PE.

## Usage in ESPHome

```yaml
micro_wake_word:
  models:
    - model: https://github.com/lukcz/zgredek-wakeword/releases/download/v1.0.0/zgredek.json
      probability_cutoff: 0.7
      sliding_window_size: 5
```

## Training Details

- Trained with microWakeWord
- 2000 Polish TTS samples (Edge TTS: pl-PL-MarekNeural, pl-PL-ZofiaNeural)
- Model: inception architecture
- Training steps: 15000
- Recall: ~90%+ at cutoff 0.5

## Author

Łukasz Czarnecki
