# CSE 253 Assignment 2 — Music Generation with ML

## Overview

We chose **Task 2 (Symbolic Conditioned Generation)** and **Task 4 (Continuous Conditioned Generation)**.

The two tasks form an end-to-end pipeline:
```
Chord Progression → [Task 2: Transformer] → MIDI melody → [Task 4: DDSP] → Audio waveform
```

## Current Progress

### Task 2: Chord → Melody (Symbolic Conditioned Generation) ✅ Baseline Done

- **Model**: GPT-style Transformer Decoder (Pre-LN, 4 layers, 256-dim, 4 heads, ~3.3M params)
- **Dataset**: POP909 (909 Chinese pop songs, melody + chord annotations)
- **Representation**: REMI-like tokens — chord/bar/pitch/duration/timeshift interleaved in one sequence
- **Training result**: Train Loss 1.24 / Val Loss 1.26, PPL ~3.5 (50 epochs on RTX 4090)
- **Generation**: `generate_melody_forced_chords()` injects chord tokens at each bar, model generates melody autoregressively (top-k + temperature sampling)

**Files:**
- `task2_chord2melody_baseline.py` — complete pipeline (data processing, model, training, generation, MIDI export)
- `checkpoints/` — saved model weights + generated MIDI samples

### Task 4: MIDI → Audio (Continuous Conditioned Generation) 🔧 Baseline Code Ready

- **Model**: Simplified MIDI-DDSP (MidiEncoder → DDSPDecoder → Differentiable Synthesizer)
- **Synthesizer**: Additive harmonic synth (60 harmonics) + filtered noise, trained with multi-scale spectral loss
- **Dataset**: URMP (University of Rochester Multi-modal Music Performance) — download in progress
- **Status**: Code written and verified, pending data download and training

**Files:**
- `task4_midi2audio_ddsp_baseline.py` — complete pipeline (DDSP synth, MIDI encoder, training, WAV export)
- Includes a `SyntheticDataset` for testing without URMP

## Repository Structure

```
assignment2/
├── README.md
├── task2_chord2melody_baseline.py    # Task 2 main code
├── task4_midi2audio_ddsp_baseline.py # Task 4 main code
├── checkpoints/                      # Task 2 trained model + generated MIDI
├── 153 _ 253 2026 Assignment 2.pdf   # Assignment spec
├── module3.pdf                       # Course slides: Symbolic Music Generation
├── module4.pdf                       # Course slides: Audio-Domain Music Generation
├── workbook3/                        # Course workbook 3 (REMI + Markov + RNN)
└── workbook4.ipynb                   # Course workbook 4
```

## How to Run

### Environment Setup

```bash
# Create conda environment
conda create -n PA3 python=3.10
conda activate PA3

# Install dependencies
pip install torch symusic midiutil numpy scipy librosa soundfile
```

### Task 2: Train & Generate

```bash
# 1. Clone POP909 dataset
git clone https://github.com/music-x-lab/POP909-Dataset.git

# 2. Train (takes ~5 min on RTX 4090)
python task2_chord2melody_baseline.py \
  --data_dir ./POP909-Dataset/POP909 \
  --epochs 50 --device cuda

# 3. Generated MIDI files will be in ./checkpoints/
```

### Task 4: Train & Synthesize

```bash
# 1. Download URMP dataset
pip install gsutil
gsutil -m cp -r gs://magentadata/datasets/urmp/urmp_20210324/ ./URMP/

# 2. (Optional) Test pipeline with synthetic data first
python task4_midi2audio_ddsp_baseline.py --epochs 30 --device cuda

# 3. Train on URMP
python task4_midi2audio_ddsp_baseline.py \
  --data_dir ./URMP --epochs 100 --device cuda

# 4. Synthesize Task 2's output
python task4_midi2audio_ddsp_baseline.py \
  --data_dir ./URMP \
  --test_midi ./checkpoints/generated_melody.mid \
  --device cuda
```

## TODO

- [ ] Improve Task 2: try larger model (d_model=512, 6 layers), add velocity tokens, relative position encoding
- [ ] Train Task 4 on URMP (or MAESTRO as fallback)
- [ ] Evaluation metrics: perplexity, pitch/rhythm statistics, chord-melody consistency
- [ ] End-to-end demo: chord → MIDI → audio
- [ ] Record 20-min presentation video
- [ ] Export notebook as HTML for submission

## Submission Checklist

- [ ] `workbook.html` — Jupyter notebook exported as HTML
- [ ] `video_url.txt` — Google Drive / YouTube link to ~20 min presentation
- [ ] `symbolic_conditioned.mid` — Task 2 generated MIDI
- [ ] `continuous_conditioned.mp3` — Task 4 generated audio

## References

- POP909 Dataset: [music-x-lab/POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset)
- DDSP (Engel et al., ICLR 2020): [arxiv.org/abs/2001.04643](https://arxiv.org/abs/2001.04643)
- MIDI-DDSP (Wu et al., 2022): [arxiv.org/abs/2112.09312](https://arxiv.org/abs/2112.09312)
- REMI (Huang & Yang, ACM MM 2020): [YatingMusic/remi](https://github.com/YatingMusic/remi)
- MusicGen / AudioCraft: [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
