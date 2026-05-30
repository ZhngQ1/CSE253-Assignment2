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

### Task 4: MIDI → Audio (Continuous Conditioned Generation) ✅ Trained on Real Violin

- **Model**: Simplified MIDI-DDSP (MidiEncoder → DDSPDecoder → Differentiable Synthesizer, ~0.74M params)
- **Synthesizer**: Additive harmonic synth (60 harmonics) + 65-band filtered noise, trained with multi-scale spectral loss
- **Dataset**: URMP **violin solo**, using the Magenta DDSP-preprocessed release (16 kHz audio + 250 fps f0 + note onsets/velocities). A pure-Python TFRecord parser reads it — **no TensorFlow / no raw-URMP download needed**. Light subset: 7 recordings (~293 MB).
- **Training**: 80 epochs on CPU (~25 min). Multi-scale spectral loss 5.73 → **best val 1.3644**.
- **Verification**: on a sustained B4 (493.9 Hz) the output spectrum peaks at 494.0 Hz with clean harmonics at 988/1482/1976 Hz — the learned synth reproduces the conditioned pitch with a violin-like timbre; rests are correctly silenced.
- **End-to-end**: Task 2's `generated_melody.mid` → `continuous_conditioned.wav/.mp3`.

**Files:**
- `task4_midi2audio_ddsp_baseline.py` — DDSP synth, MIDI encoder, training, WAV export, `SyntheticDataset`
- `urmp_tfrecord.py` — pure-Python URMP DDSP-conditioning TFRecord parser + `URMPTFRecordDataset`
- `task4_train_urmp.py` — train on real violin, then auto-synthesize the Task 2 melody
- `task4_result_summary.txt` — metrics and verification details
- `checkpoints_task4/` — trained weights (`best_model_urmp_vn.pt`), `continuous_conditioned.{wav,mp3}`, per-style syntheses, and per-epoch reconstructions

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

### Task 4: Train & Synthesize (real URMP violin, no TensorFlow needed)

```bash
# 1. Download a light violin subset (~293 MB) of the Magenta DDSP-preprocessed URMP.
#    These TFRecords already contain 16 kHz audio + f0 + note onsets/velocities.
mkdir -p URMP_tf
B=gs://magentadata/datasets/urmp/urmp_20210324
for s in 00004 00012 00017 00027 00036 00041 00056; do
  gsutil cp "$B/urmp_vn_solo_ddsp_conditioning_train_unbatched.tfrecord-${s}-of-00064" URMP_tf/
done
for s in 00001 00005; do
  gsutil cp "$B/urmp_vn_solo_ddsp_conditioning_test_unbatched.tfrecord-${s}-of-00008" URMP_tf/
done

# 2. Train on real violin, then auto-synthesize Task 2's melody (end-to-end).
#    CPU is faster here than MPS because the multi-scale STFT loss falls back to CPU.
PYTORCH_ENABLE_MPS_FALLBACK=1 python task4_train_urmp.py --device cpu --epochs 80

# Output: checkpoints_task4/best_model_urmp_vn.pt
#         checkpoints_task4/continuous_conditioned.wav   (Task 2 melody -> violin audio)

# (Optional) sanity-check the pipeline on synthetic data with the original baseline:
python task4_midi2audio_ddsp_baseline.py --epochs 30 --device cpu

# (Optional) synthesize any MIDI with a trained checkpoint:
python -c "import torch; from task4_midi2audio_ddsp_baseline import MidiDDSP, synthesize_midi; \
m=MidiDDSP(); m.load_state_dict(torch.load('checkpoints_task4/best_model_urmp_vn.pt',map_location='cpu')['model_state_dict']); \
synthesize_midi(m,'checkpoints_task2_optimized/generated_melody.mid','out.wav',torch.device('cpu'))"
```

## TODO

- [x] Improve Task 2: velocity tokens, tempo/time-signature-aware alignment, chord-tone metric (see `task2_result_comparison.txt`)
- [x] Train Task 4 on URMP (violin solo, best val 1.3644)
- [x] Evaluation metrics: spectral loss curve + pitch/rest verification (see `task4_result_summary.txt`); Task 2 perplexity + chord-tone ratio
- [x] End-to-end demo: chord → MIDI → audio (`continuous_conditioned.wav/.mp3`)
- [ ] (Optional) Train more URMP instruments / more epochs for higher fidelity
- [ ] Record 20-min presentation video
- [ ] Export notebook as HTML for submission

## Submission Checklist

- [ ] `workbook.html` — Jupyter notebook exported as HTML
- [ ] `video_url.txt` — Google Drive / YouTube link to ~20 min presentation
- [x] `symbolic_conditioned.mid` — Task 2 generated MIDI (`checkpoints_task2_optimized/generated_melody.mid`)
- [x] `continuous_conditioned.mp3` — Task 4 generated audio (`checkpoints_task4/continuous_conditioned.mp3`)

## References

- POP909 Dataset: [music-x-lab/POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset)
- DDSP (Engel et al., ICLR 2020): [arxiv.org/abs/2001.04643](https://arxiv.org/abs/2001.04643)
- MIDI-DDSP (Wu et al., 2022): [arxiv.org/abs/2112.09312](https://arxiv.org/abs/2112.09312)
- REMI (Huang & Yang, ACM MM 2020): [YatingMusic/remi](https://github.com/YatingMusic/remi)
- MusicGen / AudioCraft: [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
