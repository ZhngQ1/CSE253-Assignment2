"""
Task 4: Continuous Conditioned Generation — MIDI → Audio (DDSP) Baseline
=========================================================================
Dataset: URMP (University of Rochester Multi-modal Music Performance)
Model: Simplified MIDI-DDSP (MIDI → synth parameters → differentiable synthesizer → audio)

Architecture (following Module 4 slides + MIDI-DDSP paper):
  1. MIDI Encoder:  MIDI notes (pitch, velocity, timing) → frame-level features
  2. Decoder:       frame-level features → DDSP parameters (f0, amplitudes, harmonics, noise)
  3. DDSP Synth:    differentiable additive synthesizer + filtered noise → waveform
  4. Loss:          multi-scale spectral loss (compare synthesized vs. target audio)

References:
  - DDSP: Engel et al., "DDSP: Differentiable Digital Signal Processing", ICLR 2020
    https://arxiv.org/abs/2001.04643
  - MIDI-DDSP: Wu et al., "MIDI-DDSP: Detailed Control of Musical Performance
    via Hierarchical Modeling", 2022  https://arxiv.org/abs/2112.09312
  - PyTorch DDSP implementations: github.com/acids-ircam/ddsp_pytorch

Usage:
  python task4_midi2audio_ddsp_baseline.py --data_dir /path/to/URMP --epochs 100
"""

import os
import glob
import argparse
import math
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 0. CONSTANTS
# ============================================================

SAMPLE_RATE = 16000       # 16kHz (standard for DDSP, saves GPU memory)
FRAME_RATE = 250          # frames per second (4ms per frame)
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE  # = 64 samples
N_HARMONICS = 60          # number of harmonics for additive synth
N_NOISE_BANDS = 65        # frequency bands for filtered noise
SEGMENT_SECONDS = 4       # training segment length in seconds
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS  # = 64000
SEGMENT_FRAMES = FRAME_RATE * SEGMENT_SECONDS    # = 1000


# ============================================================
# 1. DIFFERENTIABLE DSP COMPONENTS
# ============================================================

def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(x + eps)


def midi_to_hz(midi: torch.Tensor) -> torch.Tensor:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def hz_to_midi(hz: torch.Tensor) -> torch.Tensor:
    """Convert Hz to MIDI note number."""
    return 69.0 + 12.0 * torch.log2(hz / 440.0 + 1e-7)


def harmonic_synth(f0: torch.Tensor, amplitudes: torch.Tensor,
                   harmonic_distribution: torch.Tensor,
                   sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Differentiable additive (harmonic) synthesizer.

    Args:
        f0: (batch, n_frames, 1) fundamental frequency in Hz
        amplitudes: (batch, n_frames, 1) overall amplitude
        harmonic_distribution: (batch, n_frames, n_harmonics) relative harmonic amplitudes
        sample_rate: audio sample rate

    Returns:
        audio: (batch, n_samples) synthesized waveform
    """
    batch_size, n_frames, n_harmonics = harmonic_distribution.shape

    # Normalize harmonic distribution
    harmonic_distribution = F.softmax(harmonic_distribution, dim=-1)

    # Per-harmonic amplitudes
    harmonic_amps = amplitudes * harmonic_distribution  # (B, F, H)

    # Harmonic frequencies: f0 * [1, 2, 3, ..., H]
    harmonic_numbers = torch.arange(1, n_harmonics + 1, device=f0.device, dtype=f0.dtype)
    harmonic_freqs = f0 * harmonic_numbers.unsqueeze(0).unsqueeze(0)  # (B, F, H)

    # Upsample from frame rate to sample rate
    n_samples = n_frames * (sample_rate // FRAME_RATE)

    harmonic_freqs = F.interpolate(
        harmonic_freqs.permute(0, 2, 1),  # (B, H, F)
        size=n_samples, mode='linear', align_corners=False
    ).permute(0, 2, 1)  # (B, N, H)

    harmonic_amps = F.interpolate(
        harmonic_amps.permute(0, 2, 1),
        size=n_samples, mode='linear', align_corners=False
    ).permute(0, 2, 1)  # (B, N, H)

    # Phase accumulation (cumulative sum of instantaneous frequency)
    omega = 2.0 * math.pi * harmonic_freqs / sample_rate  # (B, N, H)
    phases = torch.cumsum(omega, dim=1)  # (B, N, H)

    # Sum of sinusoids
    audio = (harmonic_amps * torch.sin(phases)).sum(dim=-1)  # (B, N)

    return audio


def filtered_noise(magnitudes: torch.Tensor,
                   n_samples: int,
                   sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Differentiable filtered noise synthesizer.

    Generates white noise and filters it with a frequency-varying filter
    specified by the magnitudes of frequency bands.

    Args:
        magnitudes: (batch, n_frames, n_bands) filter magnitudes per frame
        n_samples: number of output audio samples
        sample_rate: audio sample rate

    Returns:
        audio: (batch, n_samples) filtered noise
    """
    batch_size, n_frames, n_bands = magnitudes.shape

    # Generate white noise
    noise = torch.randn(batch_size, n_samples, device=magnitudes.device)

    # STFT of noise
    window_size = (n_bands - 1) * 2  # = 128
    hop = n_samples // n_frames

    # Pad noise for STFT
    noise_stft = torch.stft(
        noise, n_fft=window_size, hop_length=hop,
        window=torch.hann_window(window_size, device=magnitudes.device),
        return_complex=True
    )  # (B, n_fft//2+1, T)

    # Interpolate magnitudes to match STFT frames
    n_stft_frames = noise_stft.shape[-1]
    mags = F.interpolate(
        magnitudes.permute(0, 2, 1),  # (B, n_bands, n_frames)
        size=n_stft_frames, mode='linear', align_corners=False
    )  # (B, n_bands, n_stft_frames)

    # Apply filter (multiply in frequency domain)
    mags = F.softplus(mags)  # ensure positive
    noise_stft = noise_stft * mags

    # ISTFT back to time domain
    audio = torch.istft(
        noise_stft, n_fft=window_size, hop_length=hop,
        window=torch.hann_window(window_size, device=magnitudes.device),
        length=n_samples
    )

    return audio


# ============================================================
# 2. MULTI-SCALE SPECTRAL LOSS
# ============================================================

class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale spectrogram loss from DDSP paper.

    Computes L1 loss on log-magnitude spectrograms at multiple FFT sizes.
    This encourages matching at both fine (time) and coarse (frequency) scales.
    """

    def __init__(self, fft_sizes: List[int] = [64, 128, 256, 512, 1024, 2048],
                 overlap: float = 0.75, eps: float = 1e-7):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.overlap = overlap
        self.eps = eps

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: (batch, n_samples) synthesized audio
            target: (batch, n_samples) target audio
        Returns:
            loss: scalar
        """
        loss = 0.0

        for fft_size in self.fft_sizes:
            hop = int(fft_size * (1.0 - self.overlap))
            window = torch.hann_window(fft_size, device=predicted.device)

            pred_stft = torch.stft(
                predicted, n_fft=fft_size, hop_length=hop,
                window=window, return_complex=True
            )
            target_stft = torch.stft(
                target, n_fft=fft_size, hop_length=hop,
                window=window, return_complex=True
            )

            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            # Log-magnitude loss
            log_loss = F.l1_loss(safe_log(pred_mag), safe_log(target_mag))

            # Linear magnitude loss
            lin_loss = F.l1_loss(pred_mag, target_mag)

            loss += log_loss + lin_loss

        return loss / len(self.fft_sizes)


# ============================================================
# 3. MIDI ENCODER
# ============================================================

def midi_to_frames(notes: List[Dict], n_frames: int,
                   frame_rate: int = FRAME_RATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert MIDI note list to frame-level features.

    Args:
        notes: list of dicts with keys 'pitch', 'start', 'end', 'velocity'
               (times in seconds)
        n_frames: number of output frames
        frame_rate: frames per second

    Returns:
        f0_hz: (n_frames,) fundamental frequency in Hz (0 = silence)
        velocity: (n_frames,) note velocity (0-1)
        onset: (n_frames,) binary onset indicator
    """
    f0_hz = np.zeros(n_frames, dtype=np.float32)
    velocity = np.zeros(n_frames, dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)

    for note in notes:
        start_frame = int(note['start'] * frame_rate)
        end_frame = int(note['end'] * frame_rate)
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(0, min(end_frame, n_frames))

        if start_frame >= end_frame:
            continue

        freq = 440.0 * 2.0 ** ((note['pitch'] - 69.0) / 12.0)
        vel = note['velocity'] / 127.0

        f0_hz[start_frame:end_frame] = freq
        velocity[start_frame:end_frame] = vel

        # Mark onset (first 2 frames of each note)
        onset_end = min(start_frame + 2, end_frame)
        onset[start_frame:onset_end] = 1.0

    return f0_hz, velocity, onset


class MidiEncoder(nn.Module):
    """Encodes MIDI frame-level features into conditioning for the DDSP decoder.

    Input features per frame:
      - f0 (Hz, log-scaled)
      - velocity (0-1)
      - onset indicator (0/1)

    Output: (batch, n_frames, d_model) conditioning vectors
    """

    def __init__(self, d_model: int = 256, n_layers: int = 2):
        super().__init__()

        # Input: 3 features (log_f0, velocity, onset)
        self.input_proj = nn.Linear(3, d_model)

        # Bidirectional GRU for temporal context
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if n_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, f0_hz: torch.Tensor, velocity: torch.Tensor,
                onset: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0_hz: (batch, n_frames) fundamental frequency in Hz
            velocity: (batch, n_frames) note velocity 0-1
            onset: (batch, n_frames) onset indicator 0/1

        Returns:
            conditioning: (batch, n_frames, d_model)
        """
        # Log-scale f0 (normalized)
        log_f0 = safe_log(f0_hz + 1.0) / 8.0  # roughly 0-1 range

        # Stack features
        features = torch.stack([log_f0, velocity, onset], dim=-1)  # (B, F, 3)

        h = self.input_proj(features)  # (B, F, d_model)
        h, _ = self.gru(h)             # (B, F, d_model)
        h = self.norm(h)

        return h


# ============================================================
# 4. DDSP DECODER
# ============================================================

class DDSPDecoder(nn.Module):
    """Decodes conditioning vectors into DDSP synthesizer parameters.

    Outputs:
      - amplitudes: (batch, n_frames, 1) overall amplitude
      - harmonic_distribution: (batch, n_frames, n_harmonics)
      - noise_magnitudes: (batch, n_frames, n_noise_bands)
    """

    def __init__(self, d_model: int = 256, n_harmonics: int = N_HARMONICS,
                 n_noise_bands: int = N_NOISE_BANDS):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Output heads
        self.amp_head = nn.Linear(d_model, 1)
        self.harmonic_head = nn.Linear(d_model, n_harmonics)
        self.noise_head = nn.Linear(d_model, n_noise_bands)

    def forward(self, conditioning: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            conditioning: (batch, n_frames, d_model)
        Returns:
            amplitudes: (batch, n_frames, 1)
            harmonic_distribution: (batch, n_frames, n_harmonics)
            noise_magnitudes: (batch, n_frames, n_noise_bands)
        """
        h = self.mlp(conditioning)

        amplitudes = torch.sigmoid(self.amp_head(h))  # 0-1
        harmonic_distribution = self.harmonic_head(h)  # logits, will be softmaxed in synth
        noise_magnitudes = self.noise_head(h)           # will be softplus'd in synth

        return amplitudes, harmonic_distribution, noise_magnitudes


# ============================================================
# 5. FULL MODEL
# ============================================================

class MidiDDSP(nn.Module):
    """Simplified MIDI-DDSP: MIDI → DDSP parameters → Audio

    Pipeline:
      MIDI notes → MidiEncoder → conditioning → DDSPDecoder → synth params
      synth params → harmonic_synth + filtered_noise → audio
    """

    def __init__(self, d_model: int = 256, n_harmonics: int = N_HARMONICS,
                 n_noise_bands: int = N_NOISE_BANDS, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder = MidiEncoder(d_model=d_model)
        self.decoder = DDSPDecoder(d_model=d_model, n_harmonics=n_harmonics,
                                    n_noise_bands=n_noise_bands)

    def forward(self, f0_hz: torch.Tensor, velocity: torch.Tensor,
                onset: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            f0_hz: (batch, n_frames) fundamental frequency in Hz
            velocity: (batch, n_frames) note velocity 0-1
            onset: (batch, n_frames) onset indicator 0/1

        Returns:
            audio: (batch, n_samples) synthesized audio
            synth_params: dict of intermediate parameters (for analysis)
        """
        n_frames = f0_hz.shape[1]
        n_samples = n_frames * (self.sample_rate // FRAME_RATE)

        # Encode MIDI
        conditioning = self.encoder(f0_hz, velocity, onset)

        # Decode to synth params
        amplitudes, harmonic_dist, noise_mags = self.decoder(conditioning)

        # Synthesize audio
        f0_expanded = f0_hz.unsqueeze(-1)  # (B, F, 1)
        harmonic_audio = harmonic_synth(
            f0_expanded, amplitudes, harmonic_dist,
            sample_rate=self.sample_rate
        )

        noise_audio = filtered_noise(noise_mags, n_samples, self.sample_rate)

        # Mix
        audio = harmonic_audio + noise_audio

        # Normalize to prevent clipping
        peak = audio.abs().max(dim=-1, keepdim=True)[0] + 1e-7
        audio = audio / peak * 0.9

        synth_params = {
            'amplitudes': amplitudes,
            'harmonic_distribution': harmonic_dist,
            'noise_magnitudes': noise_mags,
            'harmonic_audio': harmonic_audio,
            'noise_audio': noise_audio,
        }

        return audio, synth_params


# ============================================================
# 6. DATASET
# ============================================================

def parse_midi_file(midi_path: str) -> List[Dict]:
    """Parse a MIDI file into a list of note events using symusic."""
    try:
        from symusic import Score
        score = Score(midi_path)
        notes = []
        for track in score.tracks:
            tpq = score.ticks_per_quarter
            for n in track.notes:
                notes.append({
                    'pitch': n.pitch,
                    'start': n.start / (tpq * 2.0),   # ticks → seconds (assume 120 BPM)
                    'end': n.end / (tpq * 2.0),
                    'velocity': n.velocity,
                })
        notes.sort(key=lambda x: x['start'])
        return notes
    except Exception as e:
        print(f"Error parsing {midi_path}: {e}")
        return []


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE,
               max_seconds: float = 300.0) -> Optional[np.ndarray]:
    """Load audio file and resample to target sample rate."""
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True,
                                  duration=max_seconds)
        return audio
    except ImportError:
        # Fallback: use scipy
        try:
            from scipy.io import wavfile
            from scipy.signal import resample
            sr, audio = wavfile.read(audio_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                n_samples = int(len(audio) * sample_rate / sr)
                audio = resample(audio, n_samples).astype(np.float32)
            return audio[:int(max_seconds * sample_rate)]
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None


class URMPDataset(Dataset):
    """URMP dataset loader for MIDI-conditioned audio synthesis.

    Expected URMP directory structure:
      URMP/
        01_Jupiter_vn_vc/
          AuSep_1_vn_01_Jupiter.wav    (individual instrument audio)
          Notes_1_vn_01_Jupiter.mid     (MIDI score)
          F0s_1_vn_01_Jupiter.txt       (frame-level pitch, optional)
          ...

    Each piece has multiple instrument tracks. We train on individual
    instrument tracks (mono audio + MIDI pairs).
    """

    def __init__(self, data_dir: str, segment_seconds: float = SEGMENT_SECONDS,
                 sample_rate: int = SAMPLE_RATE, frame_rate: int = FRAME_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.segment_frames = int(segment_seconds * frame_rate)

        self.pairs = []  # list of (audio_path, midi_path)
        self._scan_urmp(data_dir)
        print(f"URMPDataset: found {len(self.pairs)} MIDI-audio pairs")

    def _scan_urmp(self, data_dir: str):
        """Scan URMP directory for MIDI-audio pairs."""
        for piece_dir in sorted(glob.glob(os.path.join(data_dir, '*'))):
            if not os.path.isdir(piece_dir):
                continue

            # Find audio files (individual separations)
            audio_files = glob.glob(os.path.join(piece_dir, 'AuSep_*.wav'))
            midi_files = glob.glob(os.path.join(piece_dir, 'Notes_*.mid'))

            # Match by instrument index
            audio_by_idx = {}
            for af in audio_files:
                basename = os.path.basename(af)
                # Format: AuSep_{idx}_{instrument}_{piece}.wav
                parts = basename.replace('.wav', '').split('_')
                if len(parts) >= 2:
                    idx = parts[1]
                    audio_by_idx[idx] = af

            for mf in midi_files:
                basename = os.path.basename(mf)
                # Format: Notes_{idx}_{instrument}_{piece}.mid
                parts = basename.replace('.mid', '').split('_')
                if len(parts) >= 2:
                    idx = parts[1]
                    if idx in audio_by_idx:
                        self.pairs.append((audio_by_idx[idx], mf))

    def __len__(self):
        return len(self.pairs) * 10  # 10 random segments per piece per epoch

    def __getitem__(self, idx):
        pair_idx = idx % len(self.pairs)
        audio_path, midi_path = self.pairs[pair_idx]

        # Load audio
        audio = load_audio(audio_path, self.sample_rate)
        if audio is None or len(audio) < self.segment_samples:
            # Return silence if loading fails
            return self._empty_sample()

        # Parse MIDI
        notes = parse_midi_file(midi_path)
        if not notes:
            return self._empty_sample()

        # Random segment
        max_start = len(audio) - self.segment_samples
        if max_start <= 0:
            start_sample = 0
        else:
            start_sample = np.random.randint(0, max_start)

        start_sec = start_sample / self.sample_rate
        end_sec = start_sec + (self.segment_samples / self.sample_rate)

        # Crop audio
        audio_segment = audio[start_sample:start_sample + self.segment_samples]

        # Normalize audio
        peak = np.abs(audio_segment).max()
        if peak > 0:
            audio_segment = audio_segment / peak * 0.9

        # Filter notes in this segment
        segment_notes = []
        for n in notes:
            if n['end'] > start_sec and n['start'] < end_sec:
                segment_notes.append({
                    'pitch': n['pitch'],
                    'start': max(0, n['start'] - start_sec),
                    'end': min(end_sec - start_sec, n['end'] - start_sec),
                    'velocity': n['velocity'],
                })

        # Convert MIDI to frame-level features
        f0_hz, velocity, onset = midi_to_frames(
            segment_notes, self.segment_frames, self.frame_rate
        )

        return {
            'audio': torch.tensor(audio_segment, dtype=torch.float32),
            'f0_hz': torch.tensor(f0_hz, dtype=torch.float32),
            'velocity': torch.tensor(velocity, dtype=torch.float32),
            'onset': torch.tensor(onset, dtype=torch.float32),
        }

    def _empty_sample(self):
        return {
            'audio': torch.zeros(self.segment_samples, dtype=torch.float32),
            'f0_hz': torch.zeros(self.segment_frames, dtype=torch.float32),
            'velocity': torch.zeros(self.segment_frames, dtype=torch.float32),
            'onset': torch.zeros(self.segment_frames, dtype=torch.float32),
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing/debugging without URMP.

    Generates sine wave audio from MIDI-like note sequences.
    Useful for verifying the pipeline works end-to-end before
    training on real data.
    """

    def __init__(self, n_samples: int = 200, segment_seconds: float = SEGMENT_SECONDS,
                 sample_rate: int = SAMPLE_RATE, frame_rate: int = FRAME_RATE):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.segment_frames = int(segment_seconds * frame_rate)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        np.random.seed(idx)

        # Random notes
        n_notes = np.random.randint(3, 10)
        notes = []
        t = np.random.uniform(0, 0.5)
        for _ in range(n_notes):
            pitch = np.random.randint(55, 80)
            dur = np.random.uniform(0.2, 1.5)
            vel = np.random.randint(60, 120)
            notes.append({'pitch': pitch, 'start': t, 'end': t + dur, 'velocity': vel})
            t += dur + np.random.uniform(0, 0.3)

        # Generate frame features
        f0_hz, velocity, onset = midi_to_frames(notes, self.segment_frames, self.frame_rate)

        # Generate "target" audio (simple sine waves)
        audio = np.zeros(self.segment_samples, dtype=np.float32)
        for note in notes:
            freq = 440.0 * 2.0 ** ((note['pitch'] - 69.0) / 12.0)
            start_s = int(note['start'] * self.sample_rate)
            end_s = int(note['end'] * self.sample_rate)
            end_s = min(end_s, self.segment_samples)
            start_s = max(0, start_s)
            if start_s >= end_s:
                continue
            t_arr = np.arange(start_s, end_s) / self.sample_rate
            vel_amp = note['velocity'] / 127.0

            # Add some harmonics for richer timbre
            signal = vel_amp * 0.5 * np.sin(2 * np.pi * freq * t_arr)
            signal += vel_amp * 0.25 * np.sin(2 * np.pi * 2 * freq * t_arr)
            signal += vel_amp * 0.125 * np.sin(2 * np.pi * 3 * freq * t_arr)

            # Simple envelope
            n_env = end_s - start_s
            attack = min(int(0.02 * self.sample_rate), n_env)
            release = min(int(0.05 * self.sample_rate), n_env)
            envelope = np.ones(n_env)
            if attack > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if release > 0:
                envelope[-release:] = np.linspace(1, 0, release)
            signal *= envelope

            audio[start_s:end_s] += signal

        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9

        return {
            'audio': torch.tensor(audio, dtype=torch.float32),
            'f0_hz': torch.tensor(f0_hz, dtype=torch.float32),
            'velocity': torch.tensor(velocity, dtype=torch.float32),
            'onset': torch.tensor(onset, dtype=torch.float32),
        }


# ============================================================
# 7. TRAINING
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        audio_target = batch['audio'].to(device)
        f0_hz = batch['f0_hz'].to(device)
        velocity = batch['velocity'].to(device)
        onset = batch['onset'].to(device)

        # Forward
        audio_pred, synth_params = model(f0_hz, velocity, onset)

        # Match lengths
        min_len = min(audio_pred.shape[-1], audio_target.shape[-1])
        audio_pred = audio_pred[:, :min_len]
        audio_target = audio_target[:, :min_len]

        loss = criterion(audio_pred, audio_target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        audio_target = batch['audio'].to(device)
        f0_hz = batch['f0_hz'].to(device)
        velocity = batch['velocity'].to(device)
        onset = batch['onset'].to(device)

        audio_pred, _ = model(f0_hz, velocity, onset)

        min_len = min(audio_pred.shape[-1], audio_target.shape[-1])
        loss = criterion(audio_pred[:, :min_len], audio_target[:, :min_len])

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# 8. INFERENCE — MIDI file → audio
# ============================================================

@torch.no_grad()
def synthesize_midi(model, midi_path: str, output_path: str,
                    device: torch.device, sample_rate: int = SAMPLE_RATE):
    """Synthesize a MIDI file into audio using the trained model.

    Args:
        model: trained MidiDDSP model
        midi_path: path to input MIDI file
        output_path: path to save output WAV file
        device: torch device
        sample_rate: output sample rate
    """
    model.eval()

    # Parse MIDI
    notes = parse_midi_file(midi_path)
    if not notes:
        print(f"No notes found in {midi_path}")
        return

    # Determine duration
    max_time = max(n['end'] for n in notes) + 1.0  # add 1 second padding
    n_frames = int(max_time * FRAME_RATE)
    n_samples = n_frames * (sample_rate // FRAME_RATE)

    # Convert to frame features
    f0_hz, velocity, onset = midi_to_frames(notes, n_frames, FRAME_RATE)

    # Process in chunks to save memory
    chunk_frames = SEGMENT_FRAMES
    all_audio = []

    for start in range(0, n_frames, chunk_frames):
        end = min(start + chunk_frames, n_frames)
        actual_frames = end - start

        # Pad to full chunk if needed
        f0_chunk = np.zeros(chunk_frames, dtype=np.float32)
        vel_chunk = np.zeros(chunk_frames, dtype=np.float32)
        on_chunk = np.zeros(chunk_frames, dtype=np.float32)

        f0_chunk[:actual_frames] = f0_hz[start:end]
        vel_chunk[:actual_frames] = velocity[start:end]
        on_chunk[:actual_frames] = onset[start:end]

        # To tensors
        f0_t = torch.tensor(f0_chunk, dtype=torch.float32).unsqueeze(0).to(device)
        vel_t = torch.tensor(vel_chunk, dtype=torch.float32).unsqueeze(0).to(device)
        on_t = torch.tensor(on_chunk, dtype=torch.float32).unsqueeze(0).to(device)

        audio_chunk, _ = model(f0_t, vel_t, on_t)
        audio_chunk = audio_chunk[0].cpu().numpy()

        # Only keep the actual frames' worth of audio
        actual_samples = actual_frames * (sample_rate // FRAME_RATE)
        all_audio.append(audio_chunk[:actual_samples])

    audio = np.concatenate(all_audio)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    # Save as WAV
    try:
        import soundfile as sf
        sf.write(output_path, audio, sample_rate)
    except ImportError:
        from scipy.io import wavfile
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_path, sample_rate, audio_int16)

    duration = len(audio) / sample_rate
    print(f"Saved {duration:.1f}s audio to {output_path}")


# ============================================================
# 9. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to URMP dataset directory. If None, uses synthetic data.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_task4')
    parser.add_argument('--test_midi', type=str, default=None,
                        help='Path to a MIDI file for synthesis after training')
    args = parser.parse_args()

    print("=" * 60)
    print("Task 4: MIDI → Audio (DDSP) Baseline")
    print("=" * 60)

    # Dataset
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"\nLoading URMP dataset from {args.data_dir}")
        full_dataset = URMPDataset(args.data_dir)
        # Split
        n_total = len(full_dataset.pairs)
        n_train = int(0.8 * n_total)
        # We create two datasets pointing to different pairs
        train_pairs = full_dataset.pairs[:n_train]
        test_pairs = full_dataset.pairs[n_train:]

        train_dataset = URMPDataset.__new__(URMPDataset)
        train_dataset.__dict__.update(full_dataset.__dict__)
        train_dataset.pairs = train_pairs

        test_dataset = URMPDataset.__new__(URMPDataset)
        test_dataset.__dict__.update(full_dataset.__dict__)
        test_dataset.pairs = test_pairs
    else:
        print("\nNo URMP data_dir provided — using synthetic dataset for testing")
        print("(To use URMP, download from https://labsites.rochester.edu/air/projects/URMP.html)")
        train_dataset = SyntheticDataset(n_samples=400)
        test_dataset = SyntheticDataset(n_samples=50)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Model
    model = MidiDDSP(d_model=args.d_model).to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"d_model={args.d_model}, harmonics={N_HARMONICS}, noise_bands={N_NOISE_BANDS}")
    print(f"Sample rate={SAMPLE_RATE}Hz, frame rate={FRAME_RATE}Hz")

    # Training setup
    criterion = MultiScaleSpectralLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nTraining for {args.epochs} epochs on {args.device}...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = evaluate(model, test_loader, criterion, args.device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

        # Save a sample every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(test_loader))
                f0 = sample_batch['f0_hz'][:1].to(args.device)
                vel = sample_batch['velocity'][:1].to(args.device)
                on = sample_batch['onset'][:1].to(args.device)
                audio_pred, _ = model(f0, vel, on)
                audio_np = audio_pred[0].cpu().numpy()

                # Save as WAV
                try:
                    import soundfile as sf
                    sf.write(os.path.join(args.save_dir, f'sample_epoch{epoch}.wav'),
                             audio_np, SAMPLE_RATE)
                except ImportError:
                    from scipy.io import wavfile
                    wavfile.write(os.path.join(args.save_dir, f'sample_epoch{epoch}.wav'),
                                  SAMPLE_RATE, (audio_np * 32767).astype(np.int16))

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")

    # Synthesize from a test MIDI file
    if args.test_midi and os.path.exists(args.test_midi):
        print(f"\nSynthesizing from {args.test_midi}...")
        synthesize_midi(model, args.test_midi,
                       os.path.join(args.save_dir, 'synthesized.wav'),
                       torch.device(args.device))
    else:
        print("\nTo synthesize a MIDI file, use: --test_midi path/to/file.mid")

    print("\nAll files saved to:", args.save_dir)
    print("Done!")


if __name__ == '__main__':
    main()
