"""
Task 2: Symbolic Conditioned Generation — Chord → Melody Baseline
=================================================================
Dataset: POP909 (909 Chinese pop songs with melody/bridge/piano tracks + chord annotations)
Model: Transformer Decoder with chord prefix conditioning
Representation: REMI-like token sequence

Pipeline:
  1. Parse POP909 MIDI → extract melody notes + chord labels (time-aligned)
  2. Quantize into bar-level segments: each bar has chord label + melody note sequence
  3. Encode as token sequence: [BOS] [CHORD_1] [BAR] [note tokens...] [CHORD_2] [BAR] [note tokens...] ... [EOS]
  4. Train a Transformer decoder (next-token prediction)
  5. At inference: provide chord tokens as prompt → autoregressively generate melody

Usage:
  python task2_chord2melody_baseline.py --data_dir /path/to/POP909 --epochs 50
"""

import os
import json
import glob
import random
import argparse
import math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from symusic import Score
from midiutil import MIDIFile


# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

# --- Chord vocabulary ---
# Simplify POP909 chord labels to root + quality
CHORD_QUALITIES = ['maj', 'min', 'dim', 'aug', '7', 'maj7', 'min7', 'dim7', 'hdim7',
                    'sus2', 'sus4', 'aug', 'N']  # N = no chord
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ROOT_ALIASES = {'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B'}


def simplify_chord(chord_label: str) -> str:
    """Simplify POP909 chord label like 'Bb:min7/b7' → 'A#:min7'"""
    if chord_label == 'N' or chord_label == '':
        return 'N'

    # Remove slash (inversion) info
    if '/' in chord_label:
        chord_label = chord_label.split('/')[0]

    parts = chord_label.split(':')
    if len(parts) != 2:
        return 'N'

    root, quality = parts

    # Normalize root
    if root in ROOT_ALIASES:
        root = ROOT_ALIASES[root]
    if root not in ROOTS:
        return 'N'

    # Remove parenthetical extensions like sus4(b7)
    if '(' in quality:
        quality = quality.split('(')[0]

    # Map to simplified quality
    quality_map = {
        'maj': 'maj', 'min': 'min', 'dim': 'dim', 'aug': 'aug',
        '7': '7', 'maj7': 'maj7', 'min7': 'min7', 'dim7': 'dim7',
        'hdim7': 'hdim7', 'sus2': 'sus2', 'sus4': 'sus4',
        'maj6': 'maj', 'min6': 'min',  # simplify 6th chords
        'minmaj7': 'min7',  # simplify
        '9': '7', 'maj9': 'maj7', 'min9': 'min7',  # simplify 9th
    }
    quality = quality_map.get(quality, 'maj')  # default to maj

    return f'{root}:{quality}'


def parse_chord_file(chord_path: str, tpq: int) -> List[Tuple[int, int, str]]:
    """Parse chord_midi.txt → list of (start_tick, end_tick, chord_label)"""
    chords = []
    with open(chord_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            start_sec, end_sec, label = float(parts[0]), float(parts[1]), parts[2]
            # Convert seconds to ticks (approximate: assume 120 BPM)
            # POP909 tempo varies, but for alignment we use the MIDI tick info
            start_tick = int(start_sec * tpq * 2)  # 120 BPM → 2 beats/sec
            end_tick = int(end_sec * tpq * 2)
            simplified = simplify_chord(label)
            chords.append((start_tick, end_tick, simplified))
    return chords


def get_chord_at_tick(chords: List[Tuple[int, int, str]], tick: int) -> str:
    """Get the chord active at a given tick"""
    for start, end, label in chords:
        if start <= tick < end:
            return label
    return 'N'


# --- Note / Melody tokenization ---
# We use a simplified REMI-like scheme:
#   - Pitch tokens: Pitch_0 ... Pitch_127
#   - Duration tokens: Duration_1 ... Duration_32 (in units of 1/16 note = 30 ticks at TPQ=480)
#   - TimeShift tokens: TimeShift_0 ... TimeShift_32 (same unit)
#   - Chord tokens: Chord_C:maj, Chord_A:min, etc.
#   - Special: BOS, EOS, PAD, BAR

DURATION_UNIT = 120  # ticks per unit (1/8 note at TPQ=480)
MAX_DURATION = 16    # max duration in units
MAX_TIMESHIFT = 16   # max time shift in units
BAR_TICKS = 1920     # 4 beats * 480 TPQ


def build_vocab():
    """Build the full token vocabulary"""
    tokens = ['PAD', 'BOS', 'EOS', 'BAR', 'SEP']

    # Chord tokens
    chord_tokens = ['Chord_N']
    for root in ROOTS:
        for qual in ['maj', 'min', 'dim', 'aug', '7', 'maj7', 'min7', 'dim7', 'hdim7', 'sus2', 'sus4']:
            chord_tokens.append(f'Chord_{root}:{qual}')
    tokens += chord_tokens

    # Pitch tokens (we limit to a reasonable range for melody: 48-84)
    for p in range(48, 85):
        tokens.append(f'Pitch_{p}')

    # Duration tokens
    for d in range(1, MAX_DURATION + 1):
        tokens.append(f'Duration_{d}')

    # TimeShift tokens
    for t in range(1, MAX_TIMESHIFT + 1):
        tokens.append(f'TimeShift_{t}')

    # Rest token
    tokens.append('Rest')

    token2id = {t: i for i, t in enumerate(tokens)}
    id2token = {i: t for t, i in token2id.items()}
    return token2id, id2token


def quantize_tick(tick: int, unit: int = DURATION_UNIT) -> int:
    """Quantize a tick value to the nearest unit"""
    return max(1, round(tick / unit))


def melody_to_tokens(notes, bar_start: int, bar_end: int, token2id: dict) -> List[int]:
    """Convert melody notes in a bar to token IDs"""
    tokens = []
    # Filter notes in this bar
    bar_notes = []
    for n in notes:
        # Include notes that start in this bar
        if bar_start <= n.start < bar_end:
            bar_notes.append(n)

    if not bar_notes:
        tokens.append(token2id['Rest'])
        return tokens

    # Sort by start time
    bar_notes.sort(key=lambda n: (n.start, n.pitch))

    prev_start = bar_start
    for note in bar_notes:
        # Time shift from previous event
        dt = note.start - prev_start
        if dt > 0:
            shift = min(quantize_tick(dt), MAX_TIMESHIFT)
            ts_token = f'TimeShift_{shift}'
            if ts_token in token2id:
                tokens.append(token2id[ts_token])

        # Pitch (clamp to range)
        pitch = max(48, min(84, note.pitch))
        tokens.append(token2id[f'Pitch_{pitch}'])

        # Duration
        dur = min(quantize_tick(note.end - note.start), MAX_DURATION)
        tokens.append(token2id[f'Duration_{dur}'])

        prev_start = note.start

    return tokens


def process_song(midi_path: str, chord_path: str, token2id: dict) -> Optional[List[int]]:
    """Process one POP909 song into a token sequence
    Format: [BOS] [Chord_X] [BAR] [melody tokens...] [Chord_Y] [BAR] [melody tokens...] ... [EOS]
    """
    try:
        score = Score(midi_path)
    except Exception:
        return None

    if len(score.tracks) == 0:
        return None

    # Find MELODY track (track 0 in POP909)
    melody_track = score.tracks[0]
    if len(melody_track.notes) < 10:
        return None

    tpq = score.ticks_per_quarter
    bar_ticks = tpq * 4  # assume 4/4 time

    # Parse chords
    chords = parse_chord_file(chord_path, tpq)
    if not chords:
        return None

    # Find the range of the song
    all_starts = [n.start for n in melody_track.notes]
    song_start = min(all_starts)
    song_end = max(n.end for n in melody_track.notes)

    # Align to bar boundaries
    first_bar = (song_start // bar_ticks) * bar_ticks
    last_bar = ((song_end // bar_ticks) + 1) * bar_ticks

    tokens = [token2id['BOS']]

    for bar_start in range(first_bar, last_bar, bar_ticks):
        bar_end = bar_start + bar_ticks

        # Get chord for this bar
        bar_mid = bar_start + bar_ticks // 2
        chord = get_chord_at_tick(chords, bar_mid)
        chord_token = f'Chord_{chord}'
        if chord_token not in token2id:
            chord_token = 'Chord_N'
        tokens.append(token2id[chord_token])

        # Add BAR marker
        tokens.append(token2id['BAR'])

        # Add melody tokens for this bar
        mel_tokens = melody_to_tokens(melody_track.notes, bar_start, bar_end, token2id)
        tokens.extend(mel_tokens)

    tokens.append(token2id['EOS'])
    return tokens


def prepare_dataset(pop909_dir: str, token2id: dict) -> Tuple[List[List[int]], List[List[int]]]:
    """Process all POP909 songs and split into train/test"""
    all_sequences = []
    song_dirs = sorted([d for d in os.listdir(pop909_dir) if d.isdigit()])

    for song_id in song_dirs:
        song_dir = os.path.join(pop909_dir, song_id)
        midi_path = os.path.join(song_dir, f'{song_id}.mid')
        chord_path = os.path.join(song_dir, 'chord_midi.txt')

        if not os.path.exists(midi_path) or not os.path.exists(chord_path):
            continue

        seq = process_song(midi_path, chord_path, token2id)
        if seq and len(seq) > 20:  # minimum length filter
            all_sequences.append(seq)

    print(f"Successfully processed {len(all_sequences)} / {len(song_dirs)} songs")

    # Split 90/10
    random.seed(42)
    random.shuffle(all_sequences)
    split = int(0.9 * len(all_sequences))
    train_seqs = all_sequences[:split]
    test_seqs = all_sequences[split:]

    print(f"Train: {len(train_seqs)}, Test: {len(test_seqs)}")
    print(f"Avg sequence length: {np.mean([len(s) for s in all_sequences]):.0f}")

    return train_seqs, test_seqs


# ============================================================
# 2. PYTORCH DATASET
# ============================================================

class ChordMelodyDataset(Dataset):
    """Sliding window dataset over token sequences"""

    def __init__(self, sequences: List[List[int]], max_seq_len: int = 512, pad_id: int = 0):
        self.samples = []
        self.pad_id = pad_id

        for seq in sequences:
            # Sliding window with stride
            if len(seq) <= max_seq_len:
                self.samples.append(seq)
            else:
                stride = max_seq_len // 2
                for i in range(0, len(seq) - max_seq_len + 1, stride):
                    self.samples.append(seq[i:i + max_seq_len])
                # Always include the ending
                self.samples.append(seq[-max_seq_len:])

        print(f"  Dataset: {len(self.samples)} samples from {len(sequences)} songs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


def collate_fn(batch, pad_id=0):
    """Pad sequences to same length in batch"""
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# ============================================================
# 3. TRANSFORMER MODEL
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MelodyTransformer(nn.Module):
    """GPT-style Transformer Decoder for chord-conditioned melody generation.

    The model uses a decoder-only (causal) architecture, similar to GPT.
    Chord tokens are naturally interleaved in the sequence:
      [BOS] [Chord_X] [BAR] [note...] [Chord_Y] [BAR] [note...] [EOS]

    During inference, we provide chord tokens as a prompt and let the
    model generate melody tokens autoregressively.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1,
                 max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Use TransformerEncoder with causal mask → standard GPT-style decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token IDs
            pad_mask: (batch, seq_len) True where padded
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        device = x.device

        # Causal mask: upper triangular = True (blocked positions)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Embed + positional encoding
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        # GPT-style: TransformerEncoder with causal mask
        out = self.transformer(
            src=h,
            mask=causal_mask,
            src_key_padding_mask=pad_mask
        )

        out = self.ln_f(out)
        logits = self.fc_out(out)
        return logits


# ============================================================
# 4. TRAINING
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, device, pad_id=0):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        pad_mask = (inputs == pad_id)

        logits = model(inputs, pad_mask)
        logits = logits.reshape(-1, model.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = criterion(logits, targets_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Count non-pad tokens for accurate loss reporting
        non_pad = (targets_flat != pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, pad_id=0):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        pad_mask = (inputs == pad_id)
        logits = model(inputs, pad_mask)
        logits = logits.reshape(-1, model.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = criterion(logits, targets_flat)

        non_pad = (targets_flat != pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


# ============================================================
# 5. GENERATION
# ============================================================

def generate_melody(model, chord_sequence: List[str], token2id: dict, id2token: dict,
                    device: torch.device, max_bars: int = 16,
                    temperature: float = 0.9, top_k: int = 50) -> List[int]:
    """Generate melody given a chord progression.

    Args:
        chord_sequence: list of chord labels, e.g. ['C:maj', 'G:maj', 'A:min', 'F:maj']
        token2id: vocabulary mapping
        id2token: reverse vocabulary
        device: torch device
        max_bars: number of bars to generate (should match chord_sequence length)
        temperature: sampling temperature
        top_k: top-k filtering

    Returns:
        list of generated token IDs
    """
    model.eval()

    # Build the prompt: [BOS] [Chord_X] [BAR]
    tokens = [token2id['BOS']]
    chord_token = f'Chord_{chord_sequence[0]}'
    if chord_token not in token2id:
        chord_token = 'Chord_N'
    tokens.append(token2id[chord_token])
    tokens.append(token2id['BAR'])

    chord_idx = 0  # which chord we're on
    bars_generated = 0

    max_ctx = 512  # match training max_seq_len
    for step in range(1000):  # max generation steps
        ctx = tokens[-max_ctx:] if len(tokens) > max_ctx else tokens
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)
        logits = logits[0, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, top_k)
            filtered = torch.full_like(logits, float('-inf'))
            filtered.scatter_(0, topk_idx, topk_vals)
            logits = filtered

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        tokens.append(next_token)

        token_str = id2token.get(next_token, '')

        # If we generated a BAR token, that means we finished a bar
        if token_str == 'BAR':
            bars_generated += 1
            if bars_generated >= len(chord_sequence):
                tokens.append(token2id['EOS'])
                break

        # If we reach EOS, stop
        if token_str == 'EOS':
            break

        # When we encounter BAR, the next chord should be injected
        # Actually, let the model generate naturally — the chord tokens are part of the learned sequence

    return tokens


def generate_melody_forced_chords(model, chord_sequence: List[str], token2id: dict, id2token: dict,
                                   device: torch.device, temperature: float = 0.9,
                                   top_k: int = 50) -> List[int]:
    """Generate melody with forced chord injection at each bar.

    This ensures the generated melody follows the specified chord progression.
    At each bar boundary, we force-inject the next chord token.
    """
    model.eval()

    tokens = [token2id['BOS']]

    for bar_idx, chord in enumerate(chord_sequence):
        # Inject chord token
        chord_token = f'Chord_{chord}'
        if chord_token not in token2id:
            chord_token = 'Chord_N'
        tokens.append(token2id[chord_token])
        tokens.append(token2id['BAR'])

        # Generate melody tokens until next chord/BAR or max steps
        max_ctx = 512  # match training max_seq_len
        for step in range(50):  # max tokens per bar
            # Truncate to last max_ctx tokens to stay within positional encoding range
            ctx = tokens[-max_ctx:] if len(tokens) > max_ctx else tokens
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)
            logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, top_k)
                filtered = torch.full_like(logits, float('-inf'))
                filtered.scatter_(0, topk_idx, topk_vals)
                logits = filtered

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            token_str = id2token.get(next_token, '')

            # If model wants to output a Chord or BAR token, this bar is done
            if token_str.startswith('Chord_') or token_str == 'BAR' or token_str == 'EOS':
                break

            tokens.append(next_token)

    tokens.append(token2id['EOS'])
    return tokens


# ============================================================
# 6. TOKEN SEQUENCE → MIDI
# ============================================================

def analyze_generated(token_ids: List[int], id2token: dict) -> dict:
    """Analyze generated token sequence for quality metrics"""
    pitches = []
    durations = []
    n_bars = 0
    n_rests = 0
    n_notes = 0

    for tid in token_ids:
        t = id2token.get(tid, '')
        if t.startswith('Pitch_'):
            pitches.append(int(t.split('_')[1]))
            n_notes += 1
        elif t.startswith('Duration_'):
            durations.append(int(t.split('_')[1]))
        elif t == 'BAR':
            n_bars += 1
        elif t == 'Rest':
            n_rests += 1

    stats = {
        'n_bars': n_bars,
        'n_notes': n_notes,
        'n_rests': n_rests,
        'notes_per_bar': n_notes / max(n_bars, 1),
    }
    if pitches:
        stats['pitch_range'] = max(pitches) - min(pitches)
        stats['pitch_mean'] = np.mean(pitches)
        stats['pitch_std'] = np.std(pitches)
    if durations:
        stats['avg_duration'] = np.mean(durations)

    return stats


def tokens_to_midi(token_ids: List[int], id2token: dict, output_path: str,
                   tempo: int = 120, tpq: int = 480):
    """Convert generated token sequence back to MIDI file"""
    midi = MIDIFile(1)
    track = 0
    midi.addTempo(track, 0, tempo)

    beat_unit = 0.5  # each duration/timeshift unit = 0.5 beats (1/8 note)
    current_time = 0.0  # in beats
    current_pitch = None
    current_duration = None

    for tid in token_ids:
        token = id2token.get(tid, '')

        if token.startswith('TimeShift_'):
            shift = int(token.split('_')[1])
            current_time += shift * beat_unit

        elif token.startswith('Pitch_'):
            current_pitch = int(token.split('_')[1])

        elif token.startswith('Duration_'):
            dur = int(token.split('_')[1])
            current_duration = dur * beat_unit
            if current_pitch is not None:
                midi.addNote(track, 0, current_pitch, current_time, current_duration, 80)
            current_pitch = None
            current_duration = None

        elif token == 'BAR':
            # Align to next bar boundary (4 beats)
            bar_beats = 4.0
            current_time = math.ceil(current_time / bar_beats) * bar_beats

        elif token == 'Rest':
            pass  # no notes in this bar

    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    print(f"Saved MIDI to {output_path}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/POP909-Dataset/POP909',
                        help='Path to POP909 directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    print("=" * 60)
    print("Task 2: Chord → Melody Generation Baseline")
    print("=" * 60)

    # Build vocabulary
    token2id, id2token = build_vocab()
    vocab_size = len(token2id)
    print(f"Vocabulary size: {vocab_size}")

    # Process dataset
    print("\nProcessing POP909 dataset...")
    train_seqs, test_seqs = prepare_dataset(args.data_dir, token2id)

    # Create datasets and dataloaders
    print("\nCreating datasets...")
    train_dataset = ChordMelodyDataset(train_seqs, max_seq_len=args.max_seq_len)
    test_dataset = ChordMelodyDataset(test_seqs, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # Create model
    print(f"\nModel config: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_layers}")
    model = MelodyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    print(f"\nTraining for {args.epochs} epochs on {args.device}...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss = evaluate(model, test_loader, criterion, args.device)
        scheduler.step()

        perplexity_train = math.exp(min(train_loss, 10))
        perplexity_val = math.exp(min(val_loss, 10))

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} (PPL: {perplexity_train:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL: {perplexity_val:.2f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab': token2id,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Generate sample
    print("\n" + "=" * 60)
    print("Generating sample melody...")

    # Classic pop chord progression: C - G - Am - F (repeated)
    chord_prog = ['C:maj', 'G:maj', 'A:min', 'F:maj'] * 4  # 16 bars

    generated = generate_melody_forced_chords(
        model, chord_prog, token2id, id2token,
        device=torch.device(args.device), temperature=0.9, top_k=50
    )

    output_path = os.path.join(args.save_dir, 'generated_melody.mid')
    tokens_to_midi(generated, id2token, output_path)

    print("\nGenerated token sequence (first 50 tokens):")
    print([id2token[t] for t in generated[:50]])
    print(f"\nTotal tokens generated: {len(generated)}")

    # Analyze generated melody
    stats = analyze_generated(generated, id2token)
    print(f"\nGeneration stats: {json.dumps(stats, indent=2, default=str)}")

    # Generate with more chord progressions
    progressions = {
        'pop_classic': ['C:maj', 'G:maj', 'A:min', 'F:maj'] * 4,
        'jazz_251': ['D:min7', 'G:7', 'C:maj7', 'C:maj7'] * 4,
        'sad_ballad': ['A:min', 'F:maj', 'C:maj', 'G:maj'] * 4,
    }

    for name, chords in progressions.items():
        gen = generate_melody_forced_chords(
            model, chords, token2id, id2token,
            device=torch.device(args.device), temperature=0.9, top_k=50
        )
        out_path = os.path.join(args.save_dir, f'generated_{name}.mid')
        tokens_to_midi(gen, id2token, out_path)
        s = analyze_generated(gen, id2token)
        print(f"\n{name}: {s['n_notes']} notes, {s['n_bars']} bars, "
              f"notes/bar={s['notes_per_bar']:.1f}, pitch_range={s.get('pitch_range', 'N/A')}")

    print("\n" + "=" * 60)
    print("All files saved to:", args.save_dir)
    print("Done!")


if __name__ == '__main__':
    main()