"""
Pure-Python reader for the Magenta URMP DDSP-conditioning TFRecords.
=====================================================================
The Magenta URMP release (gs://magentadata/datasets/urmp/urmp_20210324/)
stores each recording as a serialized tf.train.Example. We do NOT want a
TensorFlow dependency, so this module parses the protobuf wire format
directly.

Each record (one solo recording, ~35s) contains, among others:
    audio                 float[N]   16 kHz waveform
    f0_hz                 float[T]   250 fps fundamental frequency (0 = unvoiced)
    loudness_db           float[T]   250 fps A-weighted loudness
    f0_confidence         float[T]
    note_onsets           float[2N]  sample-level onset impulses (0/1)
    note_offsets          float[2N]  sample-level offset impulses (0/1)
    note_active_velocities float[2N] sample-level velocity (0..~0.8)
    instrument_id         bytes      e.g. b'vn'
    recording_id          bytes

Audio is 16 kHz, conditioning is 250 fps -> hop = 64 samples, matching the
Task-4 model's SAMPLE_RATE / FRAME_RATE constants.

We expose a Dataset that yields the same dict the Task-4 model expects:
    {'audio', 'f0_hz', 'velocity', 'onset'}  (all torch tensors)
where f0/velocity/onset are at 250 fps and audio is at 16 kHz, cropped to a
fixed training segment.
"""

import os
import glob
import struct
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------
# Minimal TFRecord + protobuf-Example parsing (no TensorFlow needed)
# ------------------------------------------------------------------

def _read_tfrecords(path: str):
    """Yield the raw serialized Example bytes from a .tfrecord file.

    TFRecord framing: uint64 length, uint32 length-crc, payload, uint32 crc.
    We skip CRC validation (data is from a trusted GCS bucket).
    """
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            length = struct.unpack('<Q', hdr)[0]
            f.read(4)                 # length crc
            data = f.read(length)
            f.read(4)                 # data crc
            if len(data) < length:
                break
            yield data


def _parse_varint(b: bytes, i: int):
    shift = 0
    result = 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7f) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result, i


def _parse_fields(b: bytes) -> Dict[int, list]:
    """Parse a protobuf message into {tag: [(wire_type, value), ...]}."""
    i = 0
    n = len(b)
    out: Dict[int, list] = {}
    while i < n:
        key, i = _parse_varint(b, i)
        tag = key >> 3
        wire = key & 7
        if wire == 0:                      # varint
            v, i = _parse_varint(b, i)
        elif wire == 2:                    # length-delimited
            ln, i = _parse_varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wire == 5:                    # 32-bit
            v = b[i:i + 4]
            i += 4
        elif wire == 1:                    # 64-bit
            v = b[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.setdefault(tag, []).append((wire, v))
    return out


def _float_list(feature_bytes: bytes) -> np.ndarray:
    """Decode a Feature{FloatList} message into a float32 numpy array.

    Feature   { FloatList float_list = 2 }
    FloatList { repeated float value = 1 [packed]; }
    """
    fm = _parse_fields(feature_bytes)
    if 2 not in fm:
        return np.zeros(0, dtype=np.float32)
    inner = _parse_fields(fm[2][0][1])
    if 1 not in inner:
        return np.zeros(0, dtype=np.float32)
    raw = inner[1][0][1]
    return np.frombuffer(raw, dtype='<f4').astype(np.float32)


def _bytes_value(feature_bytes: bytes) -> bytes:
    """Decode the first value of a Feature{BytesList}."""
    fm = _parse_fields(feature_bytes)
    if 1 not in fm:
        return b''
    inner = _parse_fields(fm[1][0][1])
    if 1 not in inner:
        return b''
    return inner[1][0][1]


def parse_example(record: bytes) -> Dict[str, object]:
    """Parse one serialized tf.train.Example into {feature_name: array/bytes}."""
    ex = _parse_fields(record)
    features_bytes = ex[1][0][1]            # Example { Features features = 1 }
    fmap = _parse_fields(features_bytes)    # Features { map feature = 1 }
    out: Dict[str, object] = {}
    for _wire, entry in fmap.get(1, []):
        e = _parse_fields(entry)            # MapEntry { key=1, value=2 }
        key = e[1][0][1].decode('utf-8', 'replace')
        feat = e[2][0][1]
        fm = _parse_fields(feat)
        if 2 in fm:                         # FloatList
            out[key] = _float_list(feat)
        elif 1 in fm:                       # BytesList
            out[key] = _bytes_value(feat)
        elif 3 in fm:                       # Int64List
            inner = _parse_fields(fm[3][0][1])
            raw = inner[1][0][1] if 1 in inner else b''
            # packed varints -> decode lazily; not needed for our features
            out[key] = raw
    return out


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class URMPTFRecordDataset(Dataset):
    """URMP solo-instrument dataset from Magenta DDSP-conditioning TFRecords.

    Yields fixed-length training segments compatible with the Task-4 MidiDDSP
    model: f0/velocity/onset at `frame_rate` fps, audio at `sample_rate` Hz.

    Each underlying record is one full recording (~30-40s). We pre-load all
    records into memory (a light subset is only a few hundred MB of float32)
    and draw random crops, mirroring the original URMPDataset behaviour.
    """

    def __init__(self, tfrecord_glob: str,
                 segment_seconds: float = 4.0,
                 sample_rate: int = 16000,
                 frame_rate: int = 250,
                 segments_per_record: int = 12):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.hop = sample_rate // frame_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.segment_frames = int(segment_seconds * frame_rate)
        self.segments_per_record = segments_per_record

        if isinstance(tfrecord_glob, (list, tuple)):
            paths: List[str] = []
            for g in tfrecord_glob:
                paths.extend(sorted(glob.glob(g)))
        else:
            paths = sorted(glob.glob(tfrecord_glob))

        self.records = []   # list of dicts with float32 arrays
        n_skipped = 0
        for p in paths:
            if os.path.getsize(p) == 0:
                continue
            for raw in _read_tfrecords(p):
                try:
                    ex = parse_example(raw)
                except Exception:
                    n_skipped += 1
                    continue
                audio = ex.get('audio')
                f0 = ex.get('f0_hz')
                if audio is None or f0 is None or len(audio) == 0 or len(f0) == 0:
                    n_skipped += 1
                    continue
                f0_arr = np.asarray(f0, dtype=np.float32)
                n_frames = len(f0_arr)
                # velocity / onset are at 2x audio sample rate, stored as sparse
                # impulses near frame boundaries -> max-pool to the f0 frame grid
                # once, here, so crops in __getitem__ stay cheap.
                vel = self._pool_to_frames(
                    np.asarray(ex.get('note_active_velocities', np.zeros(0)),
                               dtype=np.float32), n_frames)
                onset = self._pool_to_frames(
                    np.asarray(ex.get('note_onsets', np.zeros(0)),
                               dtype=np.float32), n_frames)
                self.records.append({
                    'audio': np.asarray(audio, dtype=np.float32),
                    'f0_hz': f0_arr,
                    'velocity': np.clip(vel, 0.0, 1.0),
                    'onset': (onset > 0.5).astype(np.float32),
                    'rec_id': ex.get('recording_id', b'').decode('utf-8', 'replace'),
                })
        print(f"URMPTFRecordDataset: loaded {len(self.records)} recordings "
              f"from {len(paths)} shards (skipped {n_skipped})")

    def __len__(self):
        return max(1, len(self.records) * self.segments_per_record)

    def _pool_to_frames(self, arr: np.ndarray, n_total_frames: int) -> np.ndarray:
        """Max-pool a per-sample (2x audio) control array onto the frame grid.

        note_* arrays are at 2x the audio sample rate, so one 250-fps frame
        spans 2*hop = 128 array steps. Velocity/onset values are stored as
        sparse impulses near each frame boundary (not exactly index-aligned),
        so we MAX-POOL over each 128-step window to capture the impulse
        regardless of its sub-window position. Vectorized via reshape.
        """
        step = 2 * self.hop  # = 128
        if len(arr) == 0:
            return np.zeros(n_total_frames, dtype=np.float32)
        usable = (len(arr) // step) * step
        if usable == 0:
            return np.zeros(n_total_frames, dtype=np.float32)
        pooled = arr[:usable].reshape(-1, step).max(axis=1)   # (n_pooled,)
        out = np.zeros(n_total_frames, dtype=np.float32)
        m = min(n_total_frames, len(pooled))
        out[:m] = pooled[:m]
        return out

    def __getitem__(self, idx):
        rec = self.records[idx % len(self.records)]
        audio = rec['audio']
        f0 = rec['f0_hz']
        vel_full = rec['velocity']
        onset_full = rec['onset']

        n_frames_total = len(f0)
        # Random crop in frame space, then map to samples
        if n_frames_total <= self.segment_frames:
            start_f = 0
        else:
            start_f = np.random.randint(0, n_frames_total - self.segment_frames)

        f0_seg = np.zeros(self.segment_frames, dtype=np.float32)
        vel_seg = np.zeros(self.segment_frames, dtype=np.float32)
        on_seg = np.zeros(self.segment_frames, dtype=np.float32)
        avail = min(self.segment_frames, n_frames_total - start_f)
        f0_seg[:avail] = f0[start_f:start_f + avail]
        vel_seg[:avail] = vel_full[start_f:start_f + avail]
        on_seg[:avail] = onset_full[start_f:start_f + avail]

        start_s = start_f * self.hop
        audio_seg = np.zeros(self.segment_samples, dtype=np.float32)
        avail_s = min(self.segment_samples, len(audio) - start_s)
        if avail_s > 0:
            audio_seg[:avail_s] = audio[start_s:start_s + avail_s]

        # Peak-normalize audio segment (model output is peak-normalized too)
        peak = np.abs(audio_seg).max()
        if peak > 1e-6:
            audio_seg = audio_seg / peak * 0.9

        return {
            'audio': torch.from_numpy(audio_seg),
            'f0_hz': torch.from_numpy(f0_seg),
            'velocity': torch.from_numpy(vel_seg),
            'onset': torch.from_numpy(on_seg),
        }
