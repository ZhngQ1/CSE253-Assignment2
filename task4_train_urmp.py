"""
Task 4 — Train MIDI-DDSP on real URMP violin, then synthesize Task 2 melodies.
==============================================================================
Uses the DDSP-preprocessed URMP TFRecords (parsed by urmp_tfrecord.py) instead
of raw AuSep_*.wav files, so no TensorFlow / raw-URMP download is needed.

Pipeline (Task 4 — continuous conditioned generation):
    MIDI conditioning (f0 / velocity / onset, 250 fps)
        -> MidiEncoder -> DDSPDecoder
        -> differentiable additive + filtered-noise synth
        -> 16 kHz waveform
trained against the real violin audio with a multi-scale spectral loss.

End-to-end demo: feed Task 2's generated melody MIDI through the trained
synthesizer to render audio (the `continuous_conditioned` deliverable).

Run:
    python task4_train_urmp.py --epochs 80 --device mps
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from task4_midi2audio_ddsp_baseline import (
    MidiDDSP, MultiScaleSpectralLoss, synthesize_midi,
    SAMPLE_RATE, FRAME_RATE, N_HARMONICS, N_NOISE_BANDS,
)
from urmp_tfrecord import URMPTFRecordDataset


def _save_wav(path, audio_np, sr=SAMPLE_RATE):
    try:
        import soundfile as sf
        sf.write(path, audio_np, sr)
    except ImportError:
        from scipy.io import wavfile
        wavfile.write(path, sr, (np.clip(audio_np, -1, 1) * 32767).astype(np.int16))


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        f0 = batch['f0_hz'].to(device)
        vel = batch['velocity'].to(device)
        on = batch['onset'].to(device)
        target = batch['audio'].to(device)

        with torch.set_grad_enabled(train):
            pred, _ = model(f0, vel, on)
            m = min(pred.shape[-1], target.shape[-1])
            loss = criterion(pred[:, :m], target[:, :m])

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_glob', type=str, default='URMP_tf/*train*')
    ap.add_argument('--test_glob', type=str, default='URMP_tf/*test*')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--segments_per_record', type=int, default=16)
    ap.add_argument('--device', type=str,
                    default='mps' if torch.backends.mps.is_available()
                    else ('cuda' if torch.cuda.is_available() else 'cpu'))
    ap.add_argument('--save_dir', type=str, default='./checkpoints_task4')
    ap.add_argument('--test_midi', type=str,
                    default='checkpoints_task2_optimized/generated_melody.mid')
    ap.add_argument('--resume', type=str, default=None,
                    help='checkpoint to warm-start from')
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 64)
    print("Task 4: MIDI -> Audio (DDSP) — training on real URMP violin")
    print("=" * 64)
    print(f"device={device}  sr={SAMPLE_RATE}  fps={FRAME_RATE}  "
          f"harmonics={N_HARMONICS}  noise_bands={N_NOISE_BANDS}")

    train_ds = URMPTFRecordDataset(args.train_glob,
                                   segments_per_record=args.segments_per_record)
    test_ds = URMPTFRecordDataset(args.test_glob,
                                  segments_per_record=4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    model = MidiDDSP(d_model=args.d_model).to(device)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        print(f"Warm-started from {args.resume} (epoch {ck.get('epoch')})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    criterion = MultiScaleSpectralLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = float('inf')
    print("-" * 64)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, test_loader, criterion, device, None)
        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | train {tr:.4f} | val {va:.4f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e} | {dt:.1f}s")

        if va < best:
            best = va
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': va, 'instrument': 'vn',
                        'dataset': 'URMP_vn_solo'},
                       os.path.join(args.save_dir, 'best_model_urmp_vn.pt'))

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                b = next(iter(test_loader))
                pred, _ = model(b['f0_hz'][:1].to(device),
                                b['velocity'][:1].to(device),
                                b['onset'][:1].to(device))
                _save_wav(os.path.join(args.save_dir, f'vn_recon_epoch{epoch}.wav'),
                          pred[0].cpu().numpy())
                if epoch == 1:
                    _save_wav(os.path.join(args.save_dir, 'vn_target_ref.wav'),
                              b['audio'][0].cpu().numpy())

    print("-" * 64)
    print(f"Done. best val loss = {best:.4f}")
    print(f"saved: {os.path.join(args.save_dir, 'best_model_urmp_vn.pt')}")

    # End-to-end: Task 2 MIDI -> violin audio
    ckpt = os.path.join(args.save_dir, 'best_model_urmp_vn.pt')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    if args.test_midi and os.path.exists(args.test_midi):
        out = os.path.join(args.save_dir, 'continuous_conditioned.wav')
        print(f"\nEnd-to-end: synthesizing {args.test_midi} -> {out}")
        synthesize_midi(model, args.test_midi, out, device)


if __name__ == '__main__':
    main()
