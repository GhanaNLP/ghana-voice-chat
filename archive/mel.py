"""Wav2Lip's mel spectrogram, reimplemented against a modern librosa.

The upstream `audio.py` calls `librosa.filters.mel(sr, n_fft, n_mels=...)` positionally, which
librosa made keyword-only in 0.10. Rather than pin librosa 0.7 (and with it an old numpy and a
numba build that fights everything else in the image), the handful of functions Wav2Lip's
inference path actually touches are reproduced here.

The numbers below are Wav2Lip's `hparams.py` verbatim. They are not free parameters: the
checkpoint was trained on mels built exactly this way, and a mismatch in `ref_level_db` or
`max_abs_value` shifts the whole input distribution, which shows up as a mouth that moves
plausibly but never quite matches the phonemes -- no error, just bad output.
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy import signal

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 800
HOP_SIZE = 200
WIN_SIZE = 800
FMIN = 55
FMAX = 7600
MIN_LEVEL_DB = -100.0
REF_LEVEL_DB = 20.0
MAX_ABS_VALUE = 4.0
PREEMPHASIS = 0.97

# Wav2Lip feeds the network 16 mel frames per video frame. At hop 200 / 16 kHz that is 80 mel
# frames per second of audio, which is where the 80.0/fps stride in chunking comes from.
MEL_STEP_SIZE = 16
MELS_PER_SECOND = SAMPLE_RATE / HOP_SIZE  # 80.0

_mel_basis: np.ndarray | None = None


def _build_mel_basis() -> np.ndarray:
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return _mel_basis


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """float32 mono waveform at 16 kHz -> (80, T) mel in [-4, 4]."""
    pre = signal.lfilter([1.0, -PREEMPHASIS], [1.0], wav)
    spec = np.abs(librosa.stft(y=pre, n_fft=N_FFT, hop_length=HOP_SIZE, win_length=WIN_SIZE))
    mel = np.dot(_build_mel_basis(), spec)
    db = 20.0 * np.log10(np.maximum(1e-5, mel)) - REF_LEVEL_DB
    # symmetric_mels=True with max_abs_value=4.0
    norm = (2 * MAX_ABS_VALUE) * ((db - MIN_LEVEL_DB) / -MIN_LEVEL_DB) - MAX_ABS_VALUE
    return np.clip(norm, -MAX_ABS_VALUE, MAX_ABS_VALUE).astype(np.float32)


def mel_chunks(mel: np.ndarray, fps: float, n_frames: int | None = None) -> list[np.ndarray]:
    """Slice a mel into one (80, 16) window per output video frame.

    `n_frames` pins the output length so the rendered video matches the audio duration. Upstream
    stops as soon as a full 16-frame window no longer fits, which silently drops up to ~200 ms
    off the end -- fine for its offline demo, but here every segment is a clause the listener
    needs to hear, and ffmpeg's `-shortest` would cut exactly that tail off. The shortfall is
    made up with mel floor (-4.0, i.e. silence) rather than by repeating the last window, so the
    mouth closes at the end of the clause instead of holding the final shape.
    """
    stride = MELS_PER_SECOND / fps
    if n_frames is None:
        n_frames = max(1, int(np.ceil((mel.shape[1] - MEL_STEP_SIZE) / stride)) + 1)

    need = int((n_frames - 1) * stride) + MEL_STEP_SIZE
    if mel.shape[1] < need:
        mel = np.pad(mel, ((0, 0), (0, need - mel.shape[1])),
                     mode="constant", constant_values=-MAX_ABS_VALUE)

    return [mel[:, int(i * stride):int(i * stride) + MEL_STEP_SIZE] for i in range(n_frames)]


def frames_for(n_samples: int, sr: int, fps: float) -> int:
    """How many video frames a waveform of this length should render to."""
    return max(1, int(round(n_samples / sr * fps)))


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """The TTS emits 22.05 kHz (Twi voices are 24 kHz upstream); the mel front-end wants 16 k."""
    wav = np.asarray(wav, dtype=np.float32)
    if sr == SAMPLE_RATE:
        return wav
    return librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
