"""Wav2Lip inference against a fixed avatar loop, plus per-segment MP4 muxing.

Two things here are specific to driving one known avatar rather than an arbitrary uploaded
video, and both matter for a live stream:

*Face detection happens once, offline.* The avatar is a single 10-second clip, so the S3FD pass
and the 96x96 crops are precomputed into a volume by `precompute` in modal_app.py. At request
time there is no detector in the loop at all -- which removes the slowest and least predictable
stage, and means the serving image does not need the detector weights resident.

*The loop ping-pongs, and the caller owns its place in it.* Playing frames 0..N-1 on repeat puts
a visible jump-cut at the wrap, because frame N-1 and frame 0 are unrelated head positions.
Walking forwards then backwards makes the seam continuous. The cursor is passed in and handed
back rather than kept on the object, for two reasons: one container serves several websockets at
once, so a shared cursor would have concurrent turns fighting over the head position; and each
session wants its own continuity, picking up where its own last clause ended instead of snapping
back to the opening pose every time the avatar speaks.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

import mel as melmod

FPS = 25.0          # what the checkpoint was trained at; the source clip is resampled to it
FACE_SIZE = 96      # Wav2Lip's input/output face resolution

# Where the generated face is allowed to replace the original, as a fraction of box height.
# Wav2Lip is handed the face with its bottom half blacked out and generates that half, so only
# the region below MOUTH_SPLIT is genuinely new -- above it the output is the model's own
# reconstruction of pixels we already have at full resolution. Pasting the whole box therefore
# throws away the real eyes, nose and forehead and replaces them with a 96x96 upscale, which is
# the blur that makes Wav2Lip output recognisable at a glance. Pasting only the mouth keeps the
# rest of the face sharp.
MOUTH_SPLIT = 0.52
MOUTH_FEATHER = 0.16   # height of the crossfade band, so no hard horizontal line appears
BOX_FEATHER = 0.10     # taper at the left/right/bottom edges, so the crop outline never shows

_masks: dict[tuple[int, int], np.ndarray] = {}


def blend_mask(h: int, w: int) -> np.ndarray:
    """(h, w, 1) alpha: 0 over the upper face, 1 over the mouth, tapered at the box edges."""
    key = (h, w)
    if key in _masks:
        return _masks[key]

    ys = np.arange(h, dtype=np.float32)
    start = MOUTH_SPLIT * h - MOUTH_FEATHER * h / 2.0
    t = np.clip((ys - start) / max(1e-6, MOUTH_FEATHER * h), 0.0, 1.0)
    m = (t * t * (3.0 - 2.0 * t))[:, None].repeat(w, axis=1)   # smoothstep, not a linear ramp

    ex = max(1, int(BOX_FEATHER * w))
    ey = max(1, int(BOX_FEATHER * h))
    cols = np.arange(w)
    m *= np.clip(np.minimum(cols, w - 1 - cols) / ex, 0.0, 1.0).astype(np.float32)[None, :]
    m *= np.clip((h - 1 - np.arange(h)) / ey, 0.0, 1.0).astype(np.float32)[:, None]

    _masks[key] = m[:, :, None]
    return _masks[key]


class AvatarLoop:
    """Precomputed avatar frames, face boxes and face crops, with a persistent read cursor."""

    def __init__(self, asset_dir: str | Path):
        d = Path(asset_dir)
        meta = json.loads((d / "avatar.json").read_text())
        self.n = int(meta["n_frames"])
        self.width = int(meta["width"])
        self.height = int(meta["height"])
        self.boxes = [tuple(b) for b in meta["boxes"]]          # (y1, y2, x1, x2) per frame

        # Memory-mapped rather than decoded: 241 PNGs cost seconds of cold start, and a cold
        # start happens on the first thing a user says. The full frames are ~300 MB and are
        # touched sequentially as the loop walks, which is the access pattern a volume read
        # handles best. Crops are 6 MB, so those just live in RAM.
        self.frames = np.load(d / "frames.npy", mmap_mode="r")
        self.crops = np.load(d / "crops.npy")
        if len(self.frames) != self.n or len(self.crops) != self.n:
            raise RuntimeError(f"avatar.json says {self.n} frames, arrays disagree")

        self.period = max(1, 2 * (self.n - 1))

    def frame_at(self, i: int) -> int:
        """Map an unbounded step count onto a ping-ponged frame index."""
        if self.n == 1:
            return 0
        j = i % self.period
        return j if j < self.n else self.period - j

    def take(self, cursor: int, count: int) -> tuple[list[int], int]:
        """Frame indices for `count` frames starting at `cursor`, plus the next cursor."""
        return ([self.frame_at(cursor + k) for k in range(count)],
                (cursor + count) % self.period)


class LipSync:
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        import threading

        import torch
        from models import Wav2Lip           # from the pinned Wav2Lip clone on sys.path

        self.torch = torch
        self.device = device
        # The published checkpoints are plain state dicts under a "state_dict" key, but they were
        # saved with training-time module prefixes.
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model = Wav2Lip()
        model.load_state_dict(state)
        self.model = model.to(device).eval()
        self.half = device == "cuda"
        if self.half:
            self.model = self.model.half()
        # TTS, encoding and the Gemini call are where most of a turn's wall-clock goes and they
        # parallelise freely; only the forward pass is serialised.
        self._gpu = threading.Lock()

    def render(self, wav16k: np.ndarray, avatar: AvatarLoop, cursor: int = 0,
               batch_size: int = 64) -> tuple[list[np.ndarray], int]:
        """Waveform (16 kHz float32) -> (full-size BGR frames with the mouth driven, next cursor)."""
        torch = self.torch
        n_frames = melmod.frames_for(len(wav16k), melmod.SAMPLE_RATE, FPS)
        chunks = melmod.mel_chunks(melmod.melspectrogram(wav16k), FPS, n_frames)
        indices, next_cursor = avatar.take(cursor, len(chunks))

        out: list[np.ndarray] = []
        for s in range(0, len(chunks), batch_size):
            sl = slice(s, s + batch_size)
            batch_idx = indices[sl]

            faces = avatar.crops[batch_idx].astype(np.float32)
            masked = faces.copy()
            masked[:, FACE_SIZE // 2:] = 0                       # hide the real mouth
            img = np.concatenate((masked, faces), axis=3) / 255.0  # (B, 96, 96, 6)
            mels = np.stack(chunks[sl])[..., None]                 # (B, 80, 16, 1)

            it = torch.from_numpy(img).permute(0, 3, 1, 2).to(self.device)
            mt = torch.from_numpy(mels).permute(0, 3, 1, 2).to(self.device)
            if self.half:
                it, mt = it.half(), mt.half()

            with self._gpu, torch.inference_mode():
                pred = self.model(mt, it)
            pred = (pred.float().permute(0, 2, 3, 1).cpu().numpy() * 255.0)
            pred = np.clip(pred, 0, 255).astype(np.uint8)

            for k, fi in enumerate(batch_idx):
                frame = np.array(avatar.frames[fi])   # materialise this frame off the mmap
                y1, y2, x1, x2 = avatar.boxes[fi]
                bh, bw = y2 - y1, x2 - x1
                gen = cv2.resize(pred[k], (bw, bh),
                                 interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
                orig = frame[y1:y2, x1:x2].astype(np.float32)
                a = blend_mask(bh, bw)
                frame[y1:y2, x1:x2] = np.clip(gen * a + orig * (1.0 - a), 0, 255).astype(np.uint8)
                out.append(frame)
        return out, next_cursor


def write_wav(path: str | Path, wav: np.ndarray, sr: int) -> None:
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def mux_segment(frames: list[np.ndarray], wav: np.ndarray, sr: int,
                preset: str = "veryfast", crf: int = 20) -> bytes:
    """Frames + audio -> a standalone, immediately-playable MP4.

    `+faststart` puts the moov atom first so the browser can decode from the first byte instead
    of waiting for the whole blob.

    The encoder settings are a real trade, not a default. `ultrafast -crf 23` was visibly
    costing detail in the face -- enough that it, not Wav2Lip, was the limit on how sharp the
    output looked. Rendering a clause runs a little under realtime, so there is headroom to
    spend: `veryfast -crf 20` uses some of it and still finishes comfortably inside the
    clause's own duration, which is the constraint that actually matters. The encoder must
    never become the thing that gates the stream.
    """
    if not frames:
        raise ValueError("no frames to mux")
    h, w = frames[0].shape[:2]

    with tempfile.TemporaryDirectory() as td:
        apath = Path(td) / "seg.wav"
        vpath = Path(td) / "seg.mp4"
        write_wav(apath, wav, sr)

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
            "-i", str(apath),
            "-c:v", "libx264", "-preset", preset, "-tune", "zerolatency",
            "-pix_fmt", "yuv420p", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "96k", "-ar", "44100",
            "-movflags", "+faststart", "-shortest", str(vpath),
        ]
        p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames),
                           capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {p.stderr.decode()[:500]}")
        return vpath.read_bytes()


def idle_clip(avatar: AvatarLoop, seconds: float = 8.0) -> bytes:
    """A silent, seamlessly loopable clip of the avatar just breathing and moving.

    The frontend loops this whenever the avatar is not speaking, so the portrait never freezes
    into a still. It is a full ping-pong cycle, so its last frame is adjacent to its first.
    """
    count = min(avatar.period, max(1, int(seconds * FPS)))
    frames = [np.array(avatar.frames[avatar.frame_at(i)]) for i in range(count)]
    silence = np.zeros(int(count / FPS * melmod.SAMPLE_RATE), dtype=np.float32)
    # Encoded once at precompute time and then served to every visitor on repeat, so this one
    # can afford a slow preset and a low crf.
    return mux_segment(frames, silence, melmod.SAMPLE_RATE, preset="slow", crf=17)
