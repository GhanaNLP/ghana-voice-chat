# Parked: the Wav2Lip talking-head pipeline

This is the video version, kept because it worked *server side* — `/say` produced correct,
lip-synced MP4s at 0.4–0.75x realtime, and the websocket delivered them. What failed was
playback in the browser: the page received the reply text and the segment blobs, but no
`<video>` element ever started playing, and that was still undiagnosed when the project moved
to audio.

If you come back to it:

- `modal_app.py` is the old Modal-hosted backend, kept for reference. It billed continuously for
  idle containers, so it's no longer used; the local FastAPI app (`app.py`) and the Docker Space
  replaced it.
- `modal_app_wav2lip.py` is the full Modal app (L4 GPU, `precompute` + `TalkingHead`).
- `lipsync.py` holds the avatar frame store, the batched Wav2Lip forward, the feathered
  mouth-only blend, and MP4 muxing.
- `mel.py` is Wav2Lip's mel front-end rewritten against a modern librosa.
- The project README's "Things that will bite" section documents the traps that cost real
  time: the `africa-g2p` pin, librosa's ~22 s numba JIT, Modal's 2 MiB websocket cap,
  `from __future__ import annotations` breaking FastAPI websocket injection, and the
  watermark crop.

The unexplained part is browser playback. Start by checking whether `play()` rejects on a
blob-URL MP4 with audio, and whether the segment arrives as an ArrayBuffer the `Blob`
constructor accepts.
