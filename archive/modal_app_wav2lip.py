"""Ghana Talking Head -- a Twi-speaking avatar served from Modal.

One turn of the conversation:

    mic audio (webm/opus)  ->  ffmpeg -> 16 kHz wav
                           ->  Gemini: understand English *or* Twi, answer in Twi, <= 2 sentences
                           ->  stable-twi-tts (ONNX, CPU): Twi text -> 22.05 kHz speech
                           ->  Wav2Lip (GPU): drive the avatar's mouth from that speech
                           ->  one MP4 per sentence, pushed over the websocket as it is ready

The model and the web server share a container. Modal can run an ASGI app as a method on a GPU
class, so the websocket, the ONNX voice and the Wav2Lip weights all sit in one process -- the
alternative (web frontend calling a GPU function) would put a container hop in the middle of
every segment, which is the one place in this design that cannot afford one.

Two components are deliberately off the GPU. The TTS is ONNX on CPU because it runs many times
realtime there and would only contend with Wav2Lip for the device. And face detection does not
run at request time at all: the avatar is a single fixed clip, so `precompute` runs S3FD over its
241 frames once and stores the boxes: see lipsync.py.

Deploy:
    modal run  modal_app.py::precompute     # once: fetch weights, build the avatar cache
    modal deploy modal_app.py
"""
from __future__ import annotations

import modal

APP_NAME = "ghana-talking-head"

# Pinned, not tracking master: the repo supplies the Wav2Lip module definition that has to match
# the checkpoint's state dict, so a silent upstream refactor would break loading.
WAV2LIP_SHA = "bac9a81e63ecc153202353372e5724b83d9e6322"
WAV2LIP_GAN_URL = ("https://huggingface.co/camenduru/Wav2Lip/resolve/main"
                   "/checkpoints/wav2lip_gan.pth")
S3FD_URL = ("https://huggingface.co/camenduru/Wav2Lip/resolve/main"
            "/checkpoints/s3fd-619a316812.pth")

GEMINI_MODEL = "gemini-2.5-flash"
# Modal caps a single websocket message at 2 MiB. Segments are normally a few hundred KB, but a
# long clause can approach the ceiling, and exceeding it would drop the frame rather than fail
# loudly -- so oversized renders are split at frame boundaries before they are sent.
MAX_WS_BYTES = 1_800_000
# Lowest measured Twi-only phoneme error in the published set (27%, tied with twi-7).
TTS_VOICE = "twi-6"

# The source clip was rendered by MagicHour, which burns a logo into the bottom-right corner.
# Cropping it off is not cosmetic housekeeping: the watermark would otherwise be in every frame
# of a public demo, implying a tool endorsement nobody agreed to. 70 px clears it with margin
# and leaves the head and shoulders untouched (the face box sits around y 60-280).
WATERMARK_CROP_PX = 70

ASSETS = "/assets"
AVATAR_DIR = f"{ASSETS}/avatar"
CKPT = f"{ASSETS}/models/wav2lip_gan.pth"
TTS_CACHE = f"{ASSETS}/tts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "curl")
    # cu124 wheels from PyPI; 2.5.1 still defaults torch.load to weights_only=False, which the
    # published Wav2Lip and S3FD pickles need.
    .pip_install("torch==2.5.1", "torchvision==0.20.1")
    .pip_install(
        "numpy==1.26.4",                    # librosa, opencv and onnxruntime all agree here
        "opencv-python-headless==4.10.0.84",
        "librosa==0.11.0",
        "scipy==1.14.1",
        "tqdm==4.67.1",                     # imported by face_detection/detection/core.py
        # Twi front-end only. The `eng` extra bundles espeak-ng, which is GPL-3.0, and this
        # avatar answers in Twi -- so the obligation buys nothing.
        "stable-twi-tts[twi]==0.2.1",
        "ghana-g2p==0.1.1",
        # Pinned, and the pin is load-bearing. ghana-g2p only asks for `africa-g2p>=0.1.1`, so
        # a fresh build picks up 0.2.0, which writes affricates with a combining tie bar --
        # 'd͡ʑ' (U+0361) where this checkpoint's symbol table has 'dʑ', and 't͡ɕʰ' where it has
        # 'tɕʰ'. Those are the sounds Twi spells `gy` and `ky`, so with 0.2.0 installed almost
        # every real Twi sentence dies in the front-end with PhonemeError and the avatar has
        # nothing to say. 0.1.1 is the inventory the model was trained on.
        "africa-g2p==0.1.1",
        "google-genai==2.22.0",
        "fastapi==0.115.6",
        "websockets==14.1",
    )
    .run_commands(
        f"git clone https://github.com/Rudrabha/Wav2Lip /opt/wav2lip"
        f" && cd /opt/wav2lip && git checkout -q {WAV2LIP_SHA}",
        # face_detection loads this path if it exists and otherwise reaches for a long-dead
        # university URL, so placing it here is what keeps the detector working at all.
        f"curl -fsSL -o /opt/wav2lip/face_detection/detection/sfd/s3fd.pth '{S3FD_URL}'",
    )
    .env({"PYTHONPATH": "/opt/wav2lip", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file("assets/avatar_src.mp4", "/avatar_src.mp4")
    .add_local_python_source("mel", "lipsync")
)

# FastAPI resolves endpoint annotations against the *module* globals, and this file uses
# `from __future__ import annotations`, so every annotation is a string by the time FastAPI
# looks at it. Importing FastAPI inside the method that builds the app therefore left it unable
# to resolve `sock: WebSocket`, and an unresolvable annotation is treated as a query parameter:
# every websocket connect closed with 1008 "Field required: query.sock" before reaching the
# handler. Hoisting these into module scope is what makes the annotations resolvable.
with image.imports():
    from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("gth-assets", create_if_missing=True)
gemini_secret = modal.Secret.from_name("gemini-api-key")

SYSTEM_PROMPT = """\
You are a warm, plain-spoken Ghanaian person appearing as a talking-head video avatar. Someone \
is speaking to you out loud.

The person may speak to you in Twi (Akan) or in English, or mix the two. Understand whichever \
they used.

Your reply must obey all of these:
- Reply in Twi ONLY. Never reply in English, no matter which language they spoke.
- Use proper Twi orthography, including the letters ɛ and ɔ where they belong.
- Do not put English words in your reply. If a concept has no everyday Twi word, describe it in \
plain Twi instead of borrowing the English term.
- At most 2 sentences, and keep them short. Your reply is going to be spoken aloud, so write it \
the way someone would actually say it, not the way it would be written down.
- No emoji, no markdown, no bullet points, no parentheses, no quotation marks.

Return JSON with two fields:
  "transcript": what the person said, verbatim, in the language they said it in.
  "reply": your Twi reply.
"""


# --------------------------------------------------------------------------- one-off precompute

@app.function(image=image, volumes={ASSETS: volume}, gpu="L4", timeout=3600)
def precompute(force: bool = False):
    """Fetch the weights and turn the avatar clip into a ready-to-serve frame cache.

    Run once (and again only if the avatar clip changes). Everything this writes lands on the
    volume, so the serving containers start with no downloads and no detector.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import cv2
    import numpy as np

    sys.path.insert(0, "/opt/wav2lip")
    import lipsync

    Path(f"{ASSETS}/models").mkdir(parents=True, exist_ok=True)
    Path(AVATAR_DIR).mkdir(parents=True, exist_ok=True)

    if force or not Path(CKPT).exists():
        print("fetching wav2lip_gan.pth (435 MB) ...")
        subprocess.run(["curl", "-fsSL", "-o", CKPT, WAV2LIP_GAN_URL], check=True)
        volume.commit()
    print(f"checkpoint {os.path.getsize(CKPT) / 1e6:.0f} MB")

    # 24 -> 25 fps. Wav2Lip was trained at 25 and the mel stride assumes it; resampling the
    # source once is far safer than telling the mel chunker a different fps, which would leave
    # the mouth running slightly ahead of the audio for the whole clip.
    norm = "/tmp/avatar25.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", "/avatar_src.mp4", "-r", "25", "-an",
                    # crop before scale/encode so the logo never reaches a stored frame
                    "-vf", f"crop=iw:ih-{WATERMARK_CROP_PX}:0:0",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", norm], check=True)

    cap = cv2.VideoCapture(norm)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded from the avatar clip")
    h, w = frames[0].shape[:2]
    print(f"{len(frames)} frames at {w}x{h}, 25 fps")

    # ---- S3FD once over the whole loop ----
    import face_detection

    det = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device="cuda")

    raw, batch = [], 16
    for s in range(0, len(frames), batch):
        chunk = frames[s:s + batch]
        for rect, img in zip(det.get_detections_for_batch(np.array(chunk)), chunk):
            if rect is None:
                raise RuntimeError(f"no face found in frame {len(raw)} -- check the avatar clip")
            raw.append(rect)
        print(f"  detected {len(raw)}/{len(frames)}")

    # Wav2Lip's default pads: 10 px below the detected box, so the chin stays inside the crop.
    pady1, pady2, padx1, padx2 = 0, 10, 0, 0
    boxes = np.array([[max(0, x1 - padx1), max(0, y1 - pady1),
                       min(w, x2 + padx2), min(h, y2 + pady2)]
                      for (x1, y1, x2, y2) in raw], dtype=np.float64)

    # Temporal smoothing over 5 frames. S3FD jitters by a pixel or two between frames, and
    # because the generated mouth is pasted back into the box, that jitter becomes a visible
    # shimmer around the mouth in the output.
    T = 5
    smoothed = boxes.copy()
    for i in range(len(boxes)):
        window = boxes[len(boxes) - T:] if i + T > len(boxes) else boxes[i:i + T]
        smoothed[i] = window.mean(axis=0)

    out_boxes, crops = [], []
    for (x1, y1, x2, y2), img in zip(smoothed.astype(int), frames):
        out_boxes.append([int(y1), int(y2), int(x1), int(x2)])       # lipsync.py's order
        crops.append(cv2.resize(img[y1:y2, x1:x2], (lipsync.FACE_SIZE, lipsync.FACE_SIZE)))

    np.save(f"{AVATAR_DIR}/frames.npy", np.asarray(frames, dtype=np.uint8))
    np.save(f"{AVATAR_DIR}/crops.npy", np.asarray(crops, dtype=np.uint8))
    Path(f"{AVATAR_DIR}/avatar.json").write_text(json.dumps(
        {"n_frames": len(frames), "width": w, "height": h,
         "fps": 25, "boxes": out_boxes}))

    # Bake the idle loop now so a serving container never has to encode it.
    avatar = lipsync.AvatarLoop(AVATAR_DIR)
    Path(f"{AVATAR_DIR}/idle.mp4").write_bytes(lipsync.idle_clip(avatar))

    # The TTS voice is ~80 MB from a GitHub release; pull it onto the volume too.
    from stable_twi_tts import StableTwiTTS
    StableTwiTTS.from_pretrained(cache_dir=TTS_CACHE)

    volume.commit()
    box = out_boxes[0]
    print(f"done. face box on frame 0 (y1,y2,x1,x2) = {box}, "
          f"crop {box[3]-box[2]}x{box[1]-box[0]} -> 96x96")
    return {"frames": len(frames), "size": [w, h]}


# ------------------------------------------------------------------------------------ serving

class WebsocketCloseShim:
    """Coerce `websocket.close` fields to the types Modal's protobuf bridge accepts.

    Modal serialises outgoing ASGI messages straight into protobuf, and for a close frame it
    passes `reason` into a string field and `code` into a uint32 without checking either. A
    `reason` that is not a str raises `TypeError: bad argument type for built-in operation`
    inside Modal's own serialiser, which kills the connection before the handshake completes --
    so the browser sees only an opaque HTTP 500 and no route, handler or log of ours is
    implicated. Normalising the two fields here makes every close frame serialisable.

    RFC 6455's 1000 (normal closure) and an empty reason are the right defaults.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "websocket":
            return await self.app(scope, receive, send)

        async def guarded(message):
            if message.get("type") == "websocket.close":
                code, reason = message.get("code"), message.get("reason")
                if not isinstance(code, int) or isinstance(code, bool) or not isinstance(reason, str):
                    print(f"normalising unserialisable close frame: {message!r}")
                message = dict(message)
                message["code"] = code if isinstance(code, int) and not isinstance(code, bool) else 1000
                message["reason"] = reason if isinstance(reason, str) else ""
            await send(message)

        return await self.app(scope, receive, guarded)


REPLY_SCHEMA = {
    "type": "OBJECT",
    "properties": {"transcript": {"type": "STRING"}, "reply": {"type": "STRING"}},
    "required": ["transcript", "reply"],
}


@app.cls(
    image=image,
    gpu="L4",
    volumes={ASSETS: volume},
    secrets=[gemini_secret],
    # Deliberately 0. Set to 1 to keep an L4 resident and take the ~40 s cold start off the
    # first thing a user says -- but that is a GPU billed around the clock whether or not anyone
    # is talking to the avatar, so it should be a decision, not a default.
    min_containers=0,
    max_containers=4,
    scaledown_window=300,
    timeout=3600,
)
@modal.concurrent(max_inputs=8)
class TalkingHead:
    @modal.enter()
    def load(self):
        import os
        import sys
        import threading

        sys.path.insert(0, "/opt/wav2lip")
        import lipsync
        from google import genai
        from stable_twi_tts import StableTwiTTS

        self.lipsync_mod = lipsync
        self.avatar = lipsync.AvatarLoop(AVATAR_DIR)
        self.model = lipsync.LipSync(CKPT, device="cuda")
        # 4 ONNX threads is plenty: a two-sentence reply is well under a second of CPU, and
        # leaving cores free matters more, since ffmpeg encodes segments alongside it.
        self.tts = StableTwiTTS.from_pretrained(cache_dir=TTS_CACHE, quiet=True, num_threads=4)
        self.gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.idle = open(f"{AVATAR_DIR}/idle.mp4", "rb").read()
        self.tts_lock = threading.Lock()
        self._warm()
        print(f"ready: {self.avatar.n} avatar frames, "
              f"{self.avatar.width}x{self.avatar.height}, voice {TTS_VOICE}")

    def _warm(self):
        """Pay the one-off costs now instead of inside somebody's first sentence.

        Measured: the first render through a fresh container took ~26 s against ~1.8 s for every
        one after it, and warming only the GPU and the voice barely dented it. The cost is not
        CUDA -- it is numba. librosa JIT-compiles the kernels behind `stft` and `resample` on
        first use, and the mel front-end in mel.py is the only thing that calls them, so nothing
        else in startup triggers it.

        So this renders one throwaway clause through the real `segment` path rather than poking
        the pieces individually: it is the only way to be sure every lazily-initialised stage
        (numba, cuDNN, the ONNX voice, libx264) is actually hot, and it cannot drift out of date
        as the pipeline changes. The dummy forwards come first because cuDNN caches per input
        shape, and a full 64-frame batch is a shape a short warmup clause would never produce.
        """
        import time

        import numpy as np

        t = time.time()
        self.warm_seconds = None
        size = self.lipsync_mod.FACE_SIZE
        for n in (64, 8):
            faces = np.zeros((n, size, size, 6), dtype=np.float32)
            mels = np.zeros((n, 80, 16, 1), dtype=np.float32)
            it = self.model.torch.from_numpy(faces).permute(0, 3, 1, 2).to("cuda").half()
            mt = self.model.torch.from_numpy(mels).permute(0, 3, 1, 2).to("cuda").half()
            with self.model.torch.inference_mode():
                self.model.model(mt, it)
        self.model.torch.cuda.synchronize()

        try:
            self.segment("Akwaaba.", 0)
        except Exception as exc:      # never let warmup stop the container from serving
            print(f"warmup render failed ({type(exc).__name__}: {exc}) -- serving anyway")

        self.warm_seconds = round(time.time() - t, 1)
        print(f"warmup took {self.warm_seconds}s")

    # ---------------------------------------------------------------- pipeline stages

    def _to_wav16k(self, blob: bytes) -> bytes:
        """Whatever the browser recorded -> 16 kHz mono wav, which is what Gemini takes.

        MediaRecorder gives webm/opus on Chrome and mp4/aac on Safari, and Gemini accepts
        neither container reliably, so this is not optional.
        """
        import subprocess

        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0",
             "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
            input=blob, capture_output=True)
        if p.returncode != 0 or not p.stdout:
            raise RuntimeError(f"could not decode the recording: {p.stderr.decode()[:300]}")
        return p.stdout

    def _ask_gemini(self, wav_bytes: bytes, history: list[dict]) -> dict:
        """Audio in, {transcript, reply} out -- one call does the listening and the answering.

        Doing it in a single multimodal call rather than transcribe-then-generate halves the
        round trips, and matters more than that for Twi: a separate ASR step would have to be
        good at Twi to hand the model anything useful, whereas here the audio reaches the model
        directly. Thinking is switched off because this is a voice loop and the budget would
        show up as dead air on the avatar's face.
        """
        import json

        from google.genai import types

        contents = []
        for turn in history[-6:]:
            contents.append(types.Content(role="user",
                                          parts=[types.Part.from_text(text=turn["user"])]))
            contents.append(types.Content(role="model",
                                          parts=[types.Part.from_text(text=turn["reply"])]))
        contents.append(types.Content(role="user", parts=[
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")]))

        resp = self.gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=REPLY_SCHEMA,
                temperature=0.8,
                max_output_tokens=400,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        out = json.loads(resp.text)
        return {"transcript": (out.get("transcript") or "").strip(),
                "reply": (out.get("reply") or "").strip()}

    def _sentences(self, text: str) -> list[str]:
        """Split the reply so the first clause can start playing while the second still renders."""
        from stable_twi_tts.g2p import split_sentences

        parts = [s.strip() for s in split_sentences(text) if s.strip()]
        return parts or ([text.strip()] if text.strip() else [])

    def _speak(self, text: str) -> tuple["object", int]:
        with self.tts_lock:
            s = self.tts.synthesize(text, voice=TTS_VOICE, language="twi")
        return s.audio, s.sample_rate

    def _render(self, text: str, cursor: int):
        """One clause -> (frames, audio, sample rate, next cursor, partial timings)."""
        import time

        import mel as melmod

        t0 = time.time()
        audio, sr = self._speak(text)
        t1 = time.time()
        wav16 = melmod.resample_to_16k(audio, sr)
        frames, cursor = self.model.render(wav16, self.avatar, cursor)
        t2 = time.time()
        return frames, audio, sr, cursor, {"tts": round(t1 - t0, 2),
                                           "lipsync": round(t2 - t1, 2), "_t0": t0}

    def _finish(self, timings: dict, encode_start: float, duration: float) -> dict:
        """Fill in the encode and ratio figures once muxing is done.

        The ratio is the number that matters for streaming: rendering a clause has to outrun
        playing one, or the queue drains and the avatar stalls mid-reply. Reporting it on every
        segment means the cause is visible the moment playback starts stuttering.
        """
        import time

        now = time.time()
        total = now - timings.pop("_t0")
        timings["encode"] = round(now - encode_start, 2)
        timings["total"] = round(total, 2)
        timings["realtime_ratio"] = round(total / max(duration, 1e-6), 2)
        return timings

    def segment(self, text: str, cursor: int) -> tuple[bytes, float, int, dict]:
        """One clause -> (a single mp4, duration, next avatar cursor, per-stage timings)."""
        import time

        frames, audio, sr, cursor, timings = self._render(text, cursor)
        t = time.time()
        # Muxed against the TTS's own sample rate, not the 16 k the mel front-end needed --
        # downsampling is for the lip model's benefit, and the listener should get full quality.
        data = self.lipsync_mod.mux_segment(frames, audio, sr)
        duration = len(frames) / self.lipsync_mod.FPS
        return data, duration, cursor, self._finish(timings, t, duration)

    def segment_pieces(self, text: str, cursor: int) -> tuple[list, int, dict]:
        """One clause -> ([(mp4, duration), ...], next cursor, timings), each under the cap."""
        import math
        import time

        L = self.lipsync_mod
        frames, audio, sr, cursor, timings = self._render(text, cursor)
        t = time.time()
        data = L.mux_segment(frames, audio, sr)

        if len(data) <= MAX_WS_BYTES:
            pieces = [(data, len(frames) / L.FPS)]
        else:
            # Re-mux as equal runs of frames. Splitting on a frame boundary keeps audio and
            # video aligned within each piece, and the pieces then play back-to-back through
            # the same queue the sentences already use.
            n = math.ceil(len(data) / MAX_WS_BYTES)
            per = math.ceil(len(frames) / n)
            print(f"segment is {len(data)} bytes, over the {MAX_WS_BYTES} cap: splitting into {n}")
            pieces = []
            for begin in range(0, len(frames), per):
                chunk = frames[begin:begin + per]
                a0 = int(begin / L.FPS * sr)
                a1 = int((begin + len(chunk)) / L.FPS * sr)
                pieces.append((L.mux_segment(chunk, audio[a0:a1], sr), len(chunk) / L.FPS))

        return pieces, cursor, self._finish(timings, t, sum(d for _, d in pieces))


    # ---------------------------------------------------------------------- web surface

    @modal.asgi_app()
    def web(self):
        import asyncio
        import json
        import traceback

        api = FastAPI(title="Ghana Talking Head")
        # The frontend is a static Hugging Face Space on a different origin, so the browser will
        # preflight everything here.
        api.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"https://.*\.hf\.space|https://huggingface\.co"
                               r"|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?",
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/health")
        def health():
            L = self.lipsync_mod
            return {"ok": True, "frames": self.avatar.n, "voice": TTS_VOICE,
                    "size": [self.avatar.width, self.avatar.height],
                    "fps": L.FPS, "model": GEMINI_MODEL,
                    # Reported so it is possible to tell from outside which code a container is
                    # actually running -- after a redeploy, an old container can still be
                    # serving, and that is otherwise invisible.
                    "blend": {"split": L.MOUTH_SPLIT, "feather": L.MOUTH_FEATHER},
                    "warm": self.warm_seconds}

        @api.get("/idle.mp4")
        def idle():
            """The silent breathing loop the frontend plays whenever the avatar is not talking."""
            return Response(self.idle, media_type="video/mp4",
                            headers={"Cache-Control": "public, max-age=86400"})

        @api.get("/say")
        async def say(text: str):
            """Text -> one MP4. No mic and no Gemini, so it isolates TTS + Wav2Lip when debugging.

            A query parameter rather than a form field: it keeps python-multipart out of the
            image, and `curl` can hit it directly.
            """
            data, dur, _, timings = await asyncio.to_thread(self.segment, text[:400], 0)
            print(f"/say {timings} for {dur:.2f}s of video")
            return Response(data, media_type="video/mp4", headers={
                "X-Timings": json.dumps(timings), "X-Duration": f"{dur:.3f}"})

        @api.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            # Per-connection, so concurrent callers neither share conversation history nor
            # fight over the avatar's head position.
            cursor, history = 0, []

            async def send_segment(index: int, text: str):
                nonlocal cursor
                pieces, cursor, timings = await asyncio.to_thread(
                    self.segment_pieces, text, cursor)
                print(f"segment {index}: {len(pieces)} piece(s) {timings}")
                for part, (data, dur) in enumerate(pieces):
                    await sock.send_text(json.dumps(
                        {"type": "segment", "index": index, "part": part,
                         "parts": len(pieces), "text": text if part == 0 else "",
                         "duration": round(dur, 3), "bytes": len(data),
                         "timings": timings if part == 0 else None}))
                    await sock.send_bytes(data)

            try:
                while True:
                    msg = await sock.receive()
                    if msg["type"] == "websocket.disconnect":
                        break

                    if msg.get("bytes"):
                        await sock.send_text(json.dumps({"type": "status", "stage": "listening"}))
                        wav = await asyncio.to_thread(self._to_wav16k, msg["bytes"])
                        await sock.send_text(json.dumps({"type": "status", "stage": "thinking"}))
                        turn = await asyncio.to_thread(self._ask_gemini, wav, history)
                        await sock.send_text(json.dumps(
                            {"type": "transcript", "text": turn["transcript"]}))
                        await sock.send_text(json.dumps({"type": "reply", "text": turn["reply"]}))
                        if not turn["reply"]:
                            await sock.send_text(json.dumps({"type": "done", "segments": 0}))
                            continue
                        history.append({"user": turn["transcript"] or "(audio)",
                                        "reply": turn["reply"]})
                        reply = turn["reply"]

                    elif msg.get("text"):
                        req = json.loads(msg["text"])
                        if req.get("type") == "ping":
                            await sock.send_text(json.dumps({"type": "pong"}))
                            continue
                        # "say" speaks given text directly, bypassing mic and Gemini.
                        reply = (req.get("text") or "").strip()[:400]
                        if not reply:
                            continue
                        await sock.send_text(json.dumps({"type": "reply", "text": reply}))
                    else:
                        continue

                    sent = 0
                    for i, sentence in enumerate(self._sentences(reply)):
                        try:
                            await send_segment(i, sentence)
                            sent += 1
                        except Exception as exc:
                            # A clause the Twi front-end cannot pronounce should cost that
                            # clause, not the rest of the reply.
                            traceback.print_exc()
                            await sock.send_text(json.dumps(
                                {"type": "warning", "index": i,
                                 "detail": f"{type(exc).__name__}: {exc}"[:200]}))
                    await sock.send_text(json.dumps({"type": "done", "segments": sent}))

            except WebSocketDisconnect:
                pass
            except Exception as exc:
                traceback.print_exc()
                try:
                    await sock.send_text(json.dumps(
                        {"type": "error", "detail": f"{type(exc).__name__}: {exc}"[:300]}))
                except Exception:
                    pass

        return WebsocketCloseShim(api)
